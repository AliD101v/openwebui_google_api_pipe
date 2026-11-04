"""
title: Google Gemini API Manifold
id: gemini_pipe
author: AliD101v
github: https://github.com/AliD101v/openwebui_google_api_pipe
version: 0.1.0
license: MIT
requirements: google-genai>=1.68.0, pydantic>=2.0.0, cryptography
environment_variables:
    - GEMINI_API_KEY (or GOOGLE_API_KEY)

Supports:
- Fetches the model list from the Gemini API (cached, with capability detection)
- Google Search, URL context and Google Maps grounding with citations
- Function tools with a multi-call loop
- Thinking/reasoning controls (level or budget, per model generation)
- Image generation on image models
- Native file, Files API and YouTube media handling
- Per-user API key override, stored encrypted

Changelog:
v0.1.0
- Initial release.
"""

import asyncio
import base64
import contextvars
import copy
import hashlib
import html
import inspect
import io
import json
import logging
import mimetypes
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse, quote, quote_plus
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# GUARDED IMPORTS
# =============================================================================

# google-genai is the hard requirement of this pipe.
try:
    from google import genai
    from google.genai import types as gtypes
    from google.genai import errors as genai_errors

    GENAI_AVAILABLE = True
    GENAI_VERSION = getattr(genai, "__version__", "")
except ImportError:
    genai = None
    gtypes = None
    genai_errors = None
    GENAI_AVAILABLE = False
    GENAI_VERSION = ""

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

# Import Pillow for media conversion (GIF/BMP/TIFF/AVIF -> PNG).
# Optional: degrade gracefully (skip the conversion) if absent.
try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except Exception:
    PILImage = None
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)
# The SDK and the transport log request headers and bodies at DEBUG; under
# GLOBAL_LOG_LEVEL=DEBUG that puts chat content and the API key into the log.
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =============================================================================
# COMPILED REGEX PATTERNS
# =============================================================================

# Invisible carrier for per-message state (file ids, media anchors, ...). A
# markdown empty link [](gemini:...) renders as nothing, while still
# round-tripping the payload through OpenWebUI storage.
PATTERN_METADATA_MARKER = r"[ ]?\[\]\(gemini:([^)]+)\)[ ]?"

# Patterns to extract memories injected by the OpenWebUI Memory System into the
# system prompt. Both forms are volatile: OpenWebUI re-retrieves and re-ranks
# the memories per request, so leaving them in `system` would change the
# prompt on every turn. They are relocated to the last user message instead.
#
# Legacy form: everything after "\nUser Context:\n" to end of string.
PATTERN_USER_CONTEXT = re.compile(r"\nUser Context:\n(.*)$", flags=re.DOTALL)
# Current form (utils/memory.py): a <memory_context> element, appended or
# prepended to the system message depending on OpenWebUI version.
PATTERN_MEMORY_CONTEXT = re.compile(
    r"<memory_context>(.*?)</memory_context>", flags=re.DOTALL
)

# Header the relocated memories are prefixed with when they are appended to the
# last user message.
MEMORY_CONTEXT_APPENDIX_HEADER = (
    "\n\n---\n**IMPORTANT:** The following is NOT part of the user's message, "
    "but context from a memory system to help answer the user's questions:\n\n"
)

PATTERN_EMPTY_CONTEXT = re.compile(r"<context>\s*</context>", flags=re.DOTALL)

# Pattern to find remaining source tags (for checking if all were removed)
PATTERN_SOURCE_TAGS = re.compile(r"<source[^>]*>.*?</source>", flags=re.DOTALL)

# RAG message detection: matches "### Task:...<context>...</context>" blocks
PATTERN_RAG_MESSAGE = re.compile(r"### Task:.*?<context>.*?</context>", re.DOTALL)

# =============================================================================
# SDK FEATURE FLAGS AND CONSTANTS
# =============================================================================

# Anything newer than google-genai 1.68 must be feature-detected, because the
# pipe supports >=1.68 and proxies may serve older API versions.
GENAI_HAS_SERVER_SIDE_INVOCATIONS = GENAI_AVAILABLE and "include_server_side_tool_invocations" in gtypes.ToolConfig.model_fields
GENAI_HAS_TOOL_CALL_PARTS = GENAI_AVAILABLE and "tool_call" in gtypes.Part.model_fields
GENAI_HAS_STORE = GENAI_AVAILABLE and "store" in gtypes.GenerateContentConfig.model_fields
DUMMY_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"
MODEL_ID_PREFIX = "gemini/"

# Which block concepts the *current* request's user hid via UserValves.HIDE_BLOCKS.
# A ContextVar rather than an attribute on the Pipe instance: OpenWebUI keeps one
# Pipe object for all users, so an attribute would leak one user's preference into
# a concurrently streaming request. Formatters are reached through a long call
# chain that has no request context.
HIDDEN_BLOCKS: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "gemini_pipe_hidden_blocks", default=frozenset()
)

# True while serving an OpenWebUI sub-agent run (request.state.internal). The
# response is then pasted verbatim into the parent agent's context, so every
# decoration is pure token cost there. Same ContextVar reasoning as
# HIDDEN_BLOCKS: one Pipe object serves concurrent requests.
SLIM_OUTPUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gemini_pipe_slim_output", default=False
)

# Human-in-the-loop tool approval. OpenWebUI enforces its own approval gate
# inside its native tool loop, which a manifold never enters, so the pipe
# re-implements it at the single point where a tool coroutine is awaited.
# Holds `(mode, event_call)` for the current request.
TOOL_APPROVAL: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "gemini_pipe_tool_approval", default=("full", None)
)

# =============================================================================
# SECRET VALVE ENCRYPTION
# =============================================================================
# OpenWebUI can encrypt valves at rest itself since 0.10.0, but only when
# ENABLE_VALVE_ENCRYPTION is set -- it defaults to False (see
# open_webui/env.py and open_webui/utils/valves.py). On a default install of any
# version the API key therefore sits in the functions table in plaintext, which
# is what this covers.
#
# The key derivation below matches OpenWebUI's own (_fernet in utils/valves.py)
# on purpose, so both layers behave identically where they overlap. With their
# flag on, the value ends up encrypted twice, which is harmless. Every step here
# is a no-op for values that are not encrypted, so existing plaintext valves
# keep working and nothing has to be migrated.
try:
    from cryptography.fernet import Fernet, InvalidToken

    VALVE_ENCRYPTION_AVAILABLE = True
except ImportError:
    Fernet = None
    InvalidToken = Exception
    VALVE_ENCRYPTION_AVAILABLE = False

from pydantic_core import core_schema

ENCRYPTED_VALVE_PREFIX = "encrypted:"

try:
    from open_webui.env import VERSION as OPENWEBUI_VERSION
except Exception:
    OPENWEBUI_VERSION = "unknown"


def _owui_version_below(target: tuple) -> bool:
    """True when the running OpenWebUI is older than `target`; False if unknown."""
    try:
        parts = tuple(int(p) for p in str(OPENWEBUI_VERSION).split("-")[0].split(".")[:3])
    except ValueError:
        return False
    return parts < target


try:
    from open_webui.models.models import Models, ModelForm

    MODELS_AVAILABLE = True
except ImportError:
    Models = None
    ModelForm = None
    MODELS_AVAILABLE = False

# Whether OpenWebUI encrypts valves at rest itself. Checked instead of comparing
# version numbers: the constant only exists from 0.10.0, and it reflects the
# actual ENABLE_VALVE_ENCRYPTION setting rather than what the release could do.
# Missing (older releases) or False means the key would land in the DB in
# plaintext, so the pipe encrypts it itself.
try:
    from open_webui.env import ENABLE_VALVE_ENCRYPTION as OPENWEBUI_ENCRYPTS_VALVES
except Exception:
    OPENWEBUI_ENCRYPTS_VALVES = False


def _valve_encryption_key() -> Optional[bytes]:
    """Fernet key from WEBUI_SECRET_KEY, derived the same way OpenWebUI does it.

    Returns None when encryption is unavailable or no secret is configured; the
    callers then pass values through unchanged rather than failing.
    """
    if not VALVE_ENCRYPTION_AVAILABLE:
        return None
    secret = os.environ.get("WEBUI_SECRET_KEY", "").strip()
    if not secret:
        return None
    # A 44-char secret may already be a valid Fernet key; use it directly.
    if len(secret) == 44:
        try:
            Fernet(secret.encode())
            return secret.encode()
        except Exception:
            pass
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())


def encrypt_valve_secret(value: str) -> str:
    """Encrypt a valve value. Idempotent, and a no-op without a secret key.

    Skipped when OpenWebUI encrypts valves itself -- doing it twice with the
    same cipher and the same key buys nothing and only adds a way to fail.

    Idempotency is load-bearing: OpenWebUI re-validates the whole Valves model
    every time an admin saves the valve page, so an already-encrypted value is
    handed back in. Encrypting it again each save would nest the ciphertext.
    """
    if not value or value.startswith(ENCRYPTED_VALVE_PREFIX):
        return value
    if OPENWEBUI_ENCRYPTS_VALVES:
        return value
    key = _valve_encryption_key()
    if not key:
        return value
    try:
        return ENCRYPTED_VALVE_PREFIX + Fernet(key).encrypt(value.encode()).decode()
    except Exception as e:
        logger.warning(f"Could not encrypt valve value, storing as-is: {e}")
        return value


def decrypt_valve_secret(value: Any) -> str:
    """Return the plaintext behind a valve value.

    Values without the marker prefix are returned unchanged, which covers
    plaintext valves from installs that predate this, defaults, and setups
    without WEBUI_SECRET_KEY.

    Deliberately not gated on OPENWEBUI_ENCRYPTS_VALVES: a key encrypted while
    that flag was off must stay readable after an admin turns it on.
    """
    text = str(value or "")
    if not text.startswith(ENCRYPTED_VALVE_PREFIX):
        return text
    token = text[len(ENCRYPTED_VALVE_PREFIX) :]
    key = _valve_encryption_key()
    if not key:
        raise ValueError(
            "Stored API key is encrypted but WEBUI_SECRET_KEY is not available. "
            "Restore the original secret key, or re-enter the API key in the valves."
        )
    try:
        return Fernet(key).decrypt(token.encode()).decode()
    except InvalidToken as e:
        raise ValueError(
            "Stored API key could not be decrypted -- WEBUI_SECRET_KEY appears to "
            "have changed. Re-enter the API key in the valves."
        ) from e


class EncryptedStr(str):
    """Valve field type that keeps its value encrypted at rest.

    OpenWebUI persists `Valves(**form_data).model_dump()`, so the value the
    validator returns is what reaches the database.
    """

    @classmethod
    def _validate(cls, value: Any) -> "EncryptedStr":
        return cls(encrypt_valve_secret(str(value or "")))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_after_validator_function(
            cls._validate, core_schema.str_schema()
        )


# Import OpenWebUI builtin tools helper
try:
    from open_webui.utils.tools import get_builtin_tools

    BUILTIN_TOOLS_AVAILABLE = True
except ImportError:
    get_builtin_tools = None
    BUILTIN_TOOLS_AVAILABLE = False

# Import process_tool_result for Rich UI (HTMLResponse, embeds, files)
try:
    from open_webui.utils.middleware import process_tool_result

    PROCESS_TOOL_RESULT_AVAILABLE = True
except ImportError:
    process_tool_result = None
    PROCESS_TOOL_RESULT_AVAILABLE = False

# Import OpenWebUI Files and Storage for native file handling
try:
    from open_webui.models.files import Files, FileForm
    from open_webui.storage.provider import Storage

    FILES_AVAILABLE = True
except ImportError:
    Files = None
    FileForm = None
    Storage = None
    FILES_AVAILABLE = False

# Import OpenWebUI Chats for persisting usage to chat_message table
try:
    from open_webui.models.chats import Chats

    CHATS_AVAILABLE = True
except ImportError:
    Chats = None
    CHATS_AVAILABLE = False

# Import OpenWebUI Users for upload attribution
try:
    from open_webui.models.users import Users

    USERS_AVAILABLE = True
except ImportError:
    Users = None
    USERS_AVAILABLE = False

# Import OpenWebUI image upload for generated images
try:
    from open_webui.routers.images import upload_image

    UPLOAD_IMAGE_AVAILABLE = True
except ImportError:
    upload_image = None
    UPLOAD_IMAGE_AVAILABLE = False

# Import OpenWebUI URL image fetcher for image URLs in the conversation
try:
    from open_webui.utils.files import get_image_base64_from_url

    IMAGE_URL_FETCH_AVAILABLE = True
except ImportError:
    get_image_base64_from_url = None
    IMAGE_URL_FETCH_AVAILABLE = False


# =============================================================================
# STREAM STATE (GENERATED SECTIONS) AND REQUEST CONTEXT
# =============================================================================

# BEGIN GENERATED SECTION: gemini_pipe.response.grounding
def _camel(name: str) -> str:
    """snake_case to camelCase (``start_index`` -> ``startIndex``)."""
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _get(obj: Any, name: str) -> Any:
    """Read an attribute or dict key by snake_case, falling back to camelCase."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        return obj.get(_camel(name))
    value = getattr(obj, name, None)
    if value is None:
        value = getattr(obj, _camel(name), None)
    return value


def _normalize_chunk(chunk: Any) -> Optional[dict]:
    web = _get(chunk, "web")
    if web is not None:
        return {
            "kind": "web",
            "uri": _get(web, "uri"),
            "title": _get(web, "title"),
            "domain": _get(web, "domain"),
            "place_id": None,
        }
    maps = _get(chunk, "maps")
    if maps is not None:
        return {
            "kind": "maps",
            "uri": _get(maps, "uri"),
            "title": _get(maps, "title"),
            "domain": None,
            "place_id": _get(maps, "place_id"),
        }
    return None


def _chunk_key(chunk: dict) -> Optional[str]:
    """A chunk's identity: its uri, falling back to its place id."""
    if chunk.get("uri"):
        return chunk["uri"]
    if chunk.get("place_id"):
        return str(chunk["place_id"])
    return None


@dataclass
class GroundingAccumulator:
    """Cumulative grounding state of one API call (chunks, supports, queries)."""

    chunks: List[dict] = field(default_factory=list)
    supports: List[dict] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    entry_point_html: str = ""

    def add(self, gm: Any) -> None:
        """Merge one GroundingMetadata (SDK object or dict); None is ignored."""
        if gm is None:
            return
        new_chunks = []
        for chunk in _get(gm, "grounding_chunks") or []:
            norm = _normalize_chunk(chunk)
            if norm is not None and _chunk_key(norm) is not None:
                new_chunks.append(norm)
        if new_chunks:
            current_keys = [_chunk_key(c) for c in self.chunks]
            new_keys = [_chunk_key(c) for c in new_chunks]
            if new_keys[: len(current_keys)] == current_keys:
                # A cumulative snapshot of the same stream: replace, don't grow.
                self.chunks = list(new_chunks)
            else:
                # Documented streaming behaviour: chunks arrive incrementally.
                self.chunks.extend(new_chunks)
        seen = {
            (s["start"], s["end"], s["text"], tuple(s["indices"])) for s in self.supports
        }
        for support in _get(gm, "grounding_supports") or []:
            segment = _get(support, "segment")
            entry = {
                "start": _get(segment, "start_index"),
                "end": _get(segment, "end_index"),
                "text": _get(segment, "text") or "",
                "indices": list(_get(support, "grounding_chunk_indices") or []),
            }
            key = (entry["start"], entry["end"], entry["text"], tuple(entry["indices"]))
            if key not in seen:
                seen.add(key)
                self.supports.append(entry)
        for query in _get(gm, "web_search_queries") or []:
            if query and query not in self.queries:
                self.queries.append(query)
        rendered = _get(_get(gm, "search_entry_point"), "rendered_content")
        if rendered:
            self.entry_point_html = rendered

    def empty(self) -> bool:
        return not self.chunks and not self.queries


def locate_support_end(text_bytes: bytes, support: dict) -> Optional[int]:
    """The byte offset where a support's segment actually ends in ``text_bytes``.

    The API's end_index can be off by a few bytes, so a support that carries
    its segment text is re-located; a text-less one is snapped forward over the
    tail of a multi-byte UTF-8 sequence it may land inside.
    """
    text = support.get("text") or ""
    end = support.get("end")
    if text:
        needle = text.encode("utf-8")
        ends = []
        start = 0
        while True:
            index = text_bytes.find(needle, start)
            if index == -1:
                break
            ends.append(index + len(needle))
            start = index + 1
        if not ends:
            return None
        if end is None:
            return ends[0]
        return min(ends, key=lambda e: (abs(e - end), e))
    if end is not None and 0 < end <= len(text_bytes):
        pos = end
        while pos < len(text_bytes) and (text_bytes[pos] & 0xC0) == 0x80:
            pos += 1
        return pos
    return None


def byte_to_char_index(text: str, byte_off: int) -> int:
    """Map a UTF-8 byte offset into ``text`` onto a character index."""
    return len(text.encode("utf-8")[:byte_off].decode("utf-8", "ignore"))


def _fence_line_spans(text: str) -> List[Tuple[int, int]]:
    """(line_start, position after the last backtick) of every ``` fence line."""
    spans = []
    offset = 0
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            spans.append((offset, offset + len(line)))
        offset += len(line) + 1
    return spans


def _line_end(text: str, pos: int) -> int:
    closer = text.find("\n", pos)
    return closer + 1 if closer != -1 else len(text)


def safe_insertion_point(text: str, pos: int) -> int:
    """Move ``pos`` so a citation marker can't split code, spans or words."""
    pos = max(0, min(int(pos), len(text)))
    fences = _fence_line_spans(text)
    opener = None
    for span in fences:
        if _line_end(text, span[1]) <= pos:
            opener = span
        else:
            break
    if opener is not None:
        index = fences.index(opener) + 1
        if index < len(fences):
            return _line_end(text, fences[index][1])
        return len(text)
    if text[:pos].count("`") % 2 == 1:
        closer = text.find("`", pos)
        return closer + 1 if closer != -1 else len(text)
    link = text.rfind("](", 0, pos)
    if link != -1:
        closer = text.find(")", link + 2)
        if closer != -1 and closer >= pos:
            return closer + 1
    while pos > 0 and text[pos - 1] in " \t\r\n":
        pos -= 1
    if 0 < pos < len(text) and text[pos - 1].isalnum() and text[pos].isalnum():
        while pos < len(text) and text[pos].isalnum():
            pos += 1
    return pos


def insert_citation_markers(text: str, markers: List[Tuple[int, List[int]]]) -> str:
    """Insert ``[n]`` citation runs at the (safe) positions of ``markers``."""
    if not markers:
        return text
    groups: Dict[int, set] = {}
    for pos, numbers in markers:
        if not numbers:
            continue
        point = safe_insertion_point(text, pos)
        groups.setdefault(point, set()).update(numbers)
    for point in sorted(groups, reverse=True):
        run = "".join(f"[{n}]" for n in sorted(groups[point]))
        if text[point : point + len(run)] == run:
            continue
        text = text[:point] + run + text[point:]
    return text


def _normalize_url(url: str) -> str:
    """Citation key of a URL: lower-cased scheme and host, trailing / stripped."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    rest = ""
    if parsed.query:
        rest += "?" + parsed.query
    if parsed.fragment:
        rest += "#" + parsed.fragment
    return f"{(parsed.scheme or '').lower()}://{(parsed.netloc or '').lower()}{path}{rest}"


class CitationRegistry:
    """Maps citation keys to stable numbers across the API calls of one turn."""

    def __init__(self, offset: int = 0):
        self.offset = offset
        self.sources: Dict[int, dict] = {}
        self._by_key: Dict[str, int] = {}

    def number_for(self, key: str, info: dict) -> Tuple[int, bool]:
        """The number for ``key`` and whether it was assigned just now."""
        if key in self._by_key:
            return self._by_key[key], False
        number = len(self.sources) + self.offset + 1
        self._by_key[key] = number
        self.sources[number] = dict(info)
        return number, True


def _chunk_citation(chunk: dict, resolved: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    uri = chunk.get("uri")
    url = resolved.get(uri, uri) if uri else None
    if chunk.get("place_id"):
        return "place:" + str(chunk["place_id"]), url
    return _normalize_url(url) if url else None, url


def cite_call_text(
    text: str,
    acc: GroundingAccumulator,
    registry: CitationRegistry,
    resolved: Dict[str, str],
) -> Tuple[str, List[int], Dict[int, List[str]]]:
    """Cite one API call's answer text with the turn's shared registry.

    Returns the text with markers, the numbers that were new in this call
    (ascending) and ``{number: [supporting segment texts]}``.
    """
    data = text.encode("utf-8")
    markers: List[Tuple[int, List[int]]] = []
    new_numbers = set()
    references: Dict[int, List[str]] = {}
    for support in acc.supports:
        end = locate_support_end(data, support)
        if end is None:
            continue
        point = safe_insertion_point(text, byte_to_char_index(text, end))
        numbers = []
        for index in support["indices"]:
            if not 0 <= index < len(acc.chunks):
                continue
            chunk = acc.chunks[index]
            key, url = _chunk_citation(chunk, resolved)
            if key is None:
                continue
            number, is_new = registry.number_for(
                key,
                {"title": chunk.get("title"), "url": url, "kind": chunk.get("kind")},
            )
            numbers.append(number)
            if is_new:
                new_numbers.add(number)
            bucket = references.setdefault(number, [])
            if support["text"] and support["text"] not in bucket:
                bucket.append(support["text"])
        if numbers:
            markers.append((point, numbers))
    for chunk in acc.chunks:
        key, url = _chunk_citation(chunk, resolved)
        if key is None:
            continue
        registry.number_for(
            key, {"title": chunk.get("title"), "url": url, "kind": chunk.get("kind")}
        )
    return insert_citation_markers(text, markers), sorted(new_numbers), references


_PATTERNS_SUGGESTION_CHIP = re.compile(
    r"<a\b([^>]*)>(.*?)</a>", flags=re.S | re.I
)
_PATTERNS_CHIP_CLASS = re.compile(r'class\s*=\s*["\']([^"\']*)["\']', flags=re.I)
_PATTERNS_CHIP_HREF = re.compile(r'href\s*=\s*["\']([^"\']*)["\']', flags=re.I)


def extract_suggestion_links(html_text: str, queries: List[str]) -> List[Tuple[str, str]]:
    """(label, href) pairs from the Search Suggestions chips of the entry point.

    Falls back to Google search URLs built from the queries when the entry
    point carries no chips (or no HTML at all).
    """
    links: List[Tuple[str, str]] = []
    for match in _PATTERNS_SUGGESTION_CHIP.finditer(html_text or ""):
        attrs, label = match.group(1), match.group(2)
        class_match = _PATTERNS_CHIP_CLASS.search(attrs)
        if not class_match or "chip" not in class_match.group(1).split():
            continue
        href_match = _PATTERNS_CHIP_HREF.search(attrs)
        if not href_match:
            continue
        clean = re.sub(r"<[^>]+>", "", label).strip()
        links.append((html.unescape(clean), html.unescape(href_match.group(1))))
    if links:
        return links
    return [
        (query, "https://www.google.com/search?q=" + quote_plus(query))
        for query in (queries or [])
    ]


def prepare_entry_point_embed(html_text: str) -> str:
    """Make the suggestion embed open links in a new tab, not the chat."""
    if not html_text:
        return ""
    index = html_text.find("<head>")
    if index != -1:
        at = index + len("<head>")
        return html_text[:at] + '<base target="_blank">' + html_text[at:]
    return '<base target="_blank">' + html_text


def _rag_citation_offset(sources: List[dict]) -> int:
    """How many RAG source ids OpenWebUI already cited, so pipe numbers follow.

    Counts unique ids exactly the way ``get_source_context`` in
    open_webui/utils/middleware.py does: ``meta.get("source") or
    source["source"]["id"]`` over ``zip(document, metadata)``.
    """
    seen = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        for _doc, meta in zip(source.get("document") or [], source.get("metadata") or []):
            meta = meta if isinstance(meta, dict) else {}
            source_id = meta.get("source") or (source.get("source") or {}).get("id") or "N/A"
            seen.add(str(source_id))
    return len(seen)
# END GENERATED SECTION: gemini_pipe.response.grounding

# BEGIN GENERATED SECTION: gemini_pipe.response.state
@dataclass
class Segment:
    """One rendered piece of the message: streamed text or a <details> block."""

    kind: str
    key: str
    call: int
    content: str


@dataclass
class RenderedMessage:
    """The message as it will be persisted, in segment order.

    Text segments stream; blocks (the <details> collapsibles) are upserted as a
    whole. ``render()`` is the single source of truth for the message content,
    and ``append_text`` returns exactly the string by which ``render()`` grew,
    so ``message`` delta events stay consistent with it.
    """

    segments: List[Segment] = field(default_factory=list)

    def _index_of(self, key: str) -> int:
        for i, seg in enumerate(self.segments):
            if seg.key == key:
                return i
        return -1

    def render(self) -> str:
        parts: List[str] = []
        for i, seg in enumerate(self.segments):
            if seg.kind == "block":
                if parts and not parts[-1].endswith("\n"):
                    parts.append("\n")
                parts.append(seg.content)
                if not seg.content.endswith("\n") and i + 1 < len(self.segments):
                    parts.append("\n")
            else:
                parts.append(seg.content)
        return "".join(parts)

    def append_text(self, call: int, s: str) -> str:
        if not s:
            return ""
        before = self.render()
        if self.segments and self.segments[-1].kind == "text" and self.segments[-1].call == call:
            self.segments[-1].content += s
        else:
            n = sum(1 for seg in self.segments if seg.kind == "text" and seg.call == call)
            self.segments.append(Segment("text", f"text:{call}:{n}", call, s))
        after = self.render()
        if after.startswith(before):
            return after[len(before):]
        return after

    def upsert_block(self, key: str, content: str, call: int, before_key: Optional[str] = None) -> None:
        index = self._index_of(key)
        if index != -1:
            self.segments[index] = Segment("block", key, call, content)
            return
        seg = Segment("block", key, call, content)
        if before_key is not None:
            before = self._index_of(before_key)
            if before != -1:
                self.segments.insert(before, seg)
                return
        self.segments.append(seg)

    def remove(self, key: str) -> None:
        index = self._index_of(key)
        if index != -1:
            del self.segments[index]

    def text_segments(self, call: int) -> List[Segment]:
        return [seg for seg in self.segments if seg.kind == "text" and seg.call == call]

    def checkpoint(self) -> List[Segment]:
        return copy.deepcopy(self.segments)

    def rollback(self, checkpoint: List[Segment]) -> None:
        self.segments = copy.deepcopy(checkpoint)


@dataclass
class CallState:
    """Everything one API call (one pass of the generation loop) contributes."""

    index: int
    raw_parts: List[Any] = field(default_factory=list)
    function_calls: List[dict] = field(default_factory=list)
    grounding: GroundingAccumulator = field(default_factory=GroundingAccumulator)
    url_meta: List[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    finish_message: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    thought_key: Optional[str] = None
    thought_text: str = ""
    thought_started: float = 0.0
    thought_runs: int = 0
    last_thought_render: float = 0.0
    image_tasks: List[Any] = field(default_factory=list)


@dataclass
class TurnState:
    """The whole turn: every call plus the aggregates shared across calls."""

    calls: List[CallState] = field(default_factory=list)
    citations: CitationRegistry = field(default_factory=CitationRegistry)
    usage_total: Dict[str, Any] = field(default_factory=dict)
    search_queries: List[str] = field(default_factory=list)
    tool_calls_made: int = 0
    degraded: set = field(default_factory=set)
    name_map: Dict[str, str] = field(default_factory=dict)
# END GENERATED SECTION: gemini_pipe.response.state

@dataclass
class PipeRequestContext:
    """Everything one ``pipe()`` call shares with its handlers.

    Built fresh per request: the Pipe object itself is shared across all users
    and concurrent requests, so nothing request-scoped may live on it.
    """

    pipe: "Pipe"
    event_emitter: Optional[Callable[[dict], Awaitable[None]]]
    msg: RenderedMessage
    turn: TurnState
    run_id: str
    status: "StatusEmitter"
    user: dict
    user_valves: "Pipe.UserValves"
    metadata: dict
    request: dict
    model: str
    info: dict
    api_key: str
    tools: dict
    builtin_tools: dict
    rag_offset: int

    async def emit_event(self, event: dict) -> None:
        """Forward one OpenWebUI event, surviving a broken emitter."""
        if self.event_emitter is None:
            return
        try:
            await self.event_emitter(event)
        except Exception as e:
            logger.warning(f"event emitter failed: {e}")

    async def emit_text(self, call: int, s: str) -> None:
        d = self.msg.append_text(call, s)
        await self.emit_event({"type": "message", "data": {"content": d}})

    async def upsert_block(
        self, key: str, content: str, call: int, before_key: Optional[str] = None
    ) -> None:
        self.msg.upsert_block(key, content, call, before_key=before_key)
        await self.emit_event({"type": "replace", "data": {"content": self.msg.render()}})

    async def emit_replace(self) -> None:
        await self.emit_event({"type": "replace", "data": {"content": self.msg.render()}})

    def text(self) -> str:
        return self.msg.render()


def new_request_context(
    pipe,
    *,
    event_emitter=None,
    model="",
    info=None,
    user=None,
    user_valves=None,
    metadata=None,
    request=None,
    api_key="",
    tools=None,
    builtin_tools=None,
    rag_offset=0,
) -> PipeRequestContext:
    """The fresh context every pipe() call builds before anything else.

    One Pipe instance serves all users and concurrent requests, so everything
    request-scoped is created here, per call, and handed down as ``ctx``.
    """
    ctx = PipeRequestContext(
        pipe=pipe,
        event_emitter=event_emitter,
        msg=RenderedMessage(),
        turn=TurnState(citations=CitationRegistry(rag_offset)),
        run_id=uuid.uuid4().hex[:8],
        status=None,
        user=dict(user or {}),
        user_valves=pipe.UserValves() if user_valves is None else user_valves,
        metadata=dict(metadata or {}),
        request=dict(request or {}),
        model=model,
        info=dict(info or {}),
        api_key=api_key,
        tools=dict(tools or {}),
        builtin_tools=dict(builtin_tools or {}),
        rag_offset=rag_offset,
    )
    ctx.status = StatusEmitter(ctx.emit_event)
    return ctx


# BEGIN GENERATED SECTION: gemini_pipe.response.status_events
class StatusEmitter:
    def __init__(self, emit_event: Callable[[dict], Awaitable[None]]):
        """Wrap an OpenWebUI ``emit_event`` callback with dedup state for status events."""
        self._emit_event = emit_event
        self._last_payload: dict | None = None

    async def emit(
        self,
        description: str,
        *,
        done: bool = False,
        hidden: bool | None = None,
        force: bool = False,
        **fields: Any,
    ) -> None:
        """Emit an OpenWebUI ``status`` event, skipping an identical repeat.

        Dedup is what keeps the phase model cheap: a phase family can announce
        itself on every block without stuttering the history when several blocks
        of the same kind arrive in a row (grounded answers split prose into many
        text blocks around their citation markers).
        """
        data: dict = {"description": description, "done": done}
        if hidden is not None:
            data["hidden"] = hidden
        data.update(fields)

        if not force and data == self._last_payload:
            return

        await self._emit_event({"type": "status", "data": data})
        self._last_payload = data

    # -- phases ------------------------------------------------------------
    # One method per thing that can actually be happening, so call sites read as
    # intent and the wording stays consistent across handlers.

    async def waiting(self) -> None:
        """Before the first byte: the request is out, nothing has come back yet."""
        await self.emit("Waiting for response...", hidden=False, force=True)

    async def thinking(self) -> None:
        """A thought is streaming."""
        await self.emit("💭 Thinking...")

    async def responding(self) -> None:
        """Text is streaming — the model is writing the answer.

        Emitted on every text run, not once per turn: after a tool runs, text
        resuming is a real phase change the user should see.
        """
        await self.emit("Responding...")

    async def searching_google(self, queries: list) -> None:
        """Google Search grounding is running for the model."""
        shown = " · ".join(queries[:3]) if queries else ""
        await self.emit(f"🔍 Searching Google: {shown}" if shown else "🔍 Searching Google...")

    async def web_search_done(self, urls: list, queries: list = None) -> None:
        """Report finished search results using OpenWebUI's native renderer.

        With ``action="web_search"`` and ``urls`` the frontend renders a
        clickable source list rather than a plain text line.
        """
        if not urls:
            return
        await self.emit(
            "Searched {{count}} sites",
            action="web_search",
            urls=list(urls),
            queries=list(queries or []),
            count=len(urls),
        )

    async def read_urls(self, urls: list, failed: int = 0) -> None:
        """URL context finished reading the pages from the conversation."""
        if not urls and not failed:
            return
        description = "Read {{count}} pages" + (f", {failed} failed" if failed else "")
        await self.emit(
            description,
            action="web_search",
            urls=list(urls or []),
            count=len(urls or []),
        )

    async def places_found(self, urls: list) -> None:
        """Google Maps grounding returned places."""
        if not urls:
            return
        await self.emit(
            "Found {{count}} places",
            action="web_search",
            urls=list(urls),
            count=len(urls),
        )

    async def fetching_url(self, url: str = "") -> None:
        """A URL from the conversation is being fetched for the model."""
        await self.emit(f"🌐 Fetching {url}" if url else "🌐 Fetching URL...")

    async def running_tool(self, tool_name: str) -> None:
        """An OpenWebUI-side tool call is being dispatched."""
        await self.emit(f"🔧 Running {tool_name}..." if tool_name else "🔧 Running tool...")

    async def activity(self, description: str) -> None:
        """Free-form in-progress phase, for one-off states without a named phase."""
        await self.emit(description, done=False)

    async def complete(self, description: str) -> None:
        """Close the status line for this turn. Only the final status is ``done``."""
        await self.emit(description, done=True, force=True)

    async def notification(self, content: str, *, type: str = "warning") -> None:
        """Emit an OpenWebUI ``notification`` event (e.g. warning/error toast)."""
        await self._emit_event(
            {"type": "notification", "data": {"type": type, "content": content}}
        )
# END GENERATED SECTION: gemini_pipe.response.status_events

class Pipe:
    """OpenWebUI manifold pipe for the Gemini Developer API."""

    # Class-level caches: one Pipe instance serves all users and concurrent
    # requests, so per-request state must not live here.
    _model_cache: Optional[list] = None
    _model_cache_ts: float = 0.0
    _model_cache_sig: str = ""
    _api_models: Dict[str, dict] = {}
    _resolved_url_cache: Dict[str, str] = {}
    _gemini_file_cache: Dict[str, dict] = {}

    # Capability pins applied between the derived defaults and the admin's
    # MODEL_CAPABILITY_OVERRIDES valve: {model-name pattern: {key: value}}.
    MODEL_CAPABILITY_PINS: Dict[str, dict] = {}

    class Valves(BaseModel):
        GEMINI_API_KEY: EncryptedStr = Field(
            default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
            description="Gemini Developer API key, stored encrypted",
        )
        GEMINI_BASE_URL: str = Field(
            default="",
            description="Optional proxy base URL; empty = Google default",
        )
        ENABLED_MODELS: str = Field(
            default="",
            description="Comma-separated model names; skips discovery",
        )
        MODEL_CACHE_TTL_MINUTES: int = Field(
            default=1440,
            description="How long the discovered model list is cached, in minutes; 0 disables the model-list cache",
        )
        MODEL_CAPABILITY_OVERRIDES: str = Field(
            default="",
            description="JSON object of {model-name pattern: {capability: value}} applied over the derived capabilities; a pattern ending in * is a prefix match",
        )
        WEB_SEARCH_DEFAULT: bool = Field(
            default=False,
            description="Google Search on every request; otherwise only via the toggle. Billed per search query",
        )
        URL_CONTEXT: Literal["auto", "with_search", "always", "off"] = Field(
            default="auto",
            description="When to enable the URL context tool: auto adds it for URLs in the last user message",
        )
        MAPS_DEFAULT: bool = Field(
            default=False,
            description="Maps grounding on every request",
        )
        MAPS_LATITUDE: str = Field(
            default="",
            description="Default location for Maps; empty = none",
        )
        MAPS_LONGITUDE: str = Field(
            default="",
            description="Default longitude for Maps; empty = none",
        )
        BUILTIN_TOOL_CONFLICT: Literal["prefer_builtin", "prefer_functions"] = Field(
            default="prefer_builtin",
            description="Which side wins when a model cannot combine function tools with Google Search/Maps",
        )
        SHOW_SEARCH_SUGGESTIONS: Literal["embed", "links", "off"] = Field(
            default="embed",
            description="Google's terms require showing Search Suggestions with grounded answers",
        )
        RESOLVE_GROUNDING_URLS: bool = Field(
            default=True,
            description="Resolve grounding redirect URLs to their real destination",
        )
        STORE_REQUESTS: bool = Field(
            default=False,
            description="Lets Google store requests (sent only when the SDK supports `store`)",
        )
        MAX_TOOL_CALLS: int = Field(
            default=15,
            description="Maximum number of tool calls per request",
        )
        MAX_RETRIES: int = Field(
            default=3,
            description="Maximum number of retries for failed requests",
        )
        REQUEST_TIMEOUT: int = Field(
            default=600,
            description="Request timeout in seconds for Gemini API calls",
        )
        TOOL_CALL_TIMEOUT: int = Field(
            default=60,
            description="Timeout in seconds for individual tool call execution",
        )
        INLINE_FILE_MAX_MB: int = Field(
            default=20,
            description="Maximum file size in MB to inline instead of using the Files API",
        )
        FILES_API: Literal["auto", "always", "never"] = Field(
            default="auto",
            description="When to upload media files through the Gemini Files API",
        )
        YOUTUBE_LINKS: Literal["most_recent", "latest_message", "all", "off"] = Field(
            default="most_recent",
            description="Which YouTube links from the conversation to send to the model",
        )
        TASK_MODEL: str = Field(
            default="gemini-3.5-flash-lite",
            description="Model for title/tags/follow-ups; empty = chat model",
        )
        MODEL_PRICING_OVERRIDES: str = Field(
            default="",
            description="JSON object of {model name: {rate key: USD per million tokens}} overriding the built-in price table",
        )

    class UserValves(BaseModel):
        GEMINI_API_KEY: EncryptedStr = Field(
            default="",
            description="Overrides the admin-configured API key; empty = use the admin key. Stored encrypted when WEBUI_SECRET_KEY is set",
        )
        THINKING_LEVEL: Literal["auto", "minimal", "low", "medium", "high"] = Field(
            default="auto",
            description="Thinking level for models that use levels; auto follows the model default",
        )
        THINKING_BUDGET: int = Field(
            default=-1,
            description="Thinking budget in tokens for Gemini 2.5 models; -1 = dynamic",
        )
        SHOW_THOUGHTS: bool = Field(
            default=True,
            description="Render the model's reasoning blocks in the chat",
        )
        MEDIA_RESOLUTION: Literal["auto", "low", "medium", "high"] = Field(
            default="auto",
            description="Resolution the model sees images at",
        )
        IMAGE_ASPECT_RATIO: Literal[
            "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
        ] = Field(
            default="auto",
            description="Aspect ratio for generated images; auto = model default",
        )
        IMAGE_SIZE: Literal["auto", "1K", "2K", "4K"] = Field(
            default="auto",
            description="Size for generated images; auto = model default",
        )
        MAPS_LATITUDE: str = Field(
            default="",
            description="Per-user Maps location; empty = use the admin default",
        )
        MAPS_LONGITUDE: str = Field(
            default="",
            description="Per-user Maps longitude; empty = use the admin default",
        )
        USE_NATIVE_FILES: bool = Field(
            default=True,
            description="Send PDFs as native inline files instead of RAG text extraction",
        )
        HIDE_BLOCKS: List[Literal["google_search", "url_context", "google_maps", "reasoning"]] = Field(
            default=[],
            json_schema_extra={
                "input": {
                    "type": "multiselect",
                    "options": [
                        "google_search",
                        "url_context",
                        "google_maps",
                        "reasoning",
                    ],
                }
            },
            description=(
                "Block types to hide from your chat. A hidden block is not "
                "rendered at all — its progress is reported by the status line "
                "instead. The block is still sent to the API, so hiding it does "
                "not change the model's view of the conversation."
            ),
        )

        @field_validator("HIDE_BLOCKS", mode="before")
        @classmethod
        def _coerce_hide_blocks(cls, v):
            """Accept the comma-separated string form from pre-multiselect valves.

            OpenWebUI validates the whole UserValves model on every save, so a
            stored string must keep parsing instead of failing every valve
            update with a 400.
            """
            if isinstance(v, str):
                return [part.strip() for part in v.split(",") if part.strip()]
            return v

        SHOW_TOKEN_COUNT: Literal["Off", "On", "With Cache"] = Field(
            default="Off",
            description="Show context window progress after each response; With Cache also shows cached tokens",
        )
        SHOW_COST: bool = Field(
            default=True,
            description="Report the estimated USD cost of the turn in the message usage",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "gemini"
        self.name = "gemini/"
        self.valves = self.Valves()

    # COMPILED PIPE METHOD GROUPS INSERTION POINT
    # BEGIN GENERATED SECTION: gemini_pipe.pipe_method_groups
    _METADATA_MARKER = re.compile(PATTERN_METADATA_MARKER)

    def _extract_and_remove_memories(self, text: str) -> tuple[str, Optional[str]]:
        """Extract the OpenWebUI Memory System's memories out of the system
        prompt and remove them from it.

        Two injection formats are recognised, because OpenWebUI changed shape:
          * ``<memory_context>...</memory_context>`` (current, utils/memory.py)
            - can sit anywhere in the system message.
          * ``\\nUser Context:\\n...`` (legacy) - runs to the end of the string.

        Both are re-retrieved and re-ranked per request, so they are never
        stable across turns. Leaving them in the system prompt costs a full
        prefix rewrite every turn; the caller relocates the return value to
        the last user message instead.

        Returns:
            tuple[str, Optional[str]]: (cleaned_text, extracted_context)
            - cleaned_text: original text with all memory blocks removed (stripped)
            - extracted_context: the extracted memories, or None if none found
        """
        # Fast path: two substring scans are far cheaper than two regex scans,
        # and presence of the marker is the ground truth. Deliberately *not*
        # gated on a config flag - OpenWebUI decides to inject based on the
        # request's features.memory plus an admin-level config, neither of
        # which the pipe can observe reliably.
        if "<memory_context>" not in text and "User Context:" not in text:
            return text.strip(), None

        extracted_parts: List[str] = []

        # <memory_context> may appear anywhere; strip every occurrence.
        def _take_memory_context(match) -> str:
            content = match.group(1).strip()
            if content:
                extracted_parts.append(content)
            return ""

        cleaned_text = PATTERN_MEMORY_CONTEXT.sub(_take_memory_context, text)

        # Legacy tail form.
        match = PATTERN_USER_CONTEXT.search(cleaned_text)
        if match:
            context_content = match.group(1).strip()
            if context_content:
                extracted_parts.append(f"User Context:\n{context_content}")
            # Remove "\nUser Context:\n" and everything after it.
            cleaned_text = cleaned_text[: match.start()]

        extracted_context = "\n\n".join(extracted_parts) if extracted_parts else None
        return cleaned_text.strip(), extracted_context

    def _create_metadata_marker(self, id: str, value: str, messagenum: int = 0) -> str:
        """Build a URL-encoded inline metadata marker for embedding in
        assistant text; it renders as nothing but round-trips the payload."""
        encoded_value = quote(value, safe="")
        return f" [](gemini:{messagenum}:{id}:{encoded_value}) "

    def _extract_metadata_marker_from_message(self, message) -> List[str]:
        """Extract the metadata payloads from the last assistant message and
        remove the markers from it in place."""
        metadata: List[str] = []
        if not isinstance(message, dict):
            return metadata
        if message.get("role") != "assistant":
            return metadata
        content = message.get("content")
        if isinstance(content, list):
            # Join all text blocks for searching, but also update blocks in-place.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block_text = block.get("text", "")
                    metadata.extend(self._METADATA_MARKER.findall(block_text))
                    block["text"] = self._METADATA_MARKER.sub("", block_text)
        elif isinstance(content, str):
            metadata.extend(self._METADATA_MARKER.findall(content))
            message["content"] = self._METADATA_MARKER.sub("", content)
        return metadata

    _IMAGE_SEND_AS_IS = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    )
    _IMAGE_CONVERT_TO_PNG = ("image/gif", "image/bmp", "image/tiff", "image/avif")

    _PATTERN_METADATA_MARKER = re.compile(PATTERN_METADATA_MARKER)
    _PATTERN_REASONING_DETAILS = re.compile(
        r'<details\b[^>]*\btype="reasoning"[^>]*>.*?</details>', flags=re.DOTALL
    )
    _PATTERN_BUILTIN_DETAILS = re.compile(
        r'<details\b[^>]*\bdata-gemini-builtin="1"[^>]*>.*?</details>',
        flags=re.DOTALL,
    )
    _PATTERN_TOOL_CALLS_DETAILS = re.compile(
        r'<details\b[^>]*\btype="tool_calls"[^>]*>.*?</details>', flags=re.DOTALL
    )
    _PATTERN_FILES_URL = re.compile(r"/api/v1/files/([0-9a-fA-F-]+)/content")
    _PATTERN_DATA_URL = re.compile(r"^data:[^,]*;base64,(.*)$", flags=re.DOTALL)
    _PATTERN_CITATION_RUN = re.compile(r"(?:\[\d+\])+")
    _PATTERN_CODE_SPANS = re.compile(r"```.*?```|`[^`\n]*`", flags=re.DOTALL)
    _SUGGESTION_LINE_PREFIX = "🔎 **Google Search suggestions:**"
    _NOTICE_LINE_PREFIX = "> ⚠️"

    async def _convert_history(
        self, messages, info, ctx
    ) -> Tuple[Optional[str], List[gtypes.Content]]:
        """The system instruction and the Gemini Contents for one request.

        Memories are extracted out of the system prompt and appended to the
        last user message instead: they are re-ranked per request, so they
        must not churn the (cacheable) system prefix.
        """
        messages = messages or []
        last_user_index = -1
        for i, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user_index = i

        system_parts: List[str] = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                text = self._block_text(message.get("content", ""))
                if text.strip():
                    system_parts.append(text.strip())
        system_instruction = "\n\n".join(system_parts).strip() or None

        memories: Optional[str] = None
        if system_instruction:
            system_instruction, memories = self._extract_and_remove_memories(
                system_instruction
            )
            if not system_instruction:
                system_instruction = None

        contents: List[gtypes.Content] = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                parts = await self._user_parts(message.get("content", ""), ctx)
                if memories is not None and i == last_user_index:
                    parts = self._append_memories(parts, memories)
                if parts:
                    contents.append(gtypes.Content(role="user", parts=parts))
            elif role == "assistant":
                text = self._assistant_text(message.get("content", ""))
                if text:
                    contents.append(
                        gtypes.Content(role="model", parts=[gtypes.Part(text=text)])
                    )

        merged: List[gtypes.Content] = []
        for content in contents:
            if not content.parts:
                continue
            if merged and merged[-1].role == content.role:
                merged[-1].parts.extend(content.parts)
            else:
                merged.append(content)
        return system_instruction, merged

    @staticmethod
    def _block_text(content: Any) -> str:
        """The prose of a message content: the string itself, or its text blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    async def _user_parts(self, content: Any, ctx) -> List[gtypes.Part]:
        """The parts of one user message: text as-is, images as inline_data."""
        parts: List[gtypes.Part] = []
        if isinstance(content, str):
            if content:
                parts.append(gtypes.Part(text=content))
            return parts
        if not isinstance(content, list):
            return parts
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text") or ""
                if text:
                    parts.append(gtypes.Part(text=text))
            elif block_type == "image_url":
                url = (block.get("image_url") or {}).get("url") or ""
                part = await self._image_part(url)
                if part is not None:
                    parts.append(part)
        return parts

    def _append_memories(
        self, parts: List[gtypes.Part], memories: str
    ) -> List[gtypes.Part]:
        """Append the relocated memories under the standard header, after the
        last text part of the message (or as a new part when there is none)."""
        suffix = MEMORY_CONTEXT_APPENDIX_HEADER + memories
        for index in range(len(parts) - 1, -1, -1):
            if parts[index].text:
                parts[index] = gtypes.Part(text=parts[index].text + suffix)
                return parts
        parts.append(gtypes.Part(text=suffix))
        return parts

    async def _image_part(self, url: str) -> Optional[gtypes.Part]:
        """The inline_data part for one image URL, or None when it cannot be
        sent (fetch failure, unrecognised type, no converter)."""
        if not url:
            return None
        file_match = self._PATTERN_FILES_URL.search(url)
        if url.startswith("data:"):
            data = self._decode_data_url(url)
        elif FILES_AVAILABLE and file_match:
            data = await self._read_file_bytes(file_match.group(1))
        elif url.startswith(("http://", "https://")):
            data = await self._fetch_url_image(url)
        else:
            return None
        if data is None:
            return None
        prepared = self._prepare_image_bytes(data, self._sniff_image_media_type(data))
        if prepared is None:
            return None
        image_bytes, mime_type = prepared
        return gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    @classmethod
    def _decode_data_url(cls, url: str) -> Optional[bytes]:
        match = cls._PATTERN_DATA_URL.match(url)
        if not match:
            logger.debug("skipping image: not a base64 data URL")
            return None
        try:
            return base64.b64decode(match.group(1))
        except Exception as e:
            logger.debug(f"skipping image: invalid base64 data URL: {e}")
            return None

    @staticmethod
    async def _read_file_bytes(file_id: str) -> Optional[bytes]:
        """The bytes behind an /api/v1/files/<uuid>/content URL, via OpenWebUI's
        Files registry and storage provider."""
        try:
            file = await Files.get_file_by_id(id=file_id)
        except Exception as e:
            logger.warning(f"could not look up file {file_id}: {e}")
            return None
        if file is None:
            logger.debug(f"skipping image: unknown file id {file_id}")
            return None
        location = getattr(file, "path", None)
        if not location:
            logger.debug(f"skipping image: file {file_id} has no storage location")
            return None
        try:
            path = Storage.get_file(location)
            with open(path, "rb") as handle:
                return handle.read()
        except Exception as e:
            logger.warning(f"could not read file {file_id}: {e}")
            return None

    @staticmethod
    async def _fetch_url_image(url: str) -> Optional[bytes]:
        """The bytes of an http(s) image URL, through OpenWebUI's SSRF-safe
        fetcher; None when the fetcher is absent or the fetch fails."""
        if not IMAGE_URL_FETCH_AVAILABLE:
            logger.debug("skipping image URL: the OpenWebUI fetcher is not available")
            return None
        try:
            encoded = await get_image_base64_from_url(url)
        except Exception as e:
            logger.warning(f"could not fetch image URL {url}: {e}")
            return None
        if not encoded:
            logger.debug(f"skipping image URL: empty fetch result for {url}")
            return None
        try:
            return base64.b64decode(encoded)
        except Exception as e:
            logger.debug(f"skipping image URL: invalid base64 from {url}: {e}")
            return None

    def _prepare_image_bytes(
        self, data: bytes, mime_type: Optional[str]
    ) -> Optional[Tuple[bytes, str]]:
        """(bytes, mime) ready to send, or None: types the API takes are sent
        as-is, GIF/BMP/TIFF/AVIF are converted to PNG when Pillow is
        installed, and anything else is skipped."""
        if not mime_type:
            logger.debug("skipping image: unrecognised image type")
            return None
        if mime_type in self._IMAGE_SEND_AS_IS:
            return data, mime_type
        if mime_type in self._IMAGE_CONVERT_TO_PNG:
            if not PIL_AVAILABLE:
                logger.warning(
                    f"skipping {mime_type} image: Pillow is not installed, "
                    "so it cannot be converted to PNG"
                )
                return None
            try:
                image = PILImage.open(io.BytesIO(data))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue(), "image/png"
            except Exception as e:
                logger.warning(f"skipping {mime_type} image: conversion to PNG failed: {e}")
                return None
        logger.debug(f"skipping image: unsupported type {mime_type}")
        return None

    @staticmethod
    def _sniff_image_media_type(raw: bytes) -> Optional[str]:
        """Identify an image from its leading bytes, ignoring any declared type.

        Necessary because OpenWebUI's label is wrong in two different ways
        (both in MessageInput.svelte):

          * Its HEIC branch tests `file.type === 'image/heic'` exactly, so
            `image/heif`, the `*-sequence` variants, and the very common case
            of an empty `file.type` skip conversion entirely.
          * When conversion DOES run, the resulting JPEG is re-wrapped with
            `new File([blob], name, { type: file.type })` -- the ORIGINAL type.
            So a successfully converted image still arrives labelled HEIC.

        The bytes are the only reliable source.
        """
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
        # ISO-BMFF: size, then "ftyp", then the brand.
        if raw[4:8] == b"ftyp":
            brand = raw[8:12]
            if brand in (b"avif", b"avis"):
                return "image/avif"
            if brand in (b"heic", b"heix", b"heim", b"heis", b"heif", b"mif1"):
                return "image/heic"
        return None

    def _assistant_text(self, content: Any) -> str:
        """The replayable prose of an assistant message: its persisted text
        minus everything this pipe rendered into it."""
        text = self._block_text(content)
        if not text:
            return ""
        had_search_block = 'name="google_search"' in text
        text = self._PATTERN_METADATA_MARKER.sub("", text)
        text = self._PATTERN_REASONING_DETAILS.sub("", text)
        text = self._PATTERN_BUILTIN_DETAILS.sub("", text)
        text = self._PATTERN_TOOL_CALLS_DETAILS.sub("", text)
        text = self._strip_decorative_lines(text)
        if had_search_block:
            # Citation numbers are only valid within the call that produced
            # them; replayed text would point at the wrong sources, so the
            # markers go away with the search block.
            text = self._strip_citation_runs(text)
        return text.strip()

    def _strip_decorative_lines(self, text: str) -> str:
        kept = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith(self._SUGGESTION_LINE_PREFIX):
                continue
            if stripped.startswith(self._NOTICE_LINE_PREFIX):
                continue
            kept.append(line)
        return "\n".join(kept)

    def _strip_citation_runs(self, text: str) -> str:
        """Drop the [n] citation runs outside inline code and fenced blocks."""
        parts = self._PATTERN_CODE_SPANS.split(text)
        return "".join(
            self._PATTERN_CITATION_RUN.sub("", seg) if idx % 2 == 0 else seg
            for idx, seg in enumerate(parts)
        )

    _PATTERN_URL = re.compile(r"https?://[^\s<>\"')\]]+")

    @staticmethod
    def _parse_lat_lng(*parts) -> Optional[Tuple[float, float]]:
        """(lat, lng) from a (lat, lng) pair or a "lat, lng" string; None when
        missing or unparsable, so callers fall through to the next source."""
        if len(parts) == 2:
            pairs = [(str(parts[0]), str(parts[1]))]
        elif len(parts) == 1:
            pieces = [p.strip() for p in str(parts[0] or "").split(",")]
            pairs = [(pieces[0], pieces[1])] if len(pieces) == 2 else []
        else:
            pairs = []
        for lat, lng in pairs:
            try:
                return (float(lat), float(lng))
            except ValueError:
                continue
        return None

    def _last_user_url(self, text: str) -> Optional[str]:
        """The first non-YouTube URL in the user's last message, if any."""
        for match in self._PATTERN_URL.finditer(text or ""):
            host = urlparse(match.group(0)).netloc.lower()
            if host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be" or host.endswith(".youtu.be"):
                continue
            return match.group(0)
        return None

    def _resolve_builtin_flags(
        self, metadata: dict, valves, user_valves, info: dict, last_user_text: str
    ) -> dict:
        """Which built-in tools are on for this request, plus the Maps location."""
        metadata = metadata or {}
        search = bool(
            (
                valves.WEB_SEARCH_DEFAULT
                or metadata.get("gemini_web_search")
                or metadata.get("web_search_enforced")
            )
            and info.get("search")
        )

        mode = valves.URL_CONTEXT or "auto"
        url_context_auto = False
        if mode == "off":
            url_context = False
        elif mode == "always":
            url_context = True
        elif mode == "with_search":
            url_context = search
        else:
            url = self._last_user_url(last_user_text)
            url_context = bool(search or url)
            url_context_auto = bool(url and not search)
        url_context = url_context and bool(info.get("url_context"))

        maps = bool(
            (valves.MAPS_DEFAULT or metadata.get("gemini_maps")) and info.get("maps")
        )

        lat_lng = None
        if user_valves is not None:
            lat_lng = self._parse_lat_lng(
                user_valves.MAPS_LATITUDE, user_valves.MAPS_LONGITUDE
            )
        if lat_lng is None:
            raw = (metadata.get("variables") or {}).get("{{USER_LOCATION}}")
            if raw:
                lat_lng = self._parse_lat_lng(raw)
        if lat_lng is None:
            lat_lng = self._parse_lat_lng(valves.MAPS_LATITUDE, valves.MAPS_LONGITUDE)

        return {
            "search": search,
            "url_context": url_context,
            "url_context_auto": url_context_auto,
            "maps": maps,
            "lat_lng": lat_lng,
        }

    def _assemble_tools(
        self, info: dict, flags: dict, decls: List
    ) -> Tuple[List, Optional[object], List[str]]:
        """The Tool list for one call, its ToolConfig, and the degradation notes."""
        model = info.get("name") or "this model"
        notes: List[str] = []
        decls = list(decls or [])

        search = bool(flags.get("search"))
        maps = bool(flags.get("maps"))
        url_context = bool(flags.get("url_context"))
        url_context_auto = bool(flags.get("url_context_auto"))
        lat_lng = flags.get("lat_lng")

        if decls and not info.get("functions", True):
            notes.append(
                f"Function tools disabled for this message: {model} does not support function calling"
            )
            decls = []

        if search and maps and not info.get("maps_with_search"):
            search = False
            notes.append(
                f"Google Search disabled for this message: {model} cannot combine it with Google Maps"
            )

        builtins = search or url_context or maps
        if decls and builtins and not info.get("builtin_with_functions"):
            # The URL context that a URL in the user's message pulled in is the
            # softest dependency, so it goes first and silently.
            if url_context_auto:
                url_context = False
                builtins = search or maps
            if builtins:
                if self.valves.BUILTIN_TOOL_CONFLICT == "prefer_functions":
                    search = url_context = maps = False
                    builtins = False
                    notes.append(
                        f"Google Search/Maps disabled for this message: {model} cannot combine them with function tools"
                    )
                else:
                    decls = []
                    notes.append(
                        f"Function tools disabled for this message: {model} cannot combine them with Google Search/Maps"
                    )

        tools: List = []
        if decls:
            tools.append(gtypes.Tool(function_declarations=decls))
        if search:
            tools.append(gtypes.Tool(google_search=gtypes.GoogleSearch()))
        if url_context:
            tools.append(gtypes.Tool(url_context=gtypes.UrlContext()))
        if maps:
            tools.append(gtypes.Tool(google_maps=gtypes.GoogleMaps()))

        retrieval = None
        if maps and lat_lng:
            retrieval = gtypes.RetrievalConfig(
                lat_lng=gtypes.LatLng(latitude=lat_lng[0], longitude=lat_lng[1])
            )

        tool_config = None
        if decls and builtins:
            # Both sides stay: the SDK executes the built-ins server-side and
            # reports them back as regular tool calls.
            kwargs: dict = {}
            if GENAI_HAS_SERVER_SIDE_INVOCATIONS:
                kwargs["include_server_side_tool_invocations"] = True
            kwargs["function_calling_config"] = gtypes.FunctionCallingConfig(
                mode="VALIDATED"
            )
            if retrieval:
                kwargs["retrieval_config"] = retrieval
            tool_config = gtypes.ToolConfig(**kwargs)
        elif builtins and retrieval:
            tool_config = gtypes.ToolConfig(retrieval_config=retrieval)

        return tools, tool_config, notes

    _LEVEL_ORDER = ("minimal", "low", "medium", "high")
    _EFFORT_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}

    @classmethod
    def _clamp_level(cls, allowed: List[str], effort: str) -> str:
        """The allowed thinking level closest to ``effort``; ties go up."""
        allowed = [level for level in (allowed or []) if level in cls._LEVEL_ORDER]
        if not allowed:
            return "low"
        rank = {name: i for i, name in enumerate(cls._LEVEL_ORDER)}
        e = str(effort or "auto").lower()
        if e in ("none", "minimal"):
            target = 0
        elif e in ("xhigh", "max", "ultra"):
            target = len(cls._LEVEL_ORDER)
        elif e in rank:
            target = rank[e]
        else:
            return max(allowed, key=lambda level: rank[level])
        return min(allowed, key=lambda level: (abs(rank[level] - target), -rank[level]))

    def _build_thinking_config(
        self, info: dict, user_valves, body: dict, purpose: str = "chat"
    ):
        """The ThinkingConfig for this call, or None when the model has no
        thinking. A config never sets both a level and a budget."""
        if info.get("thinking") == "none":
            return None
        body = body or {}
        show = bool(user_valves.SHOW_THOUGHTS) if user_valves is not None else True
        effort = body.get("reasoning_effort") or (
            user_valves.THINKING_LEVEL if user_valves is not None else "auto"
        )
        valve_budget = (
            user_valves.THINKING_BUDGET if user_valves is not None else -1
        )
        task = purpose == "task"
        mode = info.get("thinking")

        if mode == "always":
            return gtypes.ThinkingConfig(include_thoughts=show)

        if mode == "level":
            if task:
                level = self._clamp_level(info.get("thinking_levels"), "minimal")
                return gtypes.ThinkingConfig(
                    thinking_level=gtypes.ThinkingLevel[level.upper()],
                    include_thoughts=False,
                )
            if not effort or effort == "auto":
                return gtypes.ThinkingConfig(include_thoughts=show)
            level = self._clamp_level(info.get("thinking_levels"), effort)
            return gtypes.ThinkingConfig(
                thinking_level=gtypes.ThinkingLevel[level.upper()],
                include_thoughts=show,
            )

        # mode == "budget" (Gemini 2.5)
        lo, hi = (info.get("budget_range") or (0, 0))
        can_disable = bool(info.get("thinking_can_disable"))
        if task:
            budget = 0 if can_disable else lo
            return gtypes.ThinkingConfig(thinking_budget=budget, include_thoughts=False)
        e = str(effort or "").lower()
        if e in self._EFFORT_BUDGETS:
            budget = self._EFFORT_BUDGETS[e]
        elif e in ("none", "minimal"):
            budget = 0 if can_disable else lo
        else:
            budget = valve_budget
            if budget != -1:
                budget = max(lo, min(hi, budget))
        return gtypes.ThinkingConfig(thinking_budget=budget, include_thoughts=show)

    def _build_generate_config(
        self,
        *,
        info: dict,
        system_instruction,
        tools,
        tool_config,
        thinking,
        body: dict,
        user_valves,
        purpose: str = "chat",
    ):
        """The GenerateContentConfig for one call (image models extend this in
        the image-generation milestone)."""
        kwargs: dict = {
            "automatic_function_calling": gtypes.AutomaticFunctionCallingConfig(
                disable=True
            )
        }
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if tools:
            kwargs["tools"] = tools
        if tool_config is not None:
            kwargs["tool_config"] = tool_config
        if thinking is not None:
            kwargs["thinking_config"] = thinking

        body = body or {}
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        if max_tokens:
            max_output = info.get("max_output")
            try:
                kwargs["max_output_tokens"] = (
                    min(int(max_tokens), max_output) if max_output else int(max_tokens)
                )
            except (TypeError, ValueError):
                pass

        stop = body.get("stop")
        if stop:
            kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

        resolution = (
            user_valves.MEDIA_RESOLUTION if user_valves is not None else "auto"
        ) or "auto"
        if resolution in ("low", "medium", "high"):
            kwargs["media_resolution"] = gtypes.MediaResolution[
                "MEDIA_RESOLUTION_" + resolution.upper()
            ]

        if GENAI_HAS_STORE:
            kwargs["store"] = bool(self.valves.STORE_REQUESTS)

        response_format = body.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            kwargs["response_mime_type"] = "application/json"
            schema = (response_format.get("json_schema") or {}).get("schema")
            if schema:
                kwargs["response_json_schema"] = schema

        return gtypes.GenerateContentConfig(**kwargs)

    @staticmethod
    def _block_visibility_key(name: str) -> str:
        """Map a block name onto its visibility key.

        A single logical block reaches the formatters under several names, but
        the user hides *one* concept. Normalising here keeps the HIDE_BLOCKS
        valve spelled in concepts rather than wire names.
        """
        return name

    def _is_block_hidden(self, name: str) -> bool:
        """True when this block concept is not to be rendered for this request.

        Sub-agent runs (SLIM_OUTPUT) render no decoration at all; otherwise the
        request-scoped HIDDEN_BLOCKS ContextVar decides.
        """
        if SLIM_OUTPUT.get():
            return True
        hidden = HIDDEN_BLOCKS.get()
        if not hidden:
            return False
        return self._block_visibility_key(name) in hidden

    @staticmethod
    def _parse_hidden_blocks(raw: Any) -> frozenset:
        """Normalize the HIDE_BLOCKS valve into a set of block concept keys.

        The valve is a multiselect (``list[str]``); values saved under the
        previous comma-separated ``str`` form are still accepted so an upgrade
        does not silently drop a user's preference.
        """
        if isinstance(raw, str):
            raw = raw.split(",")
        if not raw:
            return frozenset()
        return frozenset(str(part).strip() for part in raw if str(part).strip())

    @staticmethod
    def _append_block_to_text(text: str, block: str) -> str:
        """Append a rendered block with a safe markdown/html separator."""
        if not text:
            return block
        if not block:
            return text
        if text.endswith(("\n", "\r")) or block.startswith(("\n", "\r")):
            return text + block
        return f"{text}\n{block}"

    def _format_thinking_block(
        self, content: str, done: bool = False, duration: Optional[float] = None
    ) -> str:
        """Format a thought with OpenWebUI native <details type="reasoning">.

        ``done`` closes the block (spinner off, "Thought for N seconds"); while
        streaming the block stays open under "Thinking…".
        """
        if self._is_block_hidden("reasoning"):
            return ""

        # Escape content and add > prefix per line (OpenWebUI quote block style)
        escaped_lines = "\n".join(
            f"> {html.escape(line)}" if not line.startswith(">") else html.escape(line)
            for line in content.splitlines()
        )

        if done:
            duration_int = int(duration or 0)
            duration_attr = f' duration="{duration_int}"' if duration is not None else ""
            return (
                f'<details type="reasoning" done="true"{duration_attr}>\n'
                f"<summary>Thought for {duration_int} seconds</summary>\n"
                f"{escaped_lines}\n"
                f"</details>\n"
            )
        return (
            '<details type="reasoning" done="false">\n'
            "<summary>Thinking…</summary>\n"
            f"{escaped_lines}\n"
            "</details>\n"
        )

    def _format_tool_result_block(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict,
        tool_output: str,
        is_error: bool = False,
        done: bool = True,
        embeds: list = None,
        files: list = None,
    ) -> str:
        """Format a tool result block with OpenWebUI native <details type="tool_calls">.

        ``done`` False shows "Executing..." with spinner; True, "Tool Executed".
        ``embeds``/``files`` carry the process_tool_result output (HTML strings,
        file chips) into the block.
        """
        if SLIM_OUTPUT.get():
            # The tool ran; only its effect on the answer matters to the parent
            # agent. Arguments, raw result, embeds and file chips are markup a
            # human would expand -- nobody will.
            return ""

        # Escape arguments for HTML attribute
        escaped_args = (
            html.escape(json.dumps(tool_input, ensure_ascii=False)) if tool_input else ""
        )

        done_str = "true" if done else "false"
        summary = "Tool Executed" if done else "Executing..."
        error_attr = ' error="true"' if is_error and done else ""

        if done:
            # Escape result for HTML attribute
            try:
                if isinstance(tool_output, str):
                    try:
                        parsed = json.loads(tool_output)
                        escaped_result = html.escape(json.dumps(parsed, ensure_ascii=False))
                    except (json.JSONDecodeError, ValueError):
                        escaped_result = html.escape(json.dumps(tool_output, ensure_ascii=False))
                else:
                    escaped_result = html.escape(json.dumps(tool_output, ensure_ascii=False))
            except Exception:
                escaped_result = html.escape(json.dumps(str(tool_output), ensure_ascii=False))

            escaped_embeds = (
                html.escape(json.dumps(embeds, ensure_ascii=False)) if embeds else ""
            )

            return (
                f'<details type="tool_calls" done="{done_str}" id="{html.escape(tool_call_id)}" name="{html.escape(tool_name)}" '
                f'arguments="{escaped_args}" result="{escaped_result}" '
                f'files="{html.escape(json.dumps(files)) if files else ""}" '
                f'embeds="{escaped_embeds}"{error_attr}>\n'
                f"<summary>{summary}</summary>\n"
                f"</details>\n"
            )
        return (
            f'<details type="tool_calls" done="{done_str}" id="{html.escape(tool_call_id)}" name="{html.escape(tool_name)}" '
            f'arguments="{escaped_args}">\n'
            f"<summary>{summary}</summary>\n"
            f"</details>\n"
        )

    def _format_google_search_block(
        self,
        call: int,
        queries: List[str],
        sources: List[Tuple[int, str, str]],
        embed_html: str = "",
    ) -> str:
        """The Google Search block: the queries asked, the sources found, and
        the Search Suggestions embed Google's terms require us to show."""
        if self._is_block_hidden("google_search"):
            return ""

        escaped_args = html.escape(json.dumps({"queries": list(queries or [])}, ensure_ascii=False))
        escaped_result = html.escape(
            json.dumps(
                [{"n": n, "title": title, "url": url} for n, title, url in (sources or [])],
                ensure_ascii=False,
            )
        )
        escaped_embeds = (
            html.escape(json.dumps([embed_html], ensure_ascii=False)) if embed_html else ""
        )

        shown = [q for q in (queries or []) if q][:3]
        summary = "Searched Google: " + " · ".join(shown) if shown else "Searched Google"
        if len([q for q in (queries or []) if q]) > 3:
            summary += f" +{len([q for q in queries if q]) - 3} more"

        body = ""
        if sources:
            body = "\n" + "\n".join(f"- [{n}] [{title}]({url})" for n, title, url in sources) + "\n"

        return (
            f'<details type="tool_calls" done="true" id="gsearch_{call}" name="google_search" '
            f'data-gemini-builtin="1" arguments="{escaped_args}" result="{escaped_result}" '
            f'embeds="{escaped_embeds}">\n'
            f"<summary>{summary}</summary>\n"
            f"{body}</details>\n"
        )

    def _format_url_context_block(self, call: int, entries: List[dict]) -> str:
        """The URL context block: one line per URL from the conversation,
        marked ✅ when the model could read it, ❌ otherwise."""
        if self._is_block_hidden("url_context"):
            return ""

        lines = []
        for entry in entries or []:
            url = entry.get("url") or ""
            status = str(entry.get("status") or "")
            if status.lower() == "ok":
                lines.append(f"- ✅ [{url}]({url})")
            else:
                lines.append(f"- ❌ {url}" + (f" ({status})" if status else ""))
        body = ("\n" + "\n".join(lines) + "\n") if lines else ""

        return (
            f'<details type="tool_calls" done="true" id="urlctx_{call}" name="url_context" '
            f'data-gemini-builtin="1">\n'
            f"<summary>Read {len(entries or [])} pages</summary>\n"
            f"{body}</details>\n"
        )

    def _format_suggestion_links_line(self, links: List[Tuple[str, str]]) -> str:
        """The plain-text fallback for Search Suggestions (valve "links")."""
        if not links:
            return ""
        return (
            "🔎 **Google Search suggestions:** "
            + " · ".join(f"[{label}]({href})" for label, href in links)
            + "\n"
        )

    def _format_notice(self, text: str) -> str:
        """A short warning line appended to the answer (degraded capability, …)."""
        return "\n> ⚠️ " + text + "\n"

    # Model names containing any of these are not chat models, whatever the
    # API metadata says (embeddings, TTS, video, audio, agents, open models).
    _CHAT_MODEL_REJECT_TOKENS = (
        "embedding",
        "aqa",
        "tts",
        "native-audio",
        "live",
        "lyria",
        "veo",
        "imagen",
        "computer-use",
        "robotics",
        "deep-research",
        "agent",
        "learnlm",
        "gemma",
        "transcribe",
        "omni",
    )

    # Served when live discovery fails with no usable cache and the endpoint
    # returned no models: the chat models a default install can expect.
    _STATIC_FALLBACK_MODELS = (
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-nano-banana-2.1",
    )

    def _build_genai_client(self, api_key: str, timeout_s: Optional[float] = None):
        """Central google-genai client factory and the single decryption point.

        Every request path (model listing, generation, Files API, tasks) builds
        its client here so GEMINI_BASE_URL and the request timeout stay
        consistent. Resolves ``genai.Client`` from module scope on purpose so
        tests can monkeypatch it.
        """
        if genai is None:
            raise RuntimeError("google-genai is not installed")
        key = decrypt_valve_secret(api_key).strip()
        if not key:
            raise ValueError("No Gemini API key configured")
        opts: Dict[str, Any] = {
            "timeout": int((timeout_s or self.valves.REQUEST_TIMEOUT) * 1000),
            "retry_options": gtypes.HttpRetryOptions(attempts=1),
        }
        base_url = (self.valves.GEMINI_BASE_URL or "").strip()
        if base_url:
            opts["base_url"] = base_url
        # Only fields the installed SDK version actually knows: proxies and
        # older releases reject unknown HttpOptions keys.
        opts = {k: v for k, v in opts.items() if k in gtypes.HttpOptions.model_fields}
        return genai.Client(
            api_key=key, vertexai=False, http_options=gtypes.HttpOptions(**opts)
        )

    def _resolve_api_key(self, user_valves) -> str:
        """The user's stored key if non-empty, else the admin key.

        Both come back still encrypted; decryption happens in
        ``_build_genai_client`` only.
        """
        user_key = ""
        if user_valves:
            value = (
                user_valves.get("GEMINI_API_KEY")
                if isinstance(user_valves, dict)
                else getattr(user_valves, "GEMINI_API_KEY", "")
            )
            user_key = str(value or "").strip()
        return user_key or str(self.valves.GEMINI_API_KEY or "")

    def _api_key_fingerprint(self, api_key: str) -> str:
        """sha1 hex of the stored (possibly encrypted) key, for logging."""
        return hashlib.sha1(str(api_key or "").encode("utf-8")).hexdigest()

    def _model_name_from_body(self, body) -> str:
        return body["model"].split("/")[-1]

    @staticmethod
    def _parse_generation(name: str) -> Optional[float]:
        """Major.minor from the first ``gemini-<N>.<M>``; None when absent."""
        match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        return float(match.group(1)) if match else None

    def _is_chat_model(self, d: dict) -> bool:
        raw_name = str(d.get("name") or "")
        name = raw_name[len("models/"):] if raw_name.startswith("models/") else raw_name
        actions = d.get("supported_actions")
        if actions and "generateContent" not in actions:
            return False
        if not name.startswith("gemini-"):
            return False
        lowered = name.lower()
        for token in self._CHAT_MODEL_REJECT_TOKENS:
            if token in lowered:
                return False
        generation = self._parse_generation(name)
        if generation is not None:
            return generation >= 2.5
        return "nano-banana" in name

    def _derive_capabilities(self, name: str, api_model: Optional[dict] = None) -> dict:
        """Capabilities for one model: rules, then pins, then the valve overrides."""
        api_model = api_model or {}
        if "image" in name or "nano-banana" in name:
            family = "image"
        elif "flash-lite" in name:
            family = "flash-lite"
        elif "flash" in name:
            family = "flash"
        elif "pro" in name:
            family = "pro"
        else:
            family = "other"

        generation = self._parse_generation(name)
        if generation is None:
            generation = 3.0

        if api_model.get("thinking") is False:
            thinking = "none"
        elif family == "image":
            thinking = "always"
        elif generation < 3:
            thinking = "budget"
        else:
            thinking = "level"

        if thinking == "level":
            thinking_levels = ["low", "medium", "high"]
            if (
                family == "flash-lite"
                or name.startswith("gemini-3.6-flash")
                or name.startswith("gemini-3-flash")
            ):
                thinking_levels = ["minimal"] + thinking_levels
        else:
            thinking_levels = []

        if generation == 2.5:
            if family == "pro":
                budget_range = [128, 32768]
            elif family == "flash":
                budget_range = [0, 24576]
            elif family == "flash-lite":
                budget_range = [512, 24576]
            else:
                budget_range = None
        else:
            budget_range = None

        image_out = family == "image"
        info = {
            "generation": generation,
            "family": family,
            "thinking": thinking,
            "thinking_levels": thinking_levels,
            "budget_range": budget_range,
            "thinking_can_disable": generation == 2.5 and family in ("flash", "flash-lite"),
            "search": not (family == "image" and "flash-lite" in name),
            "url_context": not image_out,
            "maps": not image_out,
            "maps_with_search": generation >= 3.5,
            "functions": not image_out,
            "builtin_with_functions": bool(
                not image_out
                and generation >= 3
                and GENAI_HAS_SERVER_SIDE_INVOCATIONS
                and GENAI_HAS_TOOL_CALL_PARTS
            ),
            "image_out": image_out,
            "image_sizes": ["1K", "2K", "4K"] if image_out else [],
            "context_length": api_model.get("input_token_limit") or 1048576,
            "max_output": api_model.get("output_token_limit") or 65536,
        }
        self._apply_capability_patterns(info, Pipe.MODEL_CAPABILITY_PINS, name)
        self._apply_capability_patterns(info, self._model_capability_overrides(), name)
        return info

    def _model_capability_overrides(self) -> dict:
        """The MODEL_CAPABILITY_OVERRIDES valve parsed; invalid JSON is ignored."""
        raw = (self.valves.MODEL_CAPABILITY_OVERRIDES or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as e:
            logger.warning(f"MODEL_CAPABILITY_OVERRIDES is not valid JSON; ignored: {e}")
            return {}
        if not isinstance(parsed, dict):
            logger.warning("MODEL_CAPABILITY_OVERRIDES must be a JSON object; ignored")
            return {}
        return {
            str(pattern): values
            for pattern, values in parsed.items()
            if isinstance(values, dict)
        }

    def _apply_capability_patterns(self, info: dict, table: dict, name: str) -> None:
        """Apply {pattern: {key: value}} entries; a trailing * is a prefix match.

        All matching patterns apply, shortest pattern first, so a specific
        "gemini-2.5*" rule is not clobbered by a broad "gemini-*" one.
        """
        matches = []
        for pattern in table:
            if pattern.endswith("*"):
                if name.startswith(pattern[:-1]):
                    matches.append((len(pattern), pattern))
            elif pattern == name:
                matches.append((len(pattern), pattern))
        for _length, pattern in sorted(matches):
            info.update(table[pattern])

    def get_model_info(self, name: str) -> dict:
        """Capabilities by model name, from the API cache when discovered."""
        return self._derive_capabilities(name, Pipe._api_models.get(name))

    def _model_cache_signature(self) -> str:
        """Fingerprint of the settings the model list depends on.

        Uses the stored (possibly encrypted) values rather than decrypting: the
        stored values are stable between saves, and decrypting could raise on a
        rotated WEBUI_SECRET_KEY and take the model list down with it.
        """
        parts = [
            (self.valves.GEMINI_API_KEY or "").strip(),
            (self.valves.GEMINI_BASE_URL or "").strip(),
            (self.valves.ENABLED_MODELS or "").strip(),
            (self.valves.MODEL_CAPABILITY_OVERRIDES or "").strip(),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    async def get_gemini_models(self) -> List[dict]:
        """Discover the chat models via the API and cache the result.

        The cache lives on the class: one Pipe instance serves all users, and a
        per-request cache would refetch on every ``pipes()`` call.
        """
        enabled_raw = (self.valves.ENABLED_MODELS or "").strip()
        if enabled_raw:
            # Explicit allow-list: endpoints without a model-listing route can
            # still surface their models, and no API call is spent.
            enabled = [m.strip() for m in enabled_raw.split(",") if m.strip()]
            return [
                self._build_openwebui_model_entry(name, self.get_model_info(name))
                for name in enabled
            ]

        if not str(self.valves.GEMINI_API_KEY or "").strip():
            return [{"id": "gemini/error", "name": "Gemini: set GEMINI_API_KEY"}]

        cache_sig = self._model_cache_signature()
        ttl = int(self.valves.MODEL_CACHE_TTL_MINUTES or 0) * 60
        if (
            Pipe._model_cache
            and Pipe._model_cache_sig == cache_sig
            and ttl > 0
            and time.time() - Pipe._model_cache_ts < ttl
        ):
            return list(Pipe._model_cache)

        entries: List[dict] = []
        try:
            client = self._build_genai_client(self.valves.GEMINI_API_KEY)
            async for model in await client.aio.models.list(config={"page_size": 1000}):
                d = model.model_dump()
                raw_name = str(d.get("name") or "")
                name = raw_name[len("models/"):] if raw_name.startswith("models/") else raw_name
                if not name or not self._is_chat_model(d):
                    continue
                info = self._derive_capabilities(name, d)
                Pipe._api_models[name] = d
                entries.append(
                    self._build_openwebui_model_entry(name, info, str(d.get("display_name") or ""))
                )

            if not entries:
                logger.warning("Model listing returned no chat models; using static fallback")
                return self._static_fallback_models()

            Pipe._model_cache = list(entries)
            Pipe._model_cache_ts = time.time()
            Pipe._model_cache_sig = cache_sig
            logger.info(f"Cached model list: {len(entries)} chat models")
            return entries
        except Exception as e:
            logger.warning(f"Could not fetch models from the Gemini API: {e}")
            # A stale cache from the same settings still lists this endpoint's
            # models; one from different settings would not, so a changed
            # signature drops through to the static fallback.
            if Pipe._model_cache and Pipe._model_cache_sig == cache_sig:
                logger.info("Using stale model list cache as fallback")
                return list(Pipe._model_cache)
            return self._static_fallback_models()

    @staticmethod
    def _build_openwebui_model_entry(name: str, info: dict, display_name: str = "") -> dict:
        return {
            "id": MODEL_ID_PREFIX + name,
            "name": display_name or name,
            "context_length": info["context_length"],
            "max_output_tokens": info["max_output"],
            "supports_vision": True,
            "supports_thinking": info["thinking"] != "none",
            "info": {"meta": {"capabilities": {"status_updates": True}}},
        }

    def _static_fallback_models(self) -> List[dict]:
        return [
            self._build_openwebui_model_entry(name, self.get_model_info(name))
            for name in self._STATIC_FALLBACK_MODELS
        ]

    async def pipes(self) -> List[dict]:
        """OpenWebUI entry point returning the list of available Gemini models."""
        return await self.get_gemini_models()

    # OpenWebUI's background memory review arrives as a task whose name is not
    # in the TASKS enum -- functions.py forwards metadata.task verbatim.
    MEMORY_REVIEW_TASK = "memory_review"

    # Response schemas for the task requests whose prompt asks for raw JSON.
    #
    # OpenWebUI parses these answers with json.loads and drops the whole task
    # when it fails, so every one of its prompts spends several lines begging
    # for "no markdown fences, no preamble". Structured outputs make that a
    # guarantee instead of a plea: the model cannot emit anything but a
    # conforming object.
    #
    # Keys are the task names OpenWebUI passes in metadata.task. Deliberately
    # absent: emoji_generation and moa_response_generation (their prompts ask
    # for prose, not JSON).
    #
    # `additionalProperties: false` everywhere -- OpenWebUI reads exactly one
    # key out of each of these and extra keys are pure token cost.
    TASK_RESPONSE_SCHEMAS: Dict[str, dict] = {
        "title_generation": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
        "tags_generation": {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
        "follow_up_generation": {
            "type": "object",
            "properties": {
                "follow_ups": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["follow_ups"],
            "additionalProperties": False,
        },
        "query_generation": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
        "image_prompt_generation": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "autocomplete_generation": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        MEMORY_REVIEW_TASK: {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "replace", "move", "remove"],
                            },
                            "id": {"type": "string"},
                            "type": {"type": "string", "enum": ["user", "context"]},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        # One flat object rather than a per-action union: the
                        # required field set differs per action, and structured
                        # outputs does not accept anyOf/oneOf. OpenWebUI
                        # validates the per-action requirements itself
                        # (validate_memory_operations), so pinning the action
                        # vocabulary and the field names is the useful part --
                        # it is what stops the model from inventing the
                        # "score"/"importance"/"stability" keys the prompt
                        # explicitly warns against.
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["operations"],
            "additionalProperties": False,
        },
    }

    _PATTERN_ANY_DETAILS = re.compile(
        r"<details\b[^>]*>.*?</details>", flags=re.DOTALL
    )
    _PATTERN_METADATA_MARKER = re.compile(PATTERN_METADATA_MARKER)
    _PATTERN_TRAILING_SPACES = re.compile(r"[ \t]+$", flags=re.MULTILINE)
    _PATTERN_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

    def _resolve_task_model(self, body: dict) -> str:
        """The model for a task request: the TASK_MODEL valve when set, else
        the chat model of the body."""
        chat_model = self._model_name_from_body(body)
        override = str(self.valves.TASK_MODEL or "").strip()
        if not override:
            return chat_model
        if override.startswith(MODEL_ID_PREFIX):
            override = override[len(MODEL_ID_PREFIX) :]
        return override

    @classmethod
    def _sanitize_task_text(cls, text: str) -> str:
        """Reduce persisted chat content to the prose a task model actually
        needs.

        OpenWebUI hands task requests the stored message content, which carries
        every collapsible this pipe ever wrote plus the invisible metadata
        markers used for replay. None of it round-trips: task requests are
        one-shot. Stripping it cuts the bill and stops the task model from
        reasoning about its own UI artefacts. It matters most for the memory
        review, which truncates each message and would otherwise spend that
        budget on UI artefacts.
        """
        if not text or ("<" not in text and "[" not in text):
            return text
        cleaned = cls._PATTERN_ANY_DETAILS.sub("\n", text)
        cleaned = cls._PATTERN_METADATA_MARKER.sub(" ", cleaned)
        cleaned = cls._PATTERN_TRAILING_SPACES.sub("", cleaned)
        return cls._PATTERN_EXCESS_BLANK_LINES.sub("\n\n", cleaned).strip()

    @classmethod
    def _extract_task_system(cls, messages: List[dict]) -> str:
        """The system messages of a task request, joined into one string.

        Returned as plain text: task requests are one-shot and never cached,
        so there is nothing fancier to attach them to.
        """
        parts: List[str] = []
        for msg in messages or []:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                continue
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    def _process_messages_for_task(
        self, messages: List[dict]
    ) -> List[gtypes.Content]:
        """The conversation as text-only Gemini Contents for a task request.

        UI artefacts are stripped per message (see _sanitize_task_text); a
        message that sanitises to nothing is dropped, and the same-role
        neighbours a drop leaves behind are merged instead of being sent as
        an invalid sequence.
        """
        processed: List[gtypes.Content] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                continue
            text = self._sanitize_task_text(text)
            if not text:
                continue
            if processed and processed[-1].role == gemini_role:
                previous = processed[-1].parts[0].text
                processed[-1].parts[0] = gtypes.Part(text=f"{previous}\n\n{text}")
            else:
                processed.append(
                    gtypes.Content(role=gemini_role, parts=[gtypes.Part(text=text)])
                )
        return processed

    def _task_config_kwargs(
        self, info: dict, system_instruction: Optional[str], thinking, task: Optional[str]
    ) -> Tuple[dict, bool]:
        """The GenerateContentConfig kwargs for one task call, and whether the
        structured-output fields are in them: no tools, AFC disabled, the
        task's thinking config, and the task's response schema when it has
        one."""
        schema = (
            self.TASK_RESPONSE_SCHEMAS.get(task) if isinstance(task, str) else None
        )
        kwargs: dict = {
            "automatic_function_calling": gtypes.AutomaticFunctionCallingConfig(
                disable=True
            )
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if thinking is not None:
            kwargs["thinking_config"] = thinking
        if schema:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = schema
        return kwargs, bool(schema)

    async def _task_generate(
        self,
        client,
        model: str,
        chat_model: str,
        contents: List[gtypes.Content],
        system_instruction: str,
        user_valves,
        body: dict,
        task: Optional[str],
    ):
        """The task's generate_content call, with its two self-healing
        retries: a 404 from the TASK_MODEL valve falls back to the chat
        model, and any other failure drops the structured output and retries
        once plain -- losing the schema costs a markdown fence, losing the
        request costs the whole task."""
        info = self.get_model_info(model)
        kwargs, structured = self._task_config_kwargs(
            info,
            system_instruction,
            self._build_thinking_config(info, user_valves, body, purpose="task"),
            task,
        )
        try:
            return await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=gtypes.GenerateContentConfig(**kwargs),
            )
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 404 and model != chat_model:
                logger.info(
                    f"Task model {model} not found; retrying with the chat model {chat_model}"
                )
                info = self.get_model_info(chat_model)
                kwargs, _structured = self._task_config_kwargs(
                    info,
                    system_instruction,
                    self._build_thinking_config(
                        info, user_valves, body, purpose="task"
                    ),
                    task,
                )
                return await client.aio.models.generate_content(
                    model=chat_model,
                    contents=contents,
                    config=gtypes.GenerateContentConfig(**kwargs),
                )
            if structured:
                logger.warning(
                    f"Task {task} rejected with structured outputs, retrying "
                    f"without: {getattr(e, 'message', e)}"
                )
                kwargs.pop("response_mime_type", None)
                kwargs.pop("response_json_schema", None)
                return await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gtypes.GenerateContentConfig(**kwargs),
                )
            raise

    async def _run_task(self, body, task, __user__, user_valves) -> str:
        """One non-streaming task request (title, tags, follow-ups, ...).

        Returns the answer's non-thought text, stripped, or "" when the task
        failed: OpenWebUI drops failed tasks silently, so a total task outage
        must still leave a warning behind.
        """
        client = None
        try:
            body = body or {}
            user_valves = self.UserValves() if user_valves is None else user_valves
            api_key = self._resolve_api_key(user_valves)
            client = self._build_genai_client(
                api_key, timeout_s=self.valves.REQUEST_TIMEOUT
            )
            messages = body.get("messages") or []
            contents = self._process_messages_for_task(messages)
            system_instruction = self._extract_task_system(messages)
            if not contents:
                return ""
            response = await self._task_generate(
                client,
                self._resolve_task_model(body),
                self._model_name_from_body(body),
                contents,
                system_instruction,
                user_valves,
                body,
                task,
            )
        except Exception as e:
            # Warning, not debug: returning "" makes OpenWebUI drop the task
            # silently, so a total task outage must be visible at default log
            # settings.
            logger.warning(f"Task model error ({task}): {e}")
            return ""
        finally:
            if client is not None:
                try:
                    await client.aio.aclose()
                except Exception:
                    pass

        text_parts: List[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text and not getattr(part, "thought", False):
                    text_parts.append(text)
        result = "".join(text_parts).strip()
        logger.debug(f"Task response: {len(result)}c")
        return result

    # END GENERATED SECTION: gemini_pipe.pipe_method_groups
