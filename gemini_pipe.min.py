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
from urllib.parse import urlparse, quote_plus
from pydantic import BaseModel, Field, field_validator

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

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except Exception:
    PILImage = None
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

PATTERN_METADATA_MARKER = r"[ ]?\[\]\(gemini:([^)]+)\)[ ]?"

PATTERN_USER_CONTEXT = re.compile(r"\nUser Context:\n(.*)$", flags=re.DOTALL)

PATTERN_MEMORY_CONTEXT = re.compile(
    r"<memory_context>(.*?)</memory_context>", flags=re.DOTALL
)

MEMORY_CONTEXT_APPENDIX_HEADER = (
    "\n\n---\n**IMPORTANT:** The following is NOT part of the user's message, "
    "but context from a memory system to help answer the user's questions:\n\n"
)

PATTERN_EMPTY_CONTEXT = re.compile(r"<context>\s*</context>", flags=re.DOTALL)

PATTERN_SOURCE_TAGS = re.compile(r"<source[^>]*>.*?</source>", flags=re.DOTALL)

PATTERN_RAG_MESSAGE = re.compile(r"### Task:.*?<context>.*?</context>", re.DOTALL)

GENAI_HAS_SERVER_SIDE_INVOCATIONS = GENAI_AVAILABLE and "include_server_side_tool_invocations" in gtypes.ToolConfig.model_fields
GENAI_HAS_TOOL_CALL_PARTS = GENAI_AVAILABLE and "tool_call" in gtypes.Part.model_fields
GENAI_HAS_STORE = GENAI_AVAILABLE and "store" in gtypes.GenerateContentConfig.model_fields
DUMMY_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"
MODEL_ID_PREFIX = "gemini/"

HIDDEN_BLOCKS: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "gemini_pipe_hidden_blocks", default=frozenset()
)

SLIM_OUTPUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gemini_pipe_slim_output", default=False
)

TOOL_APPROVAL: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "gemini_pipe_tool_approval", default=("full", None)
)

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

try:
    from open_webui.env import ENABLE_VALVE_ENCRYPTION as OPENWEBUI_ENCRYPTS_VALVES
except Exception:
    OPENWEBUI_ENCRYPTS_VALVES = False

def _valve_encryption_key() -> Optional[bytes]:
    if not VALVE_ENCRYPTION_AVAILABLE:
        return None
    secret = os.environ.get("WEBUI_SECRET_KEY", "").strip()
    if not secret:
        return None

    if len(secret) == 44:
        try:
            Fernet(secret.encode())
            return secret.encode()
        except Exception:
            pass
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())

def encrypt_valve_secret(value: str) -> str:
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

    @classmethod
    def _validate(cls, value: Any) -> "EncryptedStr":
        return cls(encrypt_valve_secret(str(value or "")))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_after_validator_function(
            cls._validate, core_schema.str_schema()
        )

try:
    from open_webui.utils.tools import get_builtin_tools

    BUILTIN_TOOLS_AVAILABLE = True
except ImportError:
    get_builtin_tools = None
    BUILTIN_TOOLS_AVAILABLE = False

try:
    from open_webui.utils.middleware import process_tool_result

    PROCESS_TOOL_RESULT_AVAILABLE = True
except ImportError:
    process_tool_result = None
    PROCESS_TOOL_RESULT_AVAILABLE = False

try:
    from open_webui.models.files import Files, FileForm
    from open_webui.storage.provider import Storage

    FILES_AVAILABLE = True
except ImportError:
    Files = None
    FileForm = None
    Storage = None
    FILES_AVAILABLE = False

try:
    from open_webui.models.chats import Chats

    CHATS_AVAILABLE = True
except ImportError:
    Chats = None
    CHATS_AVAILABLE = False

try:
    from open_webui.models.users import Users

    USERS_AVAILABLE = True
except ImportError:
    Users = None
    USERS_AVAILABLE = False

try:
    from open_webui.routers.images import upload_image

    UPLOAD_IMAGE_AVAILABLE = True
except ImportError:
    upload_image = None
    UPLOAD_IMAGE_AVAILABLE = False

try:
    from open_webui.utils.files import get_image_base64_from_url

    IMAGE_URL_FETCH_AVAILABLE = True
except ImportError:
    get_image_base64_from_url = None
    IMAGE_URL_FETCH_AVAILABLE = False

def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)

def _get(obj: Any, name: str) -> Any:
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
    if chunk.get("uri"):
        return chunk["uri"]
    if chunk.get("place_id"):
        return str(chunk["place_id"])
    return None

@dataclass
class GroundingAccumulator:

    chunks: List[dict] = field(default_factory=list)
    supports: List[dict] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    entry_point_html: str = ""

    def add(self, gm: Any) -> None:
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

                self.chunks = list(new_chunks)
            else:

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
    return len(text.encode("utf-8")[:byte_off].decode("utf-8", "ignore"))

def _fence_line_spans(text: str) -> List[Tuple[int, int]]:
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
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    rest = ""
    if parsed.query:
        rest += "?" + parsed.query
    if parsed.fragment:
        rest += "#" + parsed.fragment
    return f"{(parsed.scheme or '').lower()}://{(parsed.netloc or '').lower()}{path}{rest}"

class CitationRegistry:

    def __init__(self, offset: int = 0):
        self.offset = offset
        self.sources: Dict[int, dict] = {}
        self._by_key: Dict[str, int] = {}

    def number_for(self, key: str, info: dict) -> Tuple[int, bool]:
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
    if not html_text:
        return ""
    index = html_text.find("<head>")
    if index != -1:
        at = index + len("<head>")
        return html_text[:at] + '<base target="_blank">' + html_text[at:]
    return '<base target="_blank">' + html_text

def _rag_citation_offset(sources: List[dict]) -> int:
    seen = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        for _doc, meta in zip(source.get("document") or [], source.get("metadata") or []):
            meta = meta if isinstance(meta, dict) else {}
            source_id = meta.get("source") or (source.get("source") or {}).get("id") or "N/A"
            seen.add(str(source_id))
    return len(seen)

@dataclass
class Segment:

    kind: str
    key: str
    call: int
    content: str

@dataclass
class RenderedMessage:

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

    calls: List[CallState] = field(default_factory=list)
    citations: CitationRegistry = field(default_factory=CitationRegistry)
    usage_total: Dict[str, Any] = field(default_factory=dict)
    search_queries: List[str] = field(default_factory=list)
    tool_calls_made: int = 0
    degraded: set = field(default_factory=set)
    name_map: Dict[str, str] = field(default_factory=dict)

@dataclass
class PipeRequestContext:

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

class StatusEmitter:
    def __init__(self, emit_event: Callable[[dict], Awaitable[None]]):
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
        data: dict = {"description": description, "done": done}
        if hidden is not None:
            data["hidden"] = hidden
        data.update(fields)

        if not force and data == self._last_payload:
            return

        await self._emit_event({"type": "status", "data": data})
        self._last_payload = data

    async def waiting(self) -> None:
        await self.emit("Waiting for response...", hidden=False, force=True)

    async def thinking(self) -> None:
        await self.emit("💭 Thinking...")

    async def responding(self) -> None:
        await self.emit("Responding...")

    async def searching_google(self, queries: list) -> None:
        shown = " · ".join(queries[:3]) if queries else ""
        await self.emit(f"🔍 Searching Google: {shown}" if shown else "🔍 Searching Google...")

    async def web_search_done(self, urls: list, queries: list = None) -> None:
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
        if not urls:
            return
        await self.emit(
            "Found {{count}} places",
            action="web_search",
            urls=list(urls),
            count=len(urls),
        )

    async def fetching_url(self, url: str = "") -> None:
        await self.emit(f"🌐 Fetching {url}" if url else "🌐 Fetching URL...")

    async def running_tool(self, tool_name: str) -> None:
        await self.emit(f"🔧 Running {tool_name}..." if tool_name else "🔧 Running tool...")

    async def activity(self, description: str) -> None:
        await self.emit(description, done=False)

    async def complete(self, description: str) -> None:
        await self.emit(description, done=True, force=True)

    async def notification(self, content: str, *, type: str = "warning") -> None:
        await self._emit_event(
            {"type": "notification", "data": {"type": type, "content": content}}
        )

class Pipe:

    _model_cache: Optional[list] = None
    _model_cache_ts: float = 0.0
    _model_cache_sig: str = ""
    _api_models: Dict[str, dict] = {}
    _resolved_url_cache: Dict[str, str] = {}
    _gemini_file_cache: Dict[str, dict] = {}

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

    _STATIC_FALLBACK_MODELS = (
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-nano-banana-2.1",
    )

    def _build_genai_client(self, api_key: str, timeout_s: Optional[float] = None):
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

        opts = {k: v for k, v in opts.items() if k in gtypes.HttpOptions.model_fields}
        return genai.Client(
            api_key=key, vertexai=False, http_options=gtypes.HttpOptions(**opts)
        )

    def _resolve_api_key(self, user_valves) -> str:
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
        return hashlib.sha1(str(api_key or "").encode("utf-8")).hexdigest()

    def _model_name_from_body(self, body) -> str:
        return body["model"].split("/")[-1]

    @staticmethod
    def _parse_generation(name: str) -> Optional[float]:
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
        return self._derive_capabilities(name, Pipe._api_models.get(name))

    def _model_cache_signature(self) -> str:
        parts = [
            (self.valves.GEMINI_API_KEY or "").strip(),
            (self.valves.GEMINI_BASE_URL or "").strip(),
            (self.valves.ENABLED_MODELS or "").strip(),
            (self.valves.MODEL_CAPABILITY_OVERRIDES or "").strip(),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    async def get_gemini_models(self) -> List[dict]:
        enabled_raw = (self.valves.ENABLED_MODELS or "").strip()
        if enabled_raw:

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
        return await self.get_gemini_models()

    _PATTERN_URL = re.compile(r"https?://[^\s<>\"')\]]+")

    @staticmethod
    def _parse_lat_lng(*parts) -> Optional[Tuple[float, float]]:
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
        for match in self._PATTERN_URL.finditer(text or ""):
            host = urlparse(match.group(0)).netloc.lower()
            if host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be" or host.endswith(".youtu.be"):
                continue
            return match.group(0)
        return None

    def _resolve_builtin_flags(
        self, metadata: dict, valves, user_valves, info: dict, last_user_text: str
    ) -> dict:
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
        return name

    def _is_block_hidden(self, name: str) -> bool:
        if SLIM_OUTPUT.get():
            return True
        hidden = HIDDEN_BLOCKS.get()
        if not hidden:
            return False
        return self._block_visibility_key(name) in hidden

    @staticmethod
    def _parse_hidden_blocks(raw: Any) -> frozenset:
        if isinstance(raw, str):
            raw = raw.split(",")
        if not raw:
            return frozenset()
        return frozenset(str(part).strip() for part in raw if str(part).strip())

    @staticmethod
    def _append_block_to_text(text: str, block: str) -> str:
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
        if self._is_block_hidden("reasoning"):
            return ""

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
        if SLIM_OUTPUT.get():

            return ""

        escaped_args = (
            html.escape(json.dumps(tool_input, ensure_ascii=False)) if tool_input else ""
        )

        done_str = "true" if done else "false"
        summary = "Tool Executed" if done else "Executing..."
        error_attr = ' error="true"' if is_error and done else ""

        if done:

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
        if not links:
            return ""
        return (
            "🔎 **Google Search suggestions:** "
            + " · ".join(f"[{label}]({href})" for label, href in links)
            + "\n"
        )

    def _format_notice(self, text: str) -> str:
        return "\n> ⚠️ " + text + "\n"
