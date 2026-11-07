"""
title: Google Gemini API Manifold
id: gemini_pipe
author: AliD101v
github: https://github.com/AliD101v/openwebui_google_api_pipe
description: Google Gemini models via the native Gemini API: optional Google Search and Maps grounding with citations, thinking display, OpenWebUI tools, native files and YouTube, image generation, and usage/cost stats.
version: 0.1.3
license: MIT
requirements: google-genai>=1.68.0, pydantic>=2.0.0, cryptography
environment_variables:
    - GEMINI_API_KEY (or GOOGLE_API_KEY)

Supports:
- Model auto-discovery with per-generation capabilities and overrides
- Google Search, Maps and URL context grounding with citations
- Native function tools with a multi-call loop and approval prompts
- Thinking/reasoning display (level or budget, per model generation)
- Native audio, video, PDF, image and YouTube media (inline or Files API)
- Image generation on Nano Banana models
- Usage stats: context gauge, cached tokens, cost estimate
- Encrypted API keys (admin and per-user), store=false by default

Changelog:
v0.1.3
- Fix: a Maps-only answer no longer shows an empty "Google Search" block, and its queries are not counted as billed searches in the cost estimate
v0.1.2
- Fix: Google Maps is never sent with URL context either (URL context is dropped with a note); a "cannot be combined" 400 now retries once without the lower-priority built-in
v0.1.1
- Fix: Google Maps and Google Search are never sent together (the API rejects the pair with a 400); Maps wins and Search is dropped with a note
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
import random
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse, quote, quote_plus, unquote
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
    functions_offered: bool = False

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
    request: Any
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
        request=request if request is not None else {},
        model=model,
        info=dict(info or {}),
        api_key=api_key,
        tools=dict(tools or {}),
        builtin_tools=dict(builtin_tools or {}),
        rag_offset=rag_offset,
    )
    ctx.status = StatusEmitter(ctx.emit_event)
    return ctx

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

_ARGON_TIERS = [
    {"from": "2026-01-01", "input": 2.00, "output": 10.00, "cache_read": 0.10},
    {"from": "2026-12-01", "input": 4.00, "output": 20.00, "cache_read": 0.20},
]

_FLASH_3X_TIERS = [
    {"from": "2026-01-01", "input": 0.75, "output": 3.75, "cache_read": 0.075},
    {"from": "2027-01-01", "input": 1.50, "output": 7.50, "cache_read": 0.15},
]

def _single_tier(from_date, input_rate, output_rate, cache_rate, **extra):
    tier = {
        "from": from_date,
        "input": input_rate,
        "output": output_rate,
        "cache_read": cache_rate,
    }
    tier.update(extra)
    return [tier]

GEMINI_PRICES: dict = {
    "gemini-4-argon": _ARGON_TIERS,
    "gemini-4-argon*": _ARGON_TIERS,
    "gemini-3.8-flash": _FLASH_3X_TIERS,
    "gemini-3.8-flash*": _FLASH_3X_TIERS,
    "gemini-3.6-flash": _FLASH_3X_TIERS,
    "gemini-3.6-flash*": _FLASH_3X_TIERS,
    "gemini-3.5-flash-lite": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-3.5-flash-lite*": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": _single_tier("2026-01-01", 0.25, 1.50, 0.025),
    "gemini-3.1-flash-lite*": _single_tier("2026-01-01", 0.25, 1.50, 0.025),
    "gemini-3.1-pro-preview": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.20,
        long_threshold=200_000, input_long=4.00, output_long=18.00,
        cache_read_long=0.40,
    ),
    "gemini-3.1-pro-preview*": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.20,
        long_threshold=200_000, input_long=4.00, output_long=18.00,
        cache_read_long=0.40,
    ),
    "gemini-3-flash-preview": _single_tier("2026-01-01", 0.50, 3.00, 0.05),
    "gemini-3-flash-preview*": _single_tier("2026-01-01", 0.50, 3.00, 0.05),
    "gemini-2.5-pro": _single_tier(
        "2026-01-01", 1.25, 10.00, 0.125,
        long_threshold=200_000, input_long=2.50, output_long=15.00,
        cache_read_long=0.25,
    ),
    "gemini-2.5-pro*": _single_tier(
        "2026-01-01", 1.25, 10.00, 0.125,
        long_threshold=200_000, input_long=2.50, output_long=15.00,
        cache_read_long=0.25,
    ),
    "gemini-2.5-flash": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-2.5-flash*": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": _single_tier("2026-01-01", 0.10, 0.40, 0.01),
    "gemini-2.5-flash-lite*": _single_tier("2026-01-01", 0.10, 0.40, 0.01),
    "gemini-nano-banana-2.1": _single_tier(
        "2026-01-01", 1.50, 7.50, 0.0, image_output=30.00
    ),
    "gemini-nano-banana-2.1*": _single_tier(
        "2026-01-01", 1.50, 7.50, 0.0, image_output=30.00
    ),
    "gemini-3.1-flash-lite-image": _single_tier(
        "2026-01-01", 0.25, 1.50, 0.0, image_output=30.00
    ),
    "gemini-3.1-flash-lite-image*": _single_tier(
        "2026-01-01", 0.25, 1.50, 0.0, image_output=30.00
    ),
    "gemini-3-pro-image": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.0, image_output=120.00
    ),
    "gemini-3-pro-image*": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.0, image_output=120.00
    ),
}

SEARCH_PRICE_PER_1K = {"gen3": 14.0, "gen25": 35.0}
MAPS_PRICE_PER_1K = {"gen3": 14.0, "gen25": 25.0}

def _matching_pattern(table: dict, model: str):
    if model in table:
        return model
    best = None
    for pattern in table:
        if pattern.endswith("*") and model.startswith(pattern[:-1]):
            if best is None or len(pattern) > len(best):
                best = pattern
    return best

def _usage_field(entry, name):
    if isinstance(entry, dict):
        head, *rest = name.split("_")
        camel = head + "".join(part[:1].upper() + part[1:] for part in rest)
        if name in entry:
            return entry[name]
        return entry.get(camel)
    value = getattr(entry, name, None)
    if value is None:
        head, *rest = name.split("_")
        camel = head + "".join(part[:1].upper() + part[1:] for part in rest)
        value = getattr(entry, camel, None)
    return value

class GeminiPricing:

    def rates_for(self, model, on_date, overrides=None):
        if not model:
            return None
        model = str(model)
        pattern = _matching_pattern(GEMINI_PRICES, model)
        if pattern is None:
            return None
        on = on_date.isoformat()
        tier = None
        for candidate in GEMINI_PRICES[pattern]:
            if candidate["from"] <= on and (tier is None or candidate["from"] > tier["from"]):
                tier = candidate
        if tier is None:
            return None
        rates = dict(tier)
        if overrides:
            override_pattern = _matching_pattern(overrides, model)
            if override_pattern is not None and isinstance(overrides[override_pattern], dict):
                rates.update(overrides[override_pattern])
        return rates

    def call_cost(self, rates, usage) -> float:
        usage = usage or {}
        prompt = int(usage.get("prompt_token_count") or 0)
        cached = int(usage.get("cached_content_token_count") or 0)
        tool_prompt = int(usage.get("tool_use_prompt_token_count") or 0)
        candidates = int(usage.get("candidates_token_count") or 0)
        thoughts = int(usage.get("thoughts_token_count") or 0)
        image_tokens = 0
        for entry in usage.get("candidates_tokens_details") or []:
            modality = _usage_field(entry, "modality")
            if getattr(modality, "name", modality) == "IMAGE":
                image_tokens += int(_usage_field(entry, "token_count") or 0)
        out_text = candidates - image_tokens + thoughts
        threshold = rates.get("long_threshold")
        if threshold is not None and prompt > int(threshold):
            input_rate = rates.get("input_long", rates.get("input", 0.0))
            output_rate = rates.get("output_long", rates.get("output", 0.0))
            cache_rate = rates.get("cache_read_long", rates.get("cache_read", 0.0))
        else:
            input_rate = rates.get("input", 0.0)
            output_rate = rates.get("output", 0.0)
            cache_rate = rates.get("cache_read", 0.0)
        image_rate = rates.get("image_output", 0.0)
        return (
            (prompt - cached + tool_prompt) * input_rate
            + cached * cache_rate
            + out_text * output_rate
            + image_tokens * image_rate
        ) / 1e6

GEMINI_PRICING = GeminiPricing()

def format_usd(cost: float) -> str:
    if cost >= 1:
        return f"${cost:.2f}"
    if cost >= 0.01:
        return f"${cost:.3f}"
    return f"${cost:.4f}"

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
            description="When to attach the URL context tool: auto = when web search is on or the latest message contains a non-YouTube URL; with_search = only with web search; always; off",
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
            description="Send audio, video and full-context PDFs to Gemini natively (inline or via the Files API) instead of OpenWebUI's RAG text",
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
                "Collapsible blocks to hide from your chat (Google Search, URL "
                "context, reasoning). Progress still shows in the status line. "
                "With the Google Search block hidden, Search Suggestions are shown "
                "as links instead; the Google Maps attribution is always shown."
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
            description="Show the estimated USD cost of the turn in the status line and the message usage",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "gemini"
        self.name = "gemini/"
        self.valves = self.Valves()

    _METADATA_MARKER = re.compile(PATTERN_METADATA_MARKER)

    def _extract_and_remove_memories(self, text: str) -> tuple[str, Optional[str]]:

        if "<memory_context>" not in text and "User Context:" not in text:
            return text.strip(), None

        extracted_parts: List[str] = []

        def _take_memory_context(match) -> str:
            content = match.group(1).strip()
            if content:
                extracted_parts.append(content)
            return ""

        cleaned_text = PATTERN_MEMORY_CONTEXT.sub(_take_memory_context, text)

        match = PATTERN_USER_CONTEXT.search(cleaned_text)
        if match:
            context_content = match.group(1).strip()
            if context_content:
                extracted_parts.append(f"User Context:\n{context_content}")

            cleaned_text = cleaned_text[: match.start()]

        extracted_context = "\n\n".join(extracted_parts) if extracted_parts else None
        return cleaned_text.strip(), extracted_context

    def _create_metadata_marker(self, id: str, value: str, messagenum: int = 0) -> str:
        encoded_value = quote(value, safe="")
        return f" [](gemini:{messagenum}:{id}:{encoded_value}) "

    def _remove_sources_from_rag(self, rag_content: str, filenames_to_remove: List[str]) -> str:
        if not filenames_to_remove:
            return rag_content

        modified = rag_content
        for filename in filenames_to_remove:
            pattern = re.compile(
                rf'<source[^>]*name="{re.escape(filename)}"[^>]*>.*?</source>\s*',
                re.DOTALL,
            )
            modified = pattern.sub("", modified)

        if PATTERN_EMPTY_CONTEXT.search(modified) or not PATTERN_SOURCE_TAGS.search(modified):
            logger.debug("RAG: all sources removed, clearing entire RAG message")
            return ""

        logger.debug(
            f"RAG: removed {len(filenames_to_remove)} source(s) from RAG content"
        )
        return modified

    def _remove_specific_sources_from_rag_message(
        self, processed_messages: List[dict], filenames_to_remove: List[str]
    ) -> None:
        if not filenames_to_remove:
            return

        for i in range(len(processed_messages) - 1, -1, -1):
            msg = processed_messages[i]
            if not isinstance(msg, dict) or msg.get("role") not in ("system", "user"):
                continue

            content = msg.get("content")
            if isinstance(content, str):
                match = PATTERN_RAG_MESSAGE.search(content)
                if not match:
                    continue
                modified_rag = self._remove_sources_from_rag(
                    match.group(0), filenames_to_remove
                )
                start, end = match.span()
                if not modified_rag:
                    new_text = content[:start] + content[end:]
                else:
                    new_text = content[:start] + modified_rag + content[end:]
                msg["content"] = new_text.strip()
                return

            if not isinstance(content, list):
                continue

            modified = False
            new_content: List[dict] = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    new_content.append(block)
                    continue

                text = block.get("text", "")
                match = PATTERN_RAG_MESSAGE.search(text)
                if not match:
                    new_content.append(block)
                    continue

                modified_rag = self._remove_sources_from_rag(
                    match.group(0), filenames_to_remove
                )
                start, end = match.span()
                if not modified_rag:
                    new_text = text[:start] + text[end:]
                else:
                    new_text = text[:start] + modified_rag + text[end:]
                new_text = new_text.strip()
                if new_text:
                    new_block = dict(block)
                    new_block["text"] = new_text
                    new_content.append(new_block)
                modified = True

            if modified:
                msg["content"] = new_content
                return

    def _extract_metadata_marker_from_message(self, message) -> List[str]:
        metadata: List[str] = []
        if not isinstance(message, dict):
            return metadata
        if message.get("role") != "assistant":
            return metadata
        content = message.get("content")
        if isinstance(content, list):

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

    _PATTERN_DETAILS_OPEN_TAG = re.compile(r"<[^>]*>")
    _PATTERN_HTML_ATTR = re.compile(r'(\w+)="([^"]*)"')
    _PATTERN_FILES_URL = re.compile(r"/api/v1/files/([0-9a-fA-F-]+)/content")
    _PATTERN_GENERATED_IMAGE = re.compile(
        r"!\[[^\]]*\]\(/api/v1/files/([0-9a-fA-F-]+)/content\)"
    )
    _PATTERN_DATA_URL = re.compile(r"^data:[^,]*;base64,(.*)$", flags=re.DOTALL)
    _PATTERN_CITATION_RUN = re.compile(r"(?:\[\d+\])+")
    _PATTERN_CODE_SPANS = re.compile(r"```.*?```|`[^`\n]*`", flags=re.DOTALL)
    _SUGGESTION_LINE_PREFIX = "🔎 **Google Search suggestions:**"
    _NOTICE_LINE_PREFIX = "> ⚠️"

    async def _convert_history(
        self, messages, info, ctx, extra_parts_by_user_idx: Optional[dict] = None
    ) -> Tuple[Optional[str], List[gtypes.Content]]:
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
        user_index = -1
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                user_index += 1
                parts = await self._user_parts(message.get("content", ""), ctx)
                extra = (extra_parts_by_user_idx or {}).get(user_index)
                if extra:
                    parts = list(extra) + parts
                if memories is not None and i == last_user_index:
                    parts = self._append_memories(parts, memories)
                if parts:
                    contents.append(gtypes.Content(role="user", parts=parts))
            elif role == "assistant":
                contents.extend(
                    await self._assistant_contents(message.get("content", ""), info, ctx)
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
        suffix = MEMORY_CONTEXT_APPENDIX_HEADER + memories
        for index in range(len(parts) - 1, -1, -1):
            if parts[index].text:
                parts[index] = gtypes.Part(text=parts[index].text + suffix)
                return parts
        parts.append(gtypes.Part(text=suffix))
        return parts

    async def _image_part(self, url: str) -> Optional[gtypes.Part]:
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
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"

        if raw[4:8] == b"ftyp":
            brand = raw[8:12]
            if brand in (b"avif", b"avis"):
                return "image/avif"
            if brand in (b"heic", b"heix", b"heim", b"heis", b"heif", b"mif1"):
                return "image/heic"
        return None

    def _assistant_text(self, content: Any) -> str:
        text = self._block_text(content)
        if not text:
            return ""
        text, _ = self._assistant_clean(text, keep_tool_calls=False)
        return text.strip()

    def _assistant_clean(self, text: str, keep_tool_calls: bool) -> Tuple[str, bool]:
        had_search_block = 'name="google_search"' in text
        text = self._PATTERN_METADATA_MARKER.sub("", text)
        text = self._PATTERN_REASONING_DETAILS.sub("", text)
        text = self._PATTERN_BUILTIN_DETAILS.sub("", text)
        if not keep_tool_calls:
            text = self._PATTERN_TOOL_CALLS_DETAILS.sub("", text)
        text = self._strip_decorative_lines(text)
        if not keep_tool_calls and had_search_block:

            text = self._strip_citation_runs(text)
        return text, had_search_block

    async def _assistant_contents(
        self, content: Any, info, ctx
    ) -> List[gtypes.Content]:
        text = self._block_text(content)
        if not text:
            return []
        image_parts: List[gtypes.Part] = []
        if info.get("image_out"):
            text, image_parts = await self._extract_generated_images(text)
            if not text.strip() and not image_parts:
                return []
        text, had_search_block = self._assistant_clean(text, keep_tool_calls=True)

        segments: List[Tuple[str, str]] = []
        last_end = 0
        for match in self._PATTERN_TOOL_CALLS_DETAILS.finditer(text):
            segments.append(("text", text[last_end:match.start()]))
            segments.append(("tool_call", match.group(0)))
            last_end = match.end()
        segments.append(("text", text[last_end:]))

        def prose(segment: str) -> str:
            if had_search_block:

                segment = self._strip_citation_runs(segment)
            return segment.strip()

        contents: List[gtypes.Content] = []
        model_parts: Optional[List[gtypes.Part]] = None
        index = 0
        while index < len(segments):
            kind, data = segments[index]
            if kind == "text":
                segment = prose(data)
                if segment:
                    if model_parts is None:
                        model_parts = []
                    model_parts.append(gtypes.Part(text=segment))
                index += 1
                continue
            group: List[str] = []
            while index < len(segments) and segments[index][0] == "tool_call":
                group.append(segments[index][1])
                index += 1
                if (
                    index < len(segments)
                    and segments[index][0] == "text"
                    and not prose(segments[index][1])
                ):
                    index += 1
            fc_parts, fr_parts = self._tool_call_group_parts(group)
            if fc_parts:
                if model_parts is None:
                    model_parts = []
                model_parts.extend(fc_parts)
                contents.append(gtypes.Content(role="model", parts=model_parts))
                model_parts = None
            if fr_parts:
                contents.append(gtypes.Content(role="user", parts=fr_parts))
        if model_parts:
            contents.append(gtypes.Content(role="model", parts=model_parts))
        if image_parts:
            self._prepend_image_parts(contents, image_parts)
        return contents

    async def _extract_generated_images(
        self, text: str
    ) -> Tuple[str, List[gtypes.Part]]:
        parts: List[gtypes.Part] = []
        for match in self._PATTERN_GENERATED_IMAGE.finditer(text):
            data = await self._read_file_bytes(match.group(1))
            if data is None:
                continue
            prepared = self._prepare_image_bytes(
                data, self._sniff_image_media_type(data)
            )
            if prepared is None:
                continue
            image_bytes, mime_type = prepared
            parts.append(
                gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type).model_copy(
                    update={"thought_signature": DUMMY_THOUGHT_SIGNATURE}
                )
            )
        cleaned = self._PATTERN_GENERATED_IMAGE.sub("", text)
        return cleaned, parts

    def _prepend_image_parts(
        self, contents: List[gtypes.Content], image_parts: List[gtypes.Part]
    ) -> None:
        for content in contents:
            if content.role == "model":
                content.parts = list(image_parts) + list(content.parts)
                return
        contents.insert(0, gtypes.Content(role="model", parts=list(image_parts)))

    def _tool_call_group_parts(
        self, blocks: List[str]
    ) -> Tuple[List[gtypes.Part], List[gtypes.Part]]:
        fc_parts: List[gtypes.Part] = []
        fr_parts: List[gtypes.Part] = []
        for block in blocks:
            tag = self._PATTERN_DETAILS_OPEN_TAG.match(block)
            if not tag:
                continue
            attrs = dict(self._PATTERN_HTML_ATTR.findall(tag.group(0)))
            tc_id = html.unescape(attrs.get("id") or "")
            tc_name = html.unescape(attrs.get("name") or "")
            if not tc_id or not tc_name:
                logger.warning(
                    "skipping a <details type='tool_calls'> block with a "
                    "missing id or name during history replay"
                )
                continue
            if (attrs.get("done") or "true") != "true":
                continue
            name = self._sanitize_function_name(tc_name)
            try:
                args = json.loads(html.unescape(attrs.get("arguments") or ""))
                if not isinstance(args, dict):
                    args = {}
            except (ValueError, TypeError):
                args = {}
            part_kwargs: dict = {
                "function_call": gtypes.FunctionCall(id=tc_id, name=name, args=args)
            }
            if not fc_parts:
                part_kwargs["thought_signature"] = DUMMY_THOUGHT_SIGNATURE
            fc_parts.append(gtypes.Part(**part_kwargs))
            result_raw = html.unescape(attrs.get("result") or "")
            try:
                result = json.loads(result_raw)
            except (ValueError, TypeError):
                result = result_raw
            fr_parts.append(
                gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        id=tc_id, name=name, response={"result": result}
                    )
                )
            )
        return fc_parts, fr_parts

    def _strip_decorative_lines(self, text: str) -> str:
        kept = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith(self._SUGGESTION_LINE_PREFIX):
                continue
            if stripped.startswith(self._NOTICE_LINE_PREFIX):
                continue
            if stripped.startswith("\U0001f4cd ") and "Google Maps" in stripped:
                continue
            kept.append(line)
        return "\n".join(kept)

    def _strip_citation_runs(self, text: str) -> str:
        parts = self._PATTERN_CODE_SPANS.split(text)
        return "".join(
            self._PATTERN_CITATION_RUN.sub("", seg) if idx % 2 == 0 else seg
            for idx, seg in enumerate(parts)
        )

    _INLINE_TOTAL_MAX_BYTES = 90 * 1024 * 1024
    _PDF_INLINE_MAX_BYTES = 50 * 1024 * 1024
    _MEDIA_POLL_INTERVAL = 2.0
    _MEDIA_PROCESS_TIMEOUT = 300.0
    _MAX_YOUTUBE_LINKS = 10

    _PATTERN_YOUTUBE = re.compile(
        r"(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)"
        r"([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])"
    )

    async def _collect_native_media(
        self, ctx, client, messages, files, prior_entries
    ) -> dict:
        parts_by_user_idx: Dict[int, List] = {}
        native_names: List[str] = []
        entries: List[str] = []

        candidates = self._media_candidates(messages, files, prior_entries)
        if getattr(ctx.user_valves, "USE_NATIVE_FILES", True):
            inline_total = 0
            ordered = sorted(
                candidates, key=lambda fid: (candidates[fid]["anchor"], fid)
            )
            for file_id in ordered:
                info = candidates[file_id]
                hint = info["mime_hint"]
                if hint is not None and not self._is_native_media(
                    hint, info["context"], info["prior"]
                ):
                    continue
                name = info["name"]
                await ctx.status.activity(f"📎 Attaching {name}…")
                record, data = await self._read_media_file(file_id)
                if data is None:
                    logger.warning(f"native media: could not read file {file_id} ({name})")
                    continue
                mime = self._media_mime(record, info)
                if not self._is_native_media(mime, info["context"], info["prior"]):
                    continue

                part: Optional[Any] = None
                size = len(data)
                files_api = str(getattr(self.valves, "FILES_API", "auto") or "auto")
                if (
                    files_api != "always"
                    and size <= self._inline_cap(mime)
                    and inline_total + size <= self._INLINE_TOTAL_MAX_BYTES
                ):
                    part = gtypes.Part.from_bytes(data=data, mime_type=mime)
                    inline_total += size
                elif files_api != "never":
                    try:
                        gfile = await self._ensure_gemini_file(
                            ctx, client, file_id, record, data, mime, name
                        )
                        part = gtypes.Part(
                            file_data=gtypes.FileData(file_uri=gfile.uri, mime_type=mime)
                        )
                    except Exception as e:
                        logger.warning(f"native media: Files API failed for {name}: {e}")
                        await ctx.status.notification(str(e))
                else:
                    await ctx.status.notification(
                        f"{name} is too large to send inline and the Files API is disabled"
                    )
                if part is None:
                    continue

                parts_by_user_idx.setdefault(info["anchor"], []).append(part)
                native_names.append(name)
                entries.append(f'{info["anchor"]}:media:{quote(f"{file_id}:{name}", safe="")}')

        self._collect_youtube_parts(messages, parts_by_user_idx)
        return {"parts": parts_by_user_idx, "native_names": native_names, "entries": entries}

    def _media_candidates(
        self, messages, files, prior_entries
    ) -> Dict[str, dict]:
        user_msgs = [
            m for m in (messages or []) if isinstance(m, dict) and m.get("role") == "user"
        ]
        latest = max(len(user_msgs) - 1, 0)

        prior_anchor: Dict[str, int] = {}
        prior_name: Dict[str, str] = {}
        for entry in prior_entries or []:
            parts = str(entry).split(":", 2)
            if len(parts) < 3 or parts[1] != "media":
                continue
            try:
                msg_idx = int(parts[0])
            except ValueError:
                continue
            decoded = unquote(parts[2])
            file_id, _, fname = decoded.partition(":")
            if file_id:
                prior_anchor[file_id] = msg_idx
                if fname:
                    prior_name[file_id] = fname

        owning_idx: Dict[str, int] = {}
        user_idx = -1
        for message in messages or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            user_idx += 1
            for item in message.get("files") or []:
                file_id = self._media_item_id(item)
                if file_id and file_id not in owning_idx:
                    owning_idx[file_id] = user_idx

        candidates: Dict[str, dict] = {}
        items = list(files or [])
        for message in user_msgs:
            items.extend(message.get("files") or [])
        for item in items:
            file_id = self._media_item_id(item)
            if not file_id or file_id in candidates:
                continue
            name = (item or {}).get("name") or prior_name.get(file_id, file_id)
            candidates[file_id] = {
                "name": name,
                "context": (item or {}).get("context"),
                "mime_hint": self._media_item_mime(item)
                or (mimetypes.guess_type(name)[0] if name else None),
                "anchor": prior_anchor.get(file_id, owning_idx.get(file_id, latest)),
                "prior": file_id in prior_anchor,
            }

        for file_id, msg_idx in prior_anchor.items():
            if file_id in candidates:
                continue
            name = prior_name.get(file_id, file_id)
            candidates[file_id] = {
                "name": name,
                "context": None,
                "mime_hint": mimetypes.guess_type(name)[0] if name else None,
                "anchor": msg_idx,
                "prior": True,
            }
        return candidates

    @staticmethod
    def _media_item_id(item) -> Optional[str]:
        if not isinstance(item, dict) or item.get("type") not in (None, "file"):
            return None
        file_id = item.get("id")
        return file_id if isinstance(file_id, str) and file_id else None

    @staticmethod
    def _media_item_mime(item) -> Optional[str]:
        if not isinstance(item, dict):
            return None
        file = item.get("file")
        if isinstance(file, dict):
            meta = file.get("meta")
            if isinstance(meta, dict) and meta.get("content_type"):
                return str(meta["content_type"])
        return None

    @staticmethod
    def _media_mime(record, info) -> Optional[str]:
        meta = getattr(record, "meta", None)
        if isinstance(meta, dict) and meta.get("content_type"):
            return str(meta["content_type"])
        return info.get("mime_hint")

    @staticmethod
    def _is_native_media(mime: Optional[str], context, has_prior_marker: bool) -> bool:
        if not mime:
            return False
        mime = str(mime).lower()
        if mime.startswith("audio/") or mime.startswith("video/"):
            return True
        if mime == "application/pdf":
            return context == "full" or has_prior_marker
        return False

    def _inline_cap(self, mime: Optional[str]) -> int:
        cap = max(int(getattr(self.valves, "INLINE_FILE_MAX_MB", 0) or 0), 0) * 1024 * 1024
        if str(mime or "").lower() == "application/pdf":
            cap = min(cap, self._PDF_INLINE_MAX_BYTES)
        return cap

    @staticmethod
    async def _read_media_file(file_id: str) -> Tuple[Optional[Any], Optional[bytes]]:
        if not FILES_AVAILABLE:
            return None, None
        try:
            record = await Files.get_file_by_id(id=file_id)
        except Exception as e:
            logger.warning(f"native media: could not look up file {file_id}: {e}")
            return None, None
        if record is None:
            return None, None
        location = getattr(record, "path", None)
        if not location:
            return None, None
        try:
            path = Storage.get_file(location)
            with open(path, "rb") as handle:
                return record, handle.read()
        except Exception as e:
            logger.warning(f"native media: could not read file {file_id}: {e}")
            return None, None

    async def _ensure_gemini_file(
        self, ctx, client, file_id, record, data, mime, name
    ) -> gtypes.File:
        cache_key = (
            f"{self._api_key_fingerprint(ctx.api_key)}:{file_id}:"
            f"{getattr(record, 'hash', '')}"
        )
        entry = Pipe._gemini_file_cache.get(cache_key)
        if entry is None:
            meta = getattr(record, "meta", None)
            if isinstance(meta, dict):
                entry = (meta.get("gemini_files") or {}).get(cache_key)
        if entry:
            try:
                expires = float(entry.get("expires") or 0)
            except (TypeError, ValueError):
                expires = 0.0
            if expires - time.time() > 3600:
                gfile = None
                try:
                    gfile = await client.aio.files.get(name=entry["name"])
                except Exception as e:
                    logger.warning(f"Files API: could not confirm {entry['name']}: {e}")
                if gfile is not None and str(getattr(gfile.state, "name", "")) == "ACTIVE":
                    return gfile

        gfile = await client.aio.files.upload(
            file=io.BytesIO(data),
            config=gtypes.UploadFileConfig(mime_type=mime, display_name=name),
        )
        state = str(getattr(gfile.state, "name", ""))
        if state == "FAILED":
            raise RuntimeError(f"Google rejected {name} for processing")
        deadline = time.monotonic() + self._MEDIA_PROCESS_TIMEOUT
        while state == "PROCESSING":
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Google took too long to process {name}")
            await ctx.status.activity(f"⏳ Google is processing {name}…")
            await asyncio.sleep(self._MEDIA_POLL_INTERVAL)
            gfile = await client.aio.files.get(name=gfile.name)
            state = str(getattr(gfile.state, "name", ""))
            if state == "FAILED":
                raise RuntimeError(f"Google failed to process {name}")
        if state != "ACTIVE":
            raise RuntimeError(f"Google could not process {name} (state {state})")

        stored = {
            "name": gfile.name,
            "uri": gfile.uri,
            "mime": mime,
            "expires": (
                gfile.expiration_time.timestamp()
                if getattr(gfile, "expiration_time", None) is not None
                else 0
            ),
        }
        Pipe._gemini_file_cache[cache_key] = stored
        if FILES_AVAILABLE:
            try:
                meta = dict(getattr(record, "meta", None) or {})
                gemini_files = dict(meta.get("gemini_files") or {})
                gemini_files[cache_key] = stored
                await Files.update_file_metadata_by_id(
                    record.id, {"gemini_files": gemini_files}
                )
            except Exception as e:
                logger.warning(f"could not store the Files API entry on file {file_id}: {e}")
        return gfile

    def _collect_youtube_parts(self, messages, parts_by_user_idx) -> None:
        policy = str(getattr(self.valves, "YOUTUBE_LINKS", "most_recent") or "off")
        if policy == "off":
            return
        user_msgs = [
            m for m in (messages or []) if isinstance(m, dict) and m.get("role") == "user"
        ]
        if not user_msgs:
            return

        if policy == "latest_message":
            indexes = [len(user_msgs) - 1]
        elif policy == "most_recent":
            found = [
                i
                for i in range(len(user_msgs) - 1, -1, -1)
                if self._youtube_links(self._block_text(user_msgs[i].get("content")))
            ]
            indexes = found[:1]
        else:
            indexes = list(range(len(user_msgs) - 1, -1, -1))

        seen = set()
        for idx in indexes:
            for video_id in self._youtube_links(
                self._block_text(user_msgs[idx].get("content"))
            ):
                if video_id in seen:
                    continue
                seen.add(video_id)
                parts_by_user_idx.setdefault(idx, []).append(
                    gtypes.Part(
                        file_data=gtypes.FileData(
                            file_uri=f"https://www.youtube.com/watch?v={video_id}"
                        )
                    )
                )
                if len(seen) >= self._MAX_YOUTUBE_LINKS:
                    return

    @classmethod
    def _youtube_links(cls, text: str) -> List[str]:
        if not text:
            return []
        return cls._PATTERN_YOUTUBE.findall(text)

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

        if maps and url_context and not info.get("maps_with_url_context"):
            url_context = url_context_auto = False
            notes.append(
                f"URL context disabled for this message: {model} cannot combine it with Google Maps"
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

    @staticmethod
    def _sanitize_function_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(name or ""))
        if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
            cleaned = "_" + cleaned
        return cleaned[:64]

    def _build_function_declarations(self, ctx) -> list:
        decls: List[gtypes.FunctionDeclaration] = []
        offered_originals: set = set()
        taken: set = set()
        for source in (ctx.tools or {}, ctx.builtin_tools or {}):
            for original, entry in source.items():
                original = str(original)
                if original in offered_originals:
                    continue
                if original.startswith("_"):
                    continue
                if not isinstance(entry, dict) or not entry.get("callable"):
                    continue
                offered_originals.add(original)
                spec = entry.get("spec")
                spec = spec if isinstance(spec, dict) else {}
                candidate = self._sanitize_function_name(original)
                suffix = 2
                while candidate in taken:
                    candidate = f"{self._sanitize_function_name(original)}_{suffix}"
                    suffix += 1
                taken.add(candidate)
                ctx.turn.name_map[candidate] = original
                parameters = spec.get("parameters")
                if not isinstance(parameters, dict):
                    parameters = {"type": "object", "properties": {}}
                decls.append(
                    gtypes.FunctionDeclaration(
                        name=candidate,
                        description=spec.get("description") or "",
                        parameters_json_schema=parameters,
                    )
                )
        decls.sort(key=lambda d: d.name)
        ctx.turn.functions_offered = bool(decls)
        return decls

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

        if info.get("image_out"):
            kwargs["response_modalities"] = ["TEXT", "IMAGE"]
            image_kwargs: dict = {}
            aspect_ratio = (
                user_valves.IMAGE_ASPECT_RATIO if user_valves is not None else "auto"
            ) or "auto"
            if aspect_ratio != "auto":
                image_kwargs["aspect_ratio"] = aspect_ratio
            image_size = (
                user_valves.IMAGE_SIZE if user_valves is not None else "auto"
            ) or "auto"
            if image_size != "auto" and image_size in (info.get("image_sizes") or []):
                image_kwargs["image_size"] = image_size
            if image_kwargs:
                kwargs["image_config"] = gtypes.ImageConfig(**image_kwargs)

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

    def _format_maps_attribution_line(self, places: List[Tuple[str, str]]) -> str:
        if not places:
            return ""
        return (
            '📍 <span translate="no">Google Maps</span>: '
            + " · ".join(f"[{title}]({uri})" for title, uri in places)
            + "\n"
        )

    def _format_notice(self, text: str) -> str:
        return "\n> ⚠️ " + text + "\n"

    _TOOL_RESULT_CHAR_CAP = 100_000
    _OWUI_CALL_TOOL_NAME = "call_tool"
    _OWUI_SEARCH_TOOL_NAME = "search_tools"

    @staticmethod
    def _filter_tool_args(tool_entry: Any, args: dict) -> dict:
        if not isinstance(args, dict) or not args:
            return args if isinstance(args, dict) else {}
        spec = (tool_entry or {}).get("spec") if isinstance(tool_entry, dict) else None
        if not isinstance(spec, dict):
            return args
        params = spec.get("parameters")
        if not isinstance(params, dict) or not isinstance(params.get("properties"), dict):

            return args
        allowed = params["properties"].keys()
        kept = {k: v for k, v in args.items() if k in allowed}
        dropped = [k for k in args if k not in allowed]
        if dropped:
            logger.warning(
                "Tool '%s': dropped undeclared argument(s) %s not in its schema",
                spec.get("name", "?"),
                dropped,
            )
        return kept

    @classmethod
    def _resolve_owui_call_tool(
        cls, tool_input: Any, tools: dict
    ) -> Tuple[str, dict, Optional[str]]:
        try:
            if not isinstance(tool_input, dict) or set(tool_input) != {"tool_name", "params"}:
                raise ValueError("call_tool requires tool_name and params.")
            name = tool_input["tool_name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool_name must be a non-empty string.")
            if name in {cls._OWUI_SEARCH_TOOL_NAME, cls._OWUI_CALL_TOOL_NAME}:
                raise ValueError("call_tool cannot invoke search_tools or itself.")
            if name not in tools:
                raise ValueError(f'Tool "{name}" is not available. Search for an available tool.')
            params = tool_input["params"]
            if isinstance(params, str):
                params = json.loads(params) if params.strip() else {}
            if not isinstance(params, dict):
                raise ValueError("params must decode to a JSON object.")
        except (TypeError, ValueError) as error:
            return cls._OWUI_CALL_TOOL_NAME, {}, f"Error: {error}"
        return name, params, None

    async def _start_function_call(self, ctx, call, fc: dict) -> None:
        original_name = ctx.turn.name_map.get(fc["name"], fc["name"])
        await ctx.status.running_tool(original_name)
        await ctx.upsert_block(
            fc["key"],
            self._format_tool_result_block(
                fc["id"], original_name, fc["args"], "", done=False
            ),
            call.index,
        )
        if not ctx.turn.functions_offered:
            fc["error"] = "function calling is not available"
            return
        max_calls = max(int(getattr(self.valves, "MAX_TOOL_CALLS", 1) or 1), 1)
        if ctx.turn.tool_calls_made >= max_calls:
            fc["error"] = "tool call limit reached (MAX_TOOL_CALLS)"
            return
        ctx.turn.tool_calls_made += 1
        fc["task"] = asyncio.create_task(self._execute_tool_call(ctx, fc))

    async def _execute_tool_call(self, ctx, fc: dict) -> dict:
        name = ctx.turn.name_map.get(fc["name"], fc["name"])
        args = fc["args"] if isinstance(fc["args"], dict) else {}
        files: List[Any] = []
        embeds: List[Any] = []
        error_text: Optional[str] = None
        text = ""
        try:
            tools = ctx.tools or {}
            builtin = ctx.builtin_tools or {}
            target = None
            if (tools.get(name) or {}).get("tool_id") == f"builtin:{self._OWUI_CALL_TOOL_NAME}":
                name, args, dispatch_error = self._resolve_owui_call_tool(args, tools)
                if dispatch_error:
                    error_text = dispatch_error
                else:
                    target = tools.get(name)
            if error_text is None and (target is None or not (target or {}).get("callable")):
                target = tools.get(name)
                if target is not None and not target.get("callable"):
                    target = None
            if error_text is None and target is None:
                entry = builtin.get(name)
                if isinstance(entry, dict) and entry.get("callable"):
                    target = entry
            if error_text is None and target is None:
                logger.warning(
                    "Tool '%s' not found in __tools__ or builtin_tools", name
                )
                error_text = json.dumps(
                    {
                        "error": (
                            f"Tool '{name}' is not available. It may require "
                            "server context or is not configured."
                        )
                    },
                    ensure_ascii=False,
                )
            if error_text is None:
                args = self._filter_tool_args(target, args)
                approved, denial = await self._await_tool_approval(
                    {"id": fc.get("raw_id"), "name": name, "args": args}
                )
                if not approved:
                    error_text = denial
                else:
                    timeout_s = getattr(self.valves, "TOOL_CALL_TIMEOUT", 60)
                    outcome = target["callable"](**args)
                    if inspect.isawaitable(outcome):
                        try:
                            outcome = await asyncio.wait_for(
                                outcome, timeout=max(1, float(timeout_s))
                            )
                        except asyncio.TimeoutError:
                            error_text = (
                                f"Error executing tool '{name}': "
                                f"timed out after {timeout_s}s"
                            )
                    if error_text is None and PROCESS_TOOL_RESULT_AVAILABLE and ctx.request is not None:
                        try:
                            outcome, files, embeds = await process_tool_result(
                                ctx.request,
                                name,
                                outcome,
                                "pipe",
                                metadata=ctx.metadata,
                                user=ctx.user,
                            )
                        except Exception as e:
                            logger.warning(
                                f"process_tool_result failed for '{name}': {e}"
                            )
                    if error_text is None:
                        if isinstance(outcome, str):
                            text = outcome
                        elif isinstance(outcome, (dict, list)):
                            try:
                                text = json.dumps(outcome, ensure_ascii=False)
                            except (TypeError, ValueError):
                                text = str(outcome)
                        else:
                            text = str(outcome)
        except Exception as e:
            logger.error(f"Tool call '{name}' failed: {e}")
            error_text = f"Error executing tool '{name}': {e}"
            files, embeds = [], []
        if error_text is not None:
            text = error_text
        is_error = error_text is not None or text.startswith(
            ("Error:", "Error executing tool")
        )
        return {"text": text, "files": files, "embeds": embeds, "error": is_error}

    async def _await_tool_approval(self, tool_call: dict) -> Tuple[bool, Optional[str]]:
        mode, event_call = TOOL_APPROVAL.get()
        name = str(tool_call.get("name") or "tool")
        if mode != "ask":
            return True, None
        denial = json.dumps(
            {"error": f"The user denied permission to run '{name}'."},
            ensure_ascii=False,
        )
        if event_call is None:
            return False, denial
        try:
            args = json.dumps(tool_call.get("args") or {}, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            args = str(tool_call.get("args"))
        if len(args) > 2000:
            args = args[:2000] + "\n… (truncated)"
        try:
            answer = await event_call(
                {
                    "type": "confirmation",
                    "data": {
                        "title": f"Run tool: {name}?",
                        "message": (
                            f"The model wants to call `{name}` with:\n\n"
                            f"```json\n{args}\n```"
                        ),

                        "tool_call": {
                            "id": tool_call.get("id"),
                            "name": name,
                            "arguments": tool_call.get("args") or {},
                        },
                    },
                }
            )
        except Exception as e:

            logger.warning("Tool approval prompt failed for '%s': %s", name, e)
            answer = False
        if answer is True:
            logger.info("Tool '%s' approved by user", name)
            return True, None
        logger.info("Tool '%s' denied by user", name)
        return False, denial

    async def _finish_function_calls(self, ctx, call) -> gtypes.Content:

        async def _tagged(fc: dict, task) -> Tuple[dict, dict]:
            try:
                return fc, await task
            except asyncio.CancelledError:
                return fc, {"text": "tool call cancelled", "error": True}

        for fc in call.function_calls:
            if fc.get("error"):
                original_name = ctx.turn.name_map.get(fc["name"], fc["name"])
                await ctx.upsert_block(
                    fc["key"],
                    self._format_tool_result_block(
                        fc["id"],
                        original_name,
                        fc["args"],
                        fc["error"],
                        is_error=True,
                        done=True,
                    ),
                    call.index,
                )
                fc["outcome"] = {"text": fc["error"], "error": True}

        tasks = [(fc, fc["task"]) for fc in call.function_calls if fc.get("task") is not None]
        if tasks:
            for finished in asyncio.as_completed(
                [_tagged(fc, task) for fc, task in tasks]
            ):
                fc, outcome = await finished
                outcome = outcome or {"text": "", "error": False}
                text = str(outcome.get("text") or "")
                is_error = bool(outcome.get("error"))
                files = outcome.get("files") or []
                embeds = outcome.get("embeds") or []
                if files:
                    await ctx.emit_event(
                        {"type": "files", "data": {"files": files}}
                    )
                original_name = ctx.turn.name_map.get(fc["name"], fc["name"])
                await ctx.upsert_block(
                    fc["key"],
                    self._format_tool_result_block(
                        fc["id"],
                        original_name,
                        fc["args"],
                        text,
                        is_error=is_error,
                        done=True,
                        embeds=embeds,
                        files=files,
                    ),
                    call.index,
                )
                fc["outcome"] = {"text": text, "error": is_error}

        parts: List[gtypes.Part] = []
        for fc in call.function_calls:
            outcome = fc.get("outcome") or {"text": "", "error": False}
            text = str(outcome.get("text") or "")
            if len(text) > self._TOOL_RESULT_CHAR_CAP:
                text = text[: self._TOOL_RESULT_CHAR_CAP] + "\n… (truncated)"
            if outcome.get("error"):
                response: Dict[str, Any] = {"error": text}
            else:
                response = self._tool_response_payload(text)
            parts.append(
                gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        id=fc.get("raw_id"),
                        name=fc["name"],
                        response=response,
                    )
                )
            )
        return gtypes.Content(role="user", parts=parts)

    @staticmethod
    def _tool_response_payload(text: str) -> dict:
        try:
            return {"result": json.loads(text)}
        except (TypeError, ValueError):
            return {"result": text}

    async def _start_image_output(self, ctx, call, part) -> None:
        key = f"img:{call.index}:{len(call.image_tasks)}"
        await ctx.upsert_block(key, "\n_🖼️ Saving image…_\n", call.index)
        await ctx.status.activity("🖼️ Saving image…")
        inline = _get(part, "inline_data")
        data = _get(inline, "data") or b""
        mime = _get(inline, "mime_type") or "image/png"

        async def _save() -> None:
            url = await self._store_generated_image(ctx, data, mime)
            if url:
                block = f"\n![Generated image]({url})\n"
            else:
                block = self._format_notice("Generated image could not be saved")
            await ctx.upsert_block(key, block, call.index)

        call.image_tasks.append(asyncio.create_task(_save()))

    async def _store_generated_image(self, ctx, data: bytes, mime: str) -> Optional[str]:
        metadata = {
            "chat_id": (ctx.metadata or {}).get("chat_id"),
            "message_id": (ctx.metadata or {}).get("message_id"),
        }
        user_id = (ctx.user or {}).get("id")
        try:
            if UPLOAD_IMAGE_AVAILABLE and USERS_AVAILABLE and ctx.request:
                user = await Users.get_user_by_id(user_id)
                file_item, _ = await upload_image(
                    ctx.request, data, mime, metadata, user
                )
                return f"/api/v1/files/{file_item.id}/content"
            if FILES_AVAILABLE:
                extension = (
                    mimetypes.guess_extension(mime)
                    or mimetypes.guess_extension(mime.split(";")[0])
                    or ".png"
                )
                file_id = str(uuid.uuid4())
                filename = f"generated-image{extension}"
                _, file_path = Storage.upload_file(io.BytesIO(data), filename, {})
                await Files.insert_new_file(
                    user_id=user_id,
                    form_data=FileForm(
                        id=file_id,
                        hash=hashlib.sha256(data).hexdigest(),
                        filename=filename,
                        path=file_path,
                        data={},
                        meta={
                            "content_type": mime,
                            "size": len(data),
                            "source": "gemini_image_generation",
                        },
                    ),
                )
                return f"/api/v1/files/{file_id}/content"
            logger.warning(
                "no OpenWebUI file storage available for a generated image"
            )
            return None
        except Exception as e:
            logger.warning(f"could not store generated image: {e}")
            return None

    _THOUGHT_RENDER_INTERVAL = 0.3

    async def _stream_call(self, ctx, client, contents, config, call) -> None:
        async for chunk_ in await client.aio.models.generate_content_stream(
            model=ctx.model, contents=contents, config=config
        ):
            await self._handle_chunk(ctx, call, chunk_)

    async def _handle_chunk(self, ctx, call, chunk_) -> None:
        feedback = _get(chunk_, "prompt_feedback")
        reason = _get(feedback, "block_reason") if feedback is not None else None
        if reason:
            await ctx.emit_text(call.index, self._format_notice(f"Prompt blocked: {reason}"))
            call.finish_reason = "PROMPT_BLOCKED"
            return

        usage = _get(chunk_, "usage_metadata")
        if usage:
            call.usage = self._usage_to_dict(usage)

        candidates = _get(chunk_, "candidates") or []
        if not candidates:
            return
        cand = candidates[0]

        call.grounding.add(_get(cand, "grounding_metadata"))
        self._extend_url_meta(call, _get(cand, "url_context_metadata"))

        finish_reason = _get(cand, "finish_reason")
        if finish_reason:
            call.finish_reason = finish_reason
        finish_message = _get(cand, "finish_message")
        if finish_message:
            call.finish_message = finish_message

        content = _get(cand, "content")
        for part in _get(content, "parts") or []:
            call.raw_parts.append(part)
            await self._dispatch_part(ctx, call, part)

    def _extend_url_meta(self, call, ucm) -> None:
        for entry in _get(ucm, "url_metadata") or []:
            url = _get(entry, "retrieved_url")
            if not url or any(item["url"] == url for item in call.url_meta):
                continue
            status_value = _get(entry, "url_retrieval_status")

            name = str(getattr(status_value, "name", status_value) or "")
            short = name.replace("URL_RETRIEVAL_STATUS_", "").lower()
            call.url_meta.append({"url": url, "status": "ok" if short == "success" else short})

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        dump = getattr(usage, "model_dump", None)
        data = dump(exclude_none=True) if callable(dump) else dict(usage or {})
        return {
            re.sub(r"(?<!^)(?=[A-Z])", "_", str(key)).lower(): value
            for key, value in data.items()
        }

    async def _dispatch_part(self, ctx, call, part) -> None:
        text = _get(part, "text")
        thought = bool(_get(part, "thought"))
        if thought and text:
            await self._handle_thought_part(ctx, call, text)
            return
        if text:
            await self._close_thought_run(ctx, call)
            await ctx.status.responding()
            await ctx.emit_text(call.index, text)
            return

        function_call = _get(part, "function_call")
        if function_call is not None:
            await self._close_thought_run(ctx, call)
            call.function_calls.append(
                {
                    "part": part,
                    "raw_id": _get(function_call, "id"),
                    "id": _get(function_call, "id")
                    or f"call_{call.index}_{len(call.function_calls)}",
                    "name": _get(function_call, "name") or "tool",
                    "args": _get(function_call, "args") or {},
                    "key": f"fc:{call.index}:{len(call.function_calls)}",
                }
            )
            await self._start_function_call(ctx, call, call.function_calls[-1])
            return

        tool_call = _get(part, "tool_call")
        if tool_call is not None:
            args = _get(tool_call, "args")
            queries = (
                [q for q in (args.get("queries") or []) if q]
                if isinstance(args, dict)
                else []
            )
            if queries:
                await ctx.status.searching_google(queries)
            return

        inline_data = _get(part, "inline_data")
        if not thought and inline_data is not None:
            mime = str(_get(inline_data, "mime_type") or "")
            if mime.startswith("image/"):
                await self._start_image_output(ctx, call, part)
            else:
                logger.debug("skipping non-image inline_data part")

    async def _handle_thought_part(self, ctx, call, text: str) -> None:
        if not call.thought_key:
            call.thought_runs += 1
            call.thought_key = f"think:{call.index}:{call.thought_runs}"
            call.thought_started = time.monotonic()
            call.last_thought_render = 0.0
            await ctx.status.thinking()
        call.thought_text += text
        now = time.monotonic()
        if now - call.last_thought_render < self._THOUGHT_RENDER_INTERVAL:
            return
        call.last_thought_render = now
        if not getattr(ctx.user_valves, "SHOW_THOUGHTS", True):
            return
        block = self._format_thinking_block(call.thought_text, done=False)
        if block:
            await ctx.upsert_block(call.thought_key, block, call.index)

    async def _close_thought_run(self, ctx, call) -> None:
        if not call.thought_key:
            return
        duration = time.monotonic() - call.thought_started if call.thought_started else 0.0
        block = self._format_thinking_block(call.thought_text, done=True, duration=duration)
        if block:
            await ctx.upsert_block(call.thought_key, block, call.index)
        call.thought_key = None
        call.thought_text = ""

    async def _finalize_call(self, ctx, call) -> None:
        await self._close_thought_run(ctx, call)

        if call.image_tasks:
            try:
                await asyncio.gather(*call.image_tasks)
            except Exception:
                logger.warning("a generated-image store task failed", exc_info=True)

        if not call.grounding.empty():
            await self._apply_grounding(ctx, call)

        if call.url_meta:
            ok = [e["url"] for e in call.url_meta if str(e.get("status") or "").lower() == "ok"]
            failed = len(call.url_meta) - len(ok)
            await ctx.status.read_urls(ok, failed)
            block = self._format_url_context_block(call.index, call.url_meta)
            if block:
                first_text = (ctx.msg.text_segments(call.index) or [None])[0]
                await ctx.upsert_block(
                    f"urlctx:{call.index}",
                    block,
                    call.index,
                    before_key=first_text.key if first_text else None,
                )

        notice = self._finish_reason_notice(call.finish_reason, call.finish_message)
        if notice:
            await ctx.upsert_block(f"notice:{call.index}", notice, call.index)

        if call.usage:
            self._accumulate_usage(ctx.turn.usage_total, call.usage)

    def _merge_replay_parts(self, parts) -> List:
        merged: List[Any] = []
        for part in parts or []:
            text = _get(part, "text")
            function_call = _get(part, "function_call")
            tool_call = _get(part, "tool_call")
            tool_response = _get(part, "tool_response")
            inline_data = _get(part, "inline_data")
            signature = _get(part, "thought_signature")
            if not (text or function_call or tool_call or tool_response or inline_data or signature):
                continue
            plain = (
                text
                and signature is None
                and function_call is None
                and tool_call is None
                and tool_response is None
                and inline_data is None
                and _get(part, "executable_code") is None
                and _get(part, "code_execution_result") is None
            )
            thought = bool(_get(part, "thought"))
            previous = merged[-1] if merged else None
            if (
                plain
                and isinstance(previous, gtypes.Part)
                and previous.text
                and not getattr(previous, "thought_signature", None)
                and bool(getattr(previous, "thought", False)) == thought
            ):
                merged[-1] = gtypes.Part(text=previous.text + text, thought=thought)
                continue
            if plain:
                merged.append(gtypes.Part(text=text, thought=thought))
            else:
                merged.append(part)
        return merged

    _RETRY_BACKOFF_BASE = 2.0
    _RETRY_BACKOFF_CAP = 30.0
    _RESOLVED_URL_CACHE_LIMIT = 2000
    _GROUNDING_REDIRECT_PREFIX = (
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
    )
    _RETRYABLE_KINDS = ("rate_limit", "server", "transport")
    _RETRY_AFTER_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*s?$")
    _USAGE_KEYS = (
        "prompt_token_count",
        "cached_content_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
        "total_token_count",
    )

    async def _resolve_grounding_urls(self, uris: List[str]) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        todo: List[str] = []
        for uri in dict.fromkeys(uris or []):
            if not uri:
                continue
            cached = self._resolved_url_cache.get(uri)
            if cached is not None:
                resolved[uri] = cached
            elif not uri.startswith(self._GROUNDING_REDIRECT_PREFIX):
                continue
            else:
                todo.append(uri)
        if not (todo and self.valves.RESOLVE_GROUNDING_URLS and HTTPX_AVAILABLE):
            for uri in todo:
                resolved[uri] = uri
            return resolved

        results: Dict[str, str] = {}
        semaphore = asyncio.Semaphore(8)

        async def _one(uri: str) -> None:
            async with semaphore:
                try:
                    async with httpx.AsyncClient(follow_redirects=False) as client:
                        response = await client.get(uri, timeout=3.0)
                    results[uri] = response.headers.get("Location") or uri
                except Exception:
                    results[uri] = uri

        try:
            await asyncio.wait_for(
                asyncio.gather(*(_one(uri) for uri in todo)), timeout=3.0
            )
        except asyncio.TimeoutError:
            logger.debug("grounding URL resolution budget exceeded; keeping redirects")
        for uri in todo:
            destination = results.get(uri, uri)
            self._cache_resolved_url(uri, destination)
            resolved[uri] = destination
        return resolved

    def _cache_resolved_url(self, uri: str, destination: str) -> None:
        cache = self._resolved_url_cache
        if uri in cache:
            cache.pop(uri)
        cache[uri] = destination
        while len(cache) > self._RESOLVED_URL_CACHE_LIMIT:
            cache.pop(next(iter(cache)))

    async def _apply_grounding(self, ctx, call) -> None:
        acc = call.grounding
        web_chunks = [c for c in acc.chunks if c.get("kind") == "web"]
        maps_chunks = [c for c in acc.chunks if c.get("kind") == "maps"]

        queries = [q for q in acc.queries if q]
        if maps_chunks and not web_chunks:

            queries = []
        if queries:
            for query in queries:
                if query not in ctx.turn.search_queries:
                    ctx.turn.search_queries.append(query)
            await ctx.status.searching_google(queries)

        uris = [c["uri"] for c in web_chunks if c.get("uri")]
        resolved = await self._resolve_grounding_urls(uris)

        segments = ctx.msg.text_segments(call.index)
        joined = "".join(segment.content for segment in segments)
        cited, new_numbers, references = cite_call_text(
            joined, acc, ctx.turn.citations, resolved
        )
        if segments:
            segments[0].content = cited
            for segment in segments[1:]:
                segment.content = ""
            await ctx.emit_replace()

        now_iso = datetime.now(timezone.utc).isoformat()
        for number in sorted(new_numbers):
            info = ctx.turn.citations.sources.get(number) or {}
            url = info.get("url") or ""
            title = info.get("title") or url
            await ctx.emit_event(
                {
                    "type": "source",
                    "data": {
                        "source": {"name": title, "url": url, "id": str(number)},
                        "document": references.get(number) or [title],
                        "metadata": [
                            {
                                "source": f"{url}#{number}",
                                "date_accessed": now_iso,
                                "name": f"[{number}]",
                            }
                        ],
                    },
                }
            )

        web_numbers = [
            number
            for number in sorted(references)
            if (ctx.turn.citations.sources.get(number) or {}).get("kind") == "web"
        ]
        if web_chunks or queries:
            web_urls = [
                (ctx.turn.citations.sources.get(number) or {}).get("url") or ""
                for number in web_numbers
            ]
            if web_urls:
                await ctx.status.web_search_done(web_urls, queries)
            source_triples = []
            for number in web_numbers:
                info = ctx.turn.citations.sources.get(number) or {}
                url = info.get("url") or ""
                source_triples.append((number, info.get("title") or url, url))
            embed_html = ""
            if (
                self.valves.SHOW_SEARCH_SUGGESTIONS == "embed"
                and acc.entry_point_html
            ):
                embed_html = prepare_entry_point_embed(acc.entry_point_html)
            block = self._format_google_search_block(
                call.index, queries, source_triples, embed_html
            )
            if block:
                first_text = (ctx.msg.text_segments(call.index) or [None])[0]
                await ctx.upsert_block(
                    f"gsearch:{call.index}",
                    block,
                    call.index,
                    before_key=first_text.key if first_text else None,
                )

        if maps_chunks:
            place_uris = [c["uri"] for c in maps_chunks if c.get("uri")]
            await ctx.status.places_found(place_uris)
            places = [
                (c.get("title") or c.get("uri"), c.get("uri"))
                for c in maps_chunks
                if c.get("uri")
            ]
            maps_block = self._format_maps_attribution_line(places)
            if maps_block:
                await ctx.upsert_block(f"maps:{call.index}", maps_block, call.index)

        mode = self.valves.SHOW_SEARCH_SUGGESTIONS
        if mode == "links" or (
            mode == "embed" and self._is_block_hidden("google_search")
        ):
            links = extract_suggestion_links(acc.entry_point_html, queries)
            line = self._format_suggestion_links_line(links)
            if line:
                await ctx.upsert_block(f"suggest:{call.index}", line, call.index)

    @classmethod
    def _classify_error(cls, e: Exception) -> Tuple[str, Optional[float]]:
        if genai_errors is not None and isinstance(e, genai_errors.APIError):
            code = getattr(e, "code", None)
            message = str(getattr(e, "message", "") or "")
            details = getattr(e, "details", None) or {}
            if code == 429:
                lowered = message.lower()
                if (
                    "per-day" in lowered
                    or "per day" in lowered
                    or "daily" in lowered
                    or "free tier" in lowered
                    or "free-tier" in lowered
                ):
                    return "quota", None
                return "rate_limit", cls._retry_after(details, message)
            if code in (500, 502, 503, 504):
                return "server", None
            if code == 400:
                return "bad_request", None
            if code in (401, 403):
                return "auth", None
            if code == 404:
                return "not_found", None
            return "other", None
        if isinstance(e, (asyncio.TimeoutError, OSError)):
            return "transport", None
        if str(type(e).__module__ or "").startswith(("httpx", "aiohttp")):
            return "transport", None
        return "other", None

    @classmethod
    def _retry_after(cls, details: Any, message: str = "") -> Optional[float]:
        candidates: List[Any] = []
        if isinstance(details, dict):
            candidates.append(details.get("retryDelay"))
            for value in details.values():
                if isinstance(value, dict) and "retryDelay" in value:
                    candidates.append(value.get("retryDelay"))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "retryDelay" in item:
                            candidates.append(item.get("retryDelay"))
        for raw in candidates:
            if raw is None:
                continue
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return min(float(raw), cls._RETRY_BACKOFF_CAP)
            match = cls._RETRY_AFTER_RE.match(str(raw).strip())
            if match:
                return min(float(match.group(1)), cls._RETRY_BACKOFF_CAP)
        return None

    _DROPPABLE_BUILTINS = (("url_context", "URL context"), ("google_search", "Google Search"))
    _PATTERN_TOOL_PAIR = re.compile(r"'(\w+)' and '(\w+)' cannot be combined")

    def _rejected_builtin(self, e: Exception, config) -> Optional[Tuple[str, str]]:
        if self._classify_error(e)[0] != "bad_request":
            return None
        match = self._PATTERN_TOOL_PAIR.search(f"{getattr(e, 'message', '')} {e}")
        if not match:
            return None
        carried = config.tools or []
        for field, label in self._DROPPABLE_BUILTINS:
            if field in match.groups() and any(
                getattr(t, field, None) is not None for t in carried
            ):
                return field, label
        return None

    async def _stream_call_with_retries(
        self, ctx, client, contents, config, call
    ):
        max_retries = int(getattr(self.valves, "MAX_RETRIES", 0) or 0)
        attempt = 0
        retried_empty = False
        dropped_builtin = False
        while True:
            checkpoint = ctx.msg.checkpoint()
            try:
                await self._stream_call(ctx, client, contents, config, call)
            except Exception as e:
                rejected = None if dropped_builtin else self._rejected_builtin(e, config)
                if rejected:
                    field, label = rejected
                    dropped_builtin = True
                    kept = [t for t in config.tools if getattr(t, field, None) is None]
                    config = config.model_copy(update={"tools": kept or None})
                    ctx.msg.rollback(checkpoint)
                    self._reset_call(call, ctx)
                    await ctx.status.activity(
                        f"Retrying without {label}: Google rejects it together with the other tools…"
                    )
                    continue
                kind, retry_after = self._classify_error(e)
                if kind not in self._RETRYABLE_KINDS or attempt >= max_retries:
                    raise
                attempt += 1
                await self._begin_retry(
                    ctx, call, checkpoint, attempt, max_retries, retry_after
                )
                continue
            if not self._is_empty_response(call):
                return config
            if retried_empty or attempt >= max_retries:
                return config
            retried_empty = True
            attempt += 1
            await self._begin_retry(ctx, call, checkpoint, attempt, max_retries, None)

    async def _begin_retry(
        self, ctx, call, checkpoint, attempt: int, max_retries: int,
        retry_after: Optional[float],
    ) -> None:
        ctx.msg.rollback(checkpoint)
        self._reset_call(call, ctx)
        await ctx.emit_replace()
        await ctx.status.activity(f"Retrying ({attempt}/{max_retries})…")
        delay = self._retry_delay(attempt, retry_after)
        if delay > 0:
            await asyncio.sleep(delay)

    def _retry_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(float(retry_after), self._RETRY_BACKOFF_CAP)
        base = float(self._RETRY_BACKOFF_BASE)
        delay = (base ** attempt) + random.uniform(0.0, 0.25 * base)
        return min(delay, self._RETRY_BACKOFF_CAP)

    def _reset_call(self, call, ctx=None) -> None:
        started = 0
        for fc in call.function_calls:
            task = fc.get("task") if isinstance(fc, dict) else None
            if task is not None:
                started += 1
                if not task.done():
                    task.cancel()
        if ctx is not None and started:
            ctx.turn.tool_calls_made = max(0, ctx.turn.tool_calls_made - started)
        call.raw_parts = []
        call.function_calls = []
        call.grounding = GroundingAccumulator()
        call.url_meta = []
        call.finish_reason = None
        call.finish_message = None
        call.usage = {}
        call.thought_key = None
        call.thought_text = ""
        call.thought_started = 0.0
        call.thought_runs = 0
        call.last_thought_render = 0.0

    @staticmethod
    def _is_empty_response(call) -> bool:
        if call.raw_parts or call.function_calls:
            return False
        return call.finish_reason in (None, "STOP", "OTHER")

    async def _handle_errors(self, e: Exception, ctx) -> None:
        kind, _retry_after = self._classify_error(e)
        api_message = str(getattr(e, "message", "") or "").strip()
        if kind == "auth":
            message = "Invalid or unauthorised API key"
        elif kind == "not_found":
            message = "Model not found — refresh the model list"
        elif api_message:
            message = api_message
        else:
            message = str(e)
        logger.error(f"Gemini request failed ({kind}): {e}")
        await ctx.status.notification(message, type="error")
        await ctx.emit_event(
            {
                "type": "source",
                "data": {
                    "source": {"name": "Gemini Error", "url": None},
                    "document": [traceback.format_exc()],
                    "metadata": [
                        {
                            "source": "gemini api",
                            "type": "error",
                            "date_accessed": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                },
            }
        )
        await ctx.status.emit("❌ Response with Errors", done=True, force=True)

    @classmethod
    def _finish_reason_notice(cls, finish_reason, finish_message) -> Optional[str]:
        if not finish_reason:
            return None
        name = getattr(finish_reason, "name", str(finish_reason))
        if name == "MAX_TOKENS":
            text = "Output limit reached"
        elif name in (
            "SAFETY",
            "RECITATION",
            "PROHIBITED_CONTENT",
            "SPII",
            "BLOCKLIST",
            "LANGUAGE",
        ):
            text = f"Response stopped by Google ({name})"
        elif name in (
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_OTHER",
            "NO_IMAGE",
            "IMAGE_RECITATION",
        ):
            text = f"Image not generated ({name})"
        elif name in (
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MISSING_THOUGHT_SIGNATURE",
        ):
            text = f"Tool call failed ({name})"
        elif name == "OTHER":
            text = "Response ended unexpectedly"
        else:
            return None
        if finish_message:
            text += f": {finish_message}"
        return "\n> ⚠️ " + text + "\n"

    @classmethod
    def _accumulate_usage(cls, total: dict, usage: dict) -> None:
        usage = usage or {}
        for key in cls._USAGE_KEYS:
            value = usage.get(key) or 0
            total[key] = total.get(key, 0) + value
        total["_last_prompt"] = usage.get("prompt_token_count") or 0
        total["_last_output"] = (usage.get("candidates_token_count") or 0) + (
            usage.get("thoughts_token_count") or 0
        )

    @staticmethod
    def _public_usage(turn, cost: Optional[float] = None) -> dict:
        total = turn.usage_total or {}
        prompt = int(total.get("_last_prompt") or 0)
        completion = int(
            (total.get("candidates_token_count") or 0)
            + (total.get("thoughts_token_count") or 0)
        )
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cached_tokens": int(total.get("cached_content_token_count") or 0),
            "reasoning_tokens": int(total.get("thoughts_token_count") or 0),
            "tool_use_prompt_tokens": int(
                total.get("tool_use_prompt_token_count") or 0
            ),
            "search_queries": len(turn.search_queries),
            "api_calls": len(turn.calls),
        }
        if cost is not None:
            usage["cost_usd"] = cost
        return usage

    @staticmethod
    def _format_token_count(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _format_status_line(self, ctx) -> str:
        line = "✅ Response Complete"
        setting = getattr(ctx.user_valves, "SHOW_TOKEN_COUNT", "Off")
        if setting and setting != "Off":
            total = ctx.turn.usage_total or {}
            context_used = int(total.get("_last_prompt") or 0) + int(
                total.get("_last_output") or 0
            )
            context_window = int(ctx.info.get("context_length") or 200_000)
            if context_window <= 0:
                context_window = 200_000
            percentage = min((context_used / context_window) * 100, 100.0)
            filled = int(percentage / 10)
            bar = "█" * filled + "░" * (10 - filled)
            line += (
                f" [{bar}] {self._format_token_count(context_used)}/"
                f"{self._format_token_count(context_window)} ({percentage:.1f}%)"
            )
            calls = len(ctx.turn.calls)
            if calls > 1:
                line += f" · {calls} calls"
            if setting == "With Cache":
                prompt = int(total.get("_last_prompt") or 0)
                cached = int(total.get("cached_content_token_count") or 0)
                if prompt > 0:
                    line += f" | ⚡ {cached * 100 // prompt}% cached"
        if getattr(ctx.user_valves, "SHOW_COST", True):
            cost = self._turn_cost(ctx)
            if cost is not None:
                line += f" | 💵 ≈{format_usd(cost)}"
        return line

    def _model_pricing_overrides(self) -> dict:
        raw = (self.valves.MODEL_PRICING_OVERRIDES or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as e:
            logger.warning(f"MODEL_PRICING_OVERRIDES is not valid JSON; ignored: {e}")
            return {}
        if not isinstance(parsed, dict):
            logger.warning("MODEL_PRICING_OVERRIDES must be a JSON object; ignored")
            return {}
        return {
            str(pattern): values
            for pattern, values in parsed.items()
            if isinstance(values, dict)
        }

    def _turn_cost(self, ctx) -> Optional[float]:
        rates = GEMINI_PRICING.rates_for(
            ctx.model,
            datetime.now(timezone.utc).date(),
            self._model_pricing_overrides(),
        )
        if rates is None:
            return None
        total = 0.0
        for call in ctx.turn.calls:
            total += GEMINI_PRICING.call_cost(rates, call.usage)
        generation = ctx.info.get("generation")
        if generation is None:
            generation = 3.0
        key = "gen3" if float(generation) >= 3 else "gen25"
        if key == "gen3":

            search_count = len(ctx.turn.search_queries)
        else:

            search_count = sum(
                1
                for call in ctx.turn.calls
                if any(c.get("kind") == "web" for c in call.grounding.chunks)
            )
        maps_count = sum(
            1
            for call in ctx.turn.calls
            if any(c.get("kind") == "maps" for c in call.grounding.chunks)
        )
        total += search_count * SEARCH_PRICE_PER_1K[key] / 1000
        total += maps_count * MAPS_PRICE_PER_1K[key] / 1000
        return total

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

            "maps_with_search": False,

            "maps_with_url_context": False,
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

    MEMORY_REVIEW_TASK = "memory_review"

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
        chat_model = self._model_name_from_body(body)
        override = str(self.valves.TASK_MODEL or "").strip()
        if not override:
            return chat_model
        if override.startswith(MODEL_ID_PREFIX):
            override = override[len(MODEL_ID_PREFIX) :]
        return override

    @classmethod
    def _sanitize_task_text(cls, text: str) -> str:
        if not text or ("<" not in text and "[" not in text):
            return text
        cleaned = cls._PATTERN_ANY_DETAILS.sub("\n", text)
        cleaned = cls._PATTERN_METADATA_MARKER.sub(" ", cleaned)
        cleaned = cls._PATTERN_TRAILING_SPACES.sub("", cleaned)
        return cls._PATTERN_EXCESS_BLANK_LINES.sub("\n\n", cleaned).strip()

    @classmethod
    def _extract_task_system(cls, messages: List[dict]) -> str:
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

    async def pipe(
        self,
        body,
        __user__=None,
        __event_emitter__=None,
        __metadata__=None,
        __tools__=None,
        __files__=None,
        __task__=None,
        __task_body__=None,
        __request__=None,
        __event_call__=None,
    ) -> str:
        user = __user__ or {}
        user_valves = self._user_valves_from(user)

        HIDDEN_BLOCKS.set(self._parse_hidden_blocks(getattr(user_valves, "HIDE_BLOCKS", None)))
        TOOL_APPROVAL.set(
            (
                ((__metadata__ or {}).get("params") or {}).get("tool_approval_mode", "full"),
                __event_call__,
            )
        )
        SLIM_OUTPUT.set(
            bool(
                __request__ is not None
                and getattr(getattr(__request__, "state", None), "internal", False) is True
            )
        )

        api_key = self._resolve_api_key(user_valves)
        if not str(api_key or "").strip():
            logger.error("no Gemini API key configured")
            if __event_emitter__ is not None:
                try:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {
                                "type": "error",
                                "content": "No Gemini API key configured. Set it in the "
                                "pipe valves or your personal user valves.",
                            },
                        }
                    )
                except Exception as e:
                    logger.warning(f"event emitter failed: {e}")
            return "Error: no Gemini API key configured."

        if __task__:
            return await self._run_task(body, __task__, __user__, user_valves)

        if __metadata__ is not None:
            __metadata__.setdefault("params", {})["reasoning_tags"] = False

        if inspect.isawaitable(__tools__):
            __tools__ = await __tools__

        model = self._model_name_from_body(body)
        info = self.get_model_info(model)
        ctx = new_request_context(
            self,
            event_emitter=__event_emitter__,
            model=model,
            info=info,
            user=user,
            user_valves=user_valves,
            metadata=__metadata__ or {},
            request=__request__,
            api_key=api_key,
            tools=dict(__tools__ or {}),
            builtin_tools=await self._gather_builtin_tools(
                __request__, user, __event_emitter__, __metadata__, __tools__
            ),
            rag_offset=_rag_citation_offset((__metadata__ or {}).get("sources") or []),
        )

        client = None
        try:
            client = self._build_genai_client(api_key, timeout_s=self.valves.REQUEST_TIMEOUT)
            messages = (body or {}).get("messages") or []
            prior = self._collect_prior_marker_entries(messages)
            media = await self._collect_native_media(ctx, client, messages, __files__, prior)
            cleaned = copy.deepcopy(messages)
            self._remove_specific_sources_from_rag_message(cleaned, media["native_names"])
            system_instruction, contents = await self._convert_history(
                cleaned, info, ctx, extra_parts_by_user_idx=media["parts"]
            )
            flags = self._resolve_builtin_flags(
                ctx.metadata, self.valves, user_valves, info, self._last_user_text(body)
            )
            decls = self._build_function_declarations(ctx)
            tools, tool_config, notes = self._assemble_tools(info, flags, decls)
            for note in notes:
                await ctx.status.activity(note)
            thinking = self._build_thinking_config(info, user_valves, body)
            config = self._build_generate_config(
                info=info,
                system_instruction=system_instruction,
                tools=tools,
                tool_config=tool_config,
                thinking=thinking,
                body=body,
                user_valves=user_valves,
            )
            await ctx.status.waiting()
            await self._run_generation_loop(ctx, contents, config, client)
            await self._persist_media_markers(ctx, media, prior)
            return await self._finalise(ctx, __metadata__)
        except asyncio.CancelledError:
            try:
                await ctx.status.emit(
                    "⏹️ Request Cancelled", done=True, hidden=False, force=True
                )
                partial = ctx.text()
                if partial:
                    await ctx.emit_event(
                        {"type": "replace", "data": {"content": partial}}
                    )
                await self._emit_completion(ctx)
            except Exception as e:
                logger.debug(f"Cancel cleanup emit failed: {e}")
            raise
        except Exception as e:
            await self._handle_errors(e, ctx)
            return ctx.text()
        finally:
            await self._close_client(client)

    def _user_valves_from(self, user: dict) -> "Pipe.UserValves":
        raw = (user or {}).get("valves")
        if isinstance(raw, self.UserValves):
            return raw
        if not raw:
            return self.UserValves()
        fields = {k: v for k, v in dict(raw).items() if k in self.UserValves.model_fields}
        return self.UserValves(**fields)

    def _last_user_text(self, body: dict) -> str:
        for message in reversed((body or {}).get("messages") or []):
            if isinstance(message, dict) and message.get("role") == "user":
                return self._block_text(message.get("content"))
        return ""

    def _collect_prior_marker_entries(self, messages: List[dict]) -> List[str]:
        entries: List[str] = []
        seen = set()
        for message in messages or []:
            for entry in self._extract_metadata_marker_from_message(copy.deepcopy(message)):
                if entry not in seen:
                    seen.add(entry)
                    entries.append(entry)
        return entries

    async def _persist_media_markers(self, ctx, media: dict, prior: List[str]) -> None:
        entries = list((media or {}).get("entries") or [])
        seen = set(entries)
        for entry in prior or []:
            if entry not in seen:
                seen.add(entry)
                entries.append(entry)
        markers = []
        for entry in entries:
            parts = str(entry).split(":", 2)
            if len(parts) < 3:
                continue
            try:
                msgnum = int(parts[0])
            except ValueError:
                continue
            markers.append(self._create_metadata_marker(parts[1], unquote(parts[2]), msgnum))
        if not markers:
            return
        call = ctx.turn.calls[-1].index if ctx.turn.calls else 0
        await ctx.upsert_block("markers", "".join(markers), call)

    async def _run_generation_loop(self, ctx, contents, config, client) -> None:
        max_tool_calls = max(int(getattr(self.valves, "MAX_TOOL_CALLS", 1) or 1), 1)
        retried: set = set()
        for _iteration in range(max_tool_calls + 3):
            call = CallState(index=len(ctx.turn.calls))
            ctx.turn.calls.append(call)
            checkpoint = ctx.msg.checkpoint()
            config = await self._stream_call_with_retries(ctx, client, contents, config, call)
            await self._finalize_call(ctx, call)
            finish_name = (
                getattr(call.finish_reason, "name", str(call.finish_reason))
                if call.finish_reason is not None
                else ""
            )
            if (
                finish_name
                in ("MALFORMED_FUNCTION_CALL", "MISSING_THOUGHT_SIGNATURE")
                and finish_name not in retried
            ):

                self._reset_call(call, ctx)
                ctx.msg.rollback(checkpoint)
                await ctx.emit_replace()
                if finish_name == "MISSING_THOUGHT_SIGNATURE":
                    self._restore_missing_signatures(contents)
                retried.add(finish_name)
                continue
            if not call.function_calls:
                return
            replay = self._merge_replay_parts(call.raw_parts)
            if replay:
                contents.append(gtypes.Content(role="model", parts=replay))
            contents.append(await self._finish_function_calls(ctx, call))
            if ctx.turn.tool_calls_made >= max_tool_calls:
                config = self._disable_function_calling(config)

    async def _gather_builtin_tools(
        self, __request__, user, __event_emitter__, __metadata__, __tools__
    ) -> dict:
        builtin_tools: dict = {}
        if BUILTIN_TOOLS_AVAILABLE and __request__ is not None:
            try:
                memory_enabled = (
                    user.get("settings", {}).get("ui", {}).get("memory", False)
                    if user
                    else False
                )
                builtin_tools = get_builtin_tools(
                    __request__,
                    {
                        "__user__": user,
                        "__event_emitter__": __event_emitter__,
                        "__chat_id__": (__metadata__ or {}).get("chat_id"),
                        "__message_id__": (__metadata__ or {}).get("message_id"),
                    },
                    features={"memory": memory_enabled},
                    model={"info": {"meta": {"builtinTools": {"skills": False}}}},
                )
                if inspect.isawaitable(builtin_tools):
                    builtin_tools = await builtin_tools
                logger.debug(
                    f"Loaded {len(builtin_tools)} builtin tools: "
                    f"{list(builtin_tools.keys())}"
                )
            except Exception as e:
                logger.warning(f"Could not load builtin tools: {e}")
                builtin_tools = {}
        metadata_tools = (__metadata__ or {}).get("tools", {})
        if metadata_tools:
            for t_name, t_data in metadata_tools.items():
                if t_name in builtin_tools:
                    continue
                if __tools__ and t_name in __tools__:
                    continue
                if isinstance(t_data, dict) and t_data.get("callable"):
                    builtin_tools[t_name] = t_data
        if builtin_tools:
            logger.debug(f"Executable tool set: {list(builtin_tools.keys())}")
        return builtin_tools

    def _restore_missing_signatures(self, contents: List[gtypes.Content]) -> None:
        last_user = -1
        for index, content in enumerate(contents):
            if content.role == "user":
                last_user = index
        for content in contents[last_user + 1 :]:
            if content.role != "model":
                continue
            if not any(getattr(p, "text", None) for p in content.parts):
                continue
            for index, part in enumerate(content.parts):
                if (
                    getattr(part, "function_call", None) is not None
                    and getattr(part, "thought_signature", None) is None
                ):
                    content.parts[index] = part.model_copy(
                        update={"thought_signature": DUMMY_THOUGHT_SIGNATURE}
                    )
                    break

    @staticmethod
    def _disable_function_calling(config) -> gtypes.GenerateContentConfig:
        config = config.model_copy(deep=True)
        tool_config = config.tool_config
        if tool_config is None:
            tool_config = gtypes.ToolConfig(
                function_calling_config=gtypes.FunctionCallingConfig(mode="NONE")
            )
        else:
            tool_config = tool_config.model_copy(
                update={
                    "function_calling_config": gtypes.FunctionCallingConfig(
                        mode="NONE"
                    )
                }
            )
        config.tool_config = tool_config
        return config

    async def _finalise(self, ctx, metadata: Optional[dict]) -> str:
        text = ctx.text()
        if text:
            await ctx.emit_event({"type": "replace", "data": {"content": text}})
        await ctx.status.complete(self._format_status_line(ctx))
        await self._emit_completion(ctx)
        if CHATS_AVAILABLE and metadata and ctx.turn.usage_total:
            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            if chat_id and message_id and not str(chat_id).startswith("local:"):
                try:
                    cost = self._turn_cost(ctx) if getattr(
                        ctx.user_valves, "SHOW_COST", True
                    ) else None
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id,
                        {"usage": self._public_usage(ctx.turn, cost)},
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist usage to chat_message: {e}")
        return text

    async def _emit_completion(self, ctx) -> None:
        data: dict = {
            "choices": [{"finish_reason": "stop", "delta": {"content": ""}}],
            "done": True,
        }
        if ctx.turn.usage_total:
            cost = self._turn_cost(ctx) if getattr(
                ctx.user_valves, "SHOW_COST", True
            ) else None
            data["usage"] = self._public_usage(ctx.turn, cost)
        await ctx.emit_event({"type": "chat:completion", "data": data})

    @staticmethod
    async def _close_client(client) -> None:
        if client is None:
            return
        try:
            aclose = getattr(getattr(client, "aio", None), "aclose", None)
            if callable(aclose):
                await aclose()
        except BaseException as e:
            logger.debug(f"client close failed: {e}")
