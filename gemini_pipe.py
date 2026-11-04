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

    # END GENERATED SECTION: gemini_pipe.pipe_method_groups
