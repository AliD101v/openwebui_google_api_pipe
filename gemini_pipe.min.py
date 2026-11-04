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
