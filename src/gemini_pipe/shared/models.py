"""Method group: client factory, model discovery, capabilities, list cache."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _Models:
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
