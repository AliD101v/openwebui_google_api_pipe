"""Generation config assembly: the thinking/reasoning controls and the
GenerateContentConfig for one call.

Sampling is the model's own business: this pipe never sends temperature,
top_p, top_k, candidate_count or seed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _Config:
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
