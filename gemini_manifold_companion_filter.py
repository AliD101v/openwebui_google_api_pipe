"""
title: Gemini Manifold Companion
author: AliD101v
github: https://github.com/AliD101v/openwebui_google_api_pipe
id: gemini_manifold_companion
description: Routes OpenWebUI's built-in web_search button to the Gemini pipe's native Google Search grounding. Use with the Gemini pipe.
version: 0.1.0
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        ONLY_GEMINI_PIPE_MODELS: bool = True

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        features = body.get("features") or {}
        if not features.get("web_search"):
            return body

        # Only intercept requests that actually target the Gemini pipe; a
        # different manifold behind the same OpenWebUI install keeps its own
        # web_search handling.
        if self.valves.ONLY_GEMINI_PIPE_MODELS:
            candidates = [body.get("model") or ""]
            if __model__:
                candidates.append(__model__.get("id") or "")
                info = __model__.get("info") or {}
                candidates.append(info.get("base_model_id") or "")
            if not any("gemini/" in str(candidate) for candidate in candidates):
                return body

        features["web_search"] = False
        target = __metadata__ if __metadata__ is not None else body.setdefault("metadata", {})
        target["gemini_web_search"] = True
        return body

    async def outlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        return body
