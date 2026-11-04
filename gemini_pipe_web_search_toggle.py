"""
title: Gemini Web Search Toggle
author: AliD101v
github: https://github.com/AliD101v/openwebui_google_api_pipe
id: gemini_pipe_web_search_toggle
description: Force Google Search grounding for the next Gemini message. Use in combination with the Gemini pipe: https://openwebui.com/f/alid101v/gemini_pipe
version: 0.1.0
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        # Simple magnifying glass SVG icon (base64)
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgdmlld0JveD0iMCAwIDI0IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxjaXJjbGUgY3g9IjExIiBjeT0iMTEiIHI9IjgiIHN0cm9rZT0iIzAwMCIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48bGluZSB4MT0iMjAiIHkxPSIyMCIgeDI9IjE2LjY1IiB5Mj0iMTYuNjUiIHN0cm9rZT0iIzAwMCIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48L3N2Zz4="
        )

    async def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
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
