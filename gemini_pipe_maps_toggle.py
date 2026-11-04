"""
title: Gemini Maps Toggle
author: AliD101v
github: https://github.com/AliD101v/openwebui_google_api_pipe
id: gemini_pipe_maps_toggle
description: Force Google Maps grounding for the next Gemini message. Use in combination with the Gemini pipe: https://openwebui.com/f/alid101v/gemini_pipe
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
        # Simple map pin SVG icon (base64, stroke currentColor)
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxwYXRoIGQ9Ik0xMiAyMS41cy03LTUuNy03LTExLjVhNyA3IDAgMSAxIDE0IDBjMCA1LjgtNyAxMS41LTcgMTEuNXoiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEwIiByPSIyLjUiLz48L3N2Zz4="
        )

    async def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        target = __metadata__ if __metadata__ is not None else body.setdefault("metadata", {})
        target["gemini_maps"] = True
        return body

    async def outlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        return body
