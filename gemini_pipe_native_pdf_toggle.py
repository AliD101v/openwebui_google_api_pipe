"""
title: Gemini Native PDFs Toggle
author: AliD101v
github: https://github.com/AliD101v/openwebui_google_api_pipe
id: gemini_pipe_native_pdf_toggle
description: Send attached PDFs to Gemini as whole documents (tables, charts and layout included) instead of OpenWebUI's text retrieval. Use in combination with the Gemini pipe: https://openwebui.com/f/alid101v/gemini_pipe
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
        # Document icon (base64 SVG, stroke currentColor). No file_handler flag on purpose: OpenWebUI's own
        # file processing must keep handling the files the pipe does not send natively (text, Office, ...).
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xNCAySDZhMiAyIDAgMCAwLTIgMnYxNmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAwIDAgMCAyLTJWOHoiPjwvcGF0aD48cG9seWxpbmUgcG9pbnRzPSIxNCAyIDE0IDggMjAgOCI+PC9wb2x5bGluZT48bGluZSB4MT0iOSIgeTE9IjEzIiB4Mj0iMTUiIHkyPSIxMyI+PC9saW5lPjxsaW5lIHgxPSI5IiB5MT0iMTciIHgyPSIxNSIgeTI9IjE3Ij48L2xpbmU+PC9zdmc+"
        )

    async def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        target = __metadata__ if __metadata__ is not None else body.setdefault("metadata", {})
        target["gemini_native_pdf"] = True
        return body

    async def outlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        return body
