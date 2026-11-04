"""The status line: one entry per phase of work the turn actually goes through.

OpenWebUI keeps every status event it receives in ``message.statusHistory`` and
renders the list (Chat.svelte pushes unconditionally — there is no
de-duplication on the frontend). So every emission is a permanent visible line,
and a status that does not describe real work is noise the user has to scroll
past.

The rule that follows: **a status is emitted by whatever is starting, never to
clear whatever finished.** A phase that ends emits nothing; the next phase
describes itself when it starts.

``done=True`` closes the line and stops its shimmer, so it belongs only on the
final status of a turn — never mid-stream.

Recognized status fields (open-webui ``StatusHistory/StatusItem.svelte``):
``action``, ``description``, ``done``, ``hidden``, ``query``, ``queries``,
``urls``, ``items``, ``count``. ``action="web_search"`` together with ``urls``
renders the results as OpenWebUI's native clickable source list instead of a
plain line.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable


class StatusEmitter:
    def __init__(self, emit_event: Callable[[dict], Awaitable[None]]):
        """Wrap an OpenWebUI ``emit_event`` callback with dedup state for status events."""
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
        """Emit an OpenWebUI ``status`` event, skipping an identical repeat.

        Dedup is what keeps the phase model cheap: a phase family can announce
        itself on every block without stuttering the history when several blocks
        of the same kind arrive in a row (grounded answers split prose into many
        text blocks around their citation markers).
        """
        data: dict = {"description": description, "done": done}
        if hidden is not None:
            data["hidden"] = hidden
        data.update(fields)

        if not force and data == self._last_payload:
            return

        await self._emit_event({"type": "status", "data": data})
        self._last_payload = data

    # -- phases ------------------------------------------------------------
    # One method per thing that can actually be happening, so call sites read as
    # intent and the wording stays consistent across handlers.

    async def waiting(self) -> None:
        """Before the first byte: the request is out, nothing has come back yet."""
        await self.emit("Waiting for response...", hidden=False, force=True)

    async def thinking(self) -> None:
        """A thought is streaming."""
        await self.emit("💭 Thinking...")

    async def responding(self) -> None:
        """Text is streaming — the model is writing the answer.

        Emitted on every text run, not once per turn: after a tool runs, text
        resuming is a real phase change the user should see.
        """
        await self.emit("Responding...")

    async def searching_google(self, queries: list) -> None:
        """Google Search grounding is running for the model."""
        shown = " · ".join(queries[:3]) if queries else ""
        await self.emit(f"🔍 Searching Google: {shown}" if shown else "🔍 Searching Google...")

    async def web_search_done(self, urls: list, queries: list = None) -> None:
        """Report finished search results using OpenWebUI's native renderer.

        With ``action="web_search"`` and ``urls`` the frontend renders a
        clickable source list rather than a plain text line.
        """
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
        """URL context finished reading the pages from the conversation."""
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
        """Google Maps grounding returned places."""
        if not urls:
            return
        await self.emit(
            "Found {{count}} places",
            action="web_search",
            urls=list(urls),
            count=len(urls),
        )

    async def fetching_url(self, url: str = "") -> None:
        """A URL from the conversation is being fetched for the model."""
        await self.emit(f"🌐 Fetching {url}" if url else "🌐 Fetching URL...")

    async def running_tool(self, tool_name: str) -> None:
        """An OpenWebUI-side tool call is being dispatched."""
        await self.emit(f"🔧 Running {tool_name}..." if tool_name else "🔧 Running tool...")

    async def activity(self, description: str) -> None:
        """Free-form in-progress phase, for one-off states without a named phase."""
        await self.emit(description, done=False)

    async def complete(self, description: str) -> None:
        """Close the status line for this turn. Only the final status is ``done``."""
        await self.emit(description, done=True, force=True)

    async def notification(self, content: str, *, type: str = "warning") -> None:
        """Emit an OpenWebUI ``notification`` event (e.g. warning/error toast)."""
        await self._emit_event(
            {"type": "notification", "data": {"type": type, "content": content}}
        )
