"""Per-request render state: message segments, per-API-call state and the
per-turn aggregates (citations, usage, search queries, tool bookkeeping).

Stream section: compiled at module level; the template provides all imports,
and the grounding section above defines GroundingAccumulator/CitationRegistry.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Segment:
    """One rendered piece of the message: streamed text or a <details> block."""

    kind: str
    key: str
    call: int
    content: str


@dataclass
class RenderedMessage:
    """The message as it will be persisted, in segment order.

    Text segments stream; blocks (the <details> collapsibles) are upserted as a
    whole. ``render()`` is the single source of truth for the message content,
    and ``append_text`` returns exactly the string by which ``render()`` grew,
    so ``message`` delta events stay consistent with it.
    """

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
    """Everything one API call (one pass of the generation loop) contributes."""

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
    """The whole turn: every call plus the aggregates shared across calls."""

    calls: List[CallState] = field(default_factory=list)
    citations: CitationRegistry = field(default_factory=CitationRegistry)
    usage_total: Dict[str, Any] = field(default_factory=dict)
    search_queries: List[str] = field(default_factory=list)
    tool_calls_made: int = 0
    degraded: set = field(default_factory=set)
    name_map: Dict[str, str] = field(default_factory=dict)
    functions_offered: bool = False
