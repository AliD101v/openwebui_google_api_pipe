"""Stream chunk handling: part dispatch (thoughts, text, tool calls),
per-call finalisation (grounding, URL context, finish notices, usage) and
the replay-part merge for multi-call loops.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from google.genai import types as gtypes

logger = logging.getLogger(__name__)


class _StreamParts:
    _THOUGHT_RENDER_INTERVAL = 0.3

    async def _stream_call(self, ctx, client, contents, config, call) -> None:
        """One streamed generation call, handing every chunk to the dispatcher."""
        async for chunk_ in await client.aio.models.generate_content_stream(
            model=ctx.model, contents=contents, config=config
        ):
            await self._handle_chunk(ctx, call, chunk_)

    async def _handle_chunk(self, ctx, call, chunk_) -> None:
        """One streamed chunk: prompt feedback, usage, grounding, finish
        reason and the candidate's parts."""
        feedback = _get(chunk_, "prompt_feedback")
        reason = _get(feedback, "block_reason") if feedback is not None else None
        if reason:
            await ctx.emit_text(call.index, self._format_notice(f"Prompt blocked: {reason}"))
            call.finish_reason = "PROMPT_BLOCKED"
            return

        usage = _get(chunk_, "usage_metadata")
        if usage:
            call.usage = self._usage_to_dict(usage)

        candidates = _get(chunk_, "candidates") or []
        if not candidates:
            return
        cand = candidates[0]

        call.grounding.add(_get(cand, "grounding_metadata"))
        self._extend_url_meta(call, _get(cand, "url_context_metadata"))

        finish_reason = _get(cand, "finish_reason")
        if finish_reason:
            call.finish_reason = finish_reason
        finish_message = _get(cand, "finish_message")
        if finish_message:
            call.finish_message = finish_message

        content = _get(cand, "content")
        for part in _get(content, "parts") or []:
            call.raw_parts.append(part)
            await self._dispatch_part(ctx, call, part)

    def _extend_url_meta(self, call, ucm) -> None:
        """Collect the URL-context retrieval results, deduped by URL."""
        for entry in _get(ucm, "url_metadata") or []:
            url = _get(entry, "retrieved_url")
            if not url or any(item["url"] == url for item in call.url_meta):
                continue
            status_value = _get(entry, "url_retrieval_status")
            call.url_meta.append(
                {"url": url, "status": getattr(status_value, "name", status_value) or ""}
            )

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        """The chunk's usage metadata as a dict with snake_case keys."""
        dump = getattr(usage, "model_dump", None)
        data = dump(exclude_none=True) if callable(dump) else dict(usage or {})
        return {
            re.sub(r"(?<!^)(?=[A-Z])", "_", str(key)).lower(): value
            for key, value in data.items()
        }

    async def _dispatch_part(self, ctx, call, part) -> None:
        """Route one part: thoughts render blocks, text streams, tool calls
        are recorded; everything else stays in raw_parts only."""
        text = _get(part, "text")
        thought = bool(_get(part, "thought"))
        if thought and text:
            await self._handle_thought_part(ctx, call, text)
            return
        if text:
            await self._close_thought_run(ctx, call)
            await ctx.status.responding()
            await ctx.emit_text(call.index, text)
            return

        function_call = _get(part, "function_call")
        if function_call is not None:
            await self._close_thought_run(ctx, call)
            call.function_calls.append(
                {
                    "part": part,
                    "raw_id": _get(function_call, "id"),
                    "id": _get(function_call, "id")
                    or f"call_{call.index}_{len(call.function_calls)}",
                    "name": _get(function_call, "name") or "tool",
                    "args": _get(function_call, "args") or {},
                    "key": f"fc:{call.index}:{len(call.function_calls)}",
                }
            )
            await self._start_function_call(ctx, call, call.function_calls[-1])
            return

        tool_call = _get(part, "tool_call")
        if tool_call is not None:
            args = _get(tool_call, "args")
            queries = (
                [q for q in (args.get("queries") or []) if q]
                if isinstance(args, dict)
                else []
            )
            if queries:
                await ctx.status.searching_google(queries)
            return

        if not thought and _get(part, "inline_data") is not None:
            logger.debug("skipping inline_data part (image output is a later milestone)")

    async def _handle_thought_part(self, ctx, call, text: str) -> None:
        """Accumulate one thought part into the call's reasoning run and
        re-render its block at most every _THOUGHT_RENDER_INTERVAL seconds."""
        if not call.thought_key:
            call.thought_runs += 1
            call.thought_key = f"think:{call.index}:{call.thought_runs}"
            call.thought_started = time.monotonic()
            call.last_thought_render = 0.0
            await ctx.status.thinking()
        call.thought_text += text
        now = time.monotonic()
        if now - call.last_thought_render < self._THOUGHT_RENDER_INTERVAL:
            return
        call.last_thought_render = now
        if not getattr(ctx.user_valves, "SHOW_THOUGHTS", True):
            return
        block = self._format_thinking_block(call.thought_text, done=False)
        if block:
            await ctx.upsert_block(call.thought_key, block, call.index)

    async def _close_thought_run(self, ctx, call) -> None:
        """Render the open reasoning run done (with its duration) and reset it."""
        if not call.thought_key:
            return
        duration = time.monotonic() - call.thought_started if call.thought_started else 0.0
        block = self._format_thinking_block(call.thought_text, done=True, duration=duration)
        if block:
            await ctx.upsert_block(call.thought_key, block, call.index)
        call.thought_key = None
        call.thought_text = ""

    async def _finalize_call(self, ctx, call) -> None:
        """After a call's stream ends: close thoughts, apply grounding, report
        URL context, render the finish notice and accumulate usage."""
        await self._close_thought_run(ctx, call)

        if not call.grounding.empty():
            await self._apply_grounding(ctx, call)

        if call.url_meta:
            ok = [e["url"] for e in call.url_meta if str(e.get("status") or "").lower() == "ok"]
            failed = len(call.url_meta) - len(ok)
            await ctx.status.read_urls(ok, failed)
            block = self._format_url_context_block(call.index, call.url_meta)
            if block:
                first_text = (ctx.msg.text_segments(call.index) or [None])[0]
                await ctx.upsert_block(
                    f"urlctx:{call.index}",
                    block,
                    call.index,
                    before_key=first_text.key if first_text else None,
                )

        notice = self._finish_reason_notice(call.finish_reason, call.finish_message)
        if notice:
            await ctx.upsert_block(f"notice:{call.index}", notice, call.index)

        if call.usage:
            self._accumulate_usage(ctx.turn.usage_total, call.usage)

    def _merge_replay_parts(self, parts) -> List:
        """The model content to replay for a next API call: plain adjacent text
        parts are merged, empty parts are dropped, everything else is kept raw."""
        merged: List[Any] = []
        for part in parts or []:
            text = _get(part, "text")
            function_call = _get(part, "function_call")
            tool_call = _get(part, "tool_call")
            tool_response = _get(part, "tool_response")
            inline_data = _get(part, "inline_data")
            signature = _get(part, "thought_signature")
            if not (text or function_call or tool_call or tool_response or inline_data or signature):
                continue
            plain = (
                text
                and signature is None
                and function_call is None
                and tool_call is None
                and tool_response is None
                and inline_data is None
                and _get(part, "executable_code") is None
                and _get(part, "code_execution_result") is None
            )
            thought = bool(_get(part, "thought"))
            previous = merged[-1] if merged else None
            if (
                plain
                and isinstance(previous, gtypes.Part)
                and previous.text
                and not getattr(previous, "thought_signature", None)
                and bool(getattr(previous, "thought", False)) == thought
            ):
                merged[-1] = gtypes.Part(text=previous.text + text, thought=thought)
                continue
            if plain:
                merged.append(gtypes.Part(text=text, thought=thought))
            else:
                merged.append(part)
        return merged
