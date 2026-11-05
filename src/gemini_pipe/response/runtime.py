"""Response-side runtime: grounding application, redirect URL resolution,
error classification and retries, finish-reason notices, usage and the
status line.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class _Runtime:
    _RETRY_BACKOFF_BASE = 2.0
    _RETRY_BACKOFF_CAP = 30.0
    _RESOLVED_URL_CACHE_LIMIT = 2000
    _GROUNDING_REDIRECT_PREFIX = (
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
    )
    _RETRYABLE_KINDS = ("rate_limit", "server", "transport")
    _RETRY_AFTER_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*s?$")
    _USAGE_KEYS = (
        "prompt_token_count",
        "cached_content_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
        "total_token_count",
    )

    async def _resolve_grounding_urls(self, uris: List[str]) -> Dict[str, str]:
        """Map grounding redirect URIs to their real destinations.

        Only ``grounding-api-redirect`` URIs are resolved (reading just the
        Location header, never following); anything else or any failure maps
        a URI to itself. Results are cached in an LRU on the Pipe.
        """
        resolved: Dict[str, str] = {}
        todo: List[str] = []
        for uri in dict.fromkeys(uris or []):
            if not uri:
                continue
            cached = self._resolved_url_cache.get(uri)
            if cached is not None:
                resolved[uri] = cached
            elif not uri.startswith(self._GROUNDING_REDIRECT_PREFIX):
                continue
            else:
                todo.append(uri)
        if not (todo and self.valves.RESOLVE_GROUNDING_URLS and HTTPX_AVAILABLE):
            for uri in todo:
                resolved[uri] = uri
            return resolved

        results: Dict[str, str] = {}
        semaphore = asyncio.Semaphore(8)

        async def _one(uri: str) -> None:
            async with semaphore:
                try:
                    async with httpx.AsyncClient(follow_redirects=False) as client:
                        response = await client.get(uri, timeout=3.0)
                    results[uri] = response.headers.get("Location") or uri
                except Exception:
                    results[uri] = uri

        try:
            await asyncio.wait_for(
                asyncio.gather(*(_one(uri) for uri in todo)), timeout=3.0
            )
        except asyncio.TimeoutError:
            logger.debug("grounding URL resolution budget exceeded; keeping redirects")
        for uri in todo:
            destination = results.get(uri, uri)
            self._cache_resolved_url(uri, destination)
            resolved[uri] = destination
        return resolved

    def _cache_resolved_url(self, uri: str, destination: str) -> None:
        """Store a resolved URL in the Pipe-level LRU (2000 entries)."""
        cache = self._resolved_url_cache
        if uri in cache:
            cache.pop(uri)
        cache[uri] = destination
        while len(cache) > self._RESOLVED_URL_CACHE_LIMIT:
            cache.pop(next(iter(cache)))

    async def _apply_grounding(self, ctx, call) -> None:
        """Cite the call's text, emit its source events, and render the Google
        Search block (web sources) or the Google Maps attribution (places)."""
        acc = call.grounding
        web_chunks = [c for c in acc.chunks if c.get("kind") == "web"]
        maps_chunks = [c for c in acc.chunks if c.get("kind") == "maps"]

        queries = [q for q in acc.queries if q]
        if queries:
            for query in queries:
                if query not in ctx.turn.search_queries:
                    ctx.turn.search_queries.append(query)
            await ctx.status.searching_google(queries)

        uris = [c["uri"] for c in web_chunks if c.get("uri")]
        resolved = await self._resolve_grounding_urls(uris)

        segments = ctx.msg.text_segments(call.index)
        joined = "".join(segment.content for segment in segments)
        cited, new_numbers, references = cite_call_text(
            joined, acc, ctx.turn.citations, resolved
        )
        if segments:
            segments[0].content = cited
            for segment in segments[1:]:
                segment.content = ""
            await ctx.emit_replace()

        now_iso = datetime.now(timezone.utc).isoformat()
        for number in sorted(new_numbers):
            info = ctx.turn.citations.sources.get(number) or {}
            url = info.get("url") or ""
            title = info.get("title") or url
            await ctx.emit_event(
                {
                    "type": "source",
                    "data": {
                        "source": {"name": title, "url": url, "id": str(number)},
                        "document": references.get(number) or [title],
                        "metadata": [
                            {
                                "source": f"{url}#{number}",
                                "date_accessed": now_iso,
                                "name": f"[{number}]",
                            }
                        ],
                    },
                }
            )

        web_numbers = [
            number
            for number in sorted(references)
            if (ctx.turn.citations.sources.get(number) or {}).get("kind") == "web"
        ]
        if web_chunks or queries:
            web_urls = [
                (ctx.turn.citations.sources.get(number) or {}).get("url") or ""
                for number in web_numbers
            ]
            if web_urls:
                await ctx.status.web_search_done(web_urls, queries)
            source_triples = []
            for number in web_numbers:
                info = ctx.turn.citations.sources.get(number) or {}
                url = info.get("url") or ""
                source_triples.append((number, info.get("title") or url, url))
            embed_html = ""
            if (
                self.valves.SHOW_SEARCH_SUGGESTIONS == "embed"
                and acc.entry_point_html
            ):
                embed_html = prepare_entry_point_embed(acc.entry_point_html)
            block = self._format_google_search_block(
                call.index, queries, source_triples, embed_html
            )
            if block:
                first_text = (ctx.msg.text_segments(call.index) or [None])[0]
                await ctx.upsert_block(
                    f"gsearch:{call.index}",
                    block,
                    call.index,
                    before_key=first_text.key if first_text else None,
                )

        if maps_chunks:
            place_uris = [c["uri"] for c in maps_chunks if c.get("uri")]
            await ctx.status.places_found(place_uris)
            places = [
                (c.get("title") or c.get("uri"), c.get("uri"))
                for c in maps_chunks
                if c.get("uri")
            ]
            maps_block = self._format_maps_attribution_line(places)
            if maps_block:
                await ctx.upsert_block(f"maps:{call.index}", maps_block, call.index)

        mode = self.valves.SHOW_SEARCH_SUGGESTIONS
        if mode == "links" or (
            mode == "embed" and self._is_block_hidden("google_search")
        ):
            links = extract_suggestion_links(acc.entry_point_html, queries)
            line = self._format_suggestion_links_line(links)
            if line:
                await ctx.upsert_block(f"suggest:{call.index}", line, call.index)

    @classmethod
    def _classify_error(cls, e: Exception) -> Tuple[str, Optional[float]]:
        """(kind, retry-after seconds) for an exception from the API or the
        transport; retryable kinds are rate_limit, server and transport."""
        if genai_errors is not None and isinstance(e, genai_errors.APIError):
            code = getattr(e, "code", None)
            message = str(getattr(e, "message", "") or "")
            details = getattr(e, "details", None) or {}
            if code == 429:
                lowered = message.lower()
                if (
                    "per-day" in lowered
                    or "per day" in lowered
                    or "daily" in lowered
                    or "free tier" in lowered
                    or "free-tier" in lowered
                ):
                    return "quota", None
                return "rate_limit", cls._retry_after(details, message)
            if code in (500, 502, 503, 504):
                return "server", None
            if code == 400:
                return "bad_request", None
            if code in (401, 403):
                return "auth", None
            if code == 404:
                return "not_found", None
            return "other", None
        if isinstance(e, (asyncio.TimeoutError, OSError)):
            return "transport", None
        if str(type(e).__module__ or "").startswith(("httpx", "aiohttp")):
            return "transport", None
        return "other", None

    @classmethod
    def _retry_after(cls, details: Any, message: str = "") -> Optional[float]:
        """The retryDelay (google.rpc.RetryInfo, e.g. "12s"), capped at 30 s."""
        candidates: List[Any] = []
        if isinstance(details, dict):
            candidates.append(details.get("retryDelay"))
            for value in details.values():
                if isinstance(value, dict) and "retryDelay" in value:
                    candidates.append(value.get("retryDelay"))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "retryDelay" in item:
                            candidates.append(item.get("retryDelay"))
        for raw in candidates:
            if raw is None:
                continue
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return min(float(raw), cls._RETRY_BACKOFF_CAP)
            match = cls._RETRY_AFTER_RE.match(str(raw).strip())
            if match:
                return min(float(match.group(1)), cls._RETRY_BACKOFF_CAP)
        return None

    async def _stream_call_with_retries(
        self, ctx, client, contents, config, call
    ) -> None:
        """One generation call, retrying transient failures and empty
        responses; everything else is re-raised to the pipe."""
        max_retries = int(getattr(self.valves, "MAX_RETRIES", 0) or 0)
        attempt = 0
        retried_empty = False
        while True:
            checkpoint = ctx.msg.checkpoint()
            try:
                await self._stream_call(ctx, client, contents, config, call)
            except Exception as e:
                kind, retry_after = self._classify_error(e)
                if kind not in self._RETRYABLE_KINDS or attempt >= max_retries:
                    raise
                attempt += 1
                await self._begin_retry(
                    ctx, call, checkpoint, attempt, max_retries, retry_after
                )
                continue
            if not self._is_empty_response(call):
                return
            if retried_empty or attempt >= max_retries:
                return
            retried_empty = True
            attempt += 1
            await self._begin_retry(ctx, call, checkpoint, attempt, max_retries, None)

    async def _begin_retry(
        self, ctx, call, checkpoint, attempt: int, max_retries: int,
        retry_after: Optional[float],
    ) -> None:
        """Roll the message back, reset the call, and wait out the backoff."""
        ctx.msg.rollback(checkpoint)
        self._reset_call(call, ctx)
        await ctx.emit_replace()
        await ctx.status.activity(f"Retrying ({attempt}/{max_retries})…")
        delay = self._retry_delay(attempt, retry_after)
        if delay > 0:
            await asyncio.sleep(delay)

    def _retry_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(float(retry_after), self._RETRY_BACKOFF_CAP)
        base = float(self._RETRY_BACKOFF_BASE)
        delay = (base ** attempt) + random.uniform(0.0, 0.25 * base)
        return min(delay, self._RETRY_BACKOFF_CAP)

    def _reset_call(self, call, ctx=None) -> None:
        started = 0
        for fc in call.function_calls:
            task = fc.get("task") if isinstance(fc, dict) else None
            if task is not None:
                started += 1
                if not task.done():
                    task.cancel()
        if ctx is not None and started:
            ctx.turn.tool_calls_made = max(0, ctx.turn.tool_calls_made - started)
        call.raw_parts = []
        call.function_calls = []
        call.grounding = GroundingAccumulator()
        call.url_meta = []
        call.finish_reason = None
        call.finish_message = None
        call.usage = {}
        call.thought_key = None
        call.thought_text = ""
        call.thought_started = 0.0
        call.thought_runs = 0
        call.last_thought_render = 0.0

    @staticmethod
    def _is_empty_response(call) -> bool:
        if call.raw_parts or call.function_calls:
            return False
        return call.finish_reason in (None, "STOP", "OTHER")

    async def _handle_errors(self, e: Exception, ctx) -> None:
        """Terminal failure: a notification, an error source with the
        traceback, and the final status."""
        kind, _retry_after = self._classify_error(e)
        api_message = str(getattr(e, "message", "") or "").strip()
        if kind == "auth":
            message = "Invalid or unauthorised API key"
        elif kind == "not_found":
            message = "Model not found — refresh the model list"
        elif api_message:
            message = api_message
        else:
            message = str(e)
        logger.error(f"Gemini request failed ({kind}): {e}")
        await ctx.status.notification(message, type="error")
        await ctx.emit_event(
            {
                "type": "source",
                "data": {
                    "source": {"name": "Gemini Error", "url": None},
                    "document": [traceback.format_exc()],
                    "metadata": [
                        {
                            "source": "gemini api",
                            "type": "error",
                            "date_accessed": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                },
            }
        )
        await ctx.status.emit("❌ Response with Errors", done=True, force=True)

    @classmethod
    def _finish_reason_notice(cls, finish_reason, finish_message) -> Optional[str]:
        """The warning line for a non-STOP finish reason, or None (STOP and
        unknown reasons get no notice)."""
        if not finish_reason:
            return None
        name = getattr(finish_reason, "name", str(finish_reason))
        if name == "MAX_TOKENS":
            text = "Output limit reached"
        elif name in (
            "SAFETY",
            "RECITATION",
            "PROHIBITED_CONTENT",
            "SPII",
            "BLOCKLIST",
            "LANGUAGE",
        ):
            text = f"Response stopped by Google ({name})"
        elif name in (
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_OTHER",
            "NO_IMAGE",
            "IMAGE_RECITATION",
        ):
            text = f"Image not generated ({name})"
        elif name in (
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MISSING_THOUGHT_SIGNATURE",
        ):
            text = f"Tool call failed ({name})"
        elif name == "OTHER":
            text = "Response ended unexpectedly"
        else:
            return None
        if finish_message:
            text += f": {finish_message}"
        return "\n> ⚠️ " + text + "\n"

    @classmethod
    def _accumulate_usage(cls, total: dict, usage: dict) -> None:
        """Add one call's usage into the turn total (sums plus the last call's
        prompt/output for the context gauge)."""
        usage = usage or {}
        for key in cls._USAGE_KEYS:
            value = usage.get(key) or 0
            total[key] = total.get(key, 0) + value
        total["_last_prompt"] = usage.get("prompt_token_count") or 0
        total["_last_output"] = (usage.get("candidates_token_count") or 0) + (
            usage.get("thoughts_token_count") or 0
        )

    @staticmethod
    def _public_usage(turn) -> dict:
        """Project the turn's usage onto OpenWebUI's usage schema: the prompt
        gauge is the LAST call's prompt, completion is candidates+thoughts
        summed over all calls."""
        total = turn.usage_total or {}
        prompt = int(total.get("_last_prompt") or 0)
        completion = int(
            (total.get("candidates_token_count") or 0)
            + (total.get("thoughts_token_count") or 0)
        )
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cached_tokens": int(total.get("cached_content_token_count") or 0),
            "reasoning_tokens": int(total.get("thoughts_token_count") or 0),
            "tool_use_prompt_tokens": int(
                total.get("tool_use_prompt_token_count") or 0
            ),
            "search_queries": len(turn.search_queries),
            "api_calls": len(turn.calls),
        }

    @staticmethod
    def _format_token_count(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _format_status_line(self, ctx) -> str:
        """The final status line: completion plus the context gauge when the
        user asks for token counts."""
        line = "✅ Response Complete"
        setting = getattr(ctx.user_valves, "SHOW_TOKEN_COUNT", "Off")
        if not setting or setting == "Off":
            return line
        total = ctx.turn.usage_total or {}
        context_used = int(total.get("_last_prompt") or 0) + int(
            total.get("_last_output") or 0
        )
        context_window = int(ctx.info.get("context_length") or 200_000)
        if context_window <= 0:
            context_window = 200_000
        percentage = min((context_used / context_window) * 100, 100.0)
        filled = int(percentage / 10)
        bar = "█" * filled + "░" * (10 - filled)
        line += (
            f" [{bar}] {self._format_token_count(context_used)}/"
            f"{self._format_token_count(context_window)} ({percentage:.1f}%)"
        )
        calls = len(ctx.turn.calls)
        if calls > 1:
            line += f" · {calls} calls"
        if setting == "With Cache":
            prompt = int(total.get("_last_prompt") or 0)
            cached = int(total.get("cached_content_token_count") or 0)
            if prompt > 0:
                line += f" | ⚡ {cached * 100 // prompt}% cached"
        return line
