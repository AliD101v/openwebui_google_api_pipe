"""The pipe() entry point: request setup, the generation loop, finalisation,
cancellation and error handling (SPEC section 9)."""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional

from google.genai import types as gtypes

logger = logging.getLogger(__name__)


class _Orchestrator:
    async def pipe(
        self,
        body,
        __user__=None,
        __event_emitter__=None,
        __metadata__=None,
        __tools__=None,
        __files__=None,
        __task__=None,
        __task_body__=None,
        __request__=None,
        __event_call__=None,
    ) -> str:
        user = __user__ or {}
        user_valves = self._user_valves_from(user)

        HIDDEN_BLOCKS.set(self._parse_hidden_blocks(getattr(user_valves, "HIDE_BLOCKS", None)))
        TOOL_APPROVAL.set(
            (
                ((__metadata__ or {}).get("params") or {}).get("tool_approval_mode", "full"),
                __event_call__,
            )
        )
        SLIM_OUTPUT.set(
            bool(
                __request__ is not None
                and getattr(getattr(__request__, "state", None), "internal", False) is True
            )
        )

        api_key = self._resolve_api_key(user_valves)
        if not str(api_key or "").strip():
            logger.error("no Gemini API key configured")
            if __event_emitter__ is not None:
                try:
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {
                                "type": "error",
                                "content": "No Gemini API key configured. Set it in the "
                                "pipe valves or your personal user valves.",
                            },
                        }
                    )
                except Exception as e:
                    logger.warning(f"event emitter failed: {e}")
            return "Error: no Gemini API key configured."

        if __task__:
            return await self._run_task(body, __task__, __user__, user_valves)

        if __metadata__ is not None:
            __metadata__.setdefault("params", {})["reasoning_tags"] = False

        if inspect.isawaitable(__tools__):
            __tools__ = await __tools__

        model = self._model_name_from_body(body)
        info = self.get_model_info(model)
        ctx = new_request_context(
            self,
            event_emitter=__event_emitter__,
            model=model,
            info=info,
            user=user,
            user_valves=user_valves,
            metadata=__metadata__ or {},
            request={},
            api_key=api_key,
            tools=dict(__tools__ or {}),
            builtin_tools={},
            rag_offset=_rag_citation_offset((__metadata__ or {}).get("sources") or []),
        )

        client = None
        try:
            client = self._build_genai_client(api_key, timeout_s=self.valves.REQUEST_TIMEOUT)
            messages = (body or {}).get("messages") or []
            prior = self._collect_prior_marker_entries(messages)
            media = await self._collect_native_media(ctx, client, messages, __files__, prior)
            cleaned = copy.deepcopy(messages)
            self._remove_specific_sources_from_rag_message(cleaned, media["native_names"])
            system_instruction, contents = await self._convert_history(
                cleaned, info, ctx, extra_parts_by_user_idx=media["parts"]
            )
            flags = self._resolve_builtin_flags(
                ctx.metadata, self.valves, user_valves, info, self._last_user_text(body)
            )
            decls: List = []
            tools, tool_config, notes = self._assemble_tools(info, flags, decls)
            for note in notes:
                await ctx.status.activity(note)
            thinking = self._build_thinking_config(info, user_valves, body)
            config = self._build_generate_config(
                info=info,
                system_instruction=system_instruction,
                tools=tools,
                tool_config=tool_config,
                thinking=thinking,
                body=body,
                user_valves=user_valves,
            )
            await ctx.status.waiting()
            await self._run_generation_loop(ctx, contents, config, client)
            await self._persist_media_markers(ctx, media, prior)
            return await self._finalise(ctx, __metadata__)
        except asyncio.CancelledError:
            try:
                await ctx.status.emit(
                    "⏹️ Request Cancelled", done=True, hidden=False, force=True
                )
                partial = ctx.text()
                if partial:
                    await ctx.emit_event(
                        {"type": "replace", "data": {"content": partial}}
                    )
                await self._emit_completion(ctx)
            except Exception as e:
                logger.debug(f"Cancel cleanup emit failed: {e}")
            raise
        except Exception as e:
            await self._handle_errors(e, ctx)
            return ctx.text()
        finally:
            await self._close_client(client)

    def _user_valves_from(self, user: dict) -> "Pipe.UserValves":
        """The user's valves: the instance OpenWebUI hands over, the dict it
        stores, or the defaults when it set nothing."""
        raw = (user or {}).get("valves")
        if isinstance(raw, self.UserValves):
            return raw
        if not raw:
            return self.UserValves()
        fields = {k: v for k, v in dict(raw).items() if k in self.UserValves.model_fields}
        return self.UserValves(**fields)

    def _last_user_text(self, body: dict) -> str:
        for message in reversed((body or {}).get("messages") or []):
            if isinstance(message, dict) and message.get("role") == "user":
                return self._block_text(message.get("content"))
        return ""

    def _collect_prior_marker_entries(self, messages: List[dict]) -> List[str]:
        """The union, in order and deduplicated, of the metadata marker entries
        of every assistant message in the stored conversation.

        Extracted on copies, because the extraction removes the markers from
        the message in place and the caller still needs the originals.
        """
        entries: List[str] = []
        seen = set()
        for message in messages or []:
            for entry in self._extract_metadata_marker_from_message(copy.deepcopy(message)):
                if entry not in seen:
                    seen.add(entry)
                    entries.append(entry)
        return entries

    async def _persist_media_markers(self, ctx, media: dict, prior: List[str]) -> None:
        """Append this turn's media markers, plus every prior marker of
        another kind, as the last block of the message.

        The block lands in the saved message, so the next turn re-anchors the
        files at their original messages instead of the latest one.
        """
        entries = list((media or {}).get("entries") or [])
        seen = set(entries)
        for entry in prior or []:
            if entry not in seen:
                seen.add(entry)
                entries.append(entry)
        markers = []
        for entry in entries:
            parts = str(entry).split(":", 2)
            if len(parts) < 3:
                continue
            try:
                msgnum = int(parts[0])
            except ValueError:
                continue
            markers.append(self._create_metadata_marker(parts[1], unquote(parts[2]), msgnum))
        if not markers:
            return
        call = ctx.turn.calls[-1].index if ctx.turn.calls else 0
        await ctx.upsert_block("markers", "".join(markers), call)

    async def _run_generation_loop(self, ctx, contents, config, client) -> None:
        """The call loop: stream, finalise, replay. Function tools arrive in a
        later milestone; one arriving now renders a notice and stops."""
        max_calls = max(int(getattr(self.valves, "MAX_TOOL_CALLS", 1) or 1), 1) + 1
        for call_index in range(max_calls):
            call = CallState(index=call_index)
            ctx.turn.calls.append(call)
            await self._stream_call_with_retries(ctx, client, contents, config, call)
            await self._finalize_call(ctx, call)
            replay = self._merge_replay_parts(call.raw_parts)
            if replay:
                contents.append(gtypes.Content(role="model", parts=replay))
            if not call.function_calls:
                return
            await ctx.upsert_block(
                f"toolstop:{call.index}",
                self._format_notice(
                    "Tool calls are not supported in this milestone; stopping here"
                ),
                call.index,
            )
            return

    async def _finalise(self, ctx, metadata: Optional[dict]) -> str:
        """The successful end of a turn: final replace, closing status, the
        completion event and the usage persist."""
        text = ctx.text()
        if text:
            await ctx.emit_event({"type": "replace", "data": {"content": text}})
        await ctx.status.complete(self._format_status_line(ctx))
        await self._emit_completion(ctx)
        if CHATS_AVAILABLE and metadata and ctx.turn.usage_total:
            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            if chat_id and message_id and not str(chat_id).startswith("local:"):
                try:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id, {"usage": self._public_usage(ctx.turn)}
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist usage to chat_message: {e}")
        return text

    async def _emit_completion(self, ctx) -> None:
        data: dict = {
            "choices": [{"finish_reason": "stop", "delta": {"content": ""}}],
            "done": True,
        }
        if ctx.turn.usage_total:
            data["usage"] = self._public_usage(ctx.turn)
        await ctx.emit_event({"type": "chat:completion", "data": data})

    @staticmethod
    async def _close_client(client) -> None:
        if client is None:
            return
        try:
            aclose = getattr(getattr(client, "aio", None), "aclose", None)
            if callable(aclose):
                await aclose()
        except BaseException as e:
            logger.debug(f"client close failed: {e}")
