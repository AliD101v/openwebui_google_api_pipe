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
            request=__request__,
            api_key=api_key,
            tools=dict(__tools__ or {}),
            builtin_tools=await self._gather_builtin_tools(
                __request__, user, __event_emitter__, __metadata__, __tools__
            ),
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
            decls = self._build_function_declarations(ctx)
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
        """The call loop: stream, finalise, replay. Function calls are executed
        by the pipe and their responses replayed until the model answers; a
        rejected call (malformed function call, missing thought signature)
        rolls the call back and retries it once."""
        max_tool_calls = max(int(getattr(self.valves, "MAX_TOOL_CALLS", 1) or 1), 1)
        retried: set = set()
        for _iteration in range(max_tool_calls + 3):
            call = CallState(index=len(ctx.turn.calls))
            ctx.turn.calls.append(call)
            checkpoint = ctx.msg.checkpoint()
            config = await self._stream_call_with_retries(ctx, client, contents, config, call)
            await self._finalize_call(ctx, call)
            finish_name = (
                getattr(call.finish_reason, "name", str(call.finish_reason))
                if call.finish_reason is not None
                else ""
            )
            if (
                finish_name
                in ("MALFORMED_FUNCTION_CALL", "MISSING_THOUGHT_SIGNATURE")
                and finish_name not in retried
            ):
                # Roll the call back (its rendered output, any started tool
                # tasks) and retry it once.
                self._reset_call(call, ctx)
                ctx.msg.rollback(checkpoint)
                await ctx.emit_replace()
                if finish_name == "MISSING_THOUGHT_SIGNATURE":
                    self._restore_missing_signatures(contents)
                retried.add(finish_name)
                continue
            if not call.function_calls:
                return
            replay = self._merge_replay_parts(call.raw_parts)
            if replay:
                contents.append(gtypes.Content(role="model", parts=replay))
            contents.append(await self._finish_function_calls(ctx, call))
            if ctx.turn.tool_calls_made >= max_tool_calls:
                config = self._disable_function_calling(config)

    async def _gather_builtin_tools(
        self, __request__, user, __event_emitter__, __metadata__, __tools__
    ) -> dict:
        """The executable tools beyond ``__tools__``: OpenWebUI's built-in
        tools (memory, …) plus the external tool-server tools the metadata
        hands over with their callables."""
        builtin_tools: dict = {}
        if BUILTIN_TOOLS_AVAILABLE and __request__ is not None:
            try:
                memory_enabled = (
                    user.get("settings", {}).get("ui", {}).get("memory", False)
                    if user
                    else False
                )
                builtin_tools = get_builtin_tools(
                    __request__,
                    {
                        "__user__": user,
                        "__event_emitter__": __event_emitter__,
                        "__chat_id__": (__metadata__ or {}).get("chat_id"),
                        "__message_id__": (__metadata__ or {}).get("message_id"),
                    },
                    features={"memory": memory_enabled},
                    model={"info": {"meta": {"builtinTools": {"skills": False}}}},
                )
                if inspect.isawaitable(builtin_tools):
                    builtin_tools = await builtin_tools
                logger.debug(
                    f"Loaded {len(builtin_tools)} builtin tools: "
                    f"{list(builtin_tools.keys())}"
                )
            except Exception as e:
                logger.warning(f"Could not load builtin tools: {e}")
                builtin_tools = {}
        metadata_tools = (__metadata__ or {}).get("tools", {})
        if metadata_tools:
            for t_name, t_data in metadata_tools.items():
                if t_name in builtin_tools:
                    continue
                if __tools__ and t_name in __tools__:
                    continue
                if isinstance(t_data, dict) and t_data.get("callable"):
                    builtin_tools[t_name] = t_data
        if builtin_tools:
            logger.debug(f"Executable tool set: {list(builtin_tools.keys())}")
        return builtin_tools

    def _restore_missing_signatures(self, contents: List[gtypes.Content]) -> None:
        """After a MISSING_THOUGHT_SIGNATURE rejection: give the first
        unsigned function_call part of every signed-context model content the
        dummy signature so the retry passes Gemini 3+'s validator."""
        last_user = -1
        for index, content in enumerate(contents):
            if content.role == "user":
                last_user = index
        for content in contents[last_user + 1 :]:
            if content.role != "model":
                continue
            if not any(getattr(p, "text", None) for p in content.parts):
                continue
            for index, part in enumerate(content.parts):
                if (
                    getattr(part, "function_call", None) is not None
                    and getattr(part, "thought_signature", None) is None
                ):
                    content.parts[index] = part.model_copy(
                        update={"thought_signature": DUMMY_THOUGHT_SIGNATURE}
                    )
                    break

    @staticmethod
    def _disable_function_calling(config) -> gtypes.GenerateContentConfig:
        """A copy of ``config`` with function calling switched off (the tool
        call limit is reached); the tools themselves stay declared."""
        config = config.model_copy(deep=True)
        tool_config = config.tool_config
        if tool_config is None:
            tool_config = gtypes.ToolConfig(
                function_calling_config=gtypes.FunctionCallingConfig(mode="NONE")
            )
        else:
            tool_config = tool_config.model_copy(
                update={
                    "function_calling_config": gtypes.FunctionCallingConfig(
                        mode="NONE"
                    )
                }
            )
        config.tool_config = tool_config
        return config

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
                    cost = self._turn_cost(ctx) if getattr(
                        ctx.user_valves, "SHOW_COST", True
                    ) else None
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id,
                        {"usage": self._public_usage(ctx.turn, cost)},
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
            cost = self._turn_cost(ctx) if getattr(
                ctx.user_valves, "SHOW_COST", True
            ) else None
            data["usage"] = self._public_usage(ctx.turn, cost)
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
