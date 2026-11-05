"""Execution of the model's function calls: dispatch to user, tool-server and
built-in tools, the approval gate, timeouts, Rich-UI result processing, and
the user content that replays the function responses in the model's call
order.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class _ToolExec:
    _TOOL_RESULT_CHAR_CAP = 100_000
    _OWUI_CALL_TOOL_NAME = "call_tool"
    _OWUI_SEARCH_TOOL_NAME = "search_tools"

    @staticmethod
    def _filter_tool_args(tool_entry: Any, args: dict) -> dict:
        """Drop arguments the tool's own schema does not declare.

        Mirrors what OpenWebUI does before invoking a tool in its native
        function-calling loop: without it a single undeclared key makes the
        callable raise ``TypeError: got an unexpected keyword argument`` and
        the tool fails outright. A tool with no declared properties still
        gets an empty dict rather than being left alone — that is exactly
        the case stray keys show up in.
        """
        if not isinstance(args, dict) or not args:
            return args if isinstance(args, dict) else {}
        spec = (tool_entry or {}).get("spec") if isinstance(tool_entry, dict) else None
        if not isinstance(spec, dict):
            return args
        params = spec.get("parameters")
        if not isinstance(params, dict) or not isinstance(params.get("properties"), dict):
            # No usable schema — passing the args through unchanged is the
            # safer guess than dropping everything.
            return args
        allowed = params["properties"].keys()
        kept = {k: v for k, v in args.items() if k in allowed}
        dropped = [k for k in args if k not in allowed]
        if dropped:
            logger.warning(
                "Tool '%s': dropped undeclared argument(s) %s not in its schema",
                spec.get("name", "?"),
                dropped,
            )
        return kept

    @classmethod
    def _resolve_owui_call_tool(
        cls, tool_input: Any, tools: dict
    ) -> Tuple[str, dict, Optional[str]]:
        """Unwrap OpenWebUI's ``call_tool`` dispatcher into the tool it names.

        Same checks and messages as OpenWebUI's own tool search resolution,
        which only runs in its own tool loop, not in ours. ``params`` may
        also come as an object: the pipe does not send the dispatcher as
        strict. Returns ``(tool_name, args, error)``.
        """
        try:
            if not isinstance(tool_input, dict) or set(tool_input) != {"tool_name", "params"}:
                raise ValueError("call_tool requires tool_name and params.")
            name = tool_input["tool_name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool_name must be a non-empty string.")
            if name in {cls._OWUI_SEARCH_TOOL_NAME, cls._OWUI_CALL_TOOL_NAME}:
                raise ValueError("call_tool cannot invoke search_tools or itself.")
            if name not in tools:
                raise ValueError(f'Tool "{name}" is not available. Search for an available tool.')
            params = tool_input["params"]
            if isinstance(params, str):
                params = json.loads(params) if params.strip() else {}
            if not isinstance(params, dict):
                raise ValueError("params must decode to a JSON object.")
        except (TypeError, ValueError) as error:
            return cls._OWUI_CALL_TOOL_NAME, {}, f"Error: {error}"
        return name, params, None

    async def _start_function_call(self, ctx, call, fc: dict) -> None:
        """Render the in-progress block for one function call and start its
        execution as a background task (or record the refusal to run it)."""
        original_name = ctx.turn.name_map.get(fc["name"], fc["name"])
        await ctx.status.running_tool(original_name)
        await ctx.upsert_block(
            fc["key"],
            self._format_tool_result_block(
                fc["id"], original_name, fc["args"], "", done=False
            ),
            call.index,
        )
        if not ctx.turn.functions_offered:
            fc["error"] = "function calling is not available"
            return
        max_calls = max(int(getattr(self.valves, "MAX_TOOL_CALLS", 1) or 1), 1)
        if ctx.turn.tool_calls_made >= max_calls:
            fc["error"] = "tool call limit reached (MAX_TOOL_CALLS)"
            return
        ctx.turn.tool_calls_made += 1
        fc["task"] = asyncio.create_task(self._execute_tool_call(ctx, fc))

    async def _execute_tool_call(self, ctx, fc: dict) -> dict:
        """Run one function call end to end. Never raises: every denial,
        timeout or exception becomes ``{"error": True, "text": message}``."""
        name = ctx.turn.name_map.get(fc["name"], fc["name"])
        args = fc["args"] if isinstance(fc["args"], dict) else {}
        files: List[Any] = []
        embeds: List[Any] = []
        error_text: Optional[str] = None
        text = ""
        try:
            tools = ctx.tools or {}
            builtin = ctx.builtin_tools or {}
            target = None
            if (tools.get(name) or {}).get("tool_id") == f"builtin:{self._OWUI_CALL_TOOL_NAME}":
                name, args, dispatch_error = self._resolve_owui_call_tool(args, tools)
                if dispatch_error:
                    error_text = dispatch_error
                else:
                    target = tools.get(name)
            if error_text is None and (target is None or not (target or {}).get("callable")):
                target = tools.get(name)
                if target is not None and not target.get("callable"):
                    target = None
            if error_text is None and target is None:
                entry = builtin.get(name)
                if isinstance(entry, dict) and entry.get("callable"):
                    target = entry
            if error_text is None and target is None:
                logger.warning(
                    "Tool '%s' not found in __tools__ or builtin_tools", name
                )
                error_text = json.dumps(
                    {
                        "error": (
                            f"Tool '{name}' is not available. It may require "
                            "server context or is not configured."
                        )
                    },
                    ensure_ascii=False,
                )
            if error_text is None:
                args = self._filter_tool_args(target, args)
                approved, denial = await self._await_tool_approval(
                    {"id": fc.get("raw_id"), "name": name, "args": args}
                )
                if not approved:
                    error_text = denial
                else:
                    timeout_s = getattr(self.valves, "TOOL_CALL_TIMEOUT", 60)
                    outcome = target["callable"](**args)
                    if inspect.isawaitable(outcome):
                        try:
                            outcome = await asyncio.wait_for(
                                outcome, timeout=max(1, float(timeout_s))
                            )
                        except asyncio.TimeoutError:
                            error_text = (
                                f"Error executing tool '{name}': "
                                f"timed out after {timeout_s}s"
                            )
                    if error_text is None and PROCESS_TOOL_RESULT_AVAILABLE and ctx.request is not None:
                        try:
                            outcome, files, embeds = await process_tool_result(
                                ctx.request,
                                name,
                                outcome,
                                "pipe",
                                metadata=ctx.metadata,
                                user=ctx.user,
                            )
                        except Exception as e:
                            logger.warning(
                                f"process_tool_result failed for '{name}': {e}"
                            )
                    if error_text is None:
                        if isinstance(outcome, str):
                            text = outcome
                        elif isinstance(outcome, (dict, list)):
                            try:
                                text = json.dumps(outcome, ensure_ascii=False)
                            except (TypeError, ValueError):
                                text = str(outcome)
                        else:
                            text = str(outcome)
        except Exception as e:
            logger.error(f"Tool call '{name}' failed: {e}")
            error_text = f"Error executing tool '{name}': {e}"
            files, embeds = [], []
        if error_text is not None:
            text = error_text
        is_error = error_text is not None or text.startswith(
            ("Error:", "Error executing tool")
        )
        return {"text": text, "files": files, "embeds": embeds, "error": is_error}

    async def _await_tool_approval(self, tool_call: dict) -> Tuple[bool, Optional[str]]:
        """Ask the user to allow this tool call when approval mode is 'ask'.

        OpenWebUI enforces its human-in-the-loop approval only around its own
        tool loop; a pipe that runs its own loop reproduces the gate here.
        ``mode == "ask"`` without an approval channel denies, an exception
        from the channel denies, and only a literal ``True`` approves — a
        timeout or closed tab comes back as a truthy dict. Returns
        ``(approved, denial_text)``; the denial text is fed back to the model
        as this call's result so the loop continues instead of stalling.
        """
        mode, event_call = TOOL_APPROVAL.get()
        name = str(tool_call.get("name") or "tool")
        if mode != "ask":
            return True, None
        denial = json.dumps(
            {"error": f"The user denied permission to run '{name}'."},
            ensure_ascii=False,
        )
        if event_call is None:
            return False, denial
        try:
            args = json.dumps(tool_call.get("args") or {}, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            args = str(tool_call.get("args"))
        if len(args) > 2000:
            args = args[:2000] + "\n… (truncated)"
        try:
            answer = await event_call(
                {
                    "type": "confirmation",
                    "data": {
                        "title": f"Run tool: {name}?",
                        "message": (
                            f"The model wants to call `{name}` with:\n\n"
                            f"```json\n{args}\n```"
                        ),
                        # OpenWebUI >= 0.12 shows its native Tool Permissions
                        # dialog (incl. "always allow") when tool_call is set.
                        "tool_call": {
                            "id": tool_call.get("id"),
                            "name": name,
                            "arguments": tool_call.get("args") or {},
                        },
                    },
                }
            )
        except Exception as e:
            # A broken approval channel must not silently turn into free
            # execution — that is the exact failure the gate exists to prevent.
            logger.warning("Tool approval prompt failed for '%s': %s", name, e)
            answer = False
        if answer is True:
            logger.info("Tool '%s' approved by user", name)
            return True, None
        logger.info("Tool '%s' denied by user", name)
        return False, denial

    async def _finish_function_calls(self, ctx, call) -> gtypes.Content:
        """Re-render every tool block done as its task completes, emit the
        files the tools produced, and build the user content with one
        FunctionResponse per call in the order the model made them."""

        async def _tagged(fc: dict, task) -> Tuple[dict, dict]:
            try:
                return fc, await task
            except asyncio.CancelledError:
                return fc, {"text": "tool call cancelled", "error": True}

        for fc in call.function_calls:
            if fc.get("error"):
                original_name = ctx.turn.name_map.get(fc["name"], fc["name"])
                await ctx.upsert_block(
                    fc["key"],
                    self._format_tool_result_block(
                        fc["id"],
                        original_name,
                        fc["args"],
                        fc["error"],
                        is_error=True,
                        done=True,
                    ),
                    call.index,
                )
                fc["outcome"] = {"text": fc["error"], "error": True}

        tasks = [(fc, fc["task"]) for fc in call.function_calls if fc.get("task") is not None]
        if tasks:
            for finished in asyncio.as_completed(
                [_tagged(fc, task) for fc, task in tasks]
            ):
                fc, outcome = await finished
                outcome = outcome or {"text": "", "error": False}
                text = str(outcome.get("text") or "")
                is_error = bool(outcome.get("error"))
                files = outcome.get("files") or []
                embeds = outcome.get("embeds") or []
                if files:
                    await ctx.emit_event(
                        {"type": "files", "data": {"files": files}}
                    )
                original_name = ctx.turn.name_map.get(fc["name"], fc["name"])
                await ctx.upsert_block(
                    fc["key"],
                    self._format_tool_result_block(
                        fc["id"],
                        original_name,
                        fc["args"],
                        text,
                        is_error=is_error,
                        done=True,
                        embeds=embeds,
                        files=files,
                    ),
                    call.index,
                )
                fc["outcome"] = {"text": text, "error": is_error}

        parts: List[gtypes.Part] = []
        for fc in call.function_calls:
            outcome = fc.get("outcome") or {"text": "", "error": False}
            text = str(outcome.get("text") or "")
            if len(text) > self._TOOL_RESULT_CHAR_CAP:
                text = text[: self._TOOL_RESULT_CHAR_CAP] + "\n… (truncated)"
            if outcome.get("error"):
                response: Dict[str, Any] = {"error": text}
            else:
                response = self._tool_response_payload(text)
            parts.append(
                gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        id=fc.get("raw_id"),
                        name=fc["name"],
                        response=response,
                    )
                )
            )
        return gtypes.Content(role="user", parts=parts)

    @staticmethod
    def _tool_response_payload(text: str) -> dict:
        """{"result": …} for a tool's text: parsed JSON when it is JSON."""
        try:
            return {"result": json.loads(text)}
        except (TypeError, ValueError):
            return {"result": text}
