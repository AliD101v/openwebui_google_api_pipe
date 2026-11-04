"""Non-streaming task requests: titles, tags, follow-ups and the other
one-shot answers OpenWebUI asks the model for outside the chat turn.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class _Tasks:
    # OpenWebUI's background memory review arrives as a task whose name is not
    # in the TASKS enum -- functions.py forwards metadata.task verbatim.
    MEMORY_REVIEW_TASK = "memory_review"

    # Response schemas for the task requests whose prompt asks for raw JSON.
    #
    # OpenWebUI parses these answers with json.loads and drops the whole task
    # when it fails, so every one of its prompts spends several lines begging
    # for "no markdown fences, no preamble". Structured outputs make that a
    # guarantee instead of a plea: the model cannot emit anything but a
    # conforming object.
    #
    # Keys are the task names OpenWebUI passes in metadata.task. Deliberately
    # absent: emoji_generation and moa_response_generation (their prompts ask
    # for prose, not JSON).
    #
    # `additionalProperties: false` everywhere -- OpenWebUI reads exactly one
    # key out of each of these and extra keys are pure token cost.
    TASK_RESPONSE_SCHEMAS: Dict[str, dict] = {
        "title_generation": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
        "tags_generation": {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
        "follow_up_generation": {
            "type": "object",
            "properties": {
                "follow_ups": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["follow_ups"],
            "additionalProperties": False,
        },
        "query_generation": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
        "image_prompt_generation": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "autocomplete_generation": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        MEMORY_REVIEW_TASK: {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "replace", "move", "remove"],
                            },
                            "id": {"type": "string"},
                            "type": {"type": "string", "enum": ["user", "context"]},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        # One flat object rather than a per-action union: the
                        # required field set differs per action, and structured
                        # outputs does not accept anyOf/oneOf. OpenWebUI
                        # validates the per-action requirements itself
                        # (validate_memory_operations), so pinning the action
                        # vocabulary and the field names is the useful part --
                        # it is what stops the model from inventing the
                        # "score"/"importance"/"stability" keys the prompt
                        # explicitly warns against.
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["operations"],
            "additionalProperties": False,
        },
    }

    _PATTERN_ANY_DETAILS = re.compile(
        r"<details\b[^>]*>.*?</details>", flags=re.DOTALL
    )
    _PATTERN_METADATA_MARKER = re.compile(PATTERN_METADATA_MARKER)
    _PATTERN_TRAILING_SPACES = re.compile(r"[ \t]+$", flags=re.MULTILINE)
    _PATTERN_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

    def _resolve_task_model(self, body: dict) -> str:
        """The model for a task request: the TASK_MODEL valve when set, else
        the chat model of the body."""
        chat_model = self._model_name_from_body(body)
        override = str(self.valves.TASK_MODEL or "").strip()
        if not override:
            return chat_model
        if override.startswith(MODEL_ID_PREFIX):
            override = override[len(MODEL_ID_PREFIX) :]
        return override

    @classmethod
    def _sanitize_task_text(cls, text: str) -> str:
        """Reduce persisted chat content to the prose a task model actually
        needs.

        OpenWebUI hands task requests the stored message content, which carries
        every collapsible this pipe ever wrote plus the invisible metadata
        markers used for replay. None of it round-trips: task requests are
        one-shot. Stripping it cuts the bill and stops the task model from
        reasoning about its own UI artefacts. It matters most for the memory
        review, which truncates each message and would otherwise spend that
        budget on UI artefacts.
        """
        if not text or ("<" not in text and "[" not in text):
            return text
        cleaned = cls._PATTERN_ANY_DETAILS.sub("\n", text)
        cleaned = cls._PATTERN_METADATA_MARKER.sub(" ", cleaned)
        cleaned = cls._PATTERN_TRAILING_SPACES.sub("", cleaned)
        return cls._PATTERN_EXCESS_BLANK_LINES.sub("\n\n", cleaned).strip()

    @classmethod
    def _extract_task_system(cls, messages: List[dict]) -> str:
        """The system messages of a task request, joined into one string.

        Returned as plain text: task requests are one-shot and never cached,
        so there is nothing fancier to attach them to.
        """
        parts: List[str] = []
        for msg in messages or []:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                continue
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    def _process_messages_for_task(
        self, messages: List[dict]
    ) -> List[gtypes.Content]:
        """The conversation as text-only Gemini Contents for a task request.

        UI artefacts are stripped per message (see _sanitize_task_text); a
        message that sanitises to nothing is dropped, and the same-role
        neighbours a drop leaves behind are merged instead of being sent as
        an invalid sequence.
        """
        processed: List[gtypes.Content] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                continue
            text = self._sanitize_task_text(text)
            if not text:
                continue
            if processed and processed[-1].role == gemini_role:
                previous = processed[-1].parts[0].text
                processed[-1].parts[0] = gtypes.Part(text=f"{previous}\n\n{text}")
            else:
                processed.append(
                    gtypes.Content(role=gemini_role, parts=[gtypes.Part(text=text)])
                )
        return processed

    def _task_config_kwargs(
        self, info: dict, system_instruction: Optional[str], thinking, task: Optional[str]
    ) -> Tuple[dict, bool]:
        """The GenerateContentConfig kwargs for one task call, and whether the
        structured-output fields are in them: no tools, AFC disabled, the
        task's thinking config, and the task's response schema when it has
        one."""
        schema = (
            self.TASK_RESPONSE_SCHEMAS.get(task) if isinstance(task, str) else None
        )
        kwargs: dict = {
            "automatic_function_calling": gtypes.AutomaticFunctionCallingConfig(
                disable=True
            )
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if thinking is not None:
            kwargs["thinking_config"] = thinking
        if schema:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = schema
        return kwargs, bool(schema)

    async def _task_generate(
        self,
        client,
        model: str,
        chat_model: str,
        contents: List[gtypes.Content],
        system_instruction: str,
        user_valves,
        body: dict,
        task: Optional[str],
    ):
        """The task's generate_content call, with its two self-healing
        retries: a 404 from the TASK_MODEL valve falls back to the chat
        model, and any other failure drops the structured output and retries
        once plain -- losing the schema costs a markdown fence, losing the
        request costs the whole task."""
        info = self.get_model_info(model)
        kwargs, structured = self._task_config_kwargs(
            info,
            system_instruction,
            self._build_thinking_config(info, user_valves, body, purpose="task"),
            task,
        )
        try:
            return await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=gtypes.GenerateContentConfig(**kwargs),
            )
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 404 and model != chat_model:
                logger.info(
                    f"Task model {model} not found; retrying with the chat model {chat_model}"
                )
                info = self.get_model_info(chat_model)
                kwargs, _structured = self._task_config_kwargs(
                    info,
                    system_instruction,
                    self._build_thinking_config(
                        info, user_valves, body, purpose="task"
                    ),
                    task,
                )
                return await client.aio.models.generate_content(
                    model=chat_model,
                    contents=contents,
                    config=gtypes.GenerateContentConfig(**kwargs),
                )
            if structured:
                logger.warning(
                    f"Task {task} rejected with structured outputs, retrying "
                    f"without: {getattr(e, 'message', e)}"
                )
                kwargs.pop("response_mime_type", None)
                kwargs.pop("response_json_schema", None)
                return await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gtypes.GenerateContentConfig(**kwargs),
                )
            raise

    async def _run_task(self, body, task, __user__, user_valves) -> str:
        """One non-streaming task request (title, tags, follow-ups, ...).

        Returns the answer's non-thought text, stripped, or "" when the task
        failed: OpenWebUI drops failed tasks silently, so a total task outage
        must still leave a warning behind.
        """
        client = None
        try:
            body = body or {}
            user_valves = self.UserValves() if user_valves is None else user_valves
            api_key = self._resolve_api_key(user_valves)
            client = self._build_genai_client(
                api_key, timeout_s=self.valves.REQUEST_TIMEOUT
            )
            messages = body.get("messages") or []
            contents = self._process_messages_for_task(messages)
            system_instruction = self._extract_task_system(messages)
            if not contents:
                return ""
            response = await self._task_generate(
                client,
                self._resolve_task_model(body),
                self._model_name_from_body(body),
                contents,
                system_instruction,
                user_valves,
                body,
                task,
            )
        except Exception as e:
            # Warning, not debug: returning "" makes OpenWebUI drop the task
            # silently, so a total task outage must be visible at default log
            # settings.
            logger.warning(f"Task model error ({task}): {e}")
            return ""
        finally:
            if client is not None:
                try:
                    await client.aio.aclose()
                except Exception:
                    pass

        text_parts: List[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text and not getattr(part, "thought", False):
                    text_parts.append(text)
        result = "".join(text_parts).strip()
        logger.debug(f"Task response: {len(result)}c")
        return result
