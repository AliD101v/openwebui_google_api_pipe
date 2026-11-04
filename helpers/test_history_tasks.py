"""Checks for history conversion, memory extraction and task requests
(SPEC sections 9.6-9.7)."""
from __future__ import annotations

import asyncio
import base64
import sys

from google.genai import errors as genai_errors

from _testlib import FakeClient, FakeEmitter, chunk, make_ctx, make_pipe, pipe_module

# A valid 1x1 transparent PNG.
PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_convert_history_roles_parts_and_image():
    # criterion 1
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": (
                '<details type="reasoning" done="true">\n<summary>Thought</summary>\n'
                "> x\n</details>\nHello! [1]\n"
                '<details type="tool_calls" done="true" id="gsearch_0" '
                'name="google_search" data-gemini-builtin="1">\n<summary>s</summary>\n'
                "</details>\n"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{PNG_1X1_B64}"},
                },
            ],
        },
    ]
    system_instruction, contents = asyncio.run(
        pipe._convert_history(messages, info, ctx)
    )
    assert system_instruction == "Be brief."
    assert [c.role for c in contents] == ["user", "model", "user"]
    model_parts = contents[1].parts
    assert len(model_parts) == 1
    assert model_parts[0].text == "Hello!"
    last_parts = contents[2].parts
    assert [p.text for p in last_parts if getattr(p, "text", None)] == [
        "What's this?"
    ]
    inline = [p for p in last_parts if getattr(p, "inline_data", None) is not None]
    assert len(inline) == 1
    assert inline[0].inline_data.mime_type == "image/png"
    assert base64.b64decode(PNG_1X1_B64) == inline[0].inline_data.data


def test_memories_relocated_to_last_user_message():
    # criterion 2
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    messages = [
        {
            "role": "system",
            "content": "Be brief.\n<memory_context>likes tea</memory_context>",
        },
        {"role": "user", "content": "Hi"},
    ]
    system_instruction, contents = asyncio.run(
        pipe._convert_history(messages, info, ctx)
    )
    assert system_instruction is not None
    assert "likes tea" not in system_instruction
    texts = [p.text for p in contents[-1].parts if getattr(p, "text", None)]
    assert texts, "the last user content must keep a text part"
    text = "".join(texts)
    header = pipe_module(pipe).MEMORY_CONTEXT_APPENDIX_HEADER
    assert header in text
    assert text.index("likes tea") > text.index(header)


def _task_body():
    return {
        "model": "gemini_pipe.gemini/gemini-4-argon",
        "messages": [
            {"role": "system", "content": "Answer with JSON: {\"title\": ...}"},
            {"role": "user", "content": "hello, plan my weekend"},
        ],
    }


def _run(pipe, user, user_valves, fake):
    pipe._build_genai_client = lambda api_key, timeout_s=None: fake
    return asyncio.run(
        pipe._run_task(_task_body(), "title_generation", user, user_valves)
    )


def test_run_task_chat_model_json_schema_and_plain_fallback():
    # criterion 3
    pipe = make_pipe(TASK_MODEL="")
    user = {"id": "u1", "name": "U", "email": "u@example.com", "role": "user", "valves": {}}
    user_valves = type(pipe).UserValves()

    ok = chunk(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "  Weekend plans  "}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 5,
            },
        }
    )
    fake = FakeClient(responses=[ok])
    result = _run(pipe, user, user_valves, fake)
    assert result == "Weekend plans"
    assert len(fake.generate_calls) == 1
    call = fake.generate_calls[0]
    assert call["model"] == "gemini-4-argon"
    assert call["config"].response_json_schema is not None

    error = genai_errors.APIError(
        400,
        {
            "error": {
                "code": 400,
                "message": "Invalid argument",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    retry_ok = chunk(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Weekend plans"}],
                    }
                }
            ]
        }
    )
    fake2 = FakeClient(responses=[error, retry_ok])
    result = _run(pipe, user, user_valves, fake2)
    assert result == "Weekend plans"
    assert len(fake2.generate_calls) == 2
    assert fake2.generate_calls[0]["config"].response_json_schema is not None
    assert fake2.generate_calls[1]["config"].response_json_schema is None
