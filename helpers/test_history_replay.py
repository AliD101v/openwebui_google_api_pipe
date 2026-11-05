"""Checks for replaying earlier assistant tool calls from the chat history
(task 05b): persisted <details type="tool_calls"> blocks come back as
function_call / function_response pairs instead of being stripped."""
from __future__ import annotations

import asyncio

from _testlib import FakeEmitter, make_ctx, make_pipe, pipe_module


def test_replay_tool_calls_as_function_pairs():
    # criterion 1
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    assistant_content = (
        "Let me check.\n"
        + pipe._format_tool_result_block("c1", "calc", {"x": 1}, "2")
        + "\nThe answer is 2."
    )
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": "thanks"},
    ]
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
    assert [p.text for p in contents[0].parts] == ["q"]
    model_parts = contents[1].parts
    assert len(model_parts) == 2
    assert model_parts[0].text == "Let me check."
    fc = model_parts[1].function_call
    assert fc is not None
    assert fc.id == "c1"
    assert fc.name == "calc"
    assert fc.args == {"x": 1}
    assert model_parts[1].thought_signature == (
        pipe_module(pipe).DUMMY_THOUGHT_SIGNATURE
    )
    fr = contents[2].parts[0].function_response
    assert fr is not None
    assert fr.id == "c1"
    assert fr.name == "calc"
    assert fr.response == {"result": 2}
    assert [p.text for p in contents[3].parts] == ["The answer is 2."]
    assert [p.text for p in contents[4].parts] == ["thanks"]


def test_builtin_blocks_still_stripped():
    # criterion 2
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    assistant_content = (
        "Paris.[1]\n"
        + '<details type="tool_calls" done="true" id="gsearch_0" '
        'name="google_search" data-gemini-builtin="1">\n<summary>s</summary>\n'
        + "</details>\n"
    )
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": assistant_content},
    ]
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    assert [c.role for c in contents] == ["user", "model"]
    model_parts = contents[1].parts
    assert len(model_parts) == 1
    assert model_parts[0].text == "Paris."
    assert not any(getattr(p, "function_call", None) for p in model_parts)


def test_consecutive_blocks_form_one_group():
    # criterion 3
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    assistant_content = (
        pipe._format_tool_result_block("c1", "calc", {"x": 1}, "2")
        + "\n"
        + pipe._format_tool_result_block("c2", "clock", {"tz": "UTC"}, "12:00")
    )
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": assistant_content},
    ]
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    assert [c.role for c in contents] == ["user", "model", "user"]
    model_parts = contents[1].parts
    fcs = [p.function_call for p in model_parts if getattr(p, "function_call", None)]
    assert len(fcs) == 2
    assert [f.id for f in fcs] == ["c1", "c2"]
    assert fcs[0].name == "calc"
    assert fcs[1].name == "clock"
    assert model_parts[0].thought_signature == (
        pipe_module(pipe).DUMMY_THOUGHT_SIGNATURE
    )
    assert model_parts[1].thought_signature is None
    frs = [
        p.function_response for p in contents[2].parts
        if getattr(p, "function_response", None)
    ]
    assert [f.id for f in frs] == ["c1", "c2"]
    assert frs[0].response == {"result": 2}
    assert frs[1].response == {"result": "12:00"}
