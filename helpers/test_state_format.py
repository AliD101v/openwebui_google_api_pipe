"""Checks for render state, status events and <details> formatting (SPEC sections 5, 7), criteria 8-11."""
from __future__ import annotations

import asyncio

from _testlib import load_pipe_module


def test_rendered_message_sequence():
    # criterion 8
    mod = load_pipe_module()
    msg = mod.RenderedMessage()
    assert msg.append_text(0, "Hello") == "Hello"
    msg.upsert_block("b1", "<details>x</details>", 0)
    assert msg.append_text(0, "World") == "\nWorld"
    assert msg.render() == "Hello\n<details>x</details>\nWorld"
    checkpoint = msg.checkpoint()
    msg.upsert_block("b0", "B0\n", 0, before_key="text:0:0")
    assert msg.render().startswith("B0\n")
    msg.rollback(checkpoint)
    assert msg.render() == "Hello\n<details>x</details>\nWorld"


def test_status_web_search_done():
    # criterion 9
    mod = load_pipe_module()
    events = []

    async def emit(event):
        events.append(event)

    emitter = mod.StatusEmitter(emit)

    async def run():
        await emitter.web_search_done(["https://a"], ["q"])
        await emitter.web_search_done(["https://a"], ["q"])

    asyncio.run(run())
    assert len(events) == 1
    assert events[0]["type"] == "status"
    data = events[0]["data"]
    assert data["action"] == "web_search"
    assert data["urls"] == ["https://a"]
    assert data["count"] == 1
    assert data["queries"] == ["q"]


def test_format_google_search_block():
    # criterion 10
    mod = load_pipe_module()
    pipe = mod.Pipe()
    out = pipe._format_google_search_block(0, ["q1", "q2"], [(1, "T", "https://a")], "<html>")
    assert out.startswith(
        '<details type="tool_calls" done="true" id="gsearch_0" name="google_search" '
        'data-gemini-builtin="1"'
    )
    assert "embeds=" in out
    assert "- [1] [T](https://a)" in out
    token = mod.HIDDEN_BLOCKS.set(frozenset({"google_search"}))
    try:
        assert pipe._format_google_search_block(0, ["q1"], [(1, "T", "https://a")], "") == ""
    finally:
        mod.HIDDEN_BLOCKS.reset(token)


def test_format_thinking_block():
    # criterion 11
    mod = load_pipe_module()
    pipe = mod.Pipe()
    out = pipe._format_thinking_block("line1\nline2", done=True, duration=3)
    assert 'type="reasoning"' in out
    assert 'done="true"' in out
    assert 'duration="3"' in out
    assert "Thought for 3 seconds" in out
    assert "> line1" in out
