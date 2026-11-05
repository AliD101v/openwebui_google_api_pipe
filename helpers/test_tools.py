"""Checks for the OpenWebUI tools loop: declarations, execution, approval and
signature-correct continuation (task 05a)."""
from __future__ import annotations

import asyncio
import base64

from _testlib import FakeClient, FakeEmitter, chunk, make_pipe, pipe_module

BODY = {
    "model": "gemini_pipe.gemini/gemini-4-argon",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}


def make_user(pipe, **user_valves):
    return {"id": "u1", "name": "T", "valves": pipe.UserValves(**user_valves)}


def tool_entry(name, fn, params=None):
    return {
        "spec": {
            "name": name,
            "description": name,
            "parameters": params or {"type": "object", "properties": {}},
        },
        "callable": fn,
    }


def fc_part(fc_id, name, args, sig=None):
    part = {"functionCall": {"id": fc_id, "name": name, "args": args}}
    if sig is not None:
        part["thoughtSignature"] = base64.b64encode(sig).decode()
    return part


def fc_script(fc_parts, finish="STOP"):
    """One streamed chunk carrying the given functionCall parts."""
    return [
        chunk(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": fc_parts},
                        "finishReason": finish,
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        )
    ]


def answer_script(text="Done."):
    return [
        chunk(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        )
    ]


def run_pipe(pipe, fake, *, tools=None, metadata=None, request=None, event_call=None,
             user=None, messages=None):
    """Run pipe() against the fake client; (result, emitter)."""
    pipe._build_genai_client = lambda api_key, timeout_s=None: fake
    emitter = FakeEmitter()
    body = dict(BODY, messages=list(messages or BODY["messages"]))
    metadata = dict(metadata or {"chat_id": "c", "message_id": "m"})
    result = asyncio.run(
        pipe.pipe(
            body,
            __user__=user or make_user(pipe),
            __event_emitter__=emitter,
            __metadata__=metadata,
            __tools__=tools,
            __request__=request,
            __event_call__=event_call,
        )
    )
    return result, emitter


def test_declarations_sanitized_and_mapped():
    # criterion 1
    pipe = make_pipe(GEMINI_API_KEY="k")
    ctx = pipe_module(pipe).new_request_context(
        pipe,
        tools={
            "get weather!": tool_entry("get weather!", lambda **kw: "sunny"),
            "_private": tool_entry("_private", lambda **kw: "secret"),
        },
    )
    decls = pipe._build_function_declarations(ctx)
    assert [d.name for d in decls] == ["get_weather_"]
    assert ctx.turn.name_map["get_weather_"] == "get weather!"
    assert ctx.turn.functions_offered is True


def test_single_tool_call_roundtrip():
    # criterion 2
    pipe = make_pipe(GEMINI_API_KEY="k")
    calls = []

    def weather(**kwargs):
        calls.append(kwargs)
        return "sunny"

    tools = {
        "get_weather": tool_entry(
            "get_weather", weather,
            {"type": "object", "properties": {"city": {"type": "string"}}},
        )
    }
    stream1 = fc_script([fc_part("fc1", "get_weather", {"city": "Berlin"}, b"SIG-ONE")])
    fake = FakeClient(scripts=[stream1, answer_script("It is sunny.")])
    result, _emitter = run_pipe(pipe, fake, tools=tools)

    assert calls == [{"city": "Berlin"}]
    contents = fake.stream_calls[1]["contents"]
    model_content = contents[-2]
    assert model_content.role == "model"
    fc_parts = [p for p in model_content.parts if p.function_call is not None]
    assert len(fc_parts) == 1
    assert fc_parts[0].function_call.name == "get_weather"
    assert fc_parts[0].thought_signature == b"SIG-ONE"
    user_content = contents[-1]
    assert user_content.role == "user"
    fr = user_content.parts[0].function_response
    assert fr.id == "fc1"
    assert fr.response == {"result": "sunny"}
    assert 'done="true"' in result
    assert "get_weather" in result
    assert result.index("get_weather") < result.index("It is sunny.")


def test_parallel_tool_calls_preserve_order():
    # criterion 3
    pipe = make_pipe(GEMINI_API_KEY="k")

    async def weather(**kwargs):
        await asyncio.sleep(0.05)
        return "sunny"

    def get_time(**kwargs):
        return "noon"

    tools = {
        "get_weather": tool_entry(
            "get_weather", weather,
            {"type": "object", "properties": {"city": {"type": "string"}}},
        ),
        "get_time": tool_entry("get_time", get_time),
    }
    stream1 = fc_script(
        [
            fc_part("fc1", "get_weather", {"city": "Berlin"}, b"SIG-TWO"),
            fc_part("fc2", "get_time", {}),
        ]
    )
    fake = FakeClient(scripts=[stream1, answer_script("ok")])
    run_pipe(pipe, fake, tools=tools)

    contents = fake.stream_calls[1]["contents"]
    model_content = contents[-2]
    assert model_content.role == "model"
    fc_ids = [p.function_call.id for p in model_content.parts if p.function_call is not None]
    assert fc_ids == ["fc1", "fc2"]
    assert model_content.parts[0].thought_signature == b"SIG-TWO"
    assert model_content.parts[1].thought_signature is None
    user_content = contents[-1]
    assert user_content.role == "user"
    frs = [p.function_response for p in user_content.parts if p.function_response is not None]
    assert [fr.id for fr in frs] == ["fc1", "fc2"]
    assert [fr.name for fr in frs] == ["get_weather", "get_time"]


def test_tool_approval_denied():
    # criterion 4
    pipe = make_pipe(GEMINI_API_KEY="k")
    called = []

    def weather(**kwargs):
        called.append(kwargs)
        return "sunny"

    tools = {
        "get_weather": tool_entry(
            "get_weather", weather,
            {"type": "object", "properties": {"city": {"type": "string"}}},
        )
    }
    metadata = {
        "chat_id": "c",
        "message_id": "m",
        "params": {"tool_approval_mode": "ask"},
    }

    async def event_call(event):
        assert event["type"] == "confirmation"
        return False

    fake = FakeClient(
        scripts=[
            fc_script([fc_part("fc1", "get_weather", {"city": "Berlin"}, b"SIG-ONE")]),
            answer_script("Still fine."),
        ]
    )
    result, _emitter = run_pipe(pipe, fake, tools=tools, metadata=metadata, event_call=event_call)
    assert called == []
    fr = fake.stream_calls[1]["contents"][-1].parts[0].function_response
    assert "error" in fr.response
    assert "denied" in str(fr.response["error"])
    assert "Still fine." in result

    fake2 = FakeClient(
        scripts=[
            fc_script([fc_part("fc1", "get_weather", {"city": "Berlin"}, b"SIG-ONE")]),
            answer_script("Still fine."),
        ]
    )
    result2, _emitter2 = run_pipe(pipe, fake2, tools=tools, metadata=metadata)
    assert called == []
    fr2 = fake2.stream_calls[1]["contents"][-1].parts[0].function_response
    assert "denied" in str(fr2.response["error"])
    assert "Still fine." in result2


def test_tool_call_limit_disables_function_calling():
    # criterion 5
    pipe = make_pipe(GEMINI_API_KEY="k", MAX_TOOL_CALLS=1)
    ran = []

    def weather(**kwargs):
        ran.append("w")
        return "sunny"

    def get_time(**kwargs):
        ran.append("t")
        return "noon"

    tools = {
        "get_weather": tool_entry("get_weather", weather),
        "get_time": tool_entry("get_time", get_time),
    }
    stream1 = fc_script(
        [
            fc_part("fc1", "get_weather", {}, b"SIG-ONE"),
            fc_part("fc2", "get_time", {}),
        ]
    )
    fake = FakeClient(scripts=[stream1, answer_script("ok")])
    run_pipe(pipe, fake, tools=tools)

    assert ran == ["w"]
    frs = [
        p.function_response
        for p in fake.stream_calls[1]["contents"][-1].parts
        if p.function_response is not None
    ]
    assert [fr.id for fr in frs] == ["fc1", "fc2"]
    assert frs[0].response == {"result": "sunny"}
    assert "limit" in str(frs[1].response.get("error", ""))
    mode = fake.stream_calls[1]["config"].tool_config.function_calling_config.mode
    assert getattr(mode, "name", str(mode)) == "NONE"


def test_malformed_function_call_retries_without_notice():
    # criterion 6
    pipe = make_pipe(GEMINI_API_KEY="k")
    tools = {"get_weather": tool_entry("get_weather", lambda **kw: "sunny")}
    stream1 = [chunk({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]})]
    fake = FakeClient(scripts=[stream1, answer_script("Done.")])
    result, _emitter = run_pipe(pipe, fake, tools=tools)
    assert "Done." in result
    assert "Tool call failed" not in result
    assert len(fake.stream_calls) == 2


def test_builtin_tools_declared():
    # criterion 7
    pipe = make_pipe(GEMINI_API_KEY="k")
    mod = pipe_module(pipe)

    def search_memory(**kwargs):
        return "mem"

    saved_available = mod.BUILTIN_TOOLS_AVAILABLE
    saved_fn = mod.get_builtin_tools
    mod.BUILTIN_TOOLS_AVAILABLE = True
    mod.get_builtin_tools = lambda request, meta, features=None, model=None: {
        "search_memory": tool_entry("search_memory", search_memory)
    }
    try:
        fake = FakeClient(scripts=[answer_script("ok")])
        run_pipe(pipe, fake, request=object())
        decls = fake.stream_calls[0]["config"].tools[0].function_declarations
        assert "search_memory" in [d.name for d in decls]
    finally:
        mod.BUILTIN_TOOLS_AVAILABLE = saved_available
        mod.get_builtin_tools = saved_fn
