"""Checks for image generation (Nano Banana models) and Google Maps
grounding (task 06)."""
from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys

from types import SimpleNamespace
from uuid import uuid4

from google.genai import errors as genai_errors

from _testlib import (
    REPO_ROOT,
    FakeClient,
    FakeEmitter,
    chunk,
    fake_owui_files,
    make_ctx,
    make_pipe,
    pipe_module,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def make_user(pipe, **user_valves):
    return {"id": "u1", "name": "T", "valves": pipe.UserValves(**user_valves)}


def run_pipe(pipe, fake, *, model, messages=None, user=None, metadata=None):
    """Run pipe() against the fake client and return (result, emitter)."""
    pipe._build_genai_client = lambda api_key, timeout_s=None: fake
    emitter = FakeEmitter()
    body = {
        "model": f"gemini_pipe.gemini/{model}",
        "messages": list(
            messages or [{"role": "user", "content": "Hi"}]
        ),
        "stream": True,
    }
    result = asyncio.run(
        pipe.pipe(
            body,
            __user__=user or make_user(pipe),
            __event_emitter__=emitter,
            __metadata__=(
                metadata
                if metadata is not None
                else {"chat_id": "c1", "message_id": "m1"}
            ),
        )
    )
    return result, emitter


def test_image_generation_config():
    # criterion 1
    pipe = make_pipe()
    user_valves = pipe.UserValves(IMAGE_ASPECT_RATIO="16:9", IMAGE_SIZE="2K")

    info = pipe.get_model_info("gemini-nano-banana-2.1")
    config = pipe._build_generate_config(
        info=info,
        system_instruction=None,
        tools=None,
        tool_config=None,
        thinking=None,
        body={},
        user_valves=user_valves,
    )
    assert config.response_modalities == ["TEXT", "IMAGE"]
    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "16:9"
    assert config.image_config.image_size == "2K"

    info = pipe.get_model_info("gemini-4-argon")
    config = pipe._build_generate_config(
        info=info,
        system_instruction=None,
        tools=None,
        tool_config=None,
        thinking=None,
        body={},
        user_valves=user_valves,
    )
    assert config.response_modalities is None
    assert config.image_config is None


def test_image_output_stream_and_save():
    # criterion 2
    pipe = make_pipe(GEMINI_API_KEY="k")
    data = base64.b64encode(PNG).decode()
    script = [
        chunk(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": data,
                                    },
                                    "thought": True,
                                }
                            ],
                        }
                    }
                ]
            }
        ),
        chunk(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Here you go."}],
                        }
                    }
                ]
            }
        ),
        chunk(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": data,
                                    }
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ),
    ]
    fake = FakeClient(scripts=[script])
    calls = []

    async def fake_store(ctx, image_data, mime):
        calls.append((image_data, mime))
        return "/api/v1/files/abc/content"

    pipe._store_generated_image = fake_store
    result, _emitter = run_pipe(pipe, fake, model="gemini-nano-banana-2.1")

    assert len(calls) == 1
    assert calls[0] == (PNG, "image/png")
    assert "![Generated image](/api/v1/files/abc/content)" in result
    assert "Saving image" not in result


def test_store_generated_image_paths():
    # criterion 3
    pipe = make_pipe()
    mod = pipe_module(pipe)
    ctx = make_ctx(pipe, metadata={"chat_id": "c1", "message_id": "m1"})
    ctx.user = {"id": "u1"}
    ctx.request = object()

    captured = {}

    async def fake_upload_image(request, image_data, content_type, metadata, user):
        captured.update(
            request=request,
            image_data=image_data,
            content_type=content_type,
            metadata=metadata,
            user=user,
        )
        return SimpleNamespace(id="img1"), {"id": "img1"}

    class FakeUsers:
        async def get_user_by_id(self, user_id):
            return SimpleNamespace(id=user_id)

    mod.upload_image = fake_upload_image
    mod.Users = FakeUsers()
    mod.UPLOAD_IMAGE_AVAILABLE = True
    mod.USERS_AVAILABLE = True
    url = asyncio.run(pipe._store_generated_image(ctx, PNG, "image/png"))
    assert url == "/api/v1/files/img1/content"
    assert captured["request"] is ctx.request
    assert captured["image_data"] == PNG
    assert captured["content_type"] == "image/png"
    assert captured["metadata"]["chat_id"] == "c1"
    assert captured["metadata"]["message_id"] == "m1"
    assert captured["user"].id == "u1"

    mod.upload_image = None
    mod.Users = None
    mod.UPLOAD_IMAGE_AVAILABLE = False
    mod.USERS_AVAILABLE = False
    mod.FILES_AVAILABLE = False
    mod.Files = None
    mod.Storage = None
    assert asyncio.run(pipe._store_generated_image(ctx, PNG, "image/png")) is None


def test_generated_image_history_replay(tmp_path):
    # criterion 4 (changed after probing the live API): the earlier image goes on the NEW user message.
    # Replayed in a model turn with the dummy signature, gemini-nano-banana-2.1 ignored it and redrew.
    pipe = make_pipe()
    mod = pipe_module(pipe)
    file_id = str(uuid4())
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": file_id,
                "filename": "generated-image.png",
                "path": f"files/{file_id}.png",
                "hash": "h1",
                "content_type": "image/png",
                "data": PNG,
            }
        ],
    )
    assistant = f"Here.\n![Generated image](/api/v1/files/{file_id}/content)"
    messages = [
        {"role": "user", "content": "draw a cat"},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": "make it bigger"},
    ]
    emitter = FakeEmitter()
    ctx = make_ctx(pipe, emitter=emitter)

    info = pipe.get_model_info("gemini-nano-banana-2.1")
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [p.text for p in contents[1].parts] == ["Here."]
    assert all(p.inline_data is None for p in contents[1].parts)
    parts = contents[2].parts
    assert len(parts) == 2
    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "image/png"
    assert parts[0].inline_data.data == PNG
    assert parts[0].thought_signature is None
    assert parts[1].text == "make it bigger"
    assert any(
        "previous image" in (e["data"].get("description") or "")
        for e in emitter.of_type("status")
    )

    info = pipe.get_model_info("gemini-4-argon")
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    assert not any(
        getattr(p, "inline_data", None) is not None for c in contents for p in c.parts
    )


def test_maps_grounding_end_to_end():
    # criterion 5
    pipe = make_pipe(GEMINI_API_KEY="k")
    uri = "https://maps.google.com/?cid=1"
    grounding = {
        "groundingChunks": [
            {"maps": {"uri": uri, "title": "Café Einstein", "placeId": "place1"}}
        ],
        "groundingSupports": [
            {
                "segment": {
                    "startIndex": 4,
                    "endIndex": 19,
                    "text": "Café Einstein.",
                },
                "groundingChunkIndices": [0],
            }
        ],
    }
    script = [
        chunk(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Try Café Einstein."}],
                        }
                    }
                ]
            }
        ),
        chunk(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": []},
                        "finishReason": "STOP",
                        "groundingMetadata": grounding,
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        ),
    ]
    fake = FakeClient(scripts=[script])
    user = make_user(pipe, MAPS_LATITUDE="52.52", MAPS_LONGITUDE="13.40")
    result, emitter = run_pipe(
        pipe,
        fake,
        model="gemini-4-argon",
        user=user,
        metadata={
            "chat_id": "c1",
            "message_id": "m1",
            "gemini_maps": True,
        },
    )
    config = fake.stream_calls[0]["config"]
    tools = config.tools or []
    assert any(getattr(tool, "google_maps", None) is not None for tool in tools)
    assert config.tool_config.retrieval_config.lat_lng.latitude == 52.52

    sources = emitter.of_type("source")
    assert len(sources) == 1
    assert sources[0]["data"]["source"]["id"] == "1"
    assert "Einstein.[1]" in result
    assert "[Café Einstein](https://maps.google.com/?cid=1)" in result
    assert "Google Maps" in result
    assert result.index("Einstein.[1]") < result.index("Google Maps")
    assert 'name="google_search"' not in result
    assert any(
        "places" in (e["data"].get("description") or "")
        for e in emitter.of_type("status")
    )


def test_history_strips_maps_attribution_line():
    # criterion 6
    pipe = make_pipe()
    attribution = (
        '📍 <span translate="no">Google Maps</span>: '
        "[Café Einstein](https://maps.google.com/?cid=1)"
    )
    assistant = f"Here is a café.\n{attribution}"
    messages = [
        {"role": "user", "content": "recommend a café"},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": "thanks"},
    ]
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    info = pipe.get_model_info("gemini-4-argon")
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    text = "".join(p.text or "" for c in contents for p in c.parts)
    assert "Google Maps" not in text
    assert "Here is a café." in text


def test_smoke_script_scenarios_offline():
    # criterion 7
    base = [sys.executable, str(REPO_ROOT / "helpers" / "smoke_live.py")]
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    for scenario in ("all", "image", "tool", "maps"):
        proc = subprocess.run(
            base + ["--scenario", scenario],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, (scenario, proc.stdout, proc.stderr)
        assert "skipped" in (proc.stdout + proc.stderr), (scenario, proc.stdout)


def _text_script(text):
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


def test_maps_request_carries_only_the_maps_tool():
    # The reported failure: Maps toggle with Search on by default (and a URL in the message) sent
    # google_maps next to google_search/url_context and the API answered 400.
    pipe = make_pipe(GEMINI_API_KEY="k", WEB_SEARCH_DEFAULT=True, RESOLVE_GROUNDING_URLS=False)
    fake = FakeClient(scripts=[_text_script("Try Café Einstein.")])
    run_pipe(
        pipe,
        fake,
        model="gemini-4-argon",
        messages=[{"role": "user", "content": "a café near https://example.com/berlin please"}],
        metadata={"chat_id": "c1", "message_id": "m1", "gemini_maps": True},
    )
    tools = fake.stream_calls[0]["config"].tools or []
    assert [bool(t.google_maps) for t in tools] == [True]
    assert all(t.google_search is None and t.url_context is None for t in tools)


def test_rejected_builtin_combination_is_retried_without_the_lower_priority_tool():
    # An unknown combination 400 ("'a' and 'b' cannot be combined") drops the lower-priority built-in
    # (url_context before google_search; google_maps is never dropped) and retries once.
    pipe = make_pipe(
        GEMINI_API_KEY="k",
        URL_CONTEXT="always",
        RESOLVE_GROUNDING_URLS=False,
        MODEL_CAPABILITY_OVERRIDES='{"gemini-4*": {"maps_with_url_context": true}}',
    )
    error = genai_errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": "'google_maps' and 'url_context' cannot be combined in the same request. Please choose one to continue.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    fake = FakeClient(scripts=[[error], _text_script("Try Café Einstein.")])
    result, emitter = run_pipe(
        pipe,
        fake,
        model="gemini-4-argon",
        metadata={"chat_id": "c1", "message_id": "m1", "gemini_maps": True},
    )
    assert len(fake.stream_calls) == 2
    first, second = (c["config"].tools or [] for c in fake.stream_calls)
    assert any(t.url_context is not None for t in first)
    assert any(t.google_maps is not None for t in second)
    assert all(t.url_context is None for t in second)
    assert "Try Café Einstein." in result
    assert not emitter.of_type("notification")
    assert any(
        "without URL context" in (e["data"].get("description") or "")
        for e in emitter.of_type("status")
    )


def test_maps_only_grounding_with_queries_renders_no_search_block():
    # The live API reports webSearchQueries on a Maps-only response. They belong to Maps: no "Google Search"
    # block or status, and no search-query count (cost) for a request that never enabled Search.
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    grounding = {
        "groundingChunks": [
            {"maps": {"uri": "https://maps.google.com/?cid=1", "title": "Café Einstein", "placeId": "place1"}}
        ],
        "groundingSupports": [
            {
                "segment": {"startIndex": 4, "endIndex": 19, "text": "Café Einstein."},
                "groundingChunkIndices": [0],
            }
        ],
        "webSearchQueries": ["café near me"],
    }
    script = [
        chunk({"candidates": [{"content": {"role": "model", "parts": [{"text": "Try Café Einstein."}]}}]}),
        chunk(
            {
                "candidates": [
                    {"content": {"role": "model", "parts": []}, "finishReason": "STOP", "groundingMetadata": grounding}
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
        ),
    ]
    fake = FakeClient(scripts=[script])
    result, emitter = run_pipe(
        pipe,
        fake,
        model="gemini-4-argon",
        metadata={"chat_id": "c1", "message_id": "m1", "gemini_maps": True},
    )
    assert "Einstein.[1]" in result
    assert "Google Maps" in result
    assert 'name="google_search"' not in result
    descriptions = [e["data"].get("description") or "" for e in emitter.of_type("status")]
    assert not any("Searching Google" in d or "Searched" in d for d in descriptions)
    assert any("places" in d for d in descriptions)


def _two_image_chat(tmp_path, pipe, first_text="One.", second_text="Two."):
    mod = pipe_module(pipe)
    ids = [str(uuid4()), str(uuid4())]
    datas = [PNG + b"\x01" * 8, PNG + b"\x02" * 8]
    fake_owui_files(
        mod,
        tmp_path,
        [
            {
                "id": fid,
                "filename": "generated-image.png",
                "path": f"files/{fid}.png",
                "hash": f"h{n}",
                "content_type": "image/png",
                "data": data,
            }
            for n, (fid, data) in enumerate(zip(ids, datas))
        ],
    )
    messages = [
        {"role": "user", "content": "draw a cat"},
        {"role": "assistant", "content": f"{first_text}\n![Generated image](/api/v1/files/{ids[0]}/content)"},
        {"role": "user", "content": "add a hat"},
        {"role": "assistant", "content": f"{second_text}\n![Generated image](/api/v1/files/{ids[1]}/content)"},
        {"role": "user", "content": "make the hat red"},
    ]
    return messages, datas


def test_only_the_latest_generated_image_is_resent(tmp_path):
    pipe = make_pipe()
    messages, datas = _two_image_chat(tmp_path, pipe)
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    info = pipe.get_model_info("gemini-nano-banana-2.1")
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    images = [p.inline_data.data for c in contents for p in c.parts if p.inline_data is not None]
    assert images == [datas[1]]
    assert contents[-1].role == "user"
    assert contents[-1].parts[0].inline_data.data == datas[1]
    assert contents[-1].parts[-1].text == "make the hat red"


def test_image_only_assistant_message_still_resends_its_image(tmp_path):
    pipe = make_pipe()
    messages, datas = _two_image_chat(tmp_path, pipe, first_text="", second_text="")
    ctx = make_ctx(pipe, emitter=FakeEmitter())
    info = pipe.get_model_info("gemini-nano-banana-2.1")
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    assert all(c.role == "user" for c in contents)
    images = [p.inline_data.data for c in contents for p in c.parts if p.inline_data is not None]
    assert images == [datas[1]]
    assert contents[-1].parts[-1].text == "make the hat red"
