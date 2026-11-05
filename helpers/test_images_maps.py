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
    # criterion 4
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
    ctx = make_ctx(pipe, emitter=FakeEmitter())

    info = pipe.get_model_info("gemini-nano-banana-2.1")
    _system, contents = asyncio.run(pipe._convert_history(messages, info, ctx))
    assert [c.role for c in contents] == ["user", "model", "user"]
    parts = contents[1].parts
    assert len(parts) == 2
    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "image/png"
    assert parts[0].inline_data.data == PNG
    assert parts[0].thought_signature == mod.DUMMY_THOUGHT_SIGNATURE
    assert parts[1].text == "Here."

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
