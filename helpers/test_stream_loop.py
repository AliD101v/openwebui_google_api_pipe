"""Checks for the streaming pipe: pipe(), part dispatch, grounding wiring,
retries, errors and cancellation (SPEC sections 9.1-9.5, 9.8, 9.9)."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest
from google.genai import errors as genai_errors

from _testlib import REPO_ROOT, FakeClient, FakeEmitter, chunk, make_pipe, pipe_module

BODY = {
    "model": "gemini_pipe.gemini/gemini-4-argon",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}
METADATA = {"chat_id": "c1", "message_id": "m1"}


def make_user(pipe):
    return {"id": "u1", "name": "T", "valves": pipe.UserValves()}


def run_pipe(pipe, fake, *, metadata=None, task=None):
    """Run pipe() against the fake client and return (result, emitter)."""
    pipe._build_genai_client = lambda api_key, timeout_s=None: fake
    user = make_user(pipe)
    emitter = FakeEmitter()
    result = asyncio.run(
        pipe.pipe(
            dict(BODY),
            __user__=user,
            __event_emitter__=emitter,
            __metadata__=dict(METADATA, **(metadata or {})),
            __task__=task,
        )
    )
    return result, emitter


def test_thoughts_text_usage_and_final_events():
    # criterion 1
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    fake = FakeClient(
        scripts=[
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"text": "Considering options", "thought": True}
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
                                    "parts": [{"text": "Hello "}],
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
                                    "parts": [{"text": "world."}],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 120,
                            "candidatesTokenCount": 5,
                            "thoughtsTokenCount": 30,
                        },
                    }
                ),
            ]
        ]
    )
    result, emitter = run_pipe(pipe, fake)
    assert result.index('<details type="reasoning"') < result.index("Hello world.")
    replaces = emitter.of_type("replace")
    assert replaces, "expected replace events"
    assert replaces[-1]["data"]["content"] == result
    completions = emitter.of_type("chat:completion")
    assert completions, "expected a chat:completion event"
    usage = completions[-1]["data"]["usage"]
    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 35
    statuses = emitter.of_type("status")
    assert statuses, "expected status events"
    assert statuses[-1]["data"]["done"] is True


def test_search_on_grounding_citations_and_block():
    # criterion 2
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    grounding = {
        "webSearchQueries": ["capital of France"],
        "groundingChunks": [
            {
                "web": {
                    "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/X",
                    "title": "example.com",
                }
            }
        ],
        "groundingSupports": [
            {
                "segment": {
                    "startIndex": 0,
                    "endIndex": 31,
                    "text": "Paris is the capital of France.",
                },
                "groundingChunkIndices": [0],
            }
        ],
        "searchEntryPoint": {
            "renderedContent": (
                '<a class="chip" href="https://www.google.com/search?q=paris">paris</a>'
            )
        },
    }
    fake = FakeClient(
        scripts=[
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Paris is the capital of France."}],
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
        ]
    )
    result, emitter = run_pipe(pipe, fake, metadata={"gemini_web_search": True})
    tools = fake.stream_calls[0]["config"].tools or []
    kinds = {type(tool).__name__ for tool in tools}
    assert any(getattr(tool, "google_search", None) is not None for tool in tools)
    assert any(getattr(tool, "url_context", None) is not None for tool in tools)
    sources = emitter.of_type("source")
    assert len(sources) == 1
    assert sources[0]["data"]["source"]["id"] == "1"
    assert any(
        e["data"].get("action") == "web_search" for e in emitter.of_type("status")
    )
    assert "France.[1]" in result
    assert 'name="google_search"' in result
    assert "embeds=" in result
    assert result.index('name="google_search"') < result.index("Paris is")


def test_search_off_sends_no_tools():
    # criterion 3
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    fake = FakeClient(
        scripts=[
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "No search here."}],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                    }
                )
            ]
        ]
    )
    result, emitter = run_pipe(pipe, fake)
    tools = fake.stream_calls[0]["config"].tools
    assert not tools, f"expected no tools, got {tools!r}"
    assert "No search here." in result


def test_retry_transient_server_error():
    # criterion 4
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    pipe._RETRY_BACKOFF_BASE = 0
    error = genai_errors.ServerError(
        503,
        {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}},
    )
    fake = FakeClient(
        scripts=[
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Partial"}],
                                }
                            }
                        ]
                    }
                ),
                error,
            ],
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Full answer."}],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                    }
                )
            ],
        ]
    )
    result, emitter = run_pipe(pipe, fake)
    assert result.count("Full answer.") == 1
    assert "Partial" not in result
    assert any(
        e["data"]["description"].startswith("Retrying (1/3)")
        for e in emitter.of_type("status")
    )
    assert len(fake.stream_calls) == 2


def test_non_retryable_error_notifies_once():
    # criterion 5
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    error = genai_errors.ClientError(
        400,
        {"error": {"code": 400, "message": "Invalid argument", "status": "INVALID_ARGUMENT"}},
    )
    fake = FakeClient(scripts=[[error]])
    result, emitter = run_pipe(pipe, fake)
    notifications = emitter.of_type("notification")
    assert notifications, "expected a notification event"
    assert any("Invalid argument" in e["data"]["content"] for e in notifications)
    assert len(fake.stream_calls) == 1


def test_safety_finish_reason_notice():
    # criterion 6
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    fake = FakeClient(
        scripts=[
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Some text"}],
                                },
                                "finishReason": "SAFETY",
                            }
                        ],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                    }
                )
            ]
        ]
    )
    result, emitter = run_pipe(pipe, fake)
    assert "Response stopped by Google (SAFETY)" in result


def test_no_api_key_returns_error_and_notifies(monkeypatch):
    # criterion 7
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pipe = make_pipe()
    fake = FakeClient(scripts=[])
    result, emitter = run_pipe(pipe, fake)
    assert result.startswith("Error: no Gemini API key configured.")
    assert emitter.of_type("notification"), "expected a notification event"


def test_cancellation_emits_replace_and_completion():
    # criterion 8
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    fake = FakeClient(
        scripts=[
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Hel"}],
                                }
                            }
                        ]
                    }
                ),
                asyncio.CancelledError(),
            ]
        ]
    )
    pipe._build_genai_client = lambda api_key, timeout_s=None: fake
    user = make_user(pipe)
    emitter = FakeEmitter()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            pipe.pipe(
                dict(BODY),
                __user__=user,
                __event_emitter__=emitter,
                __metadata__=dict(METADATA),
            )
        )
    replaces = emitter.of_type("replace")
    assert any("Hel" in e["data"]["content"] for e in replaces), (
        "expected a replace event containing the streamed prefix"
    )
    assert emitter.of_type("chat:completion"), "expected a chat:completion event"


def test_task_branch_runs_task_and_never_streams():
    # criterion 9
    pipe = make_pipe(GEMINI_API_KEY="k", RESOLVE_GROUNDING_URLS=False)
    fake = FakeClient(
        responses=[
            chunk(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "My title"}],
                            }
                        }
                    ]
                }
            )
        ]
    )
    result, emitter = run_pipe(pipe, fake, task="title_generation")
    assert result == "My title"
    assert len(fake.stream_calls) == 0
    assert len(fake.generate_calls) == 1


def test_smoke_script_skips_without_api_key():
    # criterion 10
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    }
    proc = subprocess.run(
        [sys.executable, "helpers/smoke_live.py", "--scenario", "all"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "skipped" in proc.stdout
