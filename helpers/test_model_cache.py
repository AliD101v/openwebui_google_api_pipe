"""Checks for model discovery and the model-list cache (SPEC section 4), criteria 8-11."""
from __future__ import annotations

import asyncio
import time

import pytest

from _testlib import FakeClient, fake_model, make_pipe

DISCOVERED = [
    fake_model("gemini-4-argon"),
    fake_model("gemini-3.8-flash"),
    fake_model("text-embedding-004", supported_actions=("embedContent",)),
    fake_model("gemini-2.0-flash"),
    fake_model("gemini-3.1-flash-tts-preview"),
    fake_model("gemini-nano-banana-2.1"),
    fake_model("gemini-2.5-pro"),
    fake_model("gemma-3-27b-it"),
]

EXPECTED_IDS = [
    "gemini/gemini-4-argon",
    "gemini/gemini-3.8-flash",
    "gemini/gemini-nano-banana-2.1",
    "gemini/gemini-2.5-pro",
]

STATIC_FALLBACK_IDS = [
    "gemini/gemini-4-argon",
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-nano-banana-2.1",
]


def _wire(pipe, client):
    pipe._build_genai_client = lambda api_key, timeout_s=None: client
    return client


def test_pipes_returns_chat_models_in_api_order(monkeypatch):
    # criterion 8
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pipe = make_pipe(GEMINI_API_KEY="AIza-test")
    client = _wire(pipe, FakeClient(models=DISCOVERED))
    entries = asyncio.run(pipe.pipes())
    assert [e["id"] for e in entries] == EXPECTED_IDS
    assert len(client.calls) == 1


def test_second_call_uses_cache_and_key_change_refetches(monkeypatch):
    # criterion 9
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pipe = make_pipe(GEMINI_API_KEY="AIza-test")
    client = _wire(pipe, FakeClient(models=DISCOVERED))
    asyncio.run(pipe.pipes())
    asyncio.run(pipe.pipes())
    assert len(client.calls) == 1, "second pipes() within the TTL must not re-list"
    pipe.valves.GEMINI_API_KEY = "AIza-other"
    asyncio.run(pipe.pipes())
    assert len(client.calls) == 2, "changing the API key must refetch"


def test_stale_cache_on_error_and_fallback_on_signature_change(monkeypatch):
    # criterion 10
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pipe = make_pipe(GEMINI_API_KEY="AIza-test")
    _wire(
        pipe,
        FakeClient(
            scripts=[
                list(DISCOVERED),
                RuntimeError("API down"),
                RuntimeError("API down"),
            ]
        ),
    )
    asyncio.run(pipe.pipes())
    type(pipe)._model_cache_ts = time.time() - 25 * 3600  # beyond the 24h TTL
    entries = asyncio.run(pipe.pipes())
    assert [e["id"] for e in entries] == EXPECTED_IDS, (
        "same signature, expired TTL, failed refetch: stale cache comes back"
    )
    pipe.valves.GEMINI_BASE_URL = "https://other-endpoint.example"
    entries = asyncio.run(pipe.pipes())
    assert [e["id"] for e in entries] == STATIC_FALLBACK_IDS, (
        "changed signature: the static fallback list is used"
    )


def test_enabled_models_skip_discovery_and_missing_key_is_error(monkeypatch):
    # criterion 11
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pipe = make_pipe(GEMINI_API_KEY="AIza-test", ENABLED_MODELS="gemini-4-argon, gemini-3.8-flash")

    def _refuse(*args, **kwargs):
        raise AssertionError("list must never be called with ENABLED_MODELS")

    pipe._build_genai_client = _refuse
    entries = asyncio.run(pipe.pipes())
    assert [e["id"] for e in entries] == [
        "gemini/gemini-4-argon",
        "gemini/gemini-3.8-flash",
    ]

    no_key_pipe = make_pipe()
    assert not str(no_key_pipe.valves.GEMINI_API_KEY).strip()
    entries = asyncio.run(no_key_pipe.pipes())
    assert [e["id"] for e in entries] == ["gemini/error"]
