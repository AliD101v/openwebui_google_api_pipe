"""Checks for valves and API-key encryption (SPEC sections 2.4, 2.5), criteria 5-7."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from _testlib import load_pipe_module

SECRET = "test-secret-key-aaaa"


@pytest.fixture(autouse=True)
def secret_key(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", SECRET)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    yield


def test_valve_encryption_roundtrip():
    # criterion 5
    module = load_pipe_module()
    dumped = module.Pipe.Valves(GEMINI_API_KEY="AIza-test").model_dump()["GEMINI_API_KEY"]
    assert dumped.startswith("encrypted:")
    again = module.Pipe.Valves(GEMINI_API_KEY=dumped).model_dump()["GEMINI_API_KEY"]
    assert again == dumped
    assert module.decrypt_valve_secret(dumped) == "AIza-test"
    assert module.Pipe.UserValves(GEMINI_API_KEY="").model_dump()["GEMINI_API_KEY"] == ""


def test_hide_blocks_accepts_comma_separated_string():
    # criterion 6
    module = load_pipe_module()
    valves = module.Pipe.UserValves(HIDE_BLOCKS="google_search, reasoning")
    assert valves.HIDE_BLOCKS == ["google_search", "reasoning"]


def test_build_genai_client_decrypts_and_configures():
    # criterion 7
    module = load_pipe_module()
    pipe = module.Pipe()
    encrypted = module.encrypt_valve_secret("AIza-test")
    assert encrypted.startswith("encrypted:")

    recorded = {}

    def recorder(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(http_options=kwargs.get("http_options"))

    monkeypatched = module.genai.Client
    module.genai.Client = recorder
    try:
        client = pipe._build_genai_client(encrypted)
    finally:
        module.genai.Client = monkeypatched

    assert recorded["api_key"] == "AIza-test"
    assert recorded["vertexai"] is False
    assert client.http_options.timeout == 600000

    with pytest.raises(ValueError):
        pipe._build_genai_client("")
