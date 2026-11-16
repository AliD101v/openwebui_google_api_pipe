"""Checks for capability derivation and overrides (SPEC section 4.3), criteria 12-15."""
from __future__ import annotations

import pytest

from _testlib import make_pipe


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    yield


def test_gen4_capabilities():
    # criterion 12
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    assert info["generation"] == 4.0
    assert info["thinking"] == "level"
    assert info["thinking_levels"] == ["low", "medium", "high"]
    assert info["builtin_with_functions"] is True


def test_hydrogen_and_carbon_capabilities():
    # Google's thinking page: Hydrogen takes low..xhigh (default high, rejects
    # minimal); Carbon takes low..high (default medium). Both: 1M context,
    # text out, Search / Maps / URL context / function calling.
    pipe = make_pipe()
    hydrogen = pipe.get_model_info("gemini-4-hydrogen")
    carbon = pipe.get_model_info("gemini-4-carbon")
    assert hydrogen["thinking_levels"] == ["low", "medium", "high", "xhigh"]
    assert carbon["thinking_levels"] == ["low", "medium", "high"]
    for info in (hydrogen, carbon):
        assert info["generation"] == 4.0
        assert info["thinking"] == "level"
        assert "minimal" not in info["thinking_levels"]
        assert info["thinking_can_disable"] is False
        assert info["budget_range"] is None
        assert info["image_out"] is False
        assert info["functions"] is True
        assert info["builtin_with_functions"] is True
        assert info["search"] and info["url_context"] and info["maps"]
        assert info["maps_with_search"] is False
        assert info["maps_with_url_context"] is False
        assert info["context_length"] == 1048576
        assert info["max_output"] == 65536
    assert pipe.get_model_info("gemini-4-argon")["thinking_levels"] == ["low", "medium", "high"]


def test_hydrogen_levels_can_still_be_overridden():
    pipe = make_pipe(MODEL_CAPABILITY_OVERRIDES='{"gemini-4-hydrogen": {"thinking_levels": ["low", "high"]}}')
    assert pipe.get_model_info("gemini-4-hydrogen")["thinking_levels"] == ["low", "high"]


def test_minimal_level_and_budget_models():
    # criterion 13
    pipe = make_pipe()
    assert pipe.get_model_info("gemini-3.5-flash-lite")["thinking_levels"][0] == "minimal"
    flash = pipe.get_model_info("gemini-2.5-flash")
    assert flash["thinking"] == "budget"
    assert flash["budget_range"] == [0, 24576]
    assert flash["thinking_can_disable"] is True
    assert pipe.get_model_info("gemini-2.5-pro")["budget_range"] == [128, 32768]


def test_image_model_capabilities():
    # criterion 14
    pipe = make_pipe()
    banana = pipe.get_model_info("gemini-nano-banana-2.1")
    assert banana["image_out"] is True
    assert banana["functions"] is False
    assert banana["thinking"] == "always"
    assert pipe.get_model_info("gemini-3.1-flash-lite-image")["search"] is False


def test_capability_overrides_valve():
    # criterion 15
    pipe = make_pipe(MODEL_CAPABILITY_OVERRIDES='{"gemini-4*": {"maps": false}}')
    assert pipe.get_model_info("gemini-4-argon")["maps"] is False

    broken = make_pipe(MODEL_CAPABILITY_OVERRIDES="{bad json")
    info = broken.get_model_info("gemini-4-argon")
    assert info["maps"] is True, "invalid override JSON must be ignored, defaults kept"
