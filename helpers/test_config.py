"""Checks for thinking and generation config (SPEC section 8.2), criteria 16-17."""
from __future__ import annotations

from google.genai import types as gtypes

from _testlib import make_pipe


def _user_valves(pipe, **overrides):
    values = {
        "THINKING_LEVEL": "auto",
        "THINKING_BUDGET": -1,
        "SHOW_THOUGHTS": True,
    }
    values.update(overrides)
    return pipe.UserValves(**values)


def _assert_single_mode(config):
    assert not (
        config.thinking_level is not None and config.thinking_budget is not None
    ), "a ThinkingConfig must never set both a level and a budget"


def test_thinking_matrix():
    # criterion 16
    pipe = make_pipe()
    argon = pipe.get_model_info("gemini-4-argon")

    config = pipe._build_thinking_config(argon, _user_valves(pipe), {"reasoning_effort": "minimal"}, "chat")
    _assert_single_mode(config)
    assert config.thinking_level == gtypes.ThinkingLevel.LOW

    config = pipe._build_thinking_config(argon, _user_valves(pipe), {"reasoning_effort": "xhigh"}, "chat")
    _assert_single_mode(config)
    assert config.thinking_level == gtypes.ThinkingLevel.HIGH

    lite = pipe.get_model_info("gemini-3.5-flash-lite")
    config = pipe._build_thinking_config(lite, _user_valves(pipe), {}, "task")
    _assert_single_mode(config)
    assert config.thinking_level == gtypes.ThinkingLevel.MINIMAL
    assert config.include_thoughts is False

    config = pipe._build_thinking_config(argon, _user_valves(pipe), {}, "chat")
    _assert_single_mode(config)
    assert config.thinking_level is None
    assert config.include_thoughts is True

    flash25 = pipe.get_model_info("gemini-2.5-flash")
    config = pipe._build_thinking_config(flash25, _user_valves(pipe), {"reasoning_effort": "high"}, "chat")
    _assert_single_mode(config)
    assert config.thinking_budget == 24576
    assert config.thinking_level is None

    pro25 = pipe.get_model_info("gemini-2.5-pro")
    config = pipe._build_thinking_config(pro25, _user_valves(pipe), {}, "task")
    _assert_single_mode(config)
    assert config.thinking_budget == 128

    pipe_none = make_pipe(MODEL_CAPABILITY_OVERRIDES='{"gemini-4-argon": {"thinking": "none"}}')
    config = pipe_none._build_thinking_config(
        pipe_none.get_model_info("gemini-4-argon"), _user_valves(pipe_none), {"reasoning_effort": "high"}, "chat"
    )
    assert config is None


def _level_name(config):
    """The level as an upper-case name, whichever way the SDK models xhigh."""
    return str(getattr(config.thinking_level, "value", config.thinking_level)).upper()


def test_hydrogen_thinking_levels():
    # Hydrogen: low/medium/high/xhigh, default high, rejects minimal.
    pipe = make_pipe()
    hydrogen = pipe.get_model_info("gemini-4-hydrogen")
    for effort, expected in (
        ("xhigh", "XHIGH"),
        ("max", "XHIGH"),
        ("high", "HIGH"),
        ("medium", "MEDIUM"),
        ("low", "LOW"),
        ("minimal", "LOW"),
        ("none", "LOW"),
    ):
        config = pipe._build_thinking_config(
            hydrogen, _user_valves(pipe), {"reasoning_effort": effort}, "chat"
        )
        _assert_single_mode(config)
        assert _level_name(config) == expected, effort
        assert config.include_thoughts is True

    config = pipe._build_thinking_config(hydrogen, _user_valves(pipe, THINKING_LEVEL="xhigh"), {}, "chat")
    assert _level_name(config) == "XHIGH"
    assert config.model_dump(mode="json", exclude_none=True)["thinking_level"].upper() == "XHIGH"

    # auto sends no level, so the model default applies
    config = pipe._build_thinking_config(hydrogen, _user_valves(pipe), {}, "chat")
    assert config.thinking_level is None

    # background tasks use the lowest allowed level, never the rejected "minimal"
    config = pipe._build_thinking_config(hydrogen, _user_valves(pipe), {}, "task")
    assert _level_name(config) == "LOW"
    assert config.include_thoughts is False


def test_carbon_thinking_levels():
    # Carbon: low/medium/high, default medium; xhigh and minimal clamp inward.
    pipe = make_pipe()
    carbon = pipe.get_model_info("gemini-4-carbon")
    for effort, expected in (("xhigh", "HIGH"), ("high", "HIGH"), ("medium", "MEDIUM"), ("minimal", "LOW")):
        config = pipe._build_thinking_config(
            carbon, _user_valves(pipe), {"reasoning_effort": effort}, "chat"
        )
        _assert_single_mode(config)
        assert _level_name(config) == expected, effort
    config = pipe._build_thinking_config(carbon, _user_valves(pipe), {}, "chat")
    assert config.thinking_level is None
    config = pipe._build_thinking_config(carbon, _user_valves(pipe), {}, "task")
    assert _level_name(config) == "LOW"


def test_xhigh_degrades_to_high_when_the_sdk_cannot_express_it(monkeypatch):
    # Older google-genai releases have no XHIGH member and reject the lookup;
    # the call must still go out, one level down.
    import enum
    import sys
    import types

    class OldLevel(enum.Enum):
        MINIMAL = "MINIMAL"
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"

    pipe = make_pipe()
    module = sys.modules[type(pipe).__module__]
    monkeypatch.setattr(module, "gtypes", types.SimpleNamespace(ThinkingLevel=OldLevel))
    assert pipe._thinking_level_member("xhigh") is OldLevel.HIGH
    assert pipe._thinking_level_member("medium") is OldLevel.MEDIUM
    assert pipe._thinking_level_member("minimal") is OldLevel.MINIMAL


def test_build_generate_config():
    # criterion 17
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    user_valves = pipe.UserValves()
    body = {"max_tokens": 10**8, "stop": "END", "temperature": 0.2}
    config = pipe._build_generate_config(
        info=info, system_instruction=None, tools=None, tool_config=None,
        thinking=None, body=body, user_valves=user_valves,
    )
    assert config.max_output_tokens == info["max_output"]
    assert config.stop_sequences == ["END"]
    assert config.temperature is None
    assert config.automatic_function_calling is not None
    assert config.automatic_function_calling.disable is True
    assert config.store is False

    body = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
        }
    }
    config = pipe._build_generate_config(
        info=info, system_instruction=None, tools=None, tool_config=None,
        thinking=None, body=body, user_valves=user_valves,
    )
    assert config.response_mime_type == "application/json"
