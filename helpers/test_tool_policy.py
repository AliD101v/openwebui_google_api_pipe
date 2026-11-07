"""Checks for the built-in tool assembly policy (SPEC section 8.1), criteria 12-15."""
from __future__ import annotations

from google.genai import types as gtypes

from _testlib import make_pipe


def _decl(name="my_tool"):
    return gtypes.FunctionDeclaration(name=name, description="a tool")


def test_tool_order_argon_builtin_with_functions():
    # criterion 12
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    flags = pipe._resolve_builtin_flags({"gemini_web_search": True}, pipe.valves, None, info, "")
    tools, tool_config, notes = pipe._assemble_tools(info, flags, [_decl()])
    assert len(tools) == 3
    assert tools[0].function_declarations is not None
    assert tools[0].function_declarations[0].name == "my_tool"
    assert tools[1].google_search is not None
    assert tools[2].url_context is not None
    assert tool_config.include_server_side_tool_invocations is True
    assert tool_config.function_calling_config.mode == gtypes.FunctionCallingConfigMode.VALIDATED
    assert notes == []


def test_tool_conflict_gemini_25_flash():
    # criterion 13
    info_args = {"search": True, "url_context": True, "url_context_auto": False, "maps": False, "lat_lng": None}
    pipe_builtin = make_pipe()
    info = pipe_builtin.get_model_info("gemini-2.5-flash")
    tools, _tool_config, notes = pipe_builtin._assemble_tools(info, info_args, [_decl()])
    assert all(t.function_declarations is None for t in tools)
    assert len(notes) == 1
    assert "disabled" in notes[0]

    pipe_functions = make_pipe(BUILTIN_TOOL_CONFLICT="prefer_functions")
    tools, _tool_config, notes = pipe_functions._assemble_tools(info, info_args, [_decl()])
    assert len(tools) == 1
    assert tools[0].function_declarations is not None
    assert all(t.google_search is None and t.url_context is None for t in tools)
    assert len(notes) == 1
    assert "disabled" in notes[0]


def test_tool_assembly_further_cases():
    # criterion 14
    pipe = make_pipe()
    info25 = pipe.get_model_info("gemini-2.5-flash")
    auto_flags = {"search": False, "url_context": True, "url_context_auto": True, "maps": False, "lat_lng": None}
    tools, _tool_config, notes = pipe._assemble_tools(info25, auto_flags, [_decl()])
    assert any(t.function_declarations is not None for t in tools)
    assert all(t.url_context is None for t in tools)
    assert notes == []

    pipe31 = make_pipe(URL_CONTEXT="off")
    info31 = pipe31.get_model_info("gemini-3.1-pro-preview")
    map_search_flags = {"search": True, "url_context": False, "url_context_auto": False, "maps": True, "lat_lng": None}
    tools, _tool_config, notes = pipe31._assemble_tools(info31, map_search_flags, [])
    assert all(t.google_search is None for t in tools)
    assert any(t.google_maps is not None for t in tools)
    assert len(notes) == 1
    assert "search" in notes[0].lower()

    map_flags = {"search": False, "url_context": False, "url_context_auto": False, "maps": True, "lat_lng": (52.52, 13.4)}
    _tools, tool_config, _notes = pipe._assemble_tools(info25, map_flags, [])
    assert tool_config is not None
    assert tool_config.retrieval_config.lat_lng.latitude == 52.52
    assert tool_config.retrieval_config.lat_lng.longitude == 13.4

    no_location_flags = {"search": False, "url_context": False, "url_context_auto": False, "maps": True, "lat_lng": None}
    tools, tool_config, _notes = pipe._assemble_tools(info25, no_location_flags, [])
    assert any(t.google_maps is not None for t in tools)
    assert tool_config is None


def test_resolve_builtin_flags():
    # criterion 15
    pipe = make_pipe()
    info = pipe.get_model_info("gemini-4-argon")
    flags = pipe._resolve_builtin_flags({"web_search_enforced": True}, pipe.valves, None, info, "")
    assert flags["search"] is True

    pipe_maps = make_pipe(MAPS_DEFAULT=True)
    located = pipe_maps._resolve_builtin_flags(
        {"variables": {"{{USER_LOCATION}}": "52.52, 13.40"}}, pipe_maps.valves, None, info, ""
    )
    assert located["lat_lng"] == (52.52, 13.4)
    unparsable = pipe_maps._resolve_builtin_flags(
        {"variables": {"{{USER_LOCATION}}": "abc"}}, pipe_maps.valves, None, info, ""
    )
    assert unparsable["lat_lng"] is None

    with_url = pipe._resolve_builtin_flags({}, pipe.valves, None, info, "check https://example.com/x please")
    assert with_url["url_context"] is True
    assert with_url["url_context_auto"] is True
    with_youtube = pipe._resolve_builtin_flags({}, pipe.valves, None, info, "watch https://youtube.com/watch?v=abc")
    assert with_youtube["url_context"] is False


def test_maps_and_search_are_never_sent_together():
    # generateContent answers 400 INVALID_ARGUMENT when google_maps and google_search share a request,
    # even on the newest models: Maps wins and Search is dropped with a note.
    for model in ("gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash"):
        for valves, metadata in (
            ({}, {"gemini_web_search": True, "gemini_maps": True}),
            ({"WEB_SEARCH_DEFAULT": True}, {"gemini_maps": True}),
        ):
            pipe = make_pipe(**valves)
            info = pipe.get_model_info(model)
            flags = pipe._resolve_builtin_flags(metadata, pipe.valves, None, info, "")
            tools, _tool_config, notes = pipe._assemble_tools(info, flags, [])
            assert any(t.google_maps is not None for t in tools), (model, valves)
            assert all(t.google_search is None for t in tools), (model, valves)
            assert any("Google Search disabled" in n for n in notes), (model, valves)


def test_maps_with_search_override_restores_the_combination():
    pipe = make_pipe(MODEL_CAPABILITY_OVERRIDES='{"gemini-4*": {"maps_with_search": true}}')
    info = pipe.get_model_info("gemini-4-argon")
    flags = pipe._resolve_builtin_flags(
        {"gemini_web_search": True, "gemini_maps": True}, pipe.valves, None, info, ""
    )
    tools, _tool_config, notes = pipe._assemble_tools(info, flags, [])
    assert any(t.google_maps is not None for t in tools)
    assert any(t.google_search is not None for t in tools)
    assert notes == []


def test_maps_excludes_url_context():
    # generateContent also answers 400 for google_maps + url_context. URL context is switched on by Search
    # ("auto"/"with_search"), by "always", or by a URL in the message; with Maps on, none of them may reach the request.
    cases = (
        ({}, {"gemini_web_search": True, "gemini_maps": True}, ""),
        ({"WEB_SEARCH_DEFAULT": True}, {"gemini_maps": True}, ""),
        ({"URL_CONTEXT": "always"}, {"gemini_maps": True}, ""),
        ({}, {"gemini_maps": True}, "is https://example.com/cafes any good?"),
    )
    for model in ("gemini-4-argon", "gemini-3.8-flash", "gemini-2.5-flash"):
        for valves, metadata, text in cases:
            pipe = make_pipe(**valves)
            info = pipe.get_model_info(model)
            flags = pipe._resolve_builtin_flags(metadata, pipe.valves, None, info, text)
            tools, _tool_config, notes = pipe._assemble_tools(info, flags, [])
            assert any(t.google_maps is not None for t in tools), (model, valves, text)
            assert all(t.google_search is None for t in tools), (model, valves, text)
            assert all(t.url_context is None for t in tools), (model, valves, text)
            assert any("URL context disabled" in n for n in notes), (model, valves, text)


def test_maps_with_url_context_override_restores_the_combination():
    pipe = make_pipe(MODEL_CAPABILITY_OVERRIDES='{"gemini-4*": {"maps_with_url_context": true}}')
    info = pipe.get_model_info("gemini-4-argon")
    flags = pipe._resolve_builtin_flags({"gemini_maps": True}, pipe.valves, None, info, "see https://example.com")
    tools, _tool_config, notes = pipe._assemble_tools(info, flags, [])
    assert any(t.google_maps is not None for t in tools)
    assert any(t.url_context is not None for t in tools)
    assert notes == []
