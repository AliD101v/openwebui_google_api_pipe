"""Checks for the toggle and companion filters (SPEC section 3), criteria 16-17."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from _testlib import REPO_ROOT


def _load_filter(path: Path):
    name = f"filter_under_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_toggles_set_metadata_flags():
    # criterion 16
    web = _load_filter(REPO_ROOT / "gemini_pipe_web_search_toggle.py")
    f = web.Filter()

    metadata = {}
    asyncio.run(f.inlet({}, __metadata__=metadata))
    assert metadata["gemini_web_search"] is True

    body = {}
    asyncio.run(f.inlet(body, __metadata__=None))
    assert body["metadata"]["gemini_web_search"] is True

    maps = _load_filter(REPO_ROOT / "gemini_pipe_maps_toggle.py")
    f = maps.Filter()

    metadata = {}
    asyncio.run(f.inlet({}, __metadata__=metadata))
    assert metadata["gemini_maps"] is True

    body = {}
    asyncio.run(f.inlet(body, __metadata__=None))
    assert body["metadata"]["gemini_maps"] is True


def test_companion_filter_routes_web_search():
    # criterion 17
    comp = _load_filter(REPO_ROOT / "gemini_manifold_companion_filter.py")
    f = comp.Filter()

    body = {"model": "gemini_pipe.gemini/gemini-4-argon", "features": {"web_search": True}}
    metadata = {}
    asyncio.run(f.inlet(body, __metadata__=metadata))
    assert body["features"]["web_search"] is False
    assert metadata["gemini_web_search"] is True

    body = {"model": "gpt-4o", "features": {"web_search": True}}
    metadata = {}
    asyncio.run(f.inlet(body, __metadata__=metadata))
    assert body == {"model": "gpt-4o", "features": {"web_search": True}}
    assert metadata == {}


def test_native_pdf_toggle_sets_metadata_flag():
    toggle = _load_filter(REPO_ROOT / "gemini_pipe_native_pdf_toggle.py")
    f = toggle.Filter()
    assert f.toggle is True

    metadata = {}
    asyncio.run(f.inlet({}, __metadata__=metadata))
    assert metadata["gemini_native_pdf"] is True

    body = {}
    asyncio.run(f.inlet(body, __metadata__=None))
    assert body["metadata"]["gemini_native_pdf"] is True
    assert not getattr(f, "file_handler", False)  # OpenWebUI's RAG must still handle the non-native files
