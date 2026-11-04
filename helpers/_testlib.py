"""Shared helpers for the pytest checks in helpers/.

Tests run against the built artifact (gemini_pipe.py), so build first:
    .venv/bin/python helpers/build_gemini_pipe.py
"""
from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE_FILE = REPO_ROOT / "gemini_pipe.py"


def load_pipe_module():
    """Import the built gemini_pipe.py fresh under a unique module name.

    A fresh module means a fresh `class Pipe`, so class-level caches never
    leak between tests.
    """
    if not PIPE_FILE.exists():
        raise AssertionError(
            f"{PIPE_FILE} is missing; run helpers/build_gemini_pipe.py first"
        )
    name = f"gemini_pipe_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, PIPE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_pipe(**valves):
    """A fresh Pipe instance whose admin valves are set from ``valves``."""
    module = load_pipe_module()
    pipe = module.Pipe()
    if valves:
        pipe.valves = pipe.Valves(**valves)
    return pipe


def fake_model(name, display_name=None, supported_actions=("generateContent",),
               input_token_limit=None, output_token_limit=None, thinking=None):
    """A real google.genai types.Model; the pipe reads it via model_dump()."""
    from google.genai import types as gtypes

    kwargs = {
        "name": name if name.startswith("models/") else f"models/{name}",
        "display_name": display_name,
        "supported_actions": list(supported_actions),
        "input_token_limit": input_token_limit,
        "output_token_limit": output_token_limit,
        "thinking": thinking,
    }
    return gtypes.Model(**{k: v for k, v in kwargs.items() if v is not None})


class FakeClient:
    """Stands in for google.genai.Client (wired by patching _build_genai_client).

    ``aio.models.list`` is awaitable and returns an async iterator. With no
    scripts it always iterates ``models``; otherwise each call pops the next
    script item, where an item that is an Exception is raised instead. Every
    call is recorded in ``calls``.
    """

    def __init__(self, models=None, scripts=None):
        self.models = list(models or [])
        self.scripts = list(scripts or [])
        self.calls = []
        self.aio = SimpleNamespace(
            models=self._AioModels(self), aclose=self._aclose
        )

    @staticmethod
    async def _aclose():
        return None

    class _AioModels:
        def __init__(self, client):
            self._client = client

        async def list(self, config=None):
            self._client.calls.append(config)
            if self._client.scripts:
                item = self._client.scripts.pop(0)
            else:
                item = self._client.models
            if isinstance(item, BaseException):
                raise item
            return self._iter(list(item))

        @staticmethod
        async def _iter(items):
            for item in items:
                yield item
