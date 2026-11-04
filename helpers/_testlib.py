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


def pipe_module(pipe):
    """The built module the ``pipe`` instance was created from."""
    return sys.modules[type(pipe).__module__]


def make_ctx(pipe, model="gemini-4-argon", emitter=None, metadata=None, user_valves=None):
    """A PipeRequestContext for ``pipe`` via the built module's new_request_context."""
    return pipe_module(pipe).new_request_context(
        pipe,
        model=model,
        event_emitter=emitter,
        metadata=metadata,
        user_valves=user_valves,
    )


class FakeEmitter:
    """An async event sink: each emitted event is appended to ``.events``."""

    def __init__(self) -> None:
        self.events = []

    async def __call__(self, event: dict) -> None:
        self.events.append(event)

    def of_type(self, t: str) -> list:
        return [e for e in self.events if e.get("type") == t]


def chunk(d: dict):
    """A GenerateContentResponse from a plain dict (the SDK accepts camelCase)."""
    from google.genai import types as gtypes

    return gtypes.GenerateContentResponse.model_validate(d)


class FakeClient:
    """Stands in for google.genai.Client (wired by patching _build_genai_client).

    ``aio.models.list`` is awaitable and returns an async iterator. With no
    scripts it always iterates ``models``; otherwise each call pops the next
    script item, where an item that is an Exception is raised instead. Every
    list call is recorded in ``calls``.

    ``aio.models.generate_content_stream(**kw)`` records ``kw`` in
    ``stream_calls`` and returns an async iterator over the next script: each
    script is a list whose items are chunks, or an Exception (raised when the
    iterator reaches it). ``aio.models.generate_content(**kw)`` records ``kw``
    in ``generate_calls`` and returns the next item of ``responses`` (a
    GenerateContentResponse), or raises it when it is an Exception.
    ``aio.aclose`` is a no-op.
    """

    def __init__(self, models=None, scripts=None, responses=None):
        self.models = list(models or [])
        self.scripts = list(scripts or [])
        self.responses = list(responses or [])
        self.calls = []
        self.stream_calls = []
        self.generate_calls = []
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

        async def generate_content_stream(self, **kwargs):
            self._client.stream_calls.append(kwargs)
            if not self._client.scripts:
                raise AssertionError("no script left for generate_content_stream")
            return self._iter(self._client.scripts.pop(0))

        async def generate_content(self, **kwargs):
            self._client.generate_calls.append(kwargs)
            if not self._client.responses:
                raise AssertionError("no response left for generate_content")
            item = self._client.responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        @staticmethod
        async def _iter(items):
            for item in items:
                if isinstance(item, BaseException):
                    raise item
                yield item
