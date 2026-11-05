"""Live smoke test of the built pipe against the real Gemini API.

Requires GEMINI_API_KEY in the environment; without it, prints "skipped" and
exits 0 so CI can run it unconditionally.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE_FILE = REPO_ROOT / "gemini_pipe.py"


def _load_pipe():
    spec = importlib.util.spec_from_file_location("gemini_pipe", PIPE_FILE)
    module = importlib.util.module_from_spec(spec)
    # OpenWebUI registers plugin modules before executing them; dataclasses
    # with string annotations look their module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _PrinterEmitter:
    """One line per event: the type plus a short summary."""

    def __init__(self):
        self.source_count = 0

    async def __call__(self, event):
        etype = event.get("type")
        data = event.get("data") or {}
        if etype == "source":
            self.source_count += 1
            source = data.get("source") or {}
            print(f"  [source {self.source_count}] {source.get('name') or source.get('url')}", flush=True)
        elif etype == "status":
            print(f"  [status] {data.get('description')}", flush=True)
        elif etype == "message":
            print(f"  [message +{len(data.get('content') or '')} chars]", flush=True)
        elif etype == "replace":
            print(f"  [replace {len(data.get('content') or '')} chars]", flush=True)
        elif etype == "notification":
            print(f"  [notification] {data.get('content')}", flush=True)
        elif etype == "chat:completion":
            print("  [chat:completion done]", flush=True)
        else:
            print(f"  [{etype}]", flush=True)


def _run_scenario(pipe, scenario: str, model: str) -> bool:
    metadata = {"chat_id": "smoke", "message_id": "smoke"}
    if scenario == "search":
        metadata["gemini_web_search"] = True
    body = {
        "model": f"gemini_pipe.gemini/{model}",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one sentence.",
            }
        ],
        "stream": True,
    }
    emitter = _PrinterEmitter()
    print(f"--- scenario: {scenario} (model {model}) ---", flush=True)
    result = asyncio.run(
        pipe.pipe(
            body,
            __user__={"id": "smoke", "valves": pipe.UserValves()},
            __event_emitter__=emitter,
            __metadata__=metadata,
        )
    )
    print(f"--- answer ({len(result)} chars) ---", flush=True)
    print(result, flush=True)
    if scenario == "text":
        return bool(str(result).strip())
    return emitter.source_count >= 1 and "[1]" in result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=("text", "search", "all"), default="text")
    parser.add_argument("--model", default="gemini-3.8-flash")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("skipped: GEMINI_API_KEY is not set")
        return 0

    module = _load_pipe()
    pipe = module.Pipe()
    pipe.valves = pipe.Valves(GEMINI_API_KEY=os.environ["GEMINI_API_KEY"])

    scenarios = ["text", "search"] if args.scenario == "all" else [args.scenario]
    failures = [s for s in scenarios if not _run_scenario(pipe, s, args.model)]
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
