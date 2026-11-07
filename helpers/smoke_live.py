"""Live smoke test of the built pipe against the real Gemini API.

Requires GEMINI_API_KEY in the environment; without it, prints "skipped" and
exits 0 so CI can run it unconditionally.

Scenarios:
  text   plain generation
  search Google Search grounding
  tool   an OpenWebUI function tool the pipe must call and fold back in
  image  Nano Banana image generation (saved under .smoke_out/)
  maps   Google Maps grounding (a place near the valve coordinates)
  all    all five, in order
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
SMOKE_OUT = REPO_ROOT / ".smoke_out"
IMAGE_MODEL = "gemini-nano-banana-2.1"
SCENARIOS = ("text", "search", "tool", "image", "maps")


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


def _prompt_for(scenario):
    if scenario == "search":
        return ("What were the main world news headlines today? Search the web and "
                "answer in three short bullet points.")
    if scenario == "maps":
        return ("Which coffee shop should I visit right now? Find one nearby using "
                "Google Maps and answer in one sentence.")
    if scenario == "tool":
        return "Use the add tool to work out 19 + 23, then state the sum."
    if scenario == "image":
        return "Draw a cute cat in a tiny astronaut helmet. Just the image."
    return "What is the capital of France? Answer in one sentence."


def _add_tool(calls):
    """One function tool (add two integers); ``calls`` counts its invocations."""

    def add(a, b):
        calls["n"] += 1
        return {"sum": a + b}

    return {
        "add": {
            "callable": add,
            "spec": {
                "description": "Add two integers and return their sum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    }


SAVED_IMAGES: list = []


async def _store_generated_image_to_smoke_dir(ctx, data, mime):
    """Stand-in for OpenWebUI's file store: write under .smoke_out/ and hand
    back the local path so the image still renders inline."""
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    extension = ".png" if "png" in mime else (".jpg" if "jpeg" in mime else ".img")
    path = SMOKE_OUT / f"smoke-image{extension}"
    path.write_bytes(data)
    SAVED_IMAGES.append(path)
    print(f"  [image saved] {path.name} ({len(data)} bytes)", flush=True)
    return str(path)


def _scenario_ok(scenario, result, emitter, calls):
    result = str(result)
    if scenario == "text":
        return bool(result.strip())
    if scenario == "search":
        return emitter.source_count >= 1 and "[1]" in result
    if scenario == "maps":
        return emitter.source_count >= 1 and "Google Maps" in result
    if scenario == "tool":
        return calls["n"] >= 1 and bool(result.strip())
    if scenario == "image":
        return bool(SAVED_IMAGES) and "![Generated image]" in result
    return False


def _run_scenario(pipe, scenario: str, model: str) -> bool:
    metadata = {"chat_id": "smoke", "message_id": "smoke"}
    user_valves = pipe.UserValves()
    tools = None
    calls = {"n": 0}
    if scenario == "search":
        metadata["gemini_web_search"] = True
    elif scenario == "maps":
        metadata["gemini_maps"] = True
        user_valves = pipe.UserValves(MAPS_LATITUDE="52.52", MAPS_LONGITUDE="13.40")
    elif scenario == "tool":
        tools = _add_tool(calls)

    body = {
        "model": f"gemini_pipe.gemini/{model}",
        "messages": [{"role": "user", "content": _prompt_for(scenario)}],
        "stream": True,
    }
    emitter = _PrinterEmitter()
    print(f"--- scenario: {scenario} (model {model}) ---", flush=True)
    kwargs = {"__tools__": tools} if tools else {}
    result = asyncio.run(
        pipe.pipe(
            body,
            __user__={"id": "smoke", "valves": user_valves},
            __event_emitter__=emitter,
            __metadata__=metadata,
            **kwargs,
        )
    )
    print(f"--- answer ({len(result)} chars) ---", flush=True)
    print(result, flush=True)
    return _scenario_ok(scenario, result, emitter, calls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=SCENARIOS + ("all",), default="text")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--image-model", default=IMAGE_MODEL)
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("skipped: GEMINI_API_KEY is not set")
        return 0
    if not os.environ["GEMINI_API_KEY"].isascii():
        print("error: GEMINI_API_KEY contains non-ASCII characters (a placeholder such as '…' left in the command?)")
        return 2

    module = _load_pipe()
    pipe = module.Pipe()
    pipe.valves = pipe.Valves(GEMINI_API_KEY=os.environ["GEMINI_API_KEY"])
    pipe._store_generated_image = _store_generated_image_to_smoke_dir

    if args.scenario == "all":
        scenarios = [
            (name, args.image_model if name == "image" else args.model)
            for name in SCENARIOS
        ]
    else:
        model = args.image_model if args.scenario == "image" else args.model
        scenarios = [(args.scenario, model)]

    failures = [name for name, model in scenarios if not _run_scenario(pipe, name, model)]
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
