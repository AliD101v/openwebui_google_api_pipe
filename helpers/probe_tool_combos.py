"""Which tool combinations does generateContent accept? Needs a real key.

    GEMINI_API_KEY=YOUR_API_KEY .venv/bin/python helpers/probe_tool_combos.py [--model gemini-3.6-flash ...]

One tiny request per combination ("reply ok"), so the model won't call a tool and the cost is negligible.
Prints ok or the API's rejection. Without a key it prints "skipped" and exits 0.
"""
from __future__ import annotations

import argparse
import os
import sys

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as gtypes

BUILTINS = {
    "search": lambda: gtypes.Tool(google_search=gtypes.GoogleSearch()),
    "url": lambda: gtypes.Tool(url_context=gtypes.UrlContext()),
    "maps": lambda: gtypes.Tool(google_maps=gtypes.GoogleMaps()),
}
# (builtins, with function tool, how the function side is configured)
COMBOS = [
    (("maps",), False, ""),
    (("search",), False, ""),
    (("url",), False, ""),
    (("maps", "search"), False, ""),
    (("maps", "url"), False, ""),
    (("search", "url"), False, ""),
    (("maps", "search", "url"), False, ""),
    (("maps",), True, "default"),
    (("maps",), True, "validated"),
    (("search",), True, "validated"),
    (("search", "url"), True, "validated"),
    (("url",), True, "validated"),
]


def build_config(builtins, with_function, mode):
    tools = []
    if with_function:
        decl = gtypes.FunctionDeclaration(
            name="noop", description="does nothing", parameters_json_schema={"type": "object", "properties": {}}
        )
        tools.append(gtypes.Tool(function_declarations=[decl]))
    tools += [BUILTINS[name]() for name in builtins]
    tool_config = None
    if with_function and mode == "validated":
        kwargs = {"function_calling_config": gtypes.FunctionCallingConfig(mode="VALIDATED")}
        if "include_server_side_tool_invocations" in gtypes.ToolConfig.model_fields:
            kwargs["include_server_side_tool_invocations"] = True
        tool_config = gtypes.ToolConfig(**kwargs)
    return gtypes.GenerateContentConfig(
        tools=tools, tool_config=tool_config, max_output_tokens=64,
        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="repeatable; default gemini-3.6-flash")
    args = parser.parse_args()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("skipped: set GEMINI_API_KEY to probe the live API")
        return 0
    if not key.isascii():
        print("error: the API key contains non-ASCII characters (a placeholder such as '…' left in the command?)")
        return 2
    client = genai.Client(api_key=key, vertexai=False)
    for model in args.model or ["gemini-3.6-flash"]:
        print(f"\n{model}")
        for builtins, with_function, mode in COMBOS:
            label = "+".join(builtins) + (f" + function({mode})" if with_function else "")
            try:
                client.models.generate_content(
                    model=model, contents="Reply with the single word ok.",
                    config=build_config(builtins, with_function, mode),
                )
                verdict = "ok"
            except genai_errors.APIError as e:
                text = str(getattr(e, "message", "") or e).replace("\n", " ")
                verdict = "REJECTED: " + text[:160]
            except Exception as e:  # noqa: BLE001 - not an API verdict (network, key encoding): stop, don't repeat it
                print(f"error: {type(e).__name__}: {e}")
                return 1
            print(f"  {label:<36} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
