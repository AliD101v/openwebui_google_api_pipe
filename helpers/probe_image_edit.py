"""Which history shape makes an image model edit its earlier image? Needs a real key.

    GEMINI_API_KEY=YOUR_API_KEY .venv/bin/python helpers/probe_image_edit.py [--model ID ...]

Turn 1 generates a picture. Turn 2 asks for a small change, sent four ways:
  A  the pipe's current shape: the earlier image in a *model* turn with the dummy thought signature
  B  the earlier image attached to the new *user* message
  C  the model's own parts replayed with their real signatures (what a stateful client sends)
  D  control: no image at all (only the earlier text), i.e. what a redraw looks like
Each result is saved under .smoke_out/ and compared with turn 1 on a 32x32 grayscale grid:
a small difference means the image was edited, a large one that a new picture was drawn.
Costs five image generations per model. Without a key it prints "skipped" and exits 0.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as gtypes
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / ".smoke_out"
DUMMY_SIGNATURE = b"skip_thought_signature_validator"
PROMPT_1 = "Draw a flat illustration of a red bicycle leaning against a yellow wall, white background elsewhere."
PROMPT_2 = "Keep everything exactly the same, but add one small blue bird sitting on the handlebar."


def image_of(response):
    """(bytes, mime) of the last non-thought image part of a response, or None."""
    found = None
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if part.inline_data and part.inline_data.data and not part.thought:
                found = (part.inline_data.data, part.inline_data.mime_type or "image/png")
    return found


def text_of(response) -> str:
    texts = []
    for candidate in response.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if part.text and not part.thought:
                texts.append(part.text)
    return " ".join(texts).strip() or "Here you go."


def grid(data: bytes) -> list:
    img = Image.open(io.BytesIO(data)).convert("L").resize((32, 32))
    return list(img.tobytes())


def difference(a: bytes, b: bytes) -> float:
    ga, gb = grid(a), grid(b)
    return sum(abs(x - y) for x, y in zip(ga, gb)) / (len(ga) * 255)


def user(*parts):
    return gtypes.Content(role="user", parts=list(parts))


def model(*parts):
    return gtypes.Content(role="model", parts=list(parts))


def probe(client, name: str) -> None:
    config = gtypes.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    print(f"\n{name}")
    first = client.models.generate_content(model=name, contents=PROMPT_1, config=config)
    picture = image_of(first)
    if not picture:
        print("  turn 1 returned no image; stopping")
        return
    data, mime = picture
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe-turn1.png").write_bytes(data)
    reply = text_of(first)
    real_parts = list(first.candidates[0].content.parts)
    ask1, ask2 = user(gtypes.Part(text=PROMPT_1)), gtypes.Part(text=PROMPT_2)
    image_part = gtypes.Part.from_bytes(data=data, mime_type=mime)
    variants = {
        "A model turn + dummy signature": [
            ask1,
            model(image_part.model_copy(update={"thought_signature": DUMMY_SIGNATURE}), gtypes.Part(text=reply)),
            user(ask2),
        ],
        "B image in the new user message": [ask1, model(gtypes.Part(text=reply)), user(image_part, ask2)],
        "C model's own parts, real signatures": [ask1, model(*real_parts), user(ask2)],
        "D no image (control: a redraw)": [ask1, model(gtypes.Part(text=reply)), user(ask2)],
    }
    for label, contents in variants.items():
        try:
            result = client.models.generate_content(model=name, contents=contents, config=config)
        except genai_errors.APIError as e:
            print(f"  {label:<40} REJECTED: {str(getattr(e, 'message', '') or e)[:140]}")
            continue
        out = image_of(result)
        if not out:
            print(f"  {label:<40} no image returned")
            continue
        path = OUT / f"probe-{label[0]}.png"
        path.write_bytes(out[0])
        print(f"  {label:<40} difference from turn 1: {difference(data, out[0]):.3f}  ({path.name})")
    print("  small (about 0.10 or less) = edited the picture; large = drew a new one. Compare D with the others.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="image model id, repeatable; default gemini-nano-banana-2.1")
    args = parser.parse_args()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("skipped: set GEMINI_API_KEY to probe the live API")
        return 0
    if not key.isascii():
        print("error: the API key contains non-ASCII characters (a placeholder such as '…' left in the command?)")
        return 2
    client = genai.Client(api_key=key, vertexai=False)
    for name in args.model or ["gemini-nano-banana-2.1"]:
        try:
            probe(client, name)
        except genai_errors.APIError as e:
            print(f"  {name}: REJECTED: {str(getattr(e, 'message', '') or e)[:160]}")
        except Exception as e:  # noqa: BLE001 - not an API verdict (network, key encoding): stop
            print(f"error: {type(e).__name__}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
