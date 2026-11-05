"""Checks for the cost estimate: the dated price table, the per-call cost
math, and the pipe-level status line and usage (Part A criteria 1-4)."""
from __future__ import annotations

import asyncio
from datetime import date

from _testlib import FakeClient, FakeEmitter, chunk, load_pipe_module, make_pipe

BODY = {
    "model": "gemini_pipe.gemini/gemini-4-argon",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": True,
}
METADATA = {"chat_id": "c1", "message_id": "m1"}


def pricing():
    return load_pipe_module().GeminiPricing()


def test_rates_for_model_and_dates():
    # criterion 1
    prices = pricing()
    assert prices.rates_for("gemini-4-argon", date(2026, 11, 5))["input"] == 2.0
    assert prices.rates_for("gemini-4-argon", date(2026, 12, 1))["input"] == 4.0
    assert prices.rates_for("gemini-unknown-9", date(2026, 11, 5)) is None


def test_call_cost_argon_mixed_tokens():
    # criterion 2
    prices = pricing()
    rates = prices.rates_for("gemini-4-argon", date(2026, 11, 5))
    cost = prices.call_cost(
        rates,
        {
            "prompt_token_count": 1_000_000,
            "cached_content_token_count": 500_000,
            "candidates_token_count": 100_000,
            "thoughts_token_count": 100_000,
        },
    )
    assert abs(cost - (0.5 * 2 + 0.5 * 0.10 + 0.2 * 10)) < 1e-9


def test_call_cost_image_output_and_long_context():
    # criterion 3
    prices = pricing()
    rates = prices.rates_for("gemini-nano-banana-2.1", date(2026, 11, 5))
    cost = prices.call_cost(
        rates,
        {
            "candidates_token_count": 2000,
            "candidates_tokens_details": [
                {"modality": "IMAGE", "token_count": 1120},
                {"modality": "TEXT", "token_count": 880},
            ],
        },
    )
    assert abs(cost - (880 * 7.5 + 1120 * 30) / 1e6) < 1e-9

    rates = prices.rates_for("gemini-3.1-pro-preview", date(2026, 11, 5))
    cost = prices.call_cost(
        rates, {"prompt_token_count": 300_000, "candidates_token_count": 10_000}
    )
    assert abs(cost - (300_000 * 4.00 + 10_000 * 18.00) / 1e6) < 1e-9


def _run(pipe, model, user_valves, usage):
    """One full pipe() run against a single scripted stream chunk."""
    pipe._build_genai_client = lambda api_key, timeout_s=None: FakeClient(
        scripts=[
            [
                chunk(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Hi there"}],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": usage,
                    }
                )
            ]
        ]
    )
    user = {"id": "u1", "name": "T", "valves": user_valves}
    emitter = FakeEmitter()
    body = dict(BODY)
    body["model"] = f"gemini_pipe.gemini/{model}"
    asyncio.run(
        pipe.pipe(
            body,
            __user__=user,
            __event_emitter__=emitter,
            __metadata__=dict(METADATA),
        )
    )
    return emitter


def test_pipe_status_and_usage_cost():
    # criterion 4
    pipe = make_pipe(GEMINI_API_KEY="k")
    usage = {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 100,
        "thoughtsTokenCount": 0,
    }
    expected = (1000 * 2.0 + 100 * 10.0) / 1e6

    emitter = _run(pipe, "gemini-4-argon", pipe.UserValves(), usage)
    final = emitter.of_type("status")[-1]["data"]["description"]
    assert "💵 ≈$" in final
    completion = emitter.of_type("chat:completion")[-1]["data"]["usage"]
    assert abs(completion["cost_usd"] - expected) < 1e-9

    emitter = _run(pipe, "gemini-4-argon", pipe.UserValves(SHOW_COST=False), usage)
    assert "💵" not in emitter.of_type("status")[-1]["data"]["description"]
    assert "cost_usd" not in emitter.of_type("chat:completion")[-1]["data"]["usage"]

    emitter = _run(pipe, "gemini-unknown-9", pipe.UserValves(), usage)
    assert "💵" not in emitter.of_type("status")[-1]["data"]["description"]
    assert "cost_usd" not in emitter.of_type("chat:completion")[-1]["data"]["usage"]
