"""Cost estimation from Google's published Gemini prices.

Stream section: compiled at module level; the template provides all imports.
Everything here is pure arithmetic over the price table and the per-call
usage dicts the pipe already records, so it degrades to "no cost" rather than
guessing whenever a model has no known rates.
"""
from __future__ import annotations

# USD per 1M tokens, read from Google's published Gemini pricing on 2026-11-05.
#
# GEMINI_PRICES maps a model pattern to the list of dated tiers for it. A
# pattern ending in * is a prefix match; anything else must match exactly.
# An exact match always wins over a prefix (so gemini-3.8-flash never falls
# onto a gemini-3.8-flash-* rate meant for another name), and of the prefix
# matches the longest one wins.
#
# Each tier:
#   from:          "YYYY-MM-DD", the date the tier starts
#   input:         USD per 1M input tokens
#   output:        USD per 1M output tokens (text + thoughts)
#   cache_read:    USD per 1M cached input tokens
#   long_threshold: prompt length above which the *_long rates apply
#   input_long / output_long / cache_read_long
#   image_output:  USD per 1M image output tokens

_ARGON_TIERS = [
    {"from": "2026-01-01", "input": 2.00, "output": 10.00, "cache_read": 0.10},
    {"from": "2026-12-01", "input": 4.00, "output": 20.00, "cache_read": 0.20},
]

_FLASH_3X_TIERS = [
    {"from": "2026-01-01", "input": 0.75, "output": 3.75, "cache_read": 0.075},
    {"from": "2027-01-01", "input": 1.50, "output": 7.50, "cache_read": 0.15},
]


def _single_tier(from_date, input_rate, output_rate, cache_rate, **extra):
    tier = {
        "from": from_date,
        "input": input_rate,
        "output": output_rate,
        "cache_read": cache_rate,
    }
    tier.update(extra)
    return [tier]


GEMINI_PRICES: dict = {
    "gemini-4-argon": _ARGON_TIERS,
    "gemini-4-argon*": _ARGON_TIERS,
    "gemini-3.8-flash": _FLASH_3X_TIERS,
    "gemini-3.8-flash*": _FLASH_3X_TIERS,
    "gemini-3.6-flash": _FLASH_3X_TIERS,
    "gemini-3.6-flash*": _FLASH_3X_TIERS,
    "gemini-3.5-flash-lite": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-3.5-flash-lite*": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": _single_tier("2026-01-01", 0.25, 1.50, 0.025),
    "gemini-3.1-flash-lite*": _single_tier("2026-01-01", 0.25, 1.50, 0.025),
    "gemini-3.1-pro-preview": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.20,
        long_threshold=200_000, input_long=4.00, output_long=18.00,
        cache_read_long=0.40,
    ),
    "gemini-3.1-pro-preview*": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.20,
        long_threshold=200_000, input_long=4.00, output_long=18.00,
        cache_read_long=0.40,
    ),
    "gemini-3-flash-preview": _single_tier("2026-01-01", 0.50, 3.00, 0.05),
    "gemini-3-flash-preview*": _single_tier("2026-01-01", 0.50, 3.00, 0.05),
    "gemini-2.5-pro": _single_tier(
        "2026-01-01", 1.25, 10.00, 0.125,
        long_threshold=200_000, input_long=2.50, output_long=15.00,
        cache_read_long=0.25,
    ),
    "gemini-2.5-pro*": _single_tier(
        "2026-01-01", 1.25, 10.00, 0.125,
        long_threshold=200_000, input_long=2.50, output_long=15.00,
        cache_read_long=0.25,
    ),
    "gemini-2.5-flash": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-2.5-flash*": _single_tier("2026-01-01", 0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": _single_tier("2026-01-01", 0.10, 0.40, 0.01),
    "gemini-2.5-flash-lite*": _single_tier("2026-01-01", 0.10, 0.40, 0.01),
    "gemini-nano-banana-2.1": _single_tier(
        "2026-01-01", 1.50, 7.50, 0.0, image_output=30.00
    ),
    "gemini-nano-banana-2.1*": _single_tier(
        "2026-01-01", 1.50, 7.50, 0.0, image_output=30.00
    ),
    "gemini-3.1-flash-lite-image": _single_tier(
        "2026-01-01", 0.25, 1.50, 0.0, image_output=30.00
    ),
    "gemini-3.1-flash-lite-image*": _single_tier(
        "2026-01-01", 0.25, 1.50, 0.0, image_output=30.00
    ),
    "gemini-3-pro-image": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.0, image_output=120.00
    ),
    "gemini-3-pro-image*": _single_tier(
        "2026-01-01", 2.00, 12.00, 0.0, image_output=120.00
    ),
}

# Grounding is billed per 1,000 uses. These are the list prices BEFORE
# Google's free monthly allowance (5,000 on Gemini 3+), so the estimate is
# an upper bound for accounts inside the allowance.
SEARCH_PRICE_PER_1K = {"gen3": 14.0, "gen25": 35.0}
MAPS_PRICE_PER_1K = {"gen3": 14.0, "gen25": 25.0}


def _matching_pattern(table: dict, model: str):
    """The table's pattern for ``model``: an exact match, else the longest
    matching ``*`` prefix, else None."""
    if model in table:
        return model
    best = None
    for pattern in table:
        if pattern.endswith("*") and model.startswith(pattern[:-1]):
            if best is None or len(pattern) > len(best):
                best = pattern
    return best


def _usage_field(entry, name):
    """Read one usage-detail entry's field; entries may be dicts or SDK objects."""
    if isinstance(entry, dict):
        head, *rest = name.split("_")
        camel = head + "".join(part[:1].upper() + part[1:] for part in rest)
        if name in entry:
            return entry[name]
        return entry.get(camel)
    value = getattr(entry, name, None)
    if value is None:
        head, *rest = name.split("_")
        camel = head + "".join(part[:1].upper() + part[1:] for part in rest)
        value = getattr(entry, camel, None)
    return value


class GeminiPricing:
    """Looks up rates and computes the estimated USD cost of a turn's calls."""

    def rates_for(self, model, on_date, overrides=None):
        """The latest tier in effect for ``model`` on ``on_date``, or None.

        ``overrides`` is {pattern: {rate key: value}} (the parsed
        MODEL_PRICING_OVERRIDES valve) and is merged over the chosen tier,
        with the same exact-match-then-longest-prefix rule as the table.
        """
        if not model:
            return None
        model = str(model)
        pattern = _matching_pattern(GEMINI_PRICES, model)
        if pattern is None:
            return None
        on = on_date.isoformat()
        tier = None
        for candidate in GEMINI_PRICES[pattern]:
            if candidate["from"] <= on and (tier is None or candidate["from"] > tier["from"]):
                tier = candidate
        if tier is None:
            return None
        rates = dict(tier)
        if overrides:
            override_pattern = _matching_pattern(overrides, model)
            if override_pattern is not None and isinstance(overrides[override_pattern], dict):
                rates.update(overrides[override_pattern])
        return rates

    def call_cost(self, rates, usage) -> float:
        """The estimated USD cost of one API call's usage dict (snake_case)."""
        usage = usage or {}
        prompt = int(usage.get("prompt_token_count") or 0)
        cached = int(usage.get("cached_content_token_count") or 0)
        tool_prompt = int(usage.get("tool_use_prompt_token_count") or 0)
        candidates = int(usage.get("candidates_token_count") or 0)
        thoughts = int(usage.get("thoughts_token_count") or 0)
        image_tokens = 0
        for entry in usage.get("candidates_tokens_details") or []:
            modality = _usage_field(entry, "modality")
            if getattr(modality, "name", modality) == "IMAGE":
                image_tokens += int(_usage_field(entry, "token_count") or 0)
        out_text = candidates - image_tokens + thoughts
        threshold = rates.get("long_threshold")
        if threshold is not None and prompt > int(threshold):
            input_rate = rates.get("input_long", rates.get("input", 0.0))
            output_rate = rates.get("output_long", rates.get("output", 0.0))
            cache_rate = rates.get("cache_read_long", rates.get("cache_read", 0.0))
        else:
            input_rate = rates.get("input", 0.0)
            output_rate = rates.get("output", 0.0)
            cache_rate = rates.get("cache_read", 0.0)
        image_rate = rates.get("image_output", 0.0)
        return (
            (prompt - cached + tool_prompt) * input_rate
            + cached * cache_rate
            + out_text * output_rate
            + image_tokens * image_rate
        ) / 1e6


GEMINI_PRICING = GeminiPricing()


def format_usd(cost: float) -> str:
    """Format a cost so sub-cent turns stay legible without padding dollar turns."""
    if cost >= 1:
        return f"${cost:.2f}"
    if cost >= 0.01:
        return f"${cost:.3f}"
    return f"${cost:.4f}"
