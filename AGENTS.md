# Gemini API Manifold Pipe — agent notes

## Scope

One OpenWebUI manifold pipe for the native Gemini Developer API, plus its toggle/companion filters. Tasks touch model discovery and capabilities, request assembly (history, media, tools, config), the multi-call generation loop, grounding/citations, thinking and tool blocks, usage/cost, or the filters.

## Golden rule

`gemini_pipe.py` and `gemini_pipe.min.py` are **generated** (both committed). Edit `src/`, then rebuild. Hand-edits to the generated sections are lost on the next build.

## Build and test

| Need | Command (from repo root) |
|---|---|
| Compile the single-file artifact (+ minified) | `.venv/bin/python helpers/build_gemini_pipe.py` |
| Fail if committed artifacts are stale | `.venv/bin/python helpers/build_gemini_pipe.py --check` |
| Pull the artifact back into the template | `.venv/bin/python helpers/build_gemini_pipe.py --refresh-template` |
| Full test suite (fake client, no network) | `.venv/bin/python -m pytest helpers -q` |
| Live smoke | `GEMINI_API_KEY=YOUR_API_KEY .venv/bin/python helpers/smoke_live.py --scenario all` |

## Build mechanics

`helpers/build_gemini_pipe.py` splices `src/` into the template at marker pairs (`# BEGIN/END GENERATED SECTION: gemini_pipe.<section>`):

- `STREAM_SOURCES` — `[(section, path)]`, compiled to **module-level** sections in the order their marker pairs appear in the template. Marker order pins section order in the artifact: a module must stay above the code that uses its names, because the build strips imports (and `from __future__ import annotations`) from each section.
- `METHOD_GROUP_SOURCES` — paths spliced into one `class Pipe` body, in list order. A duplicate method name in a later file is a build error (it would otherwise silently override).
- AST guards: stream sources may not import inside a function or class body; method-group files may contain only imports, a docstring, `logger =` and exactly one class; a template marker without a listed stream source (or vice versa) fails the build.
- **All imports live in the template.** Sections reference its module-level names (`json`, `gtypes`, `_get`, …); imports in section files are stripped, they exist only so `src/` reads standalone.
- The minified artifact is written by the same build and must never lag the full one.

## File map

| Path | Role |
|---|---|
| `src/gemini_pipe/pipe_template.py` | Template: frontmatter (version, requirements, changelog), guarded imports, patterns, `class Pipe` + valves, marker pairs. |
| `src/gemini_pipe/pipe_orchestrator.py` | `pipe()` entry: setup, generation loop, finalisation, cancellation, errors. |
| `src/gemini_pipe/request/rag.py` | Memory extraction, metadata markers, removal of natively-sent files from OpenWebUI's RAG context. |
| `src/gemini_pipe/request/history.py` | OpenWebUI messages → Gemini contents: text, images, memory relocation, tool-call replay, generated-image replay. |
| `src/gemini_pipe/request/media.py` | Native media: files/YouTube collection, inline vs Files API, anchors. |
| `src/gemini_pipe/request/tools.py` | Function declarations; combining built-in tools with function tools per generation. |
| `src/gemini_pipe/request/config.py` | `GenerateContentConfig` assembly; thinking level/budget; `reasoning_effort`. |
| `src/gemini_pipe/response/grounding.py` | Chunk/support accumulation, support locating, citation markers, citation registry, suggestion links. |
| `src/gemini_pipe/response/state.py` | `Segment`/`RenderedMessage`, `CallState`/`TurnState` (per-call and per-turn bookkeeping). |
| `src/gemini_pipe/response/status_events.py` | `StatusEmitter`: one status per real phase, dedup, `done` only on the final line. |
| `src/gemini_pipe/response/formatting.py` | Collapsible blocks: thinking, tool results, URL context, search/Maps blocks. |
| `src/gemini_pipe/response/tool_exec.py` | Tool execution, approval prompts, result replay. |
| `src/gemini_pipe/response/images_out.py` | Generated-image storage into OpenWebUI files. |
| `src/gemini_pipe/response/stream_parts.py` | Chunk dispatch (thoughts/text/tool calls), per-call finalisation, replay-part merge. |
| `src/gemini_pipe/response/runtime.py` | Grounding application, redirect resolution, retries/errors, usage, status line, cost. |
| `src/gemini_pipe/shared/models.py` | Client factory, model discovery, per-generation capabilities, model-list cache. |
| `src/gemini_pipe/shared/tasks.py` | OpenWebUI tasks: title, tags, follow-ups. |
| `src/gemini_pipe/shared/pricing.py` | Dated price table, `GeminiPricing` (rates, per-call cost), `format_usd`. |
| `helpers/build_gemini_pipe.py` | Build / `--check` / `--refresh-template`. |
| `helpers/minify_pipe.py` | Strips comments/docstrings (keeps the frontmatter docstring OpenWebUI parses). |
| `helpers/_testlib.py` | Test fakes: `FakeClient`, `FakeEmitter`, `chunk`, `make_pipe`. |
| `helpers/test_*.py` | Per-feature checks, all offline against the built artifact. |
| `helpers/smoke_live.py` | Live smoke scenarios (needs a real key). |
| `helpers/probe_tool_combos.py` | Live probe: which built-in/function tool combinations `generateContent` accepts (needs a real key). |
| `gemini_pipe.py`, `gemini_pipe.min.py` | Generated artifacts (committed; install sources). |
| `gemini_pipe_web_search_toggle.py`, `gemini_pipe_maps_toggle.py`, `gemini_manifold_companion_filter.py` | Companion filters. |

## Invariants to preserve

- `_build_genai_client` is the **single decryption point** for API keys and passes `vertexai=False`; every code path gets its client from there.
- Model parts are **replayed raw**; signed parts are never merged (merging drops thought signatures and breaks the next call).
- Function responses are replayed in **call order** (FC1, FC2, FR1, FR2).
- History replay stamps `DUMMY_THOUGHT_SIGNATURE` on the first FC of each replayed group (persisted messages carry no signatures).
- Built-in blocks (`data-gemini-builtin="1"`: Google Search, URL context) are rendering only: history strips them and never replays them as function calls.
- Grounding offsets are **UTF-8 bytes**, not characters.
- Citation numbering is turn-wide and starts **after** OpenWebUI's RAG sources (`rag_offset`).
- Search Suggestions **must stay visible** in some form (Google's terms); `SHOW_SEARCH_SUGGESTIONS` changes form, not presence.
- `google_maps` never shares a request with `google_search` or `url_context` (the API answers 400); capabilities `maps_with_search` / `maps_with_url_context` default to False. A tool-combination 400 retries once without the lower-priority built-in. Google's docs describe combinations for the Interactions API, not `generateContent`: trust `probe_tool_combos.py`, not the docs. Probed 2026-11-07 on `gemini-3.6-flash` and `gemini-4-argon`: Maps alone, Search alone, URL context alone and Search+URL context are accepted; Maps+Search and Maps+URL context are rejected (3.6 Flash answers a generic "invalid argument" without naming the tools, so the retry only catches the named variant); built-ins plus function tools are accepted only with `include_server_side_tool_invocations` and mode `VALIDATED`.
- Tool approval: a missing or broken approval channel **denies**, never silently allows.
- Metadata markers **accumulate forward**: each turn appends prior turns' markers to the saved message.
- Media anchors stay on the **original user message**, so replay re-anchors files correctly.
- The Files API cache is keyed by key fingerprint, file id and hash.
- Changing a valve's *type* needs a `field_validator(mode="before")` accepting the old shape, or every stored value 400s on save.
- Never log chat content (SDK/transport loggers are capped at WARNING for this reason).

## Common failure patterns

| Symptom | First checks |
|---|---|
| Model missing from the list | discovery response, `ENABLED_MODELS`, `MODEL_CACHE_TTL_MINUTES`, capability filters |
| 401/403 on every call | key source (env vs valve), `_build_genai_client`, `GEMINI_BASE_URL` |
| Search doesn't trigger | the model decides — check toggle/`WEB_SEARCH_DEFAULT`, then `BUILTIN_TOOL_CONFLICT` on 2.5 |
| Wrong citation numbers or URLs | `rag_offset` computation, redirect resolution, registry |
| Tool loop stalls or repeats | FC/FR replay order, thought signatures, `MAX_TOOL_CALLS` |
| Files re-uploaded every turn | Files API cache keys, metadata markers |
| Stale or missing cost | model present in `GEMINI_PRICES`, `SHOW_COST`, `MODEL_PRICING_OVERRIDES` parses |
| Valve save fails with 400 | a type change without a `mode="before"` validator |
| Artifact won't build / `--check` fails | marker pairs vs `STREAM_SOURCES`, AST guards, duplicate method names |

## Release checklist

1. Change `src/` (tests in `helpers/` first, committed separately as `checks: …`).
2. Bump `version:` and add the changelog entry in `src/gemini_pipe/pipe_template.py`.
3. `.venv/bin/python helpers/build_gemini_pipe.py` (writes both artifacts; `--check` must exit 0).
4. Mirror the changelog into `README.md` (and its Status section).
5. Commit `src/` **and** both regenerated artifacts.
