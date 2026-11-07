# Google Gemini API Manifold Pipe for Open WebUI

Open WebUI manifold pipe that talks to the **native Gemini Developer API** (Developer API key, no OpenAI-compatible proxy): Google Search and Maps grounding with citations, thinking display, native function tools, native PDF input, and image generation.

> [!NOTE]
> The work done in this repository was developed using AI.

## Status

- **Pipe version:** `0.1.4`
- **Requirements:** `google-genai>=1.68.0`, `pydantic>=2`, `cryptography`
- **Recommended OpenWebUI:** `0.11.4+`. OpenWebUI 0.11.0–0.11.3 pinned `google-genai==1.66.0`; installing this pipe upgrades that shared package, so check your instance before relying on new SDK behaviour.

## Why a native pipe

OpenWebUI's OpenAI-compatible Gemini connection cannot do Google Search grounding, inline citations, or native tools. This pipe uses the `google-genai` SDK directly, so these native capabilities are available: search/Maps grounding with the attribution blocks Google's terms require, thought summaries, function calling with thought signatures, native PDF input, and image output.

## Features

| Area | What you get |
|---|---|
| Models | Auto-discovery, capabilities derived per model generation, override valve |
| Web search | Per-message toggle (Google Search with URL context), inline `[n]` citations, source list, the Search Suggestions block Google's terms require |
| Maps | Per-message toggle, location from the user/admin valves, place citations and the required "Google Maps" attribution |
| Thinking | Thought summaries in collapsible blocks |
| Tools | OpenWebUI and built-in tools as Gemini functions (together with Search on Gemini 3+); multi-call loop; thought signatures preserved; history replay |
| PDFs | Sent to Gemini natively as whole documents: per file ("Using Entire Document"), per chat (toggle), or by admin default |
| Image generation | Nano Banana models; images saved to OpenWebUI storage; follow-up edits |
| Usage and cost | Cost estimate in the status line from dated price tables |
| Privacy and security | Admin API key encrypted at rest; no chat content in logs |

## Installation

1. Admin → Functions → `+` → paste `gemini_pipe.py` (or `gemini_pipe.min.py`).
2. Add the filters the same way:
   - `gemini_pipe_web_search_toggle.py` — turns on Google Search (plus URL context) for the next message; the model decides whether to search;
   - `gemini_pipe_maps_toggle.py` — turns on Google Maps grounding for the next message;
   - `gemini_pipe_native_pdf_toggle.py` — sends the PDFs attached to this chat to Gemini as whole documents, without having to set each file to "Using Entire Document".
3. Set `GEMINI_API_KEY` in the pipe valves (or the environment variables `GEMINI_API_KEY` / `GOOGLE_API_KEY`).
4. In Admin → Models, for each Gemini model: attach the toggles, and set Function Calling to **Native** for tools.

## Configuration

### Admin valves

| Name | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | from env | Gemini Developer API key, stored encrypted |
| `GEMINI_BASE_URL` | `""` | Optional proxy base URL; empty = Google default |
| `ENABLED_MODELS` | `""` | Comma-separated model names; skips discovery |
| `MODEL_CACHE_TTL_MINUTES` | `1440` | How long the discovered model list is cached; 0 disables the cache |
| `MODEL_CAPABILITY_OVERRIDES` | `""` | JSON `{model pattern: {capability: value}}` applied over the derived capabilities; a pattern ending in `*` is a prefix match, e.g. `{"gemini-4*": {"maps": false}}` |
| `WEB_SEARCH_DEFAULT` | `False` | Google Search on every request; otherwise only via the toggle. Off keeps search opt-in because it is billed per search query |
| `MAPS_DEFAULT` | `False` | Maps grounding on every request |
| `MAPS_LATITUDE` / `MAPS_LONGITUDE` | `""` | Default Maps location; empty = none (then the per-user valves apply) |
| `SHOW_SEARCH_SUGGESTIONS` | `embed` | `embed` (inline block), `links` or `off`. Google's terms require showing Search Suggestions with grounded answers |
| `RESOLVE_GROUNDING_URLS` | `True` | Resolve grounding redirect URLs to their real destination (reads only Google's redirect headers) |
| `MAX_TOOL_CALLS` | `15` | Maximum number of tool calls per request |
| `MAX_RETRIES` | `3` | Maximum number of retries for failed requests |
| `REQUEST_TIMEOUT` | `600` | Request timeout in seconds for Gemini API calls |
| `TOOL_CALL_TIMEOUT` | `60` | Timeout in seconds for individual tool call execution |
| `NATIVE_PDF_DEFAULT` | `False` | Send every attached PDF natively, not only those set to "Using Entire Document"; the Native PDFs toggle does the same for one chat |
| `TASK_MODEL` | `gemini-3.5-flash-lite` | Model for title/tags/follow-ups; empty = chat model |
| `MODEL_PRICING_OVERRIDES` | `""` | JSON `{model pattern: {rate key: USD per million tokens}}` overriding the built-in price table, e.g. `{"gemini-2.5*": {"input": 0.20}}` |

### User valves

| Name | Default | Description |
|---|---|---|
| `THINKING_LEVEL` | `auto` | Thinking level for models that use levels; auto follows the model default |
| `SHOW_THOUGHTS` | `True` | Render the model's reasoning blocks in the chat |
| `IMAGE_ASPECT_RATIO` | `auto` | Aspect ratio for generated images (`1:1` … `21:9`); auto = model default |
| `IMAGE_SIZE` | `auto` | Size for generated images (`1K`, `2K`, `4K`); auto = model default |
| `MAPS_LATITUDE` / `MAPS_LONGITUDE` | `""` | Per-user Maps location; empty = use the admin default |
| `USE_NATIVE_FILES` | `True` | Master switch for native PDFs (full-context, or forced by the Native PDFs toggle / `NATIVE_PDF_DEFAULT`): off = OpenWebUI's retrieval handles them |
| `HIDE_BLOCKS` | `[]` | Collapsible blocks to hide (`google_search`, `url_context`, `reasoning`). Progress still shows in the status line; with the search block hidden, Search Suggestions appear as links |
| `SHOW_COST` | `True` | Show the estimated USD cost of the turn in the status line and the message usage |

## Web search and Maps in detail

- **Built-in tools can't be forced.** Google Search and Maps are Google-side tools: the model decides when to search. The toggle and `WEB_SEARCH_DEFAULT` only switch the capability on for the request.
- **Billing:** Gemini 3+ charges $14 per 1,000 search queries after 5,000 free per month. The pipe's cost estimate counts every search at the list price, so it is an upper bound inside the free allowance.
- **Search Suggestions** are shown with every grounded answer because Google's terms require them; `SHOW_SEARCH_SUGGESTIONS` only changes the form (embedded block vs plain links).
- **Redirect URLs are resolved without visiting the target site:** the pipe reads only Google's `grounding-api-redirect` `Location` header, then stops.
- **Maps is exclusive per request:** `generateContent` rejects `google_maps` together with `google_search` or `url_context` (400 `INVALID_ARGUMENT`). With Maps on, Search and URL context are dropped for that message, with a note. If Google enables a pair for a model, restore it with `MODEL_CAPABILITY_OVERRIDES`, e.g. `{"gemini-4*": {"maps_with_search": true, "maps_with_url_context": true}}`. Check what your key accepts with `helpers/probe_tool_combos.py`.
- **Combining with function tools:** Gemini 3+ can combine built-in tools with function tools (Google labels it preview).

## Tools

- OpenWebUI functions and built-in tools are sent as Gemini **native function declarations**; the multi-call loop replays results until the model stops calling.
- **Thought signatures** on function-call parts are preserved on replay, so the API's signature checks keep passing across calls.

## PDFs

PDFs go to Gemini natively when the file is set to full context ("Using Entire Document"), when the Native PDFs toggle is on for the chat, or when `NATIVE_PDF_DEFAULT` is set. Otherwise they stay with OpenWebUI's retrieval (and its knowledge tools), like all other file types.

## Image generation

- Nano Banana models (e.g. `gemini-nano-banana-2.1`, `gemini-3-pro-image`) generate images from the chat.
- `IMAGE_ASPECT_RATIO` and `IMAGE_SIZE` are passed through to the API when set; `auto` leaves the model default.
- Generated images are saved to OpenWebUI storage. On a follow-up, the most recent assistant message's images are attached to your new message, so "make it smaller / change the background" edits the picture instead of redrawing it (probed on `gemini-nano-banana-2.1` and `gemini-3-pro-image`). Older images are not re-sent; the status line shows "Attaching the previous image" when one is.

## Costs

- The `💵 ≈$…` figure in the status line and the `cost_usd` usage field are **estimates only**, computed from the published list prices in the pipe.
- The price table was read from Google's pricing page on **2026-11-05**. Scheduled increases already in the table: `gemini-4-argon` rises on 2026-12-01, and `gemini-3.8-flash` / `gemini-3.6-flash` on 2027-01-01. Newer price changes need a table update (or `MODEL_PRICING_OVERRIDES`).
- **Free allowances are not subtracted** (e.g. the 5,000 free search queries per month on Gemini 3+), so the estimate is an upper bound.
- Models without a table entry show no cost at all, never a guess.

## Privacy

- The admin API key is **encrypted at rest** with `WEBUI_SECRET_KEY` before it reaches the database; the single decryption point is the client factory.
- Chat content is **never logged**; the SDK and transport loggers are capped at WARNING, and log lines carry only model names, key fingerprints and counts.
- Beyond that, **Google's own data policies apply** to whatever you send to the API.

## Development

- **Build:** `src/` is the source of truth; `python helpers/build_gemini_pipe.py` splices it into the single-file `gemini_pipe.py` (and `gemini_pipe.min.py`). `--check` fails if the committed artifacts are stale.
- **Tests:**
  ```
  uv venv .venv && uv pip install -p .venv -r helpers/requirements-dev.txt
  .venv/bin/python helpers/build_gemini_pipe.py
  .venv/bin/python -m pytest helpers -q
  ```
- **Live smoke:** `GEMINI_API_KEY=YOUR_API_KEY .venv/bin/python helpers/smoke_live.py --scenario all` (start the command with a space so the key stays out of shell history).
- **Live probes:** `helpers/probe_tool_combos.py` and `helpers/probe_image_edit.py` show which tool combinations and history shapes the API accepts for your key and models.

## Changelog

**v0.1.4** — fix: follow-up image edits ignored the earlier picture on `gemini-nano-banana-2.1` and redrew it from the text; the previous image now rides on your new message (probed to edit on both Nano Banana models). New: Native PDFs toggle (`gemini_pipe_native_pdf_toggle.py`) and `NATIVE_PDF_DEFAULT` valve to send PDFs as whole documents without per-file settings.

**v0.1.3** — fix: a Maps-only answer showed an empty "Google Search" block, and the queries Google reports for Maps were counted as billed searches in the cost estimate. Both are gone.

**v0.1.2** — fix: the Maps toggle still failed with `400 INVALID_ARGUMENT` because Google also rejects `google_maps` together with `url_context`, which Search switches on by itself. Maps now drops URL context too (with a note), and a "cannot be combined" 400 retries once without the lower-priority built-in (URL context, then Search). `helpers/probe_tool_combos.py` shows what the live API accepts.

**v0.1.1** — fix: the Maps toggle failed with `400 INVALID_ARGUMENT` whenever Google Search was also on (by toggle, `WEB_SEARCH_DEFAULT` or OpenWebUI's web search button). `generateContent` rejects the pair on every model, so Maps now wins and Search is dropped for that message with a note.

**v0.1.0** — initial release:

- Native Gemini Developer API connection with auto model discovery and per-generation capabilities
- Google Search grounding with inline citations, source list and the Search Suggestions block; per-message toggle
- Google Maps grounding with place citations and attribution; per-message toggle
- Thought summaries in collapsible blocks
- Native function calling for OpenWebUI and built-in tools: multi-call loop, thought signatures
- Native PDF input
- Image generation on Nano Banana models with saved outputs
- Cost estimate from dated price tables (status line, `cost_usd`)
- Encrypted API key storage, no chat content in logs

## Todo

- Code execution toggle
- Deep Research

**License:** MIT. Portions derived from Podden's [openwebui_anthropic_api_manifold_pipe](https://github.com/Podden/openwebui_anthropic_api_manifold_pipe).
