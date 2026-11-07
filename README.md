# Google Gemini API Manifold Pipe for Open WebUI

Open WebUI manifold pipe that talks to the **native Gemini Developer API** (Developer API key, no OpenAI-compatible proxy): Google Search grounding with citations, thinking display, native function tools, native files and YouTube media, and image generation.

## Status

- **Pipe version:** `0.1.2`
- **Requirements:** `google-genai>=1.68.0`, `pydantic>=2`, `cryptography`
- **Recommended OpenWebUI:** `0.11.4+`. OpenWebUI 0.11.0–0.11.3 pinned `google-genai==1.66.0`; installing this pipe upgrades that shared package, so check your instance before relying on new SDK behaviour.

## Why a native pipe

OpenWebUI's OpenAI-compatible Gemini connection cannot do Google Search grounding, inline citations, or native tools. This pipe uses the `google-genai` SDK directly, so these native capabilities are available: search/Maps grounding with the attribution blocks Google's terms require, thought summaries, function calling with thought signatures, inline or Files-API media, and image output.

## Features

| Area | What you get |
|---|---|
| Models | Auto-discovery, capabilities derived per model generation, override valve |
| Web search | Per-message toggle, Google Search + URL context, inline `[n]` citations, source list, Google Search block with the Search Suggestions Google's terms require |
| Maps | Toggle, location from user/admin valves or OpenWebUI's user location, place citations and the required "Google Maps" attribution |
| Thinking | Thought summaries in collapsible blocks; level (Gemini 3+) or budget (2.5); `reasoning_effort` honoured |
| Tools | OpenWebUI / MCP / built-in tools as Gemini functions; parallel calls; approval prompts; thought signatures preserved; history replay |
| Files | Native audio, video and full-context PDF (inline or Files API with a cache); YouTube links as video; images |
| Image generation | Nano Banana models; images saved to OpenWebUI storage; follow-up edits |
| Usage and cost | Context gauge, cached-token share, cost estimate from dated price tables |
| Privacy and security | API key encrypted at rest; `store` off by default; no chat content in logs |

## Installation

1. Admin → Functions → `+` → paste `gemini_pipe.py` (or `gemini_pipe.min.py`).
2. Add the filters the same way:
   - `gemini_pipe_web_search_toggle.py` — turns on Google Search (plus URL context) for the next message; the model decides whether to search;
   - `gemini_pipe_maps_toggle.py` — turns on Google Maps grounding for the next message;
   - optionally `gemini_manifold_companion_filter.py`, which routes OpenWebUI's own Web Search button to Google Search for Gemini pipe models only.
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
| `URL_CONTEXT` | `auto` | When to attach the URL context tool: `auto` = when web search is on or the latest message has a non-YouTube URL; `with_search` = only with web search; `always`; `off` |
| `MAPS_DEFAULT` | `False` | Maps grounding on every request |
| `MAPS_LATITUDE` / `MAPS_LONGITUDE` | `""` | Default Maps location; empty = none (then the per-user valves, then OpenWebUI's user location) |
| `BUILTIN_TOOL_CONFLICT` | `prefer_builtin` | Which side wins when a model cannot combine function tools with Google Search/Maps: `prefer_builtin` or `prefer_functions` |
| `SHOW_SEARCH_SUGGESTIONS` | `embed` | `embed` (inline block), `links` or `off`. Google's terms require showing Search Suggestions with grounded answers |
| `RESOLVE_GROUNDING_URLS` | `True` | Resolve grounding redirect URLs to their real destination (reads only Google's redirect headers) |
| `STORE_REQUESTS` | `False` | Lets Google store requests (sent only when the SDK supports `store`) |
| `MAX_TOOL_CALLS` | `15` | Maximum number of tool calls per request |
| `MAX_RETRIES` | `3` | Maximum number of retries for failed requests |
| `REQUEST_TIMEOUT` | `600` | Request timeout in seconds for Gemini API calls |
| `TOOL_CALL_TIMEOUT` | `60` | Timeout in seconds for individual tool call execution |
| `FILES_API` | `auto` | When to upload media files through the Gemini Files API: `auto`, `always`, `never` |
| `INLINE_FILE_MAX_MB` | `20` | Maximum file size in MB to inline instead of using the Files API |
| `YOUTUBE_LINKS` | `most_recent` | Which YouTube links from the conversation to send to the model: `most_recent`, `latest_message`, `all`, `off` |
| `TASK_MODEL` | `gemini-3.5-flash-lite` | Model for title/tags/follow-ups; empty = chat model |
| `MODEL_PRICING_OVERRIDES` | `""` | JSON `{model pattern: {rate key: USD per million tokens}}` overriding the built-in price table, e.g. `{"gemini-2.5*": {"input": 0.20}}` |

### User valves

| Name | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | `""` | Overrides the admin-configured API key; empty = use the admin key. Stored encrypted when `WEBUI_SECRET_KEY` is set |
| `THINKING_LEVEL` | `auto` | Thinking level for models that use levels; auto follows the model default |
| `THINKING_BUDGET` | `-1` | Thinking budget in tokens for Gemini 2.5 models; -1 = dynamic |
| `SHOW_THOUGHTS` | `True` | Render the model's reasoning blocks in the chat |
| `MEDIA_RESOLUTION` | `auto` | Resolution the model sees images at |
| `IMAGE_ASPECT_RATIO` | `auto` | Aspect ratio for generated images (`1:1` … `21:9`); auto = model default |
| `IMAGE_SIZE` | `auto` | Size for generated images (`1K`, `2K`, `4K`); auto = model default |
| `MAPS_LATITUDE` / `MAPS_LONGITUDE` | `""` | Per-user Maps location; empty = use the admin default |
| `USE_NATIVE_FILES` | `True` | Send audio, video and full-context PDFs to Gemini natively (inline or via the Files API) instead of OpenWebUI's RAG text |
| `HIDE_BLOCKS` | `[]` | Collapsible blocks to hide (`google_search`, `url_context`, `reasoning`). Progress still shows in the status line; with the search block hidden, Search Suggestions appear as links. The Google Maps attribution is always shown (`google_maps` has no effect) |
| `SHOW_TOKEN_COUNT` | `Off` | Show context window progress after each response; `With Cache` also shows cached tokens |
| `SHOW_COST` | `True` | Show the estimated USD cost of the turn in the status line and the message usage |

## Web search, URL context and Maps in detail

- **Built-in tools can't be forced.** Google Search and Maps are Google-side tools: the model decides when to search. The toggle and `WEB_SEARCH_DEFAULT` only switch the capability on for the request.
- **Billing:** Gemini 3+ charges $14 per 1,000 search queries after 5,000 free per month; Gemini 2.5 charges per grounded prompt. The pipe's cost estimate counts every search at the list price, so it is an upper bound inside the free allowance.
- **Search Suggestions** are shown with every grounded answer because Google's terms require them; `SHOW_SEARCH_SUGGESTIONS` only changes the form (embedded block vs plain links).
- **Redirect URLs are resolved without visiting the target site:** the pipe reads only Google's `grounding-api-redirect` `Location` header, then stops.
- **Maps is exclusive per request:** `generateContent` rejects `google_maps` together with `google_search` or `url_context` (400 `INVALID_ARGUMENT`). With Maps on, Search and URL context are dropped for that message, with a note. If Google enables a pair for a model, restore it with `MODEL_CAPABILITY_OVERRIDES`, e.g. `{"gemini-4*": {"maps_with_search": true, "maps_with_url_context": true}}`. Check what your key accepts with `helpers/probe_tool_combos.py`.
- **Combining with function tools:** Gemini 3+ can combine built-in tools with function tools (Google labels it preview); on Gemini 2.5 it cannot, and `BUILTIN_TOOL_CONFLICT` decides which side is dropped.

## Tools

- OpenWebUI functions, MCP servers and built-in tools are sent as Gemini **native function declarations**; the multi-call loop replays results until the model stops calling.
- Tool calls run in parallel; `MAX_TOOL_CALLS` caps the total per request, after which function calling is switched off for the remaining calls.
- **Approval mode `ask` is enforced by the pipe, not by OpenWebUI** (a manifold never enters OpenWebUI's native tool loop). The approval prompt goes through OpenWebUI's event channel; a missing or broken channel **denies** the call rather than silently allowing it.
- **Thought signatures** on function-call parts are preserved on replay, so the API's signature checks keep passing across calls.

## Files and media

- **Native:** audio and video files go to the Gemini API directly, as do images. PDFs go natively when the file is set to full context ("Using Entire Document"); otherwise they stay with OpenWebUI's RAG retrieval, like all other file types.
- **Inline vs Files API:** files up to `INLINE_FILE_MAX_MB` are inlined into the request; larger ones go through the Files API, whose files are retained by Google for 48 hours. The pipe caches the upload (keyed by key fingerprint, file id and hash) in the file's OpenWebUI metadata, so the same attachment is not re-uploaded on every turn.
- **YouTube links** are converted to native video input; `YOUTUBE_LINKS` picks which links (`most_recent`, `latest_message`, `all`, `off`). Media anchors stay on the user message where the link appeared, so history replay re-anchors files correctly.

## Image generation

- Nano Banana models (e.g. `gemini-nano-banana-2.1`, `gemini-3-pro-image`) generate images from the chat.
- `IMAGE_ASPECT_RATIO` and `IMAGE_SIZE` are passed through to the API when set; `auto` leaves the model default.
- Generated images are saved to OpenWebUI storage and re-attached on follow-up, so "make it smaller / change the background" edits work across turns.
- **SynthID** watermarks are applied by Google on all generated images, not by the pipe.

## Costs

- The `💵 ≈$…` figure in the status line and the `cost_usd` usage field are **estimates only**, computed from the published list prices in the pipe.
- The price table was read from Google's pricing page on **2026-11-05**. Scheduled increases already in the table: `gemini-4-argon` rises on 2026-12-01, and `gemini-3.8-flash` / `gemini-3.6-flash` on 2027-01-01. Newer price changes need a table update (or `MODEL_PRICING_OVERRIDES`).
- **Free allowances are not subtracted** (e.g. the 5,000 free search queries per month on Gemini 3+), so the estimate is an upper bound.
- Audio input is priced at the text input rate.
- Models without a table entry show no cost at all, never a guess.

## Privacy

- API keys (admin and per-user) are **encrypted at rest** with `WEBUI_SECRET_KEY` before they reach the database; the single decryption point is the client factory.
- `STORE_REQUESTS` defaults to **off**: the pipe sends `store=false` to opt out of Google's request storage, which Google enables by default on new paid projects. The field is sent only when the installed SDK supports it (google-genai 2.33+).
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
- **Live smoke:** `GEMINI_API_KEY=… .venv/bin/python helpers/smoke_live.py --scenario all` (start the command with a space so the key stays out of shell history).

## Changelog

**v0.1.2** — fix: the Maps toggle still failed with `400 INVALID_ARGUMENT` because Google also rejects `google_maps` together with `url_context`, which Search switches on by itself. Maps now drops URL context too (with a note), and a "cannot be combined" 400 retries once without the lower-priority built-in (URL context, then Search). `helpers/probe_tool_combos.py` shows what the live API accepts.

**v0.1.1** — fix: the Maps toggle failed with `400 INVALID_ARGUMENT` whenever Google Search was also on (by toggle, `WEB_SEARCH_DEFAULT` or OpenWebUI's web search button). `generateContent` rejects the pair on every model, so Maps now wins and Search is dropped for that message with a note.

**v0.1.0** — initial release:

- Native Gemini Developer API connection with auto model discovery and per-generation capabilities
- Google Search grounding with inline citations, source list and the Search Suggestions block; per-message toggle
- Google Maps grounding with place citations and attribution; per-message toggle
- URL context for links in the conversation
- Thought summaries in collapsible blocks; thinking level/budget controls; `reasoning_effort` honoured
- Native function calling for OpenWebUI/MCP/built-in tools: multi-call loop, parallel calls, approval prompts, thought signatures
- Native media: audio, video, full-context PDF, images; inline or Files API with an upload cache; YouTube links as video
- Image generation on Nano Banana models with saved outputs and follow-up edits
- Usage stats: context gauge, cached-token share, and a cost estimate from dated price tables (`cost_usd`, status line)
- Encrypted API key storage (admin and per-user), `store=false` by default, no chat content in logs

**License:** MIT. Portions derived from Podden's [openwebui_anthropic_api_manifold_pipe](https://github.com/Podden/openwebui_anthropic_api_manifold_pipe).
