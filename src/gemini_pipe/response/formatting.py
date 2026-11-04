"""<details> formatting: the OpenWebUI-native collapsible blocks that
grounded answers, thinking and built-in tool activity render into.

The attribute encoding is parsed by OpenWebUI (block spinner, source list,
embeds), so it must stay byte-compatible with what its built-in pipes emit.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _Formatting:
    @staticmethod
    def _block_visibility_key(name: str) -> str:
        """Map a block name onto its visibility key.

        A single logical block reaches the formatters under several names, but
        the user hides *one* concept. Normalising here keeps the HIDE_BLOCKS
        valve spelled in concepts rather than wire names.
        """
        return name

    def _is_block_hidden(self, name: str) -> bool:
        """True when this block concept is not to be rendered for this request.

        Sub-agent runs (SLIM_OUTPUT) render no decoration at all; otherwise the
        request-scoped HIDDEN_BLOCKS ContextVar decides.
        """
        if SLIM_OUTPUT.get():
            return True
        hidden = HIDDEN_BLOCKS.get()
        if not hidden:
            return False
        return self._block_visibility_key(name) in hidden

    @staticmethod
    def _parse_hidden_blocks(raw: Any) -> frozenset:
        """Normalize the HIDE_BLOCKS valve into a set of block concept keys.

        The valve is a multiselect (``list[str]``); values saved under the
        previous comma-separated ``str`` form are still accepted so an upgrade
        does not silently drop a user's preference.
        """
        if isinstance(raw, str):
            raw = raw.split(",")
        if not raw:
            return frozenset()
        return frozenset(str(part).strip() for part in raw if str(part).strip())

    @staticmethod
    def _append_block_to_text(text: str, block: str) -> str:
        """Append a rendered block with a safe markdown/html separator."""
        if not text:
            return block
        if not block:
            return text
        if text.endswith(("\n", "\r")) or block.startswith(("\n", "\r")):
            return text + block
        return f"{text}\n{block}"

    def _format_thinking_block(
        self, content: str, done: bool = False, duration: Optional[float] = None
    ) -> str:
        """Format a thought with OpenWebUI native <details type="reasoning">.

        ``done`` closes the block (spinner off, "Thought for N seconds"); while
        streaming the block stays open under "Thinking…".
        """
        if self._is_block_hidden("reasoning"):
            return ""

        # Escape content and add > prefix per line (OpenWebUI quote block style)
        escaped_lines = "\n".join(
            f"> {html.escape(line)}" if not line.startswith(">") else html.escape(line)
            for line in content.splitlines()
        )

        if done:
            duration_int = int(duration or 0)
            duration_attr = f' duration="{duration_int}"' if duration is not None else ""
            return (
                f'<details type="reasoning" done="true"{duration_attr}>\n'
                f"<summary>Thought for {duration_int} seconds</summary>\n"
                f"{escaped_lines}\n"
                f"</details>\n"
            )
        return (
            '<details type="reasoning" done="false">\n'
            "<summary>Thinking…</summary>\n"
            f"{escaped_lines}\n"
            "</details>\n"
        )

    def _format_tool_result_block(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict,
        tool_output: str,
        is_error: bool = False,
        done: bool = True,
        embeds: list = None,
        files: list = None,
    ) -> str:
        """Format a tool result block with OpenWebUI native <details type="tool_calls">.

        ``done`` False shows "Executing..." with spinner; True, "Tool Executed".
        ``embeds``/``files`` carry the process_tool_result output (HTML strings,
        file chips) into the block.
        """
        if SLIM_OUTPUT.get():
            # The tool ran; only its effect on the answer matters to the parent
            # agent. Arguments, raw result, embeds and file chips are markup a
            # human would expand -- nobody will.
            return ""

        # Escape arguments for HTML attribute
        escaped_args = (
            html.escape(json.dumps(tool_input, ensure_ascii=False)) if tool_input else ""
        )

        done_str = "true" if done else "false"
        summary = "Tool Executed" if done else "Executing..."
        error_attr = ' error="true"' if is_error and done else ""

        if done:
            # Escape result for HTML attribute
            try:
                if isinstance(tool_output, str):
                    try:
                        parsed = json.loads(tool_output)
                        escaped_result = html.escape(json.dumps(parsed, ensure_ascii=False))
                    except (json.JSONDecodeError, ValueError):
                        escaped_result = html.escape(json.dumps(tool_output, ensure_ascii=False))
                else:
                    escaped_result = html.escape(json.dumps(tool_output, ensure_ascii=False))
            except Exception:
                escaped_result = html.escape(json.dumps(str(tool_output), ensure_ascii=False))

            escaped_embeds = (
                html.escape(json.dumps(embeds, ensure_ascii=False)) if embeds else ""
            )

            return (
                f'<details type="tool_calls" done="{done_str}" id="{html.escape(tool_call_id)}" name="{html.escape(tool_name)}" '
                f'arguments="{escaped_args}" result="{escaped_result}" '
                f'files="{html.escape(json.dumps(files)) if files else ""}" '
                f'embeds="{escaped_embeds}"{error_attr}>\n'
                f"<summary>{summary}</summary>\n"
                f"</details>\n"
            )
        return (
            f'<details type="tool_calls" done="{done_str}" id="{html.escape(tool_call_id)}" name="{html.escape(tool_name)}" '
            f'arguments="{escaped_args}">\n'
            f"<summary>{summary}</summary>\n"
            f"</details>\n"
        )

    def _format_google_search_block(
        self,
        call: int,
        queries: List[str],
        sources: List[Tuple[int, str, str]],
        embed_html: str = "",
    ) -> str:
        """The Google Search block: the queries asked, the sources found, and
        the Search Suggestions embed Google's terms require us to show."""
        if self._is_block_hidden("google_search"):
            return ""

        escaped_args = html.escape(json.dumps({"queries": list(queries or [])}, ensure_ascii=False))
        escaped_result = html.escape(
            json.dumps(
                [{"n": n, "title": title, "url": url} for n, title, url in (sources or [])],
                ensure_ascii=False,
            )
        )
        escaped_embeds = (
            html.escape(json.dumps([embed_html], ensure_ascii=False)) if embed_html else ""
        )

        shown = [q for q in (queries or []) if q][:3]
        summary = "Searched Google: " + " · ".join(shown) if shown else "Searched Google"
        if len([q for q in (queries or []) if q]) > 3:
            summary += f" +{len([q for q in queries if q]) - 3} more"

        body = ""
        if sources:
            body = "\n" + "\n".join(f"- [{n}] [{title}]({url})" for n, title, url in sources) + "\n"

        return (
            f'<details type="tool_calls" done="true" id="gsearch_{call}" name="google_search" '
            f'data-gemini-builtin="1" arguments="{escaped_args}" result="{escaped_result}" '
            f'embeds="{escaped_embeds}">\n'
            f"<summary>{summary}</summary>\n"
            f"{body}</details>\n"
        )

    def _format_url_context_block(self, call: int, entries: List[dict]) -> str:
        """The URL context block: one line per URL from the conversation,
        marked ✅ when the model could read it, ❌ otherwise."""
        if self._is_block_hidden("url_context"):
            return ""

        lines = []
        for entry in entries or []:
            url = entry.get("url") or ""
            status = str(entry.get("status") or "")
            if status.lower() == "ok":
                lines.append(f"- ✅ [{url}]({url})")
            else:
                lines.append(f"- ❌ {url}" + (f" ({status})" if status else ""))
        body = ("\n" + "\n".join(lines) + "\n") if lines else ""

        return (
            f'<details type="tool_calls" done="true" id="urlctx_{call}" name="url_context" '
            f'data-gemini-builtin="1">\n'
            f"<summary>Read {len(entries or [])} pages</summary>\n"
            f"{body}</details>\n"
        )

    def _format_suggestion_links_line(self, links: List[Tuple[str, str]]) -> str:
        """The plain-text fallback for Search Suggestions (valve "links")."""
        if not links:
            return ""
        return (
            "🔎 **Google Search suggestions:** "
            + " · ".join(f"[{label}]({href})" for label, href in links)
            + "\n"
        )

    def _format_notice(self, text: str) -> str:
        """A short warning line appended to the answer (degraded capability, …)."""
        return "\n> ⚠️ " + text + "\n"
