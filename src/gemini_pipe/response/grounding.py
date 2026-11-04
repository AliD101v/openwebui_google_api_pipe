"""Grounding and citations: chunk/support accumulation, support locating,
citation markers, the per-turn citation registry and suggestion links.

Pure module: no I/O, no network. All SDK objects are read defensively through
``_get`` so older SDK versions or plain dicts keep working.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)


def _camel(name: str) -> str:
    """snake_case to camelCase (``start_index`` -> ``startIndex``)."""
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _get(obj: Any, name: str) -> Any:
    """Read an attribute or dict key by snake_case, falling back to camelCase."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        return obj.get(_camel(name))
    value = getattr(obj, name, None)
    if value is None:
        value = getattr(obj, _camel(name), None)
    return value


def _normalize_chunk(chunk: Any) -> Optional[dict]:
    web = _get(chunk, "web")
    if web is not None:
        return {
            "kind": "web",
            "uri": _get(web, "uri"),
            "title": _get(web, "title"),
            "domain": _get(web, "domain"),
            "place_id": None,
        }
    maps = _get(chunk, "maps")
    if maps is not None:
        return {
            "kind": "maps",
            "uri": _get(maps, "uri"),
            "title": _get(maps, "title"),
            "domain": None,
            "place_id": _get(maps, "place_id"),
        }
    return None


def _chunk_key(chunk: dict) -> Optional[str]:
    """A chunk's identity: its uri, falling back to its place id."""
    if chunk.get("uri"):
        return chunk["uri"]
    if chunk.get("place_id"):
        return str(chunk["place_id"])
    return None


@dataclass
class GroundingAccumulator:
    """Cumulative grounding state of one API call (chunks, supports, queries)."""

    chunks: List[dict] = field(default_factory=list)
    supports: List[dict] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    entry_point_html: str = ""

    def add(self, gm: Any) -> None:
        """Merge one GroundingMetadata (SDK object or dict); None is ignored."""
        if gm is None:
            return
        new_chunks = []
        for chunk in _get(gm, "grounding_chunks") or []:
            norm = _normalize_chunk(chunk)
            if norm is not None and _chunk_key(norm) is not None:
                new_chunks.append(norm)
        if new_chunks:
            current_keys = [_chunk_key(c) for c in self.chunks]
            new_keys = [_chunk_key(c) for c in new_chunks]
            if new_keys[: len(current_keys)] == current_keys:
                # A cumulative snapshot of the same stream: replace, don't grow.
                self.chunks = list(new_chunks)
            else:
                # Documented streaming behaviour: chunks arrive incrementally.
                self.chunks.extend(new_chunks)
        seen = {
            (s["start"], s["end"], s["text"], tuple(s["indices"])) for s in self.supports
        }
        for support in _get(gm, "grounding_supports") or []:
            segment = _get(support, "segment")
            entry = {
                "start": _get(segment, "start_index"),
                "end": _get(segment, "end_index"),
                "text": _get(segment, "text") or "",
                "indices": list(_get(support, "grounding_chunk_indices") or []),
            }
            key = (entry["start"], entry["end"], entry["text"], tuple(entry["indices"]))
            if key not in seen:
                seen.add(key)
                self.supports.append(entry)
        for query in _get(gm, "web_search_queries") or []:
            if query and query not in self.queries:
                self.queries.append(query)
        rendered = _get(_get(gm, "search_entry_point"), "rendered_content")
        if rendered:
            self.entry_point_html = rendered

    def empty(self) -> bool:
        return not self.chunks and not self.queries


def locate_support_end(text_bytes: bytes, support: dict) -> Optional[int]:
    """The byte offset where a support's segment actually ends in ``text_bytes``.

    The API's end_index can be off by a few bytes, so a support that carries
    its segment text is re-located; a text-less one is snapped forward over the
    tail of a multi-byte UTF-8 sequence it may land inside.
    """
    text = support.get("text") or ""
    end = support.get("end")
    if text:
        needle = text.encode("utf-8")
        ends = []
        start = 0
        while True:
            index = text_bytes.find(needle, start)
            if index == -1:
                break
            ends.append(index + len(needle))
            start = index + 1
        if not ends:
            return None
        if end is None:
            return ends[0]
        return min(ends, key=lambda e: (abs(e - end), e))
    if end is not None and 0 < end <= len(text_bytes):
        pos = end
        while pos < len(text_bytes) and (text_bytes[pos] & 0xC0) == 0x80:
            pos += 1
        return pos
    return None


def byte_to_char_index(text: str, byte_off: int) -> int:
    """Map a UTF-8 byte offset into ``text`` onto a character index."""
    return len(text.encode("utf-8")[:byte_off].decode("utf-8", "ignore"))


def _fence_line_spans(text: str) -> List[Tuple[int, int]]:
    """(line_start, position after the last backtick) of every ``` fence line."""
    spans = []
    offset = 0
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            spans.append((offset, offset + len(line)))
        offset += len(line) + 1
    return spans


def _line_end(text: str, pos: int) -> int:
    closer = text.find("\n", pos)
    return closer + 1 if closer != -1 else len(text)


def safe_insertion_point(text: str, pos: int) -> int:
    """Move ``pos`` so a citation marker can't split code, spans or words."""
    pos = max(0, min(int(pos), len(text)))
    fences = _fence_line_spans(text)
    opener = None
    for span in fences:
        if _line_end(text, span[1]) <= pos:
            opener = span
        else:
            break
    if opener is not None:
        index = fences.index(opener) + 1
        if index < len(fences):
            return _line_end(text, fences[index][1])
        return len(text)
    if text[:pos].count("`") % 2 == 1:
        closer = text.find("`", pos)
        return closer + 1 if closer != -1 else len(text)
    link = text.rfind("](", 0, pos)
    if link != -1:
        closer = text.find(")", link + 2)
        if closer != -1 and closer >= pos:
            return closer + 1
    while pos > 0 and text[pos - 1] in " \t\r\n":
        pos -= 1
    if 0 < pos < len(text) and text[pos - 1].isalnum() and text[pos].isalnum():
        while pos < len(text) and text[pos].isalnum():
            pos += 1
    return pos


def insert_citation_markers(text: str, markers: List[Tuple[int, List[int]]]) -> str:
    """Insert ``[n]`` citation runs at the (safe) positions of ``markers``."""
    if not markers:
        return text
    groups: Dict[int, set] = {}
    for pos, numbers in markers:
        if not numbers:
            continue
        point = safe_insertion_point(text, pos)
        groups.setdefault(point, set()).update(numbers)
    for point in sorted(groups, reverse=True):
        run = "".join(f"[{n}]" for n in sorted(groups[point]))
        if text[point : point + len(run)] == run:
            continue
        text = text[:point] + run + text[point:]
    return text


def _normalize_url(url: str) -> str:
    """Citation key of a URL: lower-cased scheme and host, trailing / stripped."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    rest = ""
    if parsed.query:
        rest += "?" + parsed.query
    if parsed.fragment:
        rest += "#" + parsed.fragment
    return f"{(parsed.scheme or '').lower()}://{(parsed.netloc or '').lower()}{path}{rest}"


class CitationRegistry:
    """Maps citation keys to stable numbers across the API calls of one turn."""

    def __init__(self, offset: int = 0):
        self.offset = offset
        self.sources: Dict[int, dict] = {}
        self._by_key: Dict[str, int] = {}

    def number_for(self, key: str, info: dict) -> Tuple[int, bool]:
        """The number for ``key`` and whether it was assigned just now."""
        if key in self._by_key:
            return self._by_key[key], False
        number = len(self.sources) + self.offset + 1
        self._by_key[key] = number
        self.sources[number] = dict(info)
        return number, True


def _chunk_citation(chunk: dict, resolved: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    uri = chunk.get("uri")
    url = resolved.get(uri, uri) if uri else None
    if chunk.get("place_id"):
        return "place:" + str(chunk["place_id"]), url
    return _normalize_url(url) if url else None, url


def cite_call_text(
    text: str,
    acc: GroundingAccumulator,
    registry: CitationRegistry,
    resolved: Dict[str, str],
) -> Tuple[str, List[int], Dict[int, List[str]]]:
    """Cite one API call's answer text with the turn's shared registry.

    Returns the text with markers, the numbers that were new in this call
    (ascending) and ``{number: [supporting segment texts]}``.
    """
    data = text.encode("utf-8")
    markers: List[Tuple[int, List[int]]] = []
    new_numbers = set()
    references: Dict[int, List[str]] = {}
    for support in acc.supports:
        end = locate_support_end(data, support)
        if end is None:
            continue
        point = safe_insertion_point(text, byte_to_char_index(text, end))
        numbers = []
        for index in support["indices"]:
            if not 0 <= index < len(acc.chunks):
                continue
            chunk = acc.chunks[index]
            key, url = _chunk_citation(chunk, resolved)
            if key is None:
                continue
            number, is_new = registry.number_for(
                key,
                {"title": chunk.get("title"), "url": url, "kind": chunk.get("kind")},
            )
            numbers.append(number)
            if is_new:
                new_numbers.add(number)
            bucket = references.setdefault(number, [])
            if support["text"] and support["text"] not in bucket:
                bucket.append(support["text"])
        if numbers:
            markers.append((point, numbers))
    for chunk in acc.chunks:
        key, url = _chunk_citation(chunk, resolved)
        if key is None:
            continue
        registry.number_for(
            key, {"title": chunk.get("title"), "url": url, "kind": chunk.get("kind")}
        )
    return insert_citation_markers(text, markers), sorted(new_numbers), references


_PATTERNS_SUGGESTION_CHIP = re.compile(
    r"<a\b([^>]*)>(.*?)</a>", flags=re.S | re.I
)
_PATTERNS_CHIP_CLASS = re.compile(r'class\s*=\s*["\']([^"\']*)["\']', flags=re.I)
_PATTERNS_CHIP_HREF = re.compile(r'href\s*=\s*["\']([^"\']*)["\']', flags=re.I)


def extract_suggestion_links(html_text: str, queries: List[str]) -> List[Tuple[str, str]]:
    """(label, href) pairs from the Search Suggestions chips of the entry point.

    Falls back to Google search URLs built from the queries when the entry
    point carries no chips (or no HTML at all).
    """
    links: List[Tuple[str, str]] = []
    for match in _PATTERNS_SUGGESTION_CHIP.finditer(html_text or ""):
        attrs, label = match.group(1), match.group(2)
        class_match = _PATTERNS_CHIP_CLASS.search(attrs)
        if not class_match or "chip" not in class_match.group(1).split():
            continue
        href_match = _PATTERNS_CHIP_HREF.search(attrs)
        if not href_match:
            continue
        clean = re.sub(r"<[^>]+>", "", label).strip()
        links.append((html.unescape(clean), html.unescape(href_match.group(1))))
    if links:
        return links
    return [
        (query, "https://www.google.com/search?q=" + quote_plus(query))
        for query in (queries or [])
    ]


def prepare_entry_point_embed(html_text: str) -> str:
    """Make the suggestion embed open links in a new tab, not the chat."""
    if not html_text:
        return ""
    index = html_text.find("<head>")
    if index != -1:
        at = index + len("<head>")
        return html_text[:at] + '<base target="_blank">' + html_text[at:]
    return '<base target="_blank">' + html_text


def _rag_citation_offset(sources: List[dict]) -> int:
    """How many RAG source ids OpenWebUI already cited, so pipe numbers follow.

    Counts unique ids exactly the way ``get_source_context`` in
    open_webui/utils/middleware.py does: ``meta.get("source") or
    source["source"]["id"]`` over ``zip(document, metadata)``.
    """
    seen = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        for _doc, meta in zip(source.get("document") or [], source.get("metadata") or []):
            meta = meta if isinstance(meta, dict) else {}
            source_id = meta.get("source") or (source.get("source") or {}).get("id") or "N/A"
            seen.add(str(source_id))
    return len(seen)
