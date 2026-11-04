"""Checks for grounding and citations (SPEC section 6), criteria 1-7."""
from __future__ import annotations

from google.genai import types as gtypes

from _testlib import load_pipe_module


def _web(uri, title, domain):
    return gtypes.GroundingChunk(
        web=gtypes.GroundingChunkWeb(uri=uri, title=title, domain=domain)
    )


def test_grounding_accumulator_add():
    # criterion 1
    mod = load_pipe_module()
    acc = mod.GroundingAccumulator()
    acc.add(
        gtypes.GroundingMetadata(
            grounding_chunks=[
                _web("https://a.example/1", "A", "a.example"),
                _web("https://b.example/2", "B", "b.example"),
            ],
            grounding_supports=[
                gtypes.GroundingSupport(
                    segment=gtypes.Segment(
                        start_index=0, end_index=5, text="hello"
                    ),
                    grounding_chunk_indices=[0],
                )
            ],
        )
    )
    acc.add(
        gtypes.GroundingMetadata(
            grounding_chunks=[_web("https://c.example/3", "C", "c.example")],
            grounding_supports=[
                gtypes.GroundingSupport(
                    segment=gtypes.Segment(
                        start_index=6, end_index=11, text="world"
                    ),
                    grounding_chunk_indices=[2],
                )
            ],
        )
    )
    assert len(acc.chunks) == 3
    assert len(acc.supports) == 2
    acc.add(
        gtypes.GroundingMetadata(
            grounding_chunks=[
                _web("https://a.example/1", "A", "a.example"),
                _web("https://b.example/2", "B", "b.example"),
                _web("https://c.example/3", "C", "c.example"),
            ]
        )
    )
    assert len(acc.chunks) == 3


def test_locate_support_end_and_byte_to_char():
    # criterion 2
    mod = load_pipe_module()
    text = "Café 😀 是 Paris is big. Next."
    data = text.encode("utf-8")
    support = {"start": 15, "end": 26, "text": "Paris is big.", "indices": []}
    end = mod.locate_support_end(data, support)
    assert end == 28
    assert mod.byte_to_char_index(text, end) == 22
    emoji_support = {"start": 6, "end": 8, "text": "", "indices": []}
    assert mod.locate_support_end(data, emoji_support) == 10


def test_cite_call_text():
    # criterion 3
    mod = load_pipe_module()
    acc = mod.GroundingAccumulator()
    acc.add(
        {
            "grounding_chunks": [
                {"web": {"uri": "https://a.example/1", "title": "A", "domain": "a.example"}},
                {"web": {"uri": "https://b.example/2", "title": "B", "domain": "b.example"}},
            ],
            "grounding_supports": [
                {
                    "segment": {
                        "start_index": 0,
                        "end_index": 32,
                        "text": "Paris is the capital of France.",
                    },
                    "grounding_chunk_indices": [0],
                },
                {
                    "segment": {
                        "start_index": 33,
                        "end_index": 50,
                        "text": "It has 2M people.",
                    },
                    "grounding_chunk_indices": [0, 1],
                },
            ],
        }
    )
    text = "Paris is the capital of France. It has 2M people.\n"
    registry = mod.CitationRegistry(offset=2)
    new_text, new_numbers, _references = mod.cite_call_text(text, acc, registry, {})
    assert new_text == "Paris is the capital of France.[3] It has 2M people.[3][4]\n"
    assert new_numbers == [3, 4]


def test_insert_citation_markers_safe_points():
    # criterion 4
    mod = load_pipe_module()
    fenced = mod.insert_citation_markers("```\ncode\n```\n", [(5, [1])])
    assert fenced == "```\ncode\n```\n[1]"
    linked = mod.insert_citation_markers("See [x](https://a.b/c) now", [(12, [2])])
    assert linked == "See [x](https://a.b/c)[2] now"


def test_registry_across_calls():
    # criterion 5
    mod = load_pipe_module()
    registry = mod.CitationRegistry()
    uri_a = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/aaa"
    uri_b = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/bbb"
    uri_c = "https://other.example/page"
    resolved = {uri_a: "https://real.example/page", uri_b: "https://real.example/page"}
    acc1 = mod.GroundingAccumulator()
    acc1.add(
        {
            "grounding_chunks": [
                {"web": {"uri": uri_a, "title": "A", "domain": "a.example"}}
            ],
            "grounding_supports": [
                {
                    "segment": {"start_index": 0, "end_index": 6, "text": "Alpha."},
                    "grounding_chunk_indices": [0],
                }
            ],
        }
    )
    _text1, new1, _refs1 = mod.cite_call_text("Alpha.\n", acc1, registry, resolved)
    assert new1 == [1]
    acc2 = mod.GroundingAccumulator()
    acc2.add(
        {
            "grounding_chunks": [
                {"web": {"uri": uri_b, "title": "B", "domain": "b.example"}},
                {"web": {"uri": uri_c, "title": "C", "domain": "other.example"}},
            ],
            "grounding_supports": [
                {
                    "segment": {"start_index": 0, "end_index": 5, "text": "Beta."},
                    "grounding_chunk_indices": [0],
                },
                {
                    "segment": {"start_index": 6, "end_index": 11, "text": "Gamma."},
                    "grounding_chunk_indices": [1],
                },
            ],
        }
    )
    text2, new2, _refs2 = mod.cite_call_text("Beta. Gamma.\n", acc2, registry, resolved)
    assert new2 == [2]
    assert text2 == "Beta.[1] Gamma.[2]\n"


def test_suggestion_links_and_entry_point():
    # criterion 6
    mod = load_pipe_module()
    html_text = (
        "<html><body>"
        '<a class="chip" href="https://www.google.com/search?q=a&amp;b=1">Label A</a>'
        '<a href="https://www.google.com/search?q=c" class="chip">Label B</a>'
        "</body></html>"
    )
    links = mod.extract_suggestion_links(html_text, [])
    assert links == [
        ("Label A", "https://www.google.com/search?q=a&b=1"),
        ("Label B", "https://www.google.com/search?q=c"),
    ]
    assert mod.extract_suggestion_links("", ["x y"]) == [
        ("x y", "https://www.google.com/search?q=x+y")
    ]
    embed = mod.prepare_entry_point_embed("<html><head></head><body></body></html>")
    assert '<head><base target="_blank">' in embed


def test_rag_citation_offset():
    # criterion 7
    mod = load_pipe_module()
    sources = [
        {
            "source": {"name": "a.pdf", "id": "s1"},
            "document": ["doc1", "doc2"],
            "metadata": [{"source": "a.pdf"}, {"source": "a.pdf"}],
        },
        {
            "source": {"name": "w", "id": "w"},
            "document": ["doc3"],
            "metadata": [{}],
        },
    ]
    assert mod._rag_citation_offset(sources) == 2
