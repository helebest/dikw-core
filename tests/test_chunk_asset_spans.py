"""Tests for span-aware markdown chunking.

The contract: when ``chunk_markdown`` is given a list of ``atomic_spans``
(typically the byte offsets of image references), every span must be fully
contained in at least one chunk — the chunker may never split the literal
``![…](…)`` / ``![[…]]`` syntax across chunk boundaries.

In the v1 paragraph-aligned chunker this is naturally true for any
single-line reference; the explicit parameter exists to (a) document the
guarantee on the API surface, (b) hard-fail (rather than silently corrupt)
if a future chunker variant or pathological multi-line reference would
violate it.
"""

from __future__ import annotations

from collections.abc import Sequence

from dikw_core.domains.data.backends.markdown import extract_image_refs
from dikw_core.domains.info.chunk import chunk_markdown


def _spans_from_refs(body: str) -> list[tuple[int, int]]:
    return [(r.start, r.end) for r in extract_image_refs(body)]


def _assert_each_span_in_some_chunk(
    body: str, spans: Sequence[tuple[int, int]], chunks: list
) -> None:
    """Every span must lie fully inside at least one chunk's [start, end)."""
    for s_start, s_end in spans:
        contained = any(c.start <= s_start and s_end <= c.end for c in chunks)
        assert contained, (
            f"span [{s_start}:{s_end}] = {body[s_start:s_end]!r} was split "
            f"across chunk boundaries: {[(c.start, c.end) for c in chunks]}"
        )


def test_no_atomic_spans_preserves_existing_behavior() -> None:
    body = "Para one.\n\nPara two.\n\nPara three."
    a = chunk_markdown(body)
    b = chunk_markdown(body, atomic_spans=())
    assert [c.text for c in a] == [c.text for c in b]
    assert [(c.start, c.end) for c in a] == [(c.start, c.end) for c in b]


def test_image_in_short_paragraph_is_preserved_intact() -> None:
    body = "Intro paragraph.\n\nHere is a diagram: ![arch](arch.png) inline.\n\nMore text."
    spans = _spans_from_refs(body)
    chunks = chunk_markdown(body, atomic_spans=spans)
    _assert_each_span_in_some_chunk(body, spans, chunks)


def test_image_at_paragraph_boundary_stays_in_one_chunk() -> None:
    """Heading right before, image at start of next paragraph, then prose —
    chunker forces a boundary before headings; the image-bearing paragraph
    must still hold its ref atomically."""
    body = (
        "# Section A\n\n"
        "Lots of text in A. " * 30
        + "\n\n# Section B\n\n"
        "![hero](hero.png) Body of B.\n\n"
        "More B prose."
    )
    spans = _spans_from_refs(body)
    chunks = chunk_markdown(body, atomic_spans=spans)
    _assert_each_span_in_some_chunk(body, spans, chunks)


def test_multiple_images_across_many_paragraphs_each_intact() -> None:
    body = "\n\n".join(
        [
            "Intro.",
            "First fig: ![a](a.png).",
            "Words " * 80,  # forces a chunk boundary between figs
            "Second fig: ![b](b.png).",
            "Words " * 80,
            "Third: ![[c.svg|150]] inline.",
            "Tail prose.",
        ]
    )
    spans = _spans_from_refs(body)
    assert len(spans) == 3
    chunks = chunk_markdown(body, atomic_spans=spans, max_tokens=200)
    _assert_each_span_in_some_chunk(body, spans, chunks)


def test_consecutive_refs_in_one_paragraph_share_one_chunk() -> None:
    body = (
        "Intro.\n\n"
        "A row of figures: ![one](1.png) ![two](2.png) ![[three.png|200]] all together.\n\n"
        "Outro."
    )
    spans = _spans_from_refs(body)
    assert len(spans) == 3
    chunks = chunk_markdown(body, atomic_spans=spans)
    _assert_each_span_in_some_chunk(body, spans, chunks)
    # All three live in the same chunk because they're in one paragraph.
    chunk_ids_for_each = []
    for s_start, s_end in spans:
        for i, c in enumerate(chunks):
            if c.start <= s_start and s_end <= c.end:
                chunk_ids_for_each.append(i)
                break
    assert len(set(chunk_ids_for_each)) == 1, (
        f"refs landed in different chunks: {chunk_ids_for_each}"
    )


def test_wikilink_with_dimension_alias_intact() -> None:
    """Wikilinks with ``|150`` dimension or ``|caption`` alias must not split
    on the pipe character."""
    body = "Embed: ![[architecture.png|400]] right here."
    spans = _spans_from_refs(body)
    chunks = chunk_markdown(body, atomic_spans=spans)
    _assert_each_span_in_some_chunk(body, spans, chunks)
    # And the literal syntax survives in chunk text.
    assert any("![[architecture.png|400]]" in c.text for c in chunks)


def test_atomic_spans_violation_raises() -> None:
    """If a caller passes a span that the chunker can't honor (e.g. one
    that crosses a paragraph boundary), the function fails loudly rather
    than silently producing a chunk that splits the span."""
    import pytest

    body = "Para A.\n\nPara B."
    # Synthetic span covers from middle of Para A through Para B — impossible
    # to honor with paragraph-aligned chunking that wants to cut on \n\n.
    bad_span = (3, len(body) - 2)
    # The chunker still needs to be told this is one atomic unit.
    with pytest.raises(ValueError, match="atomic span"):
        chunk_markdown(body, atomic_spans=[bad_span], max_tokens=2)
