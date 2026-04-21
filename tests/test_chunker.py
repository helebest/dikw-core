from __future__ import annotations

from dikw_core.info.chunk import chunk_markdown


def test_empty_body_returns_no_chunks() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  \n") == []


def test_short_body_is_single_chunk() -> None:
    body = "# Title\n\nHello world paragraph."
    chunks = chunk_markdown(body, max_tokens=900)
    assert len(chunks) == 1
    assert chunks[0].seq == 0
    assert body[chunks[0].start : chunks[0].end] == chunks[0].text


def test_heading_forces_chunk_boundary() -> None:
    para_a = "alpha " * 200
    para_b = "beta " * 200
    body = f"# A\n\n{para_a}\n\n## B\n\n{para_b}"
    chunks = chunk_markdown(body, max_tokens=300, overlap_ratio=0.1)
    assert len(chunks) >= 2
    # at least one chunk should begin with an ATX heading
    assert any(chunks[i].text.lstrip().startswith("#") for i in range(len(chunks)))


def test_overflow_splits_with_overlap() -> None:
    # 10 paragraphs of ~40 tokens each → should overflow a 100-token budget.
    paragraphs = ["lorem " * 40 for _ in range(10)]
    body = "\n\n".join(p.strip() for p in paragraphs)
    chunks = chunk_markdown(body, max_tokens=100, overlap_ratio=0.2)
    assert len(chunks) >= 3
    # consecutive chunks overlap in character range (end_N > start_{N+1})
    overlaps = [chunks[i].end > chunks[i + 1].start for i in range(len(chunks) - 1)]
    assert any(overlaps)


def test_char_offsets_round_trip() -> None:
    body = "# Intro\n\nAlpha beta.\n\n## Deep\n\nGamma delta epsilon.\n"
    for c in chunk_markdown(body, max_tokens=5):
        assert body[c.start : c.end] == c.text


def test_paragraph_larger_than_budget_does_not_loop() -> None:
    huge = "word " * 500
    body = f"{huge}\n\nsmall paragraph"
    chunks = chunk_markdown(body, max_tokens=100, overlap_ratio=0.1)
    # Just terminating is the contract; we also expect both paragraphs represented.
    joined = " ".join(c.text for c in chunks)
    assert "small paragraph" in joined
