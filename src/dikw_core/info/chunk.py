"""Heading-aware markdown chunker.

Strategy (inspired by ``qmd/src/store.ts:257-310`` and
``mineru-doc-explorer/src/store.ts``):

* Split the body into paragraphs separated by blank lines.
* Greedily pack paragraphs into chunks of up to ``max_tokens`` tokens.
* Force a chunk boundary immediately before any heading if the current chunk
  already has substantive content — this keeps headings at chunk starts,
  which matches retrieval intuition.
* Carry the tail of the previous chunk into the next as a paragraph-level
  overlap (~``overlap_ratio`` of ``max_tokens``). Overlap is by paragraph,
  not by word, so character offsets stay exact and no paragraph is split
  mid-sentence.

Token counting uses whitespace-split — cheap, deterministic, and close enough
for retrieval. Swap in a real tokenizer later if the Phase-1 eval shows drift.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from pydantic import BaseModel

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_PARA_SEP = re.compile(r"\n\s*\n")


class MarkdownChunk(BaseModel):
    """A paragraph-aligned slice of a document body."""

    seq: int
    start: int  # inclusive char offset
    end: int    # exclusive char offset
    text: str


class _Para(NamedTuple):
    start: int
    end: int
    tokens: int


def _paragraph_spans(text: str) -> list[_Para]:
    """Return (start, end, tokens) for each paragraph, preserving char offsets."""
    spans: list[_Para] = []
    if not text.strip():
        return spans
    pos = 0
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in "\r\n":
            pos += 1
        if pos >= n:
            break
        start = pos
        m = _PARA_SEP.search(text, pos)
        if m is None:
            end = n
            pos = n
        else:
            end = m.start()
            pos = m.end()
        # strip trailing whitespace within the paragraph span so tokens is accurate,
        # but keep char offsets pointing into the original body
        para_text = text[start:end]
        stripped_end = end - (len(para_text) - len(para_text.rstrip()))
        if stripped_end <= start:
            continue
        tokens = len(para_text.split())
        if tokens == 0:
            continue
        spans.append(_Para(start, stripped_end, tokens))
    return spans


def chunk_markdown(
    body: str,
    *,
    max_tokens: int = 900,
    overlap_ratio: float = 0.15,
) -> list[MarkdownChunk]:
    """Chunk ``body`` into (mostly) ``max_tokens``-sized paragraph-aligned windows."""
    if not body:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")

    paras = _paragraph_spans(body)
    if not paras:
        return []

    max_overlap = int(max_tokens * overlap_ratio)
    chunks: list[MarkdownChunk] = []
    current: list[_Para] = []
    current_tokens = 0

    def _is_heading(p: _Para) -> bool:
        return bool(_HEADING.match(body[p.start : p.end]))

    def _flush_and_overlap() -> list[_Para]:
        """Emit current chunk and return the tail paragraphs to carry forward.

        Always carries the final paragraph (unless it's oversize on its own —
        that would risk an infinite re-emission loop). Then adds earlier
        paragraphs while staying under the overlap budget.
        """
        if not current:
            return []
        start = current[0].start
        end = current[-1].end
        chunks.append(
            MarkdownChunk(
                seq=len(chunks),
                start=start,
                end=end,
                text=body[start:end],
            )
        )
        overlap: list[_Para] = []
        total = 0
        last = current[-1]
        if last.tokens < max_tokens:
            overlap.append(last)
            total = last.tokens
        for p in reversed(current[:-1]):
            if total + p.tokens > max_overlap:
                break
            overlap.insert(0, p)
            total += p.tokens
        return overlap

    for p in paras:
        heading = _is_heading(p)

        # 1) Start a new chunk before a heading when the current chunk has real content.
        if heading and current_tokens > max_overlap and current:
            carry = _flush_and_overlap()
            current = list(carry)
            current_tokens = sum(c.tokens for c in current)

        # 2) Close the current chunk if adding this paragraph would blow the budget.
        if current and current_tokens + p.tokens > max_tokens:
            carry = _flush_and_overlap()
            current = list(carry)
            current_tokens = sum(c.tokens for c in current)
            # Guard against an endless loop: if the carry itself already overflows
            # (shouldn't happen because of the overlap cap) discard it.
            if current_tokens + p.tokens > max_tokens and current:
                current = []
                current_tokens = 0

        current.append(p)
        current_tokens += p.tokens

    if current:
        start = current[0].start
        end = current[-1].end
        chunks.append(
            MarkdownChunk(
                seq=len(chunks), start=start, end=end, text=body[start:end]
            )
        )

    return chunks
