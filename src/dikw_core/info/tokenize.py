"""CJK-aware preprocessing for the BM25 / FTS5 leg of hybrid search.

SQLite FTS5's default ``unicode61`` tokenizer segments CJK one
character at a time, which collapses BM25 to single-char IDF on
Chinese corpora. This module pre-segments CJK runs with ``jieba``
so the resulting text is whitespace-separated Chinese words plus
unchanged ASCII — ``unicode61`` then splits at the whitespace
boundaries we control. No custom FTS5 tokenizer, no C extension,
no schema migration.

The same preprocessor must run on both sides of the FTS query.
``RetrievalConfig.cjk_tokenizer`` is locked at first ingest for
the same reason ``embedding_dimensions`` is: flipping the knob
post-ingest makes the query side produce tokens that don't match
what's indexed.

``cjk_tokenizer="none"`` (the default) returns the text unchanged —
zero behavioural change for ASCII-only corpora. See
``evals/BASELINES.md`` for measured impact on CMTEB / T2Retrieval.
"""

from __future__ import annotations

import re
from typing import Literal

CjkTokenizer = Literal["none", "jieba"]

# Basic CJK Unified Ideographs (U+4E00 - U+9FFF). Enough for Chinese
# and Japanese Han. Kana, Hangul, and extension-B ideographs aren't
# covered; extending to JA/KO means widening this character class.
CJK_CHAR_CLASS = r"一-鿿"
_CJK_CHAR = re.compile(f"[{CJK_CHAR_CLASS}]")
_CJK_RUN = re.compile(f"[{CJK_CHAR_CLASS}]+")


def has_cjk(text: str) -> bool:
    """True iff ``text`` contains at least one basic CJK ideograph."""
    return _CJK_CHAR.search(text) is not None


def preprocess_for_fts(text: str, *, tokenizer: CjkTokenizer) -> str:
    """Return ``text`` segmented for FTS5 indexing + query-building.

    - ``tokenizer="none"``: passthrough.
    - ``tokenizer="jieba"``: segments only the CJK runs; ASCII is
      left verbatim so code identifiers (``retrieval.rrf_k``) survive.
    """
    if tokenizer == "none":
        return text
    if tokenizer == "jieba":
        if not has_cjk(text):
            return text
        return _jieba_segment(text)
    raise ValueError(f"unknown cjk_tokenizer: {tokenizer!r}")


def _jieba_segment(text: str) -> str:
    """Word-level CJK segmentation, applied only to CJK runs.

    Feeding the whole string to jieba would mangle ASCII punctuation
    (``retrieval.rrf_k`` → ``retrieval . rrf _ k``). Instead we walk
    the text with ``_CJK_RUN.finditer`` and segment only the matched
    runs. ``cut_for_search`` yields both long words and their subwords
    (``机器学习`` → ``机器 学习 机器学习``), which gives FTS5 more
    chances to match partial-phrase queries; BM25 IDF recovers precision.

    Local ``import jieba`` so the default ``"none"`` path never pays
    the dictionary-load penalty. After the first call Python caches
    the module in ``sys.modules``.
    """
    import jieba

    parts: list[str] = []
    last = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > last:
            parts.append(text[last : m.start()])
        segmented = " ".join(
            t for t in jieba.cut_for_search(m.group(0)) if t.strip()
        )
        # Whitespace guards stop CJK tokens from fusing with adjacent ASCII.
        parts.append(f" {segmented} ")
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return "".join(parts)


def count_tokens(text: str, *, tokenizer: CjkTokenizer = "jieba") -> int:
    """Token count for chunk-budgeting.

    ``tokenizer="jieba"`` segments CJK runs with ``jieba.cut_for_search``
    and counts whitespace tokens elsewhere; an all-ASCII body returns
    exactly ``len(text.split())``. ``tokenizer="none"`` is the
    whitespace-only escape hatch for eval reproducibility.
    """
    if tokenizer == "none":
        return len(text.split())
    if tokenizer != "jieba":
        raise ValueError(f"unknown cjk_tokenizer: {tokenizer!r}")
    if not has_cjk(text):
        return len(text.split())

    import jieba

    total = 0
    last = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > last:
            total += len(text[last : m.start()].split())
        total += sum(1 for t in jieba.cut_for_search(m.group(0)) if t.strip())
        last = m.end()
    if last < len(text):
        total += len(text[last:].split())
    return total


def initialize_jieba() -> None:
    """Load jieba's dictionary now instead of lazily on first segment.

    Call from ingest entrypoints when ``cjk_tokenizer="jieba"`` so the
    ~0.3s dictionary load happens during setup rather than the first
    ``replace_chunks`` call. Safe to call multiple times — jieba
    no-ops subsequent invocations via its internal lock.
    """
    import jieba

    jieba.initialize()
