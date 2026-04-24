"""CJK-aware preprocessing for the BM25 / FTS5 leg of hybrid search.

SQLite FTS5's default ``unicode61`` tokenizer segments CJK text one
character at a time. Single-character BM25 has no useful signal on
Chinese passage retrieval — high-frequency characters dominate IDF
and discriminating multi-character terms never form. Measured on
CMTEB T2Retrieval at commit ``6a8bc46``: nDCG@10 = 0.031, 91.7 % of
queries returned zero BM25 hits.

This module lets the caller pre-segment CJK runs with ``jieba``
before they reach FTS5. The resulting text is whitespace-separated
Chinese words plus unchanged ASCII, which ``unicode61`` then splits
at the whitespace boundaries we control. No custom FTS5 tokenizer,
no C extension, no schema migration — only two symmetric call sites
(ingest writes the segmented body, ``_sanitize_fts`` segments the
query).

The same preprocessor MUST run on both sides of the FTS query to
stay consistent. ``RetrievalConfig.cjk_tokenizer`` is locked at
first ingest for the same reason ``embedding_dimensions`` is: the
indexed form of every doc reflects the tokenizer chosen when it
was written, and flipping the knob post-ingest makes the query side
produce tokens that don't match what's in the table.

``cjk_tokenizer="none"`` (the default) returns the text unchanged
— zero behavioural change for existing ASCII-only corpora.
"""

from __future__ import annotations

import re
from typing import Literal

CjkTokenizer = Literal["none", "jieba"]

# Basic CJK Unified Ideographs (U+4E00 - U+9FFF). Enough to trigger
# jieba on Chinese / Japanese Han characters. Kana, Hangul, and
# extension-B ideographs aren't covered; if we ever target those
# corpora the detection regex is the only place to extend.
_CJK_CHAR = re.compile(r"[一-鿿]")
_CJK_RUN = re.compile(r"[一-鿿]+")


def has_cjk(text: str) -> bool:
    """True iff ``text`` contains at least one basic CJK ideograph."""
    return _CJK_CHAR.search(text) is not None


def preprocess_for_fts(text: str, *, tokenizer: CjkTokenizer) -> str:
    """Return ``text`` segmented for FTS5 indexing + query-building.

    - ``tokenizer="none"``: no-op passthrough. Keeps the pre-feature
      behaviour bit-for-bit so ASCII corpora are unaffected.
    - ``tokenizer="jieba"``: pure-ASCII passthrough (jieba is only
      invoked when CJK is actually present — saves startup cost and
      avoids mangling code/identifier queries). Mixed text has its
      CJK runs replaced by ``jieba.cut_for_search`` output joined by
      spaces; ASCII runs are left alone.
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

    We explicitly **do not** feed the whole string to jieba.
    ``jieba.cut_for_search`` treats ASCII punctuation as a separator,
    so ``retrieval.rrf_k`` comes back as ``retrieval . rrf _ k``,
    which destroys identifier retrieval in mixed-language dev docs.
    Instead we slice the text into alternating CJK / non-CJK runs
    via ``_CJK_RUN``, apply jieba only to the CJK pieces, and
    concatenate the result with whitespace so the unicode61
    tokenizer picks up each piece cleanly.

    ``cut_for_search`` on a CJK run produces a finer-grained split
    than ``cut`` — it yields both long words *and* their subwords
    (e.g., ``机器学习`` → ``机器 学习 机器学习``), giving FTS5 more
    chances to match partial-phrase queries. Precision is recovered
    by BM25's IDF weighting.

    Imports are local so wikis that don't enable jieba never pay the
    ~0.5 s dictionary-load penalty at startup. The first real call
    eats it; subsequent calls within the process are cached by jieba.
    """
    import jieba  # local: keeps the default path jieba-free

    parts: list[str] = []
    last = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > last:
            # Preserve the ASCII / punctuation run verbatim so
            # identifiers like ``retrieval.rrf_k`` stay intact for
            # ``_sanitize_fts`` to process with its own ASCII rules.
            parts.append(text[last : m.start()])
        cjk_run = m.group(0)
        segmented = " ".join(
            t for t in jieba.cut_for_search(cjk_run) if t.strip()
        )
        parts.append(" " + segmented + " ")  # whitespace guards prevent
        # CJK tokens from re-fusing with adjacent ASCII at search time
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return "".join(parts)
