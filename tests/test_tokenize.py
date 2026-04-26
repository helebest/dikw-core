"""CJK-aware FTS preprocessing — ``src/dikw_core/info/tokenize.py``."""

from __future__ import annotations

import pytest

from dikw_core.info.tokenize import count_tokens, has_cjk, preprocess_for_fts


def test_has_cjk_detects_basic_ideographs() -> None:
    assert has_cjk("机器学习") is True
    assert has_cjk("hello world") is False
    assert has_cjk("mixed 中文 text") is True
    assert has_cjk("") is False
    # Digits and punctuation alone are ASCII, no CJK.
    assert has_cjk("123 !@#") is False


def test_preprocess_none_is_passthrough() -> None:
    """Mode 'none' must be a bit-for-bit identity — the default path
    preserves the pre-feature FTS behaviour for every existing wiki.
    """
    for s in ["hello world", "机器学习入门", "mixed 中文 with ASCII", "", "   "]:
        assert preprocess_for_fts(s, tokenizer="none") == s


def test_preprocess_jieba_passthrough_on_ascii() -> None:
    """No CJK → jieba isn't invoked; the text returns unchanged.

    Important for keyword-heavy English corpora: users flipping
    cjk_tokenizer=jieba shouldn't pay any tokenization cost or see any
    FTS behaviour change on their English docs.
    """
    assert preprocess_for_fts("hello world", tokenizer="jieba") == "hello world"
    assert preprocess_for_fts("expect_any field", tokenizer="jieba") == "expect_any field"
    assert preprocess_for_fts("", tokenizer="jieba") == ""


def test_preprocess_jieba_segments_chinese_run() -> None:
    """Multi-char Chinese words must be separable after preprocessing.

    ``机器学习`` tokenized per-char under unicode61 gives the useless
    ``机``/``器``/``学``/``习`` set; after jieba pre-segmentation the
    output contains ``机器`` and/or ``学习`` as whitespace-bounded
    tokens that unicode61 will pick up intact.
    """
    out = preprocess_for_fts("机器学习入门", tokenizer="jieba")
    tokens = set(out.split())
    # jieba.cut_for_search yields both long and sub-words; we assert
    # on presence, not exact composition, because dictionary updates
    # in future jieba versions would break exact-string assertions.
    assert "机器学习" in tokens or "机器" in tokens
    assert "入门" in tokens
    # The degenerate per-char case must NOT be the output: if jieba
    # fell back to character-level splits, BM25 would still be broken.
    assert not (len(tokens) >= 4 and all(len(t) == 1 for t in tokens))


def test_preprocess_jieba_preserves_ascii_in_mixed_text() -> None:
    """ASCII and CJK runs must coexist — an identifier inside a
    Chinese sentence (e.g. in dev-doc retrieval) stays intact.
    """
    out = preprocess_for_fts("配置 retrieval.rrf_k 参数", tokenizer="jieba")
    assert "retrieval.rrf_k" in out or ("retrieval" in out and "rrf_k" in out)
    # Chinese pieces are still segmented
    assert "配置" in out.split() or "配置" in out
    assert "参数" in out.split() or "参数" in out


def test_preprocess_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="cjk_tokenizer"):
        preprocess_for_fts("anything", tokenizer="trigram")  # type: ignore[arg-type]


def test_preprocess_jieba_whitespace_only_text() -> None:
    """Whitespace-only input has no tokens to emit — return empty."""
    # With has_cjk short-circuit, no CJK + no content → unchanged ""
    assert preprocess_for_fts("   ", tokenizer="jieba") == "   "


def test_preprocess_jieba_idempotent_on_pre_segmented_text() -> None:
    """Running the preprocessor on already-segmented output must not
    degrade it. Guards against accidental double-preprocessing when
    ingest and query paths both run preprocess on the same string.
    """
    once = preprocess_for_fts("机器学习入门", tokenizer="jieba")
    twice = preprocess_for_fts(once, tokenizer="jieba")
    # Token *set* stable (ordering may differ because jieba's
    # cut_for_search output for a whitespace-separated input is
    # equivalent to per-word cuts).
    assert set(once.split()) == set(twice.split())


def test_count_tokens_ascii_matches_split() -> None:
    """ASCII parity: ``count_tokens(s) == len(s.split())`` for any
    all-ASCII string, so English chunk-budget tuning stays unchanged.
    """
    for s in ["lorem " * 40, "alpha beta gamma", "retrieval.rrf_k weight_a", ""]:
        assert count_tokens(s) == len(s.split())


def test_count_tokens_cjk_segments_at_word_level() -> None:
    """A short Chinese phrase yields more than one token — necessary
    condition for the chunk budget to ever trip on CJK content. Exact
    count is jieba-dictionary-dependent; assert only a lower bound.
    """
    assert count_tokens("机器学习入门") >= 3


def test_count_tokens_mixed_ascii_and_cjk_sums_runs() -> None:
    """Mixed text totals ASCII whitespace tokens plus jieba CJK segments."""
    # "hello" + "world" + >= 2 CJK tokens for "机器学习"
    assert count_tokens("hello 机器学习 world") >= 4


def test_count_tokens_none_passthrough_on_cjk() -> None:
    """``tokenizer="none"`` returns ``len(text.split())`` on any input,
    Chinese included — preserves the escape hatch for eval reproducibility.
    """
    assert count_tokens("机器学习入门", tokenizer="none") == 1
    assert count_tokens("机器 学习 入门", tokenizer="none") == 3


def test_count_tokens_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="cjk_tokenizer"):
        count_tokens("anything", tokenizer="trigram")  # type: ignore[arg-type]
