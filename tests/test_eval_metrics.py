"""Metrics module lives in src, not tests — test the public import path."""

from __future__ import annotations

import math

import pytest

from dikw_core.eval.metrics import (
    hit_at_k,
    mean_hit_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k_matches_any_expected() -> None:
    assert hit_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert hit_at_k(["a", "b", "c"], ["x", "a"], 1) == 1.0
    assert hit_at_k(["a", "b", "c"], ["x"], 10) == 0.0


def test_hit_at_k_respects_k_cutoff() -> None:
    # "c" is position 3; k=2 excludes it
    assert hit_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0


def test_hit_at_k_zero_or_negative_k_is_zero() -> None:
    assert hit_at_k(["a"], ["a"], 0) == 0.0
    assert hit_at_k(["a"], ["a"], -1) == 0.0


def test_reciprocal_rank_uses_first_match() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["c", "b"]) == 0.5  # b is earlier
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["x"]) == 0.0


def test_mean_aggregates_preserve_semantics() -> None:
    # two queries, one full hit@1, one miss at k=10
    results: list[tuple[list[str], list[str]]] = [
        (["a", "b"], ["a"]),
        (["c", "d"], ["x"]),
    ]
    assert mean_hit_at_k(results, 1) == 0.5
    assert mean_reciprocal_rank(results) == 0.5  # 1.0 + 0.0 / 2


def test_empty_results_aggregate_to_zero() -> None:
    assert mean_hit_at_k([], 10) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


# ---- nDCG@k -----------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one() -> None:
    # All relevant docs ranked first → DCG == IDCG.
    assert ndcg_at_k(["a", "b", "c", "x"], ["a", "b", "c"], 3) == pytest.approx(1.0)


def test_ndcg_no_match_is_zero() -> None:
    assert ndcg_at_k(["x", "y", "z"], ["a"], 3) == 0.0


def test_ndcg_single_hit_at_first_position() -> None:
    # rel = [1]; DCG = 1/log2(2) = 1.0
    # IDCG (1 relevant doc, capped at k=3) = 1/log2(2) = 1.0
    assert ndcg_at_k(["a", "x", "y"], ["a"], 3) == pytest.approx(1.0)


def test_ndcg_single_hit_at_second_position() -> None:
    # rel = [0, 1]; DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k(["x", "a", "y"], ["a"], 3) == pytest.approx(expected)


def test_ndcg_partial_match_with_one_at_top_one_lower() -> None:
    # ranked = [a, x, b], expected = {a, b, c}; k = 3.
    # rel = [1, 0, 1]; DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG: 3 relevant docs, k=3 → 1/log2(2) + 1/log2(3) + 1/log2(4)
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert ndcg_at_k(["a", "x", "b"], ["a", "b", "c"], 3) == pytest.approx(dcg / idcg)


def test_ndcg_caps_idcg_at_k() -> None:
    # 5 relevant docs but k=2 → IDCG only counts 2 of them.
    # ranked all hits → DCG = 1/log2(2) + 1/log2(3); IDCG identical.
    assert ndcg_at_k(["a", "b"], ["a", "b", "c", "d", "e"], 2) == pytest.approx(1.0)


def test_ndcg_zero_or_negative_k_is_zero() -> None:
    assert ndcg_at_k(["a"], ["a"], 0) == 0.0
    assert ndcg_at_k(["a"], ["a"], -1) == 0.0


def test_ndcg_empty_expected_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], [], 3) == 0.0


# ---- Recall@k ---------------------------------------------------------------


def test_recall_full_coverage_is_one() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0


def test_recall_partial_coverage() -> None:
    # 1 of 2 expected found in top-2
    assert recall_at_k(["a", "x"], ["a", "b"], 2) == 0.5


def test_recall_respects_k_cutoff() -> None:
    # "b" is at position 3; k=2 excludes it → only "a" hit
    assert recall_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.5
    assert recall_at_k(["a", "x", "b"], ["a", "b"], 3) == 1.0


def test_recall_no_match_is_zero() -> None:
    assert recall_at_k(["x", "y"], ["a", "b"], 5) == 0.0


def test_recall_zero_or_negative_k_is_zero() -> None:
    assert recall_at_k(["a"], ["a"], 0) == 0.0
    assert recall_at_k(["a"], ["a"], -1) == 0.0


def test_recall_empty_expected_is_zero() -> None:
    assert recall_at_k(["a", "b"], [], 3) == 0.0


# ---- mean_* aggregations ----------------------------------------------------


def test_mean_ndcg_averages_per_query() -> None:
    results: list[tuple[list[str], list[str]]] = [
        (["a", "x"], ["a"]),  # nDCG@2 = 1.0
        (["x", "y"], ["a"]),  # nDCG@2 = 0.0
    ]
    assert mean_ndcg_at_k(results, 2) == pytest.approx(0.5)


def test_mean_recall_averages_per_query() -> None:
    results: list[tuple[list[str], list[str]]] = [
        (["a", "b"], ["a", "b"]),  # recall@2 = 1.0
        (["x", "y"], ["a", "b"]),  # recall@2 = 0.0
    ]
    assert mean_recall_at_k(results, 2) == pytest.approx(0.5)


def test_mean_aggregates_empty_inputs_are_zero() -> None:
    assert mean_ndcg_at_k([], 10) == 0.0
    assert mean_recall_at_k([], 10) == 0.0
