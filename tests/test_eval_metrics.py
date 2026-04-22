"""Metrics module lives in src, not tests — test the public import path."""

from __future__ import annotations

from dikw_core.eval.metrics import (
    hit_at_k,
    mean_hit_at_k,
    mean_reciprocal_rank,
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
