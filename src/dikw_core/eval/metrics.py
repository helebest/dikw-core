"""Retrieval-quality metrics.

Semantics: each query's ground truth is an ``expect_any`` set — retrieval
succeeds if **any** member of that set appears in the top-k ranked results.
This matches how dogfood Q/A is authored (paraphrased answers often live in
multiple docs; requiring all of them would be artificially punitive).

For public-benchmark calibration we also expose ``ndcg_at_k`` and
``recall_at_k``, which BEIR-family datasets report. ``expect_any`` is
treated as the binary qrels positive set — graded relevance is out of
scope (SciFact / CMTEB-T2 use binary qrels too, so this matches their
published numbers).

All functions are pure and synchronous — safe to call from any fixture.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], expected_any: Iterable[str], k: int) -> float:
    """1.0 if any ``expected_any`` is in ``ranked[:k]``, else 0.0."""
    if k <= 0:
        return 0.0
    top = set(ranked[:k])
    return 1.0 if any(e in top for e in expected_any) else 0.0


def reciprocal_rank(ranked: Sequence[str], expected_any: Iterable[str]) -> float:
    """1 / rank of the first ``expected_any`` match (1-indexed); 0.0 if none."""
    expected = set(expected_any)
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in expected:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: Sequence[str], expected_any: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG = Σ rel_i / log2(i+1) for i in 1..k (rel_i ∈ {0, 1}).
    IDCG = same with all relevant docs ranked first (capped at k).
    """
    if k <= 0:
        return 0.0
    expected = set(expected_any)
    if not expected:
        return 0.0
    dcg = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in expected:
            dcg += 1.0 / math.log2(idx + 1)
    n_rel = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], expected_any: Iterable[str], k: int) -> float:
    """|hits ∩ expected| / |expected|, capped at k."""
    if k <= 0:
        return 0.0
    expected = set(expected_any)
    if not expected:
        return 0.0
    top = set(ranked[:k])
    return len(top & expected) / len(expected)


def mean_hit_at_k(
    results: Sequence[tuple[Sequence[str], Iterable[str]]], k: int
) -> float:
    """Average ``hit_at_k`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(hit_at_k(r, e, k) for r, e in results) / len(results)


def mean_reciprocal_rank(
    results: Sequence[tuple[Sequence[str], Iterable[str]]],
) -> float:
    """Average ``reciprocal_rank`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, e) for r, e in results) / len(results)


def mean_ndcg_at_k(
    results: Sequence[tuple[Sequence[str], Iterable[str]]], k: int
) -> float:
    """Average ``ndcg_at_k`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(ndcg_at_k(r, e, k) for r, e in results) / len(results)


def mean_recall_at_k(
    results: Sequence[tuple[Sequence[str], Iterable[str]]], k: int
) -> float:
    """Average ``recall_at_k`` across queries. Empty input returns 0.0."""
    if not results:
        return 0.0
    return sum(recall_at_k(r, e, k) for r, e in results) / len(results)
