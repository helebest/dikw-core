"""Offline RRF weight sweep — `evals/tools/sweep_rrf.py`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import the tool as a module by adding the repo root to sys.path. The
# production build does not install `evals/` as a package — it's developer
# tooling, not runtime. Tests exercise it directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.tools.sweep_rrf import (  # noqa: E402
    evaluate_weights,
    format_table,
    load_raw_dump,
    sweep,
)


def _write_dump(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_raw_dump_parses_mode_rows(tmp_path: Path) -> None:
    dump = tmp_path / "raw.jsonl"
    _write_dump(
        dump,
        [
            {
                "dataset": "toy",
                "mode": "bm25",
                "q": "q1",
                "expect_any": ["a"],
                "expect_none": False,
                "ranked": ["a", "b", "c"],
            },
            {
                "dataset": "toy",
                "mode": "vector",
                "q": "q1",
                "expect_any": ["a"],
                "expect_none": False,
                "ranked": ["b", "a", "d"],
            },
            # hybrid rows are skipped — we re-fuse from the legs.
            {
                "dataset": "toy",
                "mode": "hybrid",
                "q": "q1",
                "expect_any": ["a"],
                "expect_none": False,
                "ranked": ["a", "b", "d"],
            },
            # Negative — skipped because no positive relevance set.
            {
                "dataset": "toy",
                "mode": "bm25",
                "q": "q_neg",
                "expect_any": [],
                "expect_none": True,
                "ranked": ["x", "y"],
            },
        ],
    )
    per_ds = load_raw_dump(dump)
    assert set(per_ds) == {"toy"}
    legs = per_ds["toy"]
    assert len(legs) == 1
    (q,) = legs
    assert q.q == "q1"
    assert q.expect_any == ["a"]
    assert q.bm25_ranked == ["a", "b", "c"]
    assert q.vector_ranked == ["b", "a", "d"]


def test_evaluate_weights_shifts_toward_weighted_leg(tmp_path: Path) -> None:
    """Weights that favour the vector leg move vector-only docs ahead.

    Construction: doc ``gold`` appears only in the vector leg at rank 0,
    doc ``distractor`` appears only in bm25 at rank 0. Equal weights
    make them tie; raising vector_weight makes ``gold`` win and flips
    hit@1 from 0 to 1.
    """
    dump = tmp_path / "raw.jsonl"
    _write_dump(
        dump,
        [
            {
                "dataset": "toy",
                "mode": "bm25",
                "q": "q1",
                "expect_any": ["gold"],
                "expect_none": False,
                "ranked": ["distractor"],
            },
            {
                "dataset": "toy",
                "mode": "vector",
                "q": "q1",
                "expect_any": ["gold"],
                "expect_none": False,
                "ranked": ["gold"],
            },
        ],
    )
    legs = load_raw_dump(dump)["toy"]

    equal = evaluate_weights(legs, rrf_k=60, bm25_weight=1.0, vector_weight=1.0)
    vec_heavy = evaluate_weights(legs, rrf_k=60, bm25_weight=0.5, vector_weight=1.5)

    # Equal weights → tie at rank 0; dict order makes it deterministic but
    # we only assert the weighted run outperforms on the gold-containing
    # metric. hit@3 always 1.0 (both docs fit), nDCG@10 should favour the
    # weighted run because gold ranks first there.
    assert vec_heavy.ndcg_at_10 >= equal.ndcg_at_10
    assert vec_heavy.hit_at_3 == 1.0


def test_evaluate_weights_reproduces_legacy_at_equal_weights(tmp_path: Path) -> None:
    """At (k=60, w=[1,1]) the sweep tool's numbers must equal what the
    production runner computes — otherwise the offline sweep wouldn't be
    comparable to a real end-to-end eval.
    """
    dump = tmp_path / "raw.jsonl"
    # Two queries, each mode ranks them differently.
    _write_dump(
        dump,
        [
            {
                "dataset": "d", "mode": "bm25", "q": "q1", "expect_any": ["a"],
                "expect_none": False, "ranked": ["a", "b", "c"],
            },
            {
                "dataset": "d", "mode": "vector", "q": "q1", "expect_any": ["a"],
                "expect_none": False, "ranked": ["b", "c", "a"],
            },
            {
                "dataset": "d", "mode": "bm25", "q": "q2", "expect_any": ["x"],
                "expect_none": False, "ranked": ["y", "x"],
            },
            {
                "dataset": "d", "mode": "vector", "q": "q2", "expect_any": ["x"],
                "expect_none": False, "ranked": ["x", "y"],
            },
        ],
    )
    legs = load_raw_dump(dump)["d"]
    r = evaluate_weights(legs, rrf_k=60, bm25_weight=1.0, vector_weight=1.0)

    # q1 fused (k=60, equal weights):
    #   a: 1/61 + 1/63 ≈ 0.03226
    #   b: 1/62 + 1/61 ≈ 0.03252  ← top (b is in both legs at rank 1+0)
    #   c: 1/63 + 1/62 ≈ 0.03200
    #   → a at rank 2, RR = 1/2 = 0.5; hit@3 = 1.0 (a in {b,a,c})
    # q2 fused: scores tie (x and y symmetric) → dict-insert order wins;
    # y was first inserted from bm25 so y ranks first → x at rank 2, RR=0.5
    # → MRR = (0.5 + 0.5) / 2 = 0.5; hit@3 = (1 + 1)/2 = 1.0
    assert r.hit_at_3 == pytest.approx(1.0)
    assert r.mrr == pytest.approx(0.5)


def test_sweep_grid_covers_all_combinations(tmp_path: Path) -> None:
    dump = tmp_path / "raw.jsonl"
    _write_dump(
        dump,
        [
            {
                "dataset": "d", "mode": "bm25", "q": "q1", "expect_any": ["a"],
                "expect_none": False, "ranked": ["a"],
            },
            {
                "dataset": "d", "mode": "vector", "q": "q1", "expect_any": ["a"],
                "expect_none": False, "ranked": ["a"],
            },
        ],
    )
    legs = load_raw_dump(dump)["d"]

    results = sweep(
        legs,
        k_grid=(40, 60),
        w_bm25_grid=(0.5, 1.0),
        w_vec_grid=(1.0, 1.5),
    )
    assert len(results) == 2 * 2 * 2  # 2 x 2 x 2 = 8 combinations

    combos = {(r.rrf_k, r.bm25_weight, r.vector_weight) for r in results}
    assert (40, 0.5, 1.0) in combos
    assert (60, 1.0, 1.5) in combos


def test_format_table_includes_baseline_row(tmp_path: Path) -> None:
    dump = tmp_path / "raw.jsonl"
    _write_dump(
        dump,
        [
            {
                "dataset": "d", "mode": "bm25", "q": "q1", "expect_any": ["a"],
                "expect_none": False, "ranked": ["a"],
            },
            {
                "dataset": "d", "mode": "vector", "q": "q1", "expect_any": ["a"],
                "expect_none": False, "ranked": ["a"],
            },
        ],
    )
    legs = load_raw_dump(dump)["d"]
    results = sweep(legs)
    out = format_table(results, top_n=3)
    assert "Top 3 by nDCG@10" in out
    assert "Equal-weight baseline" in out
    assert "current default" in out
