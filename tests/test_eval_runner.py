"""Runner — drive ingest + hybrid search + metrics for a single dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.eval.dataset import (
    DatasetSpec,
    Query,
    load_dataset,
)
from dikw_core.eval.runner import EvalError, EvalReport, run_eval


def _write_dataset(
    root: Path,
    *,
    queries: list[tuple[str, list[str]]],
    thresholds: dict[str, float] | None = None,
    docs: dict[str, str] | None = None,
) -> Path:
    ds = root / "toy"
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    payload = docs or {
        "alpha": "# Alpha\n\nAlpha describes foo and bar topics.\n",
        "beta": "# Beta\n\nBeta discusses baz and qux.\n",
    }
    for stem, body in payload.items():
        (ds / "corpus" / f"{stem}.md").write_text(body, encoding="utf-8")

    thr = thresholds if thresholds is not None else {
        "hit_at_3": 0.5,
        "hit_at_10": 0.5,
        "mrr": 0.3,
    }
    thr_yaml = "\n".join(f"  {k}: {v}" for k, v in thr.items())
    (ds / "dataset.yaml").write_text(
        f"name: toy\ndescription: runner smoke test\nthresholds:\n{thr_yaml}\n",
        encoding="utf-8",
    )

    q_lines: list[str] = ["queries:"]
    for q, exp in queries:
        q_lines.append(f"  - q: {q}")
        q_lines.append(f"    expect_any: [{', '.join(exp)}]")
    (ds / "queries.yaml").write_text("\n".join(q_lines) + "\n", encoding="utf-8")
    return ds


@pytest.mark.asyncio
async def test_run_eval_returns_report_with_metrics_and_passed_flag(
    tmp_path: Path,
) -> None:
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar topics", ["alpha"]),
            ("baz and qux", ["beta"]),
        ],
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    assert isinstance(report, EvalReport)
    assert set(report.metrics.keys()) == {
        "hit_at_3",
        "hit_at_10",
        "mrr",
        "ndcg_at_10",
        "recall_at_100",
    }
    assert all(0.0 <= v <= 1.0 for v in report.metrics.values())
    # FakeEmbeddings' bag-of-words on a 2-doc corpus with keyword-rich
    # queries should land cleanly in top-3.
    assert report.metrics["hit_at_3"] == 1.0
    assert report.metrics["mrr"] == 1.0
    assert report.metrics["ndcg_at_10"] == 1.0
    assert report.metrics["recall_at_100"] == 1.0
    # Single-mode default → modes carries the one mode that ran.
    assert report.modes == ["hybrid"]
    assert report.thresholds == spec.thresholds
    assert report.passed is True
    # Per-query diagnostic data preserved for the CLI's failure table.
    assert len(report.per_query) == 2
    q0 = report.per_query[0]
    assert q0["q"] == "foo and bar topics"
    assert q0["expect_any"] == ["alpha"]
    assert "alpha" in q0["ranked"][:3]


@pytest.mark.asyncio
async def test_run_eval_passed_false_when_any_metric_below_threshold(
    tmp_path: Path,
) -> None:
    # Expected stem "ghost" is not in the corpus — guaranteed miss
    # regardless of what HybridSearcher decides to rank first.
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar", ["ghost"]),
        ],
        thresholds={"hit_at_3": 1.0, "hit_at_10": 1.0, "mrr": 1.0},
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)
    assert report.passed is False
    assert report.metrics["hit_at_3"] == 0.0


@pytest.mark.asyncio
async def test_run_eval_empty_queries_guarded_at_dataset_load(
    tmp_path: Path,
) -> None:
    """Empty queries.yaml is a DatasetError — runner needn't re-validate."""
    from dikw_core.eval.dataset import DatasetError

    ds = _write_dataset(tmp_path, queries=[("x", ["alpha"])])
    (ds / "queries.yaml").write_text("queries: []\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(ds)


@pytest.mark.asyncio
async def test_run_eval_with_missing_thresholds_defaults_passed_true(
    tmp_path: Path,
) -> None:
    """A dataset without thresholds still runs — passed defaults to True.

    Useful for exploratory datasets where the user hasn't calibrated yet.
    """
    ds = _write_dataset(
        tmp_path,
        queries=[("foo", ["alpha"])],
        thresholds={},
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)
    assert report.thresholds == {}
    assert report.passed is True  # nothing to fail against


@pytest.mark.asyncio
async def test_run_eval_synthetic_spec_direct(tmp_path: Path) -> None:
    """Runner accepts a ``DatasetSpec`` constructed directly, not just via disk.

    Guards against accidentally tying runner to the yaml loader.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nalpha content\n", encoding="utf-8")
    (corpus / "b.md").write_text("# B\n\nbeta content\n", encoding="utf-8")
    spec = DatasetSpec(
        name="synthetic",
        description="",
        thresholds={"hit_at_3": 0.5},
        corpus_dir=corpus,
        queries=[Query(q="alpha content", expect_any=["a"])],
    )
    report = await run_eval(spec)
    assert report.metrics["hit_at_3"] == 1.0
    assert report.passed


@pytest.mark.asyncio
async def test_run_eval_mode_all_emits_per_mode_and_canonical_metrics(
    tmp_path: Path,
) -> None:
    """``mode='all'`` runs each retrieval leg, emits prefixed metrics for
    every (mode, key) pair, and mirrors the hybrid mode unprefixed so
    existing dataset thresholds keep gating.
    """
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar topics", ["alpha"]),
            ("baz and qux", ["beta"]),
        ],
    )
    spec = load_dataset(ds)
    report = await run_eval(spec, mode="all")

    assert sorted(report.modes) == ["bm25", "hybrid", "vector"]
    base_keys = {"hit_at_3", "hit_at_10", "mrr", "ndcg_at_10", "recall_at_100"}
    for m in ("bm25", "vector", "hybrid"):
        for k in base_keys:
            assert f"{m}/{k}" in report.metrics, f"missing {m}/{k}"
    # Unprefixed mirror equals the hybrid mode (canonical) — gating
    # against existing thresholds keeps working under --retrieval all.
    for k in base_keys:
        assert report.metrics[k] == report.metrics[f"hybrid/{k}"]
    # Sanity: this hermetic 2-doc corpus passes its 0.5 thresholds on hybrid.
    assert report.passed


@pytest.mark.asyncio
async def test_run_eval_negative_query_does_not_drag_positive_metrics(
    tmp_path: Path,
) -> None:
    """Negative queries (expect_none=True) must not contribute to hit@k/MRR.

    Two queries: one hit, one ``expect_none``. If the runner averaged
    hit@3 over both, the negative would count as a miss and halve the
    score. Correct behaviour is to compute metrics over positives only.
    """
    # Positive query with a guaranteed hit + negative query (no expected match).
    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar topics", ["alpha"]),  # positive — alpha is the answer
        ],
    )
    # Append a negative query manually (helper only supports positives).
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: foo and bar topics\n"
        "    expect_any: [alpha]\n"
        "  - q: totally unrelated out-of-domain question\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    # hit@3 = 1.0 over 1 positive, not 0.5 over positive + negative
    assert report.metrics["hit_at_3"] == 1.0
    assert report.metrics["hit_at_10"] == 1.0
    assert report.metrics["mrr"] == 1.0
    # Only the positive query lands in per_query (it's what metrics table uses).
    assert len(report.per_query) == 1
    assert report.per_query[0]["q"] == "foo and bar topics"


@pytest.mark.asyncio
async def test_run_eval_exposes_negative_diagnostics(tmp_path: Path) -> None:
    """Negative queries still get executed — their top-k is surfaced as
    observational diagnostics so humans can eyeball "what DID get retrieved
    for this out-of-domain query?".
    """
    ds = _write_dataset(tmp_path, queries=[("foo and bar", ["alpha"])])
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: foo and bar\n"
        "    expect_any: [alpha]\n"
        "  - q: totally unrelated\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    assert len(report.negative_diagnostics) == 1
    neg = report.negative_diagnostics[0]
    assert neg["q"] == "totally unrelated"
    # Retrieval always returns SOMETHING from a non-empty corpus; the point
    # is to observe what, not to pass/fail on it.
    assert "ranked" in neg
    assert isinstance(neg["ranked"], list)


@pytest.mark.asyncio
async def test_run_eval_all_negative_dataset_metrics_empty(tmp_path: Path) -> None:
    """A dataset of only negatives produces no hit@k/MRR values — nothing
    to average. The report's ``passed`` flag still works (trivially True
    if no thresholds, or would fail if the user set one for a metric the
    runner now skips).
    """
    ds = _write_dataset(tmp_path, queries=[("placeholder", ["alpha"])], thresholds={})
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: out of domain one\n"
        "    expect_none: true\n"
        "  - q: out of domain two\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    report = await run_eval(spec)

    # Metrics over an empty positive set are not meaningful — runner
    # omits them so a spurious 0.0 doesn't fail a threshold that was
    # never intended for a negatives-only corpus.
    assert "hit_at_3" not in report.metrics
    assert len(report.negative_diagnostics) == 2
    assert report.passed is True


@pytest.mark.asyncio
async def test_run_eval_dump_raw_writes_per_mode_jsonl(tmp_path: Path) -> None:
    """--dump-raw captures every (query, mode) ranked list for offline sweep.

    Shape: one JSON-per-line row per (mode, query). For a 2-positive +
    1-negative dataset at mode='all' we expect 3 queries x 3 modes = 9 rows.
    Each row must carry ``ranked`` (top-k doc stems), ``expect_any``, and
    ``expect_none``.
    """
    import json

    ds = _write_dataset(
        tmp_path,
        queries=[
            ("foo and bar topics", ["alpha"]),
            ("baz and qux", ["beta"]),
        ],
    )
    # Add a negative to prove negatives land in the dump too.
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: foo and bar topics\n"
        "    expect_any: [alpha]\n"
        "  - q: baz and qux\n"
        "    expect_any: [beta]\n"
        "  - q: unrelated noise question\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    dump = tmp_path / "raw.jsonl"
    await run_eval(spec, mode="all", raw_dump_path=dump)

    rows = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 9  # 3 queries x 3 modes

    # Every mode shows up for every query.
    by_mode: dict[str, list[dict[str, object]]] = {"bm25": [], "vector": [], "hybrid": []}
    for r in rows:
        assert set(r.keys()) >= {
            "dataset", "mode", "q", "expect_any", "expect_none", "ranked"
        }
        assert r["dataset"] == "toy"
        assert isinstance(r["ranked"], list)
        by_mode[r["mode"]].append(r)
    for m, items in by_mode.items():
        assert len(items) == 3, f"{m} has {len(items)} rows (want 3)"

    # Negative row carries expect_none=True with empty expect_any.
    neg_rows = [r for r in rows if r["q"] == "unrelated noise question"]
    assert len(neg_rows) == 3  # one per mode
    assert all(r["expect_none"] is True and r["expect_any"] == [] for r in neg_rows)

    # Positive row carries its expect_any list.
    pos_rows = [r for r in rows if r["q"] == "foo and bar topics"]
    assert all(r["expect_any"] == ["alpha"] and r["expect_none"] is False for r in pos_rows)


@pytest.mark.asyncio
async def test_run_eval_dump_raw_single_mode_is_noop(tmp_path: Path) -> None:
    """Single-mode runs can't feed the sweep tool (needs both legs) — the
    runner silently skips writing rather than producing a half-populated
    dump. The CLI warns, the runner is the defense-in-depth layer.
    """
    ds = _write_dataset(tmp_path, queries=[("foo and bar", ["alpha"])])
    spec = load_dataset(ds)
    dump = tmp_path / "raw.jsonl"
    await run_eval(spec, mode="hybrid", raw_dump_path=dump)
    # File was never written to
    assert not dump.exists() or dump.read_text(encoding="utf-8") == ""


def test_eval_error_is_raised_for_nonexistent_corpus_dir(tmp_path: Path) -> None:
    """Programmatic DatasetSpec with bad corpus path → EvalError at run time."""
    spec = DatasetSpec(
        name="bad",
        description="",
        thresholds={},
        corpus_dir=tmp_path / "nonexistent",
        queries=[Query(q="x", expect_any=["y"])],
    )
    import asyncio

    with pytest.raises(EvalError, match="corpus"):
        asyncio.run(run_eval(spec))
