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
    assert set(report.metrics.keys()) == {"hit_at_3", "hit_at_10", "mrr"}
    assert all(0.0 <= v <= 1.0 for v in report.metrics.values())
    # FakeEmbeddings' bag-of-words on a 2-doc corpus with keyword-rich
    # queries should land cleanly in top-3.
    assert report.metrics["hit_at_3"] == 1.0
    assert report.metrics["mrr"] == 1.0
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
