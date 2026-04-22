"""CLI integration for ``dikw eval``."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app


def _write_toy_dataset(
    root: Path,
    *,
    name: str = "toy",
    thresholds: dict[str, float] | None = None,
    queries: list[tuple[str, list[str]]] | None = None,
) -> Path:
    ds = root / name
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    (ds / "corpus" / "alpha.md").write_text(
        "# Alpha\n\nAlpha describes foo and bar topics.\n", encoding="utf-8"
    )
    (ds / "corpus" / "beta.md").write_text(
        "# Beta\n\nBeta discusses baz and qux.\n", encoding="utf-8"
    )
    thr = thresholds if thresholds is not None else {
        "hit_at_3": 0.5,
        "hit_at_10": 0.5,
        "mrr": 0.3,
    }
    thr_yaml = "\n".join(f"  {k}: {v}" for k, v in thr.items())
    (ds / "dataset.yaml").write_text(
        f"name: {name}\ndescription: cli test\nthresholds:\n{thr_yaml}\n",
        encoding="utf-8",
    )
    q = queries if queries is not None else [
        ("foo and bar topics", ["alpha"]),
        ("baz and qux", ["beta"]),
    ]
    q_lines = ["queries:"]
    for qtext, expects in q:
        q_lines.append(f"  - q: {qtext}")
        q_lines.append(f"    expect_any: [{', '.join(expects)}]")
    (ds / "queries.yaml").write_text("\n".join(q_lines) + "\n", encoding="utf-8")
    return ds


def test_cli_eval_exits_zero_when_all_thresholds_met(tmp_path: Path) -> None:
    ds = _write_toy_dataset(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "--dataset", str(ds)])
    assert result.exit_code == 0, result.stdout
    assert "toy" in result.stdout
    assert "hit_at_3" in result.stdout


def test_cli_eval_exits_one_when_any_threshold_fails(tmp_path: Path) -> None:
    ds = _write_toy_dataset(
        tmp_path,
        thresholds={"hit_at_3": 1.0, "mrr": 1.0},
        queries=[("foo bar", ["ghost"])],  # expected stem not in corpus → 0
    )
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "--dataset", str(ds)])
    assert result.exit_code == 1, result.stdout


def test_cli_eval_exits_two_when_dataset_not_found(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "--dataset", str(tmp_path / "missing")])
    assert result.exit_code == 2, result.stdout
    assert "not found" in result.stdout.lower() or "no such" in result.stdout.lower()


def test_cli_eval_all_datasets_iterates_packaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain ``dikw eval`` walks datasets_root() and runs every one.

    Monkeypatch datasets_root() so the test doesn't depend on whether
    evals/datasets/mvp/ exists yet (E5 populates it).
    """
    _write_toy_dataset(tmp_path, name="toy_a")
    _write_toy_dataset(tmp_path, name="toy_b")
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 0, result.stdout
    assert "toy_a" in result.stdout
    assert "toy_b" in result.stdout


def test_cli_eval_all_datasets_exits_one_when_any_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_toy_dataset(tmp_path, name="ok_ds")
    _write_toy_dataset(
        tmp_path,
        name="bad_ds",
        thresholds={"hit_at_3": 1.0},
        queries=[("foo", ["ghost"])],
    )
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 1, result.stdout


def test_cli_eval_empty_datasets_root_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No datasets to run is an error, not a silent pass."""
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2, result.stdout
    assert "no datasets" in result.stdout.lower()


def test_cli_eval_prints_negative_diagnostics_section(tmp_path: Path) -> None:
    """When the dataset has ``expect_none`` queries, the report ends with a
    diagnostic section listing each negative query and its top-k observed
    retrieval. Pure observation — does not contribute to pass/fail.
    """
    ds = _write_toy_dataset(tmp_path)
    # Replace queries.yaml with one positive + two negatives.
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: foo and bar topics\n"
        "    expect_any: [alpha]\n"
        "  - q: weather in Tokyo today\n"
        "    expect_none: true\n"
        "  - q: who wrote the Iliad\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "--dataset", str(ds)])
    assert result.exit_code == 0, result.stdout
    # Section header + both negative queries surface in the diagnostic table.
    assert "negative" in result.stdout.lower()
    assert "weather in Tokyo today" in result.stdout
    assert "who wrote the Iliad" in result.stdout
