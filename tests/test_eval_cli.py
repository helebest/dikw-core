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


def test_cli_eval_all_skips_incomplete_stub_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-discovery must tolerate stub-only dataset directories.

    Public-benchmark datasets ship as a committed dataset.yaml plus a
    converter the user runs locally. ``uv run dikw eval`` (no --dataset)
    must still succeed for users who haven't materialised every
    benchmark; the incomplete dirs surface as a yellow warning, not as
    a hard failure that takes down the whole run.
    """
    _write_toy_dataset(tmp_path, name="ok_ds")
    # Stub: dataset.yaml only, no corpus/, no queries.yaml.
    stub = tmp_path / "stub_ds"
    stub.mkdir()
    (stub / "dataset.yaml").write_text(
        "name: stub_ds\ndescription: stub\nthresholds: {}\n", encoding="utf-8"
    )
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 0, result.stdout
    assert "skipping stub_ds" in result.stdout
    assert "ok_ds" in result.stdout

    # Explicit --dataset on the same stub *does* fail clearly — auto
    # discovery is permissive, explicit selection is not.
    result_explicit = runner.invoke(app, ["eval", "--dataset", str(stub)])
    assert result_explicit.exit_code == 2, result_explicit.stdout


def test_cli_eval_dump_raw_writes_jsonl(tmp_path: Path) -> None:
    """`--dump-raw PATH --retrieval all` writes one JSONL row per
    (query, mode). The CLI truncates the file before the run so repeated
    invocations don't contaminate.
    """
    import json

    ds = _write_toy_dataset(tmp_path)
    dump = tmp_path / "raw.jsonl"

    # Pre-seed with stale content to prove CLI truncation works.
    dump.write_text('{"stale": "row"}\n', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval", "--dataset", str(ds), "--retrieval", "all", "--dump-raw", str(dump)],
    )
    assert result.exit_code == 0, result.stdout

    lines = dump.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    # No stale row survived.
    assert all("stale" not in r for r in rows)
    # 2 queries x 3 modes = 6 rows.
    assert len(rows) == 6
    assert {r["mode"] for r in rows} == {"bm25", "vector", "hybrid"}


def test_cli_eval_provider_mode_threads_retrieval_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--embedder provider`` must forward cfg.retrieval into run_eval.

    Regression guard for the 2026-04-24 CMTEB v2 rerun bug: the CLI
    read only cfg.provider from the scratch wiki's dikw.yml and
    silently dropped cfg.retrieval, so wiki-level overrides like
    ``cjk_tokenizer: jieba`` never reached the runner. Without the
    fix, the first live run against CMTEB wasted ~4h + API cost
    producing identical numbers to v1.
    """
    # Scratch wiki with a non-default retrieval block.
    from dikw_core import api

    wiki = tmp_path / "scratch-wiki"
    api.init_wiki(wiki, description="retrieval threading guard")
    (wiki / "dikw.yml").write_text(
        """provider:
  llm: anthropic
  embedding: openai_compat
  embedding_model: fake-model
  embedding_base_url: https://example.invalid
storage:
  backend: sqlite
  path: .dikw/index.sqlite
retrieval:
  rrf_k: 40
  bm25_weight: 0.7
  vector_weight: 1.3
  cjk_tokenizer: jieba
schema:
  description: regression guard
sources: []
""",
        encoding="utf-8",
    )
    # Stub out the embedding key so build_embedder doesn't error.
    monkeypatch.setenv("DIKW_EMBEDDING_API_KEY", "test-only")
    # Stub build_embedder so we don't actually call the network.
    # CLI imports it lazily inside eval_cmd, so patch at the module it
    # resolves from at that moment (providers/__init__).
    from tests.fakes import FakeEmbeddings

    monkeypatch.setattr(
        "dikw_core.providers.build_embedder", lambda _cfg: FakeEmbeddings()
    )

    # Capture the RetrievalConfig that the CLI passes to run_eval.
    captured: dict[str, object] = {}
    from dikw_core.eval import runner as runner_mod

    real_run_eval = runner_mod.run_eval

    async def spy(*args, **kwargs):
        captured["retrieval_config"] = kwargs.get("retrieval_config")
        captured["provider_config"] = kwargs.get("provider_config")
        return await real_run_eval(*args, **kwargs)

    # eval_cmd imports run_eval lazily from dikw_core.eval.runner;
    # patch the source module, not the cli reference.
    monkeypatch.setattr("dikw_core.eval.runner.run_eval", spy)

    ds = _write_toy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "--dataset", str(ds),
            "--embedder", "provider",
            "--path", str(wiki),
        ],
    )
    assert result.exit_code == 0, result.stdout
    rc = captured["retrieval_config"]
    assert rc is not None, "cfg.retrieval not threaded into run_eval"
    assert rc.rrf_k == 40  # type: ignore[attr-defined]
    assert rc.bm25_weight == 0.7  # type: ignore[attr-defined]
    assert rc.vector_weight == 1.3  # type: ignore[attr-defined]
    assert rc.cjk_tokenizer == "jieba"  # type: ignore[attr-defined]


def test_cli_eval_dump_raw_warns_and_skips_in_single_mode(tmp_path: Path) -> None:
    """Single-mode --dump-raw has no paired legs to sweep; warn + skip."""
    ds = _write_toy_dataset(tmp_path)
    dump = tmp_path / "raw.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["eval", "--dataset", str(ds), "--retrieval", "hybrid", "--dump-raw", str(dump)],
    )
    assert result.exit_code == 0, result.stdout
    assert "ignored" in result.stdout.lower() or "warning" in result.stdout.lower()
    # File either missing or empty — both acceptable.
    if dump.exists():
        assert dump.read_text(encoding="utf-8") == ""


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
