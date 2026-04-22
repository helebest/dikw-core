"""Dataset loader — the convention-over-configuration contract.

Each dataset is a directory with three things: ``dataset.yaml``, ``corpus/``,
``queries.yaml``. ``load_dataset`` accepts either a registered name (for
packaged datasets) or a directory path (user-provided); both resolve to the
same ``DatasetSpec`` shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.eval.dataset import (
    DatasetError,
    DatasetSpec,
    Query,
    load_dataset,
)


def _write_valid_dataset(root: Path, *, name: str = "toy") -> Path:
    """Build a minimal valid dataset directory; return its root."""
    ds = root / name
    (ds / "corpus").mkdir(parents=True, exist_ok=True)
    (ds / "corpus" / "alpha.md").write_text("# Alpha\n\nabout alpha.\n", encoding="utf-8")
    (ds / "corpus" / "beta.md").write_text("# Beta\n\nabout beta.\n", encoding="utf-8")
    (ds / "dataset.yaml").write_text(
        "name: toy\n"
        "description: two-doc toy dataset\n"
        "thresholds:\n"
        "  hit_at_3: 0.5\n"
        "  hit_at_10: 0.5\n"
        "  mrr: 0.3\n",
        encoding="utf-8",
    )
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: what is alpha\n"
        "    expect_any: [alpha]\n"
        "  - q: what is beta\n"
        "    expect_any: [beta]\n",
        encoding="utf-8",
    )
    return ds


def test_load_dataset_from_path_returns_populated_spec(tmp_path: Path) -> None:
    ds_dir = _write_valid_dataset(tmp_path)
    spec = load_dataset(ds_dir)
    assert isinstance(spec, DatasetSpec)
    assert spec.name == "toy"
    assert spec.description == "two-doc toy dataset"
    assert spec.thresholds == {"hit_at_3": 0.5, "hit_at_10": 0.5, "mrr": 0.3}
    assert spec.corpus_dir == ds_dir / "corpus"
    assert len(spec.queries) == 2
    assert isinstance(spec.queries[0], Query)
    assert spec.queries[0].q == "what is alpha"
    assert spec.queries[0].expect_any == ["alpha"]


def test_load_dataset_missing_dataset_yaml_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").unlink()
    with pytest.raises(DatasetError, match=r"dataset\.yaml"):
        load_dataset(ds)


def test_load_dataset_missing_corpus_dir_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    # remove corpus contents then the dir
    for f in (ds / "corpus").iterdir():
        f.unlink()
    (ds / "corpus").rmdir()
    with pytest.raises(DatasetError, match="corpus"):
        load_dataset(ds)


def test_load_dataset_empty_corpus_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    for f in (ds / "corpus").iterdir():
        f.unlink()
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(ds)


def test_load_dataset_missing_queries_yaml_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").unlink()
    with pytest.raises(DatasetError, match=r"queries\.yaml"):
        load_dataset(ds)


def test_load_dataset_empty_queries_raises(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").write_text("queries: []\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="at least one query"):
        load_dataset(ds)


def test_load_dataset_rejects_unknown_threshold_key(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "dataset.yaml").write_text(
        "name: toy\n"
        "description: bad thresholds\n"
        "thresholds:\n"
        "  hit_at_99: 0.5\n",  # not a supported metric
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="hit_at_99"):
        load_dataset(ds)


def test_query_accepts_expect_none_with_empty_expect_any() -> None:
    """``expect_none: true`` marks an out-of-domain query — no stems expected."""
    q = Query(q="weather in Tokyo?", expect_none=True)
    assert q.expect_none is True
    assert q.expect_any == []


def test_query_rejects_both_expect_any_and_expect_none() -> None:
    """Polarity must be unambiguous — can't claim both 'some doc' and 'no doc'."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Query(q="ambiguous", expect_any=["foo"], expect_none=True)


def test_query_rejects_both_fields_empty() -> None:
    """A query with neither polarity given is almost always a YAML typo."""
    with pytest.raises(ValueError, match=r"expect_any.*expect_none"):
        Query(q="nothing specified")


def test_load_dataset_parses_negative_query_yaml(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: what is alpha\n"
        "    expect_any: [alpha]\n"
        "  - q: weather in Tokyo?\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    spec = load_dataset(ds)
    assert len(spec.queries) == 2
    assert spec.queries[0].expect_none is False
    assert spec.queries[1].expect_none is True
    assert spec.queries[1].expect_any == []


def test_load_dataset_rejects_mixed_polarity_query(tmp_path: Path) -> None:
    ds = _write_valid_dataset(tmp_path)
    (ds / "queries.yaml").write_text(
        "queries:\n"
        "  - q: muddled\n"
        "    expect_any: [alpha]\n"
        "    expect_none: true\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="mutually exclusive"):
        load_dataset(ds)


def test_load_dataset_by_name_finds_packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When given a plain name, load_dataset searches datasets_root().

    We redirect datasets_root() via monkeypatch so the test doesn't depend on
    whether evals/datasets/mvp/ exists yet (E5 will move real data there).
    """
    _write_valid_dataset(tmp_path, name="toy")
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)
    spec = load_dataset("toy")
    assert spec.name == "toy"


def test_load_dataset_name_not_found_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dikw_core.eval.dataset.datasets_root", lambda: tmp_path)
    with pytest.raises(DatasetError, match="no such dataset"):
        load_dataset("does-not-exist")
