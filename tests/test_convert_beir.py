"""End-to-end test for the BEIR → three-file converter.

Uses a 5-doc / 4-query fixture under ``tests/fixtures/beir-tiny/`` that
exercises the interesting branches:

* Passage IDs with characters that need sanitising (``doc/3``,
  ``doc.4``) must collapse to safe stems (``doc_3``, ``doc_4``).
* A query whose qrels are all zero (``q-orphan``) must be dropped.
* A qrel pointing at a corpus_id that isn't in corpus.jsonl
  (``doc-missing-from-corpus``) must be filtered out without crashing.
* The output must round-trip through dikw's own ``load_dataset`` —
  if it doesn't, the converter is producing garbage no matter what
  the test asserts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.tools._common import ConverterError
from evals.tools.convert_beir import convert

from dikw_core.eval.dataset import load_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "beir-tiny"


def test_converter_emits_round_trippable_dataset(tmp_path: Path) -> None:
    out = tmp_path / "tiny"
    stats = convert(
        FIXTURE,
        out,
        qrels_split="test",
        name="beir-tiny",
        description="end-to-end test fixture",
        published_baselines=None,
    )

    # 5 corpus docs in the fixture, all should land on disk.
    assert stats["corpus_files"] == 5

    # 4 queries in the fixture: q-1 / q-2 / q-3 keep (q-3 had a 0-score
    # row that gets dropped but a 1-score row that survives), q-orphan
    # has only a 0-score row → dropped.
    assert stats["queries_kept"] == 3
    assert stats["queries_no_qrels"] == 1

    # The qrel pointing at doc-missing-from-corpus must be counted
    # but not crash anything.
    assert stats["qrels_dropped_unknown_cid"] == 1

    # File layout matches the runner's three-file contract.
    assert (out / "dataset.yaml").is_file()
    assert (out / "queries.yaml").is_file()
    corpus_files = sorted((out / "corpus").iterdir())
    stems = {p.stem for p in corpus_files}
    assert stems == {"doc-1", "doc-2", "doc_3", "doc_4", "doc-5"}

    # Round-trip via the actual runner loader — proves the emitted
    # YAML matches the schema the runner expects.
    spec = load_dataset(out)
    assert spec.name == "beir-tiny"
    assert {q.q for q in spec.queries} == {
        "How do plants convert sunlight into food?",
        "What is the path of water through the atmosphere?",
        "How does inheritance work at the molecular level?",
    }
    # q-3 mapped doc.4 → doc_4 via sanitise; q-3's expect_any uses the
    # sanitised stem (not the raw passage id).
    by_q = {q.q: list(q.expect_any) for q in spec.queries}
    assert by_q["How does inheritance work at the molecular level?"] == ["doc_4"]


def test_converter_refuses_to_clobber_existing_corpus(tmp_path: Path) -> None:
    out = tmp_path / "tiny"
    convert(
        FIXTURE,
        out,
        qrels_split="test",
        name="beir-tiny",
        description="first run",
        published_baselines=None,
    )
    # Second run on the same out dir should refuse rather than silently
    # leave stale files mixed with fresh ones.
    with pytest.raises(ConverterError, match="non-empty"):
        convert(
            FIXTURE,
            out,
            qrels_split="test",
            name="beir-tiny",
            description="second run",
            published_baselines=None,
        )


def test_converter_writes_published_baselines_block(tmp_path: Path) -> None:
    """``--baseline-bm25-ndcg10`` flows into dataset.yaml so users can
    eyeball the published reference next to their observed numbers."""
    import yaml

    out = tmp_path / "tiny"
    convert(
        FIXTURE,
        out,
        qrels_split="test",
        name="beir-tiny",
        description="with baseline",
        published_baselines={
            "source": "test",
            "bm25_anserini": {"ndcg_at_10": 0.665},
        },
    )
    payload = yaml.safe_load((out / "dataset.yaml").read_text(encoding="utf-8"))
    assert payload["published_baselines"]["bm25_anserini"]["ndcg_at_10"] == 0.665
