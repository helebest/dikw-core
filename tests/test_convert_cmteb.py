"""Tests for the CMTEB converter.

CMTEB shares the BEIR JSONL+TSV layout — the only thing convert_cmteb
adds on top is **stratified sampling** for huge corpora. Reuse the
existing ``tests/fixtures/beir-tiny`` fixture and exercise the sampling
edge cases.
"""

from __future__ import annotations

from pathlib import Path

from evals.tools.convert_cmteb import convert, stratified_sample

from dikw_core.eval.dataset import load_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "beir-tiny"


def test_stratified_sample_keeps_relevant_and_pads_with_random() -> None:
    import random

    rng = random.Random(0)
    corpus = [f"c-{i}" for i in range(20)]
    relevant = {"c-3", "c-7", "c-15"}
    sampled = stratified_sample(corpus, relevant, sample_size=10, rng=rng)

    assert relevant.issubset(sampled), "relevant set must be preserved"
    assert len(sampled) == 10, "sample size honoured exactly"


def test_stratified_sample_does_not_truncate_relevant_when_budget_too_small() -> None:
    import random

    rng = random.Random(0)
    corpus = [f"c-{i}" for i in range(20)]
    relevant = {f"c-{i}" for i in range(15)}  # 15 relevant, sample budget 10
    sampled = stratified_sample(corpus, relevant, sample_size=10, rng=rng)

    # Relevant set is preserved even if it overshoots budget — sampling
    # should never drop gold docs (would silently break recall@k).
    assert sampled == relevant
    assert len(sampled) == 15


def test_convert_with_oversized_sample_returns_full_corpus(tmp_path: Path) -> None:
    """If sample_size >= corpus size, every passage is kept (sampling is a no-op)."""
    out = tmp_path / "tiny"
    stats = convert(
        FIXTURE,
        out,
        qrels_split="test",
        name="tiny-cmteb",
        description="oversized sample test",
        sample_size=100,  # corpus has 5
        random_seed=0,
        published_baselines=None,
    )
    assert stats["corpus_files"] == 5
    assert stats["queries_kept"] == 3  # same as BEIR test
    spec = load_dataset(out)
    assert len(spec.queries) == 3


def test_convert_with_tight_sample_keeps_relevant(tmp_path: Path) -> None:
    """sample_size below corpus size still keeps every relevant passage."""
    out = tmp_path / "tiny"
    stats = convert(
        FIXTURE,
        out,
        qrels_split="test",
        name="tiny-cmteb",
        description="tight sample test",
        sample_size=3,  # corpus has 5; 3 relevant docs (doc-1, doc-2, doc_4)
        random_seed=0,
        published_baselines=None,
    )
    # All 3 relevant docs survived (even though sample_size=3 leaves no
    # room for distractors).
    spec = load_dataset(out)
    all_expected: set[str] = set()
    for q in spec.queries:
        all_expected.update(q.expect_any)
    corpus_stems = {
        p.stem for p in (out / "corpus").iterdir()
    }
    assert all_expected.issubset(corpus_stems), (
        f"relevant docs missing from sampled corpus: "
        f"expected {all_expected}, got {corpus_stems}"
    )
    assert stats["corpus_files"] >= len(all_expected)


def test_convert_records_sampling_block_in_dataset_yaml(tmp_path: Path) -> None:
    """The generated dataset.yaml pins seed + source totals so re-runs
    against the same source bundle reproduce the same sample."""
    import yaml

    out = tmp_path / "tiny"
    convert(
        FIXTURE,
        out,
        qrels_split="test",
        name="tiny-cmteb",
        description="sampling block test",
        sample_size=4,
        random_seed=7,
        published_baselines=None,
    )
    payload = yaml.safe_load((out / "dataset.yaml").read_text(encoding="utf-8"))
    sampling = payload["_sampling"]
    assert sampling["sample_size"] == 4
    assert sampling["random_seed"] == 7
    assert sampling["source_corpus_total"] == 5
