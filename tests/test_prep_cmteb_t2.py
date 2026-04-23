"""Tests for the CMTEB T2Retrieval query-first subsample function.

T2Retrieval's corpus is 100% positive-covered (every doc is a positive
qrel for at least one query). The helper must therefore:

* sample queries, not corpus docs;
* collect those queries' positive pids as the relevant set;
* pad with distractors drawn from non-referenced corpus docs;
* never silently drop gold docs — raise if the target size is
  smaller than the relevant set.

Parquet I/O is exercised only via integration (pyarrow is trusted).
"""

from __future__ import annotations

import pytest
from evals.tools.prep_cmteb_t2 import PrepError, subsample_queries


def _mini_dataset(num_corpus: int = 20, num_queries: int = 10) -> tuple[
    dict[str, str], dict[str, str], list[tuple[str, str, int]]
]:
    """Tiny synthetic dataset: each query has ~2 positive pids, corpus
    has some extra rows that are not referenced by any query (these
    serve as distractors for sampling tests)."""
    corpus = {str(i): f"doc text {i}" for i in range(num_corpus)}
    queries = {str(i): f"query text {i}" for i in range(num_queries)}
    qrels: list[tuple[str, str, int]] = []
    # Each query i → pids {i, i+1} as positives (wraps modulo corpus size).
    for q in range(num_queries):
        qrels.append((str(q), str(q % num_corpus), 1))
        qrels.append((str(q), str((q + 1) % num_corpus), 1))
    return corpus, queries, qrels


def test_subsample_preserves_relevant_pids_and_hits_target_size() -> None:
    corpus, queries, qrels = _mini_dataset(num_corpus=20, num_queries=10)
    sampled_c, sampled_q, sampled_qr, stats = subsample_queries(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        num_queries=5,
        target_corpus_size=10,
        seed=0,
    )
    assert len(sampled_q) == 5
    assert len(sampled_c) == 10, "pad to target_corpus_size exactly"
    # Every query's positive pids must be in the sampled corpus.
    for qid in sampled_q:
        for q_qid, pid, score in sampled_qr:
            if q_qid == qid and score > 0:
                assert pid in sampled_c, f"relevant pid {pid} for query {qid} was dropped"
    assert stats["final_corpus_size"] == 10
    assert stats["relevant_pids_unique"] >= 1
    assert stats["relevant_pids_unique"] + stats["distractor_pids_added"] == 10


def test_subsample_is_deterministic_under_same_seed() -> None:
    corpus, queries, qrels = _mini_dataset()
    first = subsample_queries(
        corpus=corpus, queries=queries, qrels=qrels,
        num_queries=4, target_corpus_size=8, seed=123,
    )
    second = subsample_queries(
        corpus=corpus, queries=queries, qrels=qrels,
        num_queries=4, target_corpus_size=8, seed=123,
    )
    assert set(first[0].keys()) == set(second[0].keys())
    assert set(first[1].keys()) == set(second[1].keys())


def test_subsample_raises_when_relevant_exceeds_target() -> None:
    """The tool must never silently drop gold docs — the whole point of
    query-first sampling is that the relevant set is small by
    construction. If the caller set target_corpus_size too tight, they
    must widen it."""
    corpus, queries, qrels = _mini_dataset(num_corpus=100, num_queries=50)
    with pytest.raises(PrepError, match="relevant set size"):
        subsample_queries(
            corpus=corpus, queries=queries, qrels=qrels,
            num_queries=50,            # every query's ~2 pids → ~50 unique
            target_corpus_size=10,      # way below relevant set
            seed=0,
        )


def test_subsample_qrel_filtering_drops_out_of_scope_rows() -> None:
    """Qrels targeting queries not in the sample, or pids not in the
    sampled corpus, must be filtered out of the emitted qrels list —
    otherwise the BEIR converter would flag them as orphan references."""
    corpus, queries, qrels = _mini_dataset(num_corpus=20, num_queries=10)
    _, sampled_q, sampled_qr, _ = subsample_queries(
        corpus=corpus, queries=queries, qrels=qrels,
        num_queries=3, target_corpus_size=8, seed=0,
    )
    sampled_qids = set(sampled_q.keys())
    for qid, _pid, _score in sampled_qr:
        assert qid in sampled_qids, f"qrel row for query {qid} leaked past the filter"


def test_subsample_rejects_oversized_num_queries() -> None:
    corpus, queries, qrels = _mini_dataset(num_corpus=10, num_queries=5)
    with pytest.raises(PrepError, match="exceeds queries with positive qrels"):
        subsample_queries(
            corpus=corpus, queries=queries, qrels=qrels,
            num_queries=100, target_corpus_size=5, seed=0,
        )


def test_subsample_excludes_orphan_queries_from_pool() -> None:
    """If queries.jsonl has IDs absent from the chosen qrels split (or
    with all-non-positive scores), they must be excluded from the
    sampling pool — otherwise the final bundle silently ends up with
    fewer queries than requested after convert_cmteb drops the orphans."""
    corpus = {str(i): f"doc {i}" for i in range(10)}
    # 5 queries total, but only q0/q1/q2 have positive qrels.
    queries = {str(i): f"q {i}" for i in range(5)}
    qrels: list[tuple[str, str, int]] = [
        ("0", "0", 1),
        ("1", "1", 1),
        ("2", "2", 1),
        ("3", "3", 0),  # explicit zero — judged but non-relevant
        # qid 4 has no qrel at all
    ]
    # Asking for 4 queries must fail — only 3 are judgable.
    with pytest.raises(PrepError, match="exceeds queries with positive qrels"):
        subsample_queries(
            corpus=corpus, queries=queries, qrels=qrels,
            num_queries=4, target_corpus_size=5, seed=0,
        )
    # Asking for 3 must succeed and pick exactly the judgable ones.
    _, sampled_q, _, _ = subsample_queries(
        corpus=corpus, queries=queries, qrels=qrels,
        num_queries=3, target_corpus_size=5, seed=0,
    )
    assert set(sampled_q.keys()) == {"0", "1", "2"}


def test_subsample_handles_corpus_smaller_than_target() -> None:
    """If the non-relevant pool is tight, the tool should not crash —
    it just emits whatever distractors it can."""
    corpus = {str(i): f"d{i}" for i in range(3)}
    queries = {"q1": "text", "q2": "text"}
    qrels: list[tuple[str, str, int]] = [("q1", "0", 1), ("q2", "1", 1)]
    sampled_c, _, _, stats = subsample_queries(
        corpus=corpus, queries=queries, qrels=qrels,
        num_queries=2, target_corpus_size=100, seed=0,
    )
    assert len(sampled_c) == 3  # whole corpus
    assert stats["final_corpus_size"] == 3
