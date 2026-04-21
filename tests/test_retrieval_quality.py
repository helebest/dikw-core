"""Retrieval-quality gate over the MVP dogfood corpus.

The test ingests ``tests/fixtures/mvp_corpus/`` into a temp wiki with
``FakeEmbeddings`` (deterministic bag-of-words), runs each query in
``tests/fixtures/mvp_queries.yaml`` through ``HybridSearcher``, and asserts
aggregate ``hit@3 / hit@10 / MRR`` thresholds.

Thresholds are calibrated from the first observed numbers with ~2 queries
of slack (10 queries → hit@k changes in 0.1 steps) — the point of this test
is to catch *regressions*, not to benchmark chunker/search on first run.

On failure, a per-query table is printed so misses are diagnosable without
a debugger.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from dikw_core import api
from dikw_core.info.search import HybridSearcher
from dikw_core.storage import build_storage
from tests.eval.metrics import mean_hit_at_k, mean_reciprocal_rank
from tests.fakes import FakeEmbeddings

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mvp_corpus"
QUERIES_FILE = Path(__file__).parent / "fixtures" / "mvp_queries.yaml"

# Calibrated against the first hermetic run on the MVP corpus:
# observed hit@3 = 1.000, hit@10 = 1.000, MRR = 0.833.
# With 10 queries, each single-query change shifts hit@k by 0.1 — thresholds
# keep ~2 queries of slack so corpus tweaks don't flake the gate, while still
# catching a real retrieval regression.
HIT_AT_3_THRESHOLD = 0.80
HIT_AT_10_THRESHOLD = 0.80
MRR_THRESHOLD = 0.60


def _load_queries() -> list[dict[str, object]]:
    with QUERIES_FILE.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    queries = raw.get("queries", [])
    assert isinstance(queries, list), "mvp_queries.yaml must have top-level 'queries' list"
    return queries


def _require_fixtures() -> None:
    if not FIXTURES_DIR.is_dir() or not any(FIXTURES_DIR.glob("*.md")):
        pytest.fail(
            f"MVP corpus fixtures not found at {FIXTURES_DIR}. "
            "Phase A.A2 of the plan is to populate this directory with "
            "project docs + Karpathy materials."
        )
    if not QUERIES_FILE.is_file():
        pytest.fail(
            f"MVP query ground truth not found at {QUERIES_FILE}. "
            "Phase A.A2 of the plan is to author ~10 Q/A pairs here."
        )


@pytest.fixture()
def mvp_wiki(tmp_path: Path) -> Path:
    _require_fixtures()
    wiki = tmp_path / "wiki"
    api.init_wiki(wiki, description="mvp retrieval-quality test wiki")
    dest = wiki / "sources" / "corpus"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES_DIR.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_retrieval_quality_meets_thresholds(mvp_wiki: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(mvp_wiki, embedder=embedder)

    cfg, root = api.load_wiki(mvp_wiki)
    storage = build_storage(cfg.storage, root=root)
    await storage.connect()
    await storage.migrate()
    try:
        searcher = HybridSearcher(storage, embedder, embedding_model="fake")
        queries = _load_queries()
        assert queries, "mvp_queries.yaml has no queries"

        per_query_rows: list[tuple[str, list[str], list[str]]] = []
        results_top10: list[tuple[list[str], list[str]]] = []

        for entry in queries:
            q = str(entry["q"])
            expected_any = [str(x) for x in entry.get("expect_any", [])]

            hits = await searcher.search(q, limit=10)
            ranked_stems = [Path(h.path).stem if h.path else h.doc_id for h in hits]

            per_query_rows.append((q, expected_any, ranked_stems))
            results_top10.append((ranked_stems, expected_any))

        hit3 = mean_hit_at_k(results_top10, 3)
        hit10 = mean_hit_at_k(results_top10, 10)
        mrr = mean_reciprocal_rank(results_top10)

        _print_table(per_query_rows, hit3, hit10, mrr)

        assert hit3 >= HIT_AT_3_THRESHOLD, f"hit@3 = {hit3:.3f} < {HIT_AT_3_THRESHOLD}"
        assert hit10 >= HIT_AT_10_THRESHOLD, f"hit@10 = {hit10:.3f} < {HIT_AT_10_THRESHOLD}"
        assert mrr >= MRR_THRESHOLD, f"MRR = {mrr:.3f} < {MRR_THRESHOLD}"
    finally:
        await storage.close()


def _print_table(
    rows: list[tuple[str, list[str], list[str]]],
    hit3: float,
    hit10: float,
    mrr: float,
) -> None:
    print()
    print("Retrieval quality — per-query results")
    print("-" * 88)
    for q, expected, ranked in rows:
        top5 = ranked[:5]
        hit3_mark = "✓" if any(e in ranked[:3] for e in expected) else "✗"
        hit10_mark = "✓" if any(e in ranked[:10] for e in expected) else "✗"
        q_short = q if len(q) <= 48 else q[:45] + "..."
        print(f"  {hit3_mark}@3 {hit10_mark}@10  {q_short}")
        print(f"       expected: {expected}")
        print(f"       top-5:    {top5}")
    print("-" * 88)
    print(f"hit@3 = {hit3:.3f}   hit@10 = {hit10:.3f}   MRR = {mrr:.3f}")
    print(
        f"thresholds: hit@3 ≥ {HIT_AT_3_THRESHOLD}, "
        f"hit@10 ≥ {HIT_AT_10_THRESHOLD}, MRR ≥ {MRR_THRESHOLD}"
    )
