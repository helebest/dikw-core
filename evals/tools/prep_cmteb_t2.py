"""Pipeline for CMTEB / T2Retrieval benchmark prep.

T2Retrieval on HuggingFace ships as parquet, and its corpus is *100%
positive-covered* — every passage is a positive qrel for at least one
query. That breaks the assumption in ``convert_cmteb.py``'s generic
stratified sampler (relevant-set-preservation produces no sampling when
the relevant set equals the whole corpus). Real calibration requires
sampling **queries first** and padding the corpus with synthetic
distractors drawn from docs not referenced by the sampled queries.

End-to-end this script does:

1. Read the three parquet files produced by ``hf download``:
    * ``<src-dir>/data/corpus-*.parquet`` → ``{id, text}`` rows
    * ``<src-dir>/data/queries-*.parquet`` → ``{id, text}`` rows
    * ``<qrels-dir>/data/<split>-*.parquet`` → ``{qid, pid, score}`` rows
2. Deterministically sample ``--num-queries`` query IDs.
3. Keep every corpus doc referenced by those queries' positive qrels,
   then pad with uniformly-random non-referenced corpus docs to reach
   ``--target-size`` total.
4. Emit a BEIR-shape bundle at ``--out``:
    * ``corpus.jsonl``  → ``{_id, text}``
    * ``queries.jsonl`` → ``{_id, text}``
    * ``qrels/<split>.tsv`` → header + ``query-id\\tcorpus-id\\tscore``

The output is drop-in input for ``convert_cmteb.py`` with
``--sample-size`` set to any value ≥ target-size (so the generic
stratified sampler becomes a pass-through).

Parquet reading needs ``pyarrow`` — not a runtime dep for dikw, install
with ``uv pip install pyarrow`` before running this tool. The pure
sampling function ``subsample_queries`` is stdlib-only and is what the
tests exercise; parquet I/O is a thin wrapper around pyarrow's standard
reader and trusted accordingly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class PrepError(RuntimeError):
    """Raised when the pipeline can't find, read, or produce valid output."""


# ---------------------------------------------------------------------------
# Pure sampling — no parquet, no filesystem. Exercised by tests/test_prep_cmteb_t2.
# ---------------------------------------------------------------------------


def subsample_queries(
    *,
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: Iterable[tuple[str, str, int]],
    num_queries: int,
    target_corpus_size: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, str], list[tuple[str, str, int]], dict[str, int]]:
    """Sample ``num_queries`` query IDs, collect their positive pids, pad
    the corpus with random non-referenced docs up to ``target_corpus_size``,
    and filter qrels to the sampled (qid, pid) universe.

    ``corpus`` and ``queries`` are ``{id: text}`` mappings.
    ``qrels`` is an iterable of ``(qid, pid, score)`` tuples. Only
    rows with ``score > 0`` contribute to the relevant set; all rows
    targeting a sampled (qid, pid) pair are kept in the output (so
    negative annotations survive if the upstream data has them).

    Returns ``(sampled_corpus, sampled_queries, sampled_qrels, stats)``.
    ``stats`` keys: ``relevant_pids_unique``, ``distractor_pids_added``,
    ``final_corpus_size``, ``qrel_rows_kept``.

    Raises PrepError if ``num_queries`` exceeds the query pool or if
    the relevant set would exceed ``target_corpus_size`` (we never
    silently drop gold docs — the caller must widen the target).
    """
    if num_queries > len(queries):
        raise PrepError(
            f"num_queries={num_queries} exceeds queries available ({len(queries)})"
        )

    # Materialise qrels once; we iterate twice.
    qrel_rows = list(qrels)
    qrels_by_qid: dict[str, list[tuple[str, int]]] = {}
    for qid, pid, score in qrel_rows:
        qrels_by_qid.setdefault(qid, []).append((pid, score))

    # Deterministic query sample — sort numerically when possible so
    # the order is stable across Python versions that differ on dict
    # iteration. Fallback to string sort for non-numeric IDs.
    def _key(x: str) -> tuple[int, int | str]:
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    rng = random.Random(seed)
    all_qids = sorted(queries.keys(), key=_key)
    sampled_qids_list = rng.sample(all_qids, num_queries)
    sampled_qids = set(sampled_qids_list)

    # Relevant set = union of positive pids for sampled queries.
    relevant_pids: set[str] = set()
    for qid in sampled_qids:
        for pid, score in qrels_by_qid.get(qid, []):
            if score > 0 and pid in corpus:
                relevant_pids.add(pid)

    if len(relevant_pids) > target_corpus_size:
        raise PrepError(
            f"relevant set size {len(relevant_pids)} exceeds target_corpus_size "
            f"{target_corpus_size} — would need to drop gold docs. Either raise "
            f"the target or lower num_queries."
        )

    # Pad with uniformly-random non-referenced corpus docs.
    distractor_budget = target_corpus_size - len(relevant_pids)
    non_relevant_pool = sorted(set(corpus.keys()) - relevant_pids, key=_key)
    if distractor_budget > len(non_relevant_pool):
        # Corpus smaller than target — keep whatever's available.
        distractor_pids: set[str] = set(non_relevant_pool)
    else:
        distractor_pids = set(rng.sample(non_relevant_pool, distractor_budget))
    sampled_pids = relevant_pids | distractor_pids

    sampled_corpus = {pid: corpus[pid] for pid in sampled_pids}
    sampled_queries = {qid: queries[qid] for qid in sampled_qids}

    # Filter qrels to the sampled universe. Preserve all (qid, pid)
    # rows where both ends survive, positive or negative.
    sampled_qrels: list[tuple[str, str, int]] = []
    for qid, pid, score in qrel_rows:
        if qid in sampled_qids and pid in sampled_pids:
            sampled_qrels.append((qid, pid, score))

    stats = {
        "relevant_pids_unique": len(relevant_pids),
        "distractor_pids_added": len(distractor_pids),
        "final_corpus_size": len(sampled_pids),
        "qrel_rows_kept": len(sampled_qrels),
    }
    return sampled_corpus, sampled_queries, sampled_qrels, stats


# ---------------------------------------------------------------------------
# Parquet I/O — pyarrow wrapper. Not unit-tested (trust pyarrow + the
# integration run the caller does after using this script).
# ---------------------------------------------------------------------------


def _load_parquet_rows(path: Path, cols: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PrepError(
            "pyarrow is required to read T2Retrieval parquet files. "
            "Install it with `uv pip install pyarrow` (one-time dep for "
            "this prep script, not a dikw runtime requirement)."
        ) from exc
    tbl = pq.read_table(str(path), columns=list(cols))  # type: ignore[no-untyped-call]
    col_lists = {c: tbl.column(c).to_pylist() for c in cols}
    return [dict(zip(cols, row, strict=True)) for row in zip(*col_lists.values(), strict=True)]


def _find_parquet(root: Path, pattern: str) -> Path:
    matches = sorted((root / "data").glob(pattern))
    if not matches:
        raise PrepError(f"no parquet matching {pattern!r} under {root / 'data'}")
    if len(matches) > 1:
        raise PrepError(
            f"multiple parquet shards match {pattern!r} under {root / 'data'}: "
            f"{[m.name for m in matches]}. This script expects single-shard "
            f"bundles; extend it if T2Retrieval grows shards."
        )
    return matches[0]


def read_t2_corpus(src_dir: Path) -> dict[str, str]:
    rows = _load_parquet_rows(_find_parquet(src_dir, "corpus-*.parquet"), ("id", "text"))
    return {str(r["id"]): str(r["text"]) for r in rows}


def read_t2_queries(src_dir: Path) -> dict[str, str]:
    rows = _load_parquet_rows(_find_parquet(src_dir, "queries-*.parquet"), ("id", "text"))
    return {str(r["id"]): str(r["text"]) for r in rows}


def read_t2_qrels(qrels_dir: Path, split: str) -> list[tuple[str, str, int]]:
    rows = _load_parquet_rows(
        _find_parquet(qrels_dir, f"{split}-*.parquet"),
        ("qid", "pid", "score"),
    )
    return [(str(r["qid"]), str(r["pid"]), int(r["score"])) for r in rows]


# ---------------------------------------------------------------------------
# BEIR-shape emission — stdlib only.
# ---------------------------------------------------------------------------


def _write_beir_bundle(
    out_dir: Path,
    *,
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: list[tuple[str, str, int]],
    split: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qrels").mkdir(exist_ok=True)

    def _sort_key(x: str) -> tuple[int, int | str]:
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for pid in sorted(corpus.keys(), key=_sort_key):
            f.write(json.dumps({"_id": pid, "text": corpus[pid]}, ensure_ascii=False) + "\n")

    with (out_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for qid in sorted(queries.keys(), key=_sort_key):
            f.write(json.dumps({"_id": qid, "text": queries[qid]}, ensure_ascii=False) + "\n")

    with (out_dir / "qrels" / f"{split}.tsv").open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, pid, score in qrels:
            f.write(f"{qid}\t{pid}\t{score}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="`hf download C-MTEB/T2Retrieval` local dir (has data/corpus-*.parquet + data/queries-*.parquet)",
    )
    ap.add_argument(
        "--qrels-dir",
        type=Path,
        required=True,
        help="`hf download C-MTEB/T2Retrieval-qrels` local dir (has data/<split>-*.parquet)",
    )
    ap.add_argument("--out", type=Path, required=True, help="output dir for BEIR-shape bundle")
    ap.add_argument("--split", default="dev", help="qrels split to read (T2Retrieval only ships 'dev')")
    ap.add_argument("--num-queries", type=int, default=300)
    ap.add_argument("--target-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    corpus = read_t2_corpus(args.src_dir)
    queries = read_t2_queries(args.src_dir)
    qrels = read_t2_qrels(args.qrels_dir, args.split)
    print(
        f"loaded: corpus={len(corpus):,} queries={len(queries):,} "
        f"qrels={len(qrels):,}",
        file=sys.stderr,
    )

    sampled_corpus, sampled_queries, sampled_qrels, stats = subsample_queries(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        num_queries=args.num_queries,
        target_corpus_size=args.target_size,
        seed=args.seed,
    )
    print(
        "sampled: "
        + " ".join(f"{k}={v}" for k, v in stats.items()),
        file=sys.stderr,
    )

    _write_beir_bundle(
        args.out,
        corpus=sampled_corpus,
        queries=sampled_queries,
        qrels=sampled_qrels,
        split=args.split,
    )
    print(f"wrote BEIR-shape bundle → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
