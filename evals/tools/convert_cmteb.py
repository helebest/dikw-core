"""Convert a CMTEB (Chinese MTEB) retrieval bundle into the dikw three-file shape.

CMTEB datasets on HuggingFace follow the BEIR layout once exported to
JSONL — same ``corpus.jsonl`` / ``queries.jsonl`` / ``qrels/test.tsv``
file shape, just in Chinese. Many CMTEB datasets are huge (T2Retrieval
ships ~2.3M passages, MMarcoRetrieval ~8.8M), so embedding the whole
corpus on a paid provider isn't realistic for a calibration run.

This converter therefore adds **stratified sampling** on top of the
BEIR converter:

1. Keep every corpus row referenced by at least one positive qrel
   (so all gold docs survive).
2. Fill the remaining sample budget with random non-relevant rows.
3. Drop queries whose qrels are entirely outside the sampled corpus,
   reporting the count.

If the sample budget is smaller than the relevant set, the relevant
set is *not* truncated — sampling preserves the gold docs and emits a
warning that some queries will see thinner distractor sets than at full
scale. Treat the resulting numbers as a lower bound on what a real CMTEB
benchmark run would show, not a substitute.

Usage (after ``huggingface-cli download <dataset> --local-dir /tmp/ds``
and converting any parquet shards to JSONL — pandas / pyarrow handle this
in a one-liner)::

    python evals/tools/convert_cmteb.py \\
        --source /tmp/ds/ \\
        --out evals/datasets/cmteb-t2-subset/ \\
        --sample-size 5000 \\
        --random-seed 42

Stdlib only — no HuggingFace ``datasets`` runtime dependency.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Make the tools package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.tools._common import (
    ConverterError,
    dump_dataset_yaml,
    dump_queries_yaml,
    ensure_clean_outdir,
    sanitize_stem,
    write_corpus_file,
)
from evals.tools.convert_beir import _iter_jsonl, _load_qrels


def stratified_sample(
    corpus_ids: list[str],
    relevant_ids: set[str],
    *,
    sample_size: int,
    rng: random.Random,
) -> set[str]:
    """Return a sampled subset of ``corpus_ids`` that always includes the
    relevant set, padded with random non-relevant rows up to ``sample_size``.
    """
    sampled: set[str] = set(relevant_ids)
    if len(sampled) >= sample_size:
        return sampled
    remaining = [c for c in corpus_ids if c not in sampled]
    rng.shuffle(remaining)
    fill_n = sample_size - len(sampled)
    sampled.update(remaining[:fill_n])
    return sampled


def _iter_corpus_ids(corpus_jsonl: Path) -> Iterator[str]:
    for row in _iter_jsonl(corpus_jsonl):
        cid = str(row.get("_id") or "")
        if cid:
            yield cid


def convert(
    source: Path,
    out: Path,
    *,
    qrels_split: str,
    name: str,
    description: str,
    sample_size: int,
    random_seed: int,
    published_baselines: dict[str, Any] | None,
) -> dict[str, int]:
    if not source.is_dir():
        raise ConverterError(f"--source must be a directory: {source}")
    corpus_jsonl = source / "corpus.jsonl"
    queries_jsonl = source / "queries.jsonl"
    qrels_tsv = source / "qrels" / f"{qrels_split}.tsv"
    for required in (corpus_jsonl, queries_jsonl, qrels_tsv):
        if not required.is_file():
            raise ConverterError(f"missing required file: {required}")

    out.mkdir(parents=True, exist_ok=True)
    ensure_clean_outdir(out)

    # 1. Load qrels first so we know which corpus_ids must survive sampling.
    raw_qrels = _load_qrels(qrels_tsv)
    relevant_ids = {cid for cids in raw_qrels.values() for cid in cids}

    # 2. Walk the corpus to enumerate available IDs (stream — the file is large).
    all_ids = list(_iter_corpus_ids(corpus_jsonl))

    rng = random.Random(random_seed)
    keep = stratified_sample(
        all_ids, relevant_ids, sample_size=sample_size, rng=rng
    )

    # 3. Second corpus pass — write only the sampled rows.
    cid_to_stem: dict[str, str] = {}
    corpus_dir = out / "corpus"
    n_written = 0
    for doc in _iter_jsonl(corpus_jsonl):
        cid = str(doc.get("_id") or "")
        if cid not in keep:
            continue
        title = doc.get("title")
        text = doc.get("text") or ""
        stem = sanitize_stem(cid)
        if stem in cid_to_stem.values():
            raise ConverterError(
                f"sanitised stem collision: corpus id {cid!r} → {stem!r}"
            )
        write_corpus_file(corpus_dir, stem, title=title, body=text)
        cid_to_stem[cid] = stem
        n_written += 1

    # 4. Re-filter qrels against the sampled corpus.
    qrels: dict[str, list[str]] = {}
    n_qrels_dropped = 0
    for qid, cids in raw_qrels.items():
        kept = [cid_to_stem[c] for c in cids if c in cid_to_stem]
        n_qrels_dropped += len(cids) - len(kept)
        if kept:
            qrels[qid] = kept

    # 5. Emit queries that still have at least one positive judgement.
    queries: list[dict[str, Any]] = []
    n_queries_dropped = 0
    for q in _iter_jsonl(queries_jsonl):
        qid = str(q.get("_id") or "")
        text = q.get("text") or ""
        if not qid or not text:
            continue
        expect = qrels.get(qid)
        if not expect:
            n_queries_dropped += 1
            continue
        queries.append({"q": text, "expect_any": sorted(expect)})

    if not queries:
        raise ConverterError(
            f"no queries survived sampling — sample_size={sample_size} "
            f"is too tight for the qrels coverage in {source}"
        )

    dump_queries_yaml(out, queries)

    # Pin the seed + sample_size in dataset.yaml so re-runs are
    # reproducible even after `rm -rf`.
    extra = {
        "_sampling": {
            "sample_size": sample_size,
            "random_seed": random_seed,
            "source_qrels_split": qrels_split,
            "source_corpus_total": len(all_ids),
            "relevant_kept": len(relevant_ids & set(all_ids)),
        }
    }
    dump_dataset_yaml(
        out,
        name=name,
        description=description,
        thresholds={},
        published_baselines=published_baselines,
        extra=extra,
    )

    return {
        "corpus_files": n_written,
        "queries_kept": len(queries),
        "queries_dropped_no_qrels": n_queries_dropped,
        "qrels_dropped_unknown_cid": n_qrels_dropped,
        "corpus_total_in_source": len(all_ids),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_cmteb.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--qrels-split", default="test", choices=("train", "dev", "test")
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Target corpus size after sampling (default 5000). Relevant "
        "passages are always kept; the remainder fills the budget.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed for the random fill (default 42 — reproducible).",
    )
    p.add_argument("--name", default=None)
    p.add_argument("--description", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    name = args.name or args.out.name
    desc = args.description or (
        f"CMTEB/{name} converted via evals/tools/convert_cmteb.py — "
        f"qrels split: {args.qrels_split}, sample: "
        f"{args.sample_size} (seed {args.random_seed})."
    )
    try:
        stats = convert(
            args.source,
            args.out,
            qrels_split=args.qrels_split,
            name=name,
            description=desc,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            published_baselines=None,
        )
    except ConverterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(
        f"OK — sampled {stats['corpus_files']} of "
        f"{stats['corpus_total_in_source']} passages and kept "
        f"{stats['queries_kept']} queries → {args.out}"
    )
    if stats["queries_dropped_no_qrels"]:
        print(
            f"  ({stats['queries_dropped_no_qrels']} queries dropped — "
            "qrels fell entirely outside the sampled corpus or were 0-score)"
        )
    if stats["qrels_dropped_unknown_cid"]:
        print(
            f"  ({stats['qrels_dropped_unknown_cid']} qrels rows dropped — "
            "corpus_id not in sampled corpus)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
