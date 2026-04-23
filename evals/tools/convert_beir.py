"""Convert a BEIR bundle into the dikw three-file dataset shape.

Usage::

    python evals/tools/convert_beir.py \\
        --source ~/Downloads/scifact/ \\
        --out evals/datasets/scifact/

A BEIR bundle has this layout (matches the official zips at
``https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/``):

    <source>/
    ├── corpus.jsonl       — one JSON object per passage: {_id, title, text}
    ├── queries.jsonl      — one JSON object per query:   {_id, text}
    └── qrels/
        ├── train.tsv      — optional
        ├── dev.tsv        — optional
        └── test.tsv       — TSV with header: query-id\\tcorpus-id\\tscore

The converter:

1. Walks ``corpus.jsonl`` and writes each passage to
   ``<out>/corpus/<sanitized_id>.md`` with ``--- title: ... ---`` front-
   matter.
2. Walks the qrels TSV (``--qrels-split test`` by default), dropping
   rows with score 0; binary positivity matches our ``expect_any``
   schema. Rows pointing at corpus IDs that didn't make it into the
   corpus pass are skipped with a count.
3. Walks ``queries.jsonl`` and emits ``queries.yaml`` with one entry
   per query that has at least one positive judgement. Queries with
   only negative qrels (or no qrels at all) are dropped — the
   ``expect_none`` schema is reserved for hand-authored OOD queries
   in the dogfood dataset.
4. Writes ``dataset.yaml`` with empty thresholds (the user is expected
   to calibrate after the first real run, per ``evals/README.md``)
   and an optional ``published_baselines`` block driven by ``--baseline``.

The script is stdlib + ``yaml`` only — no HuggingFace ``datasets``
dependency. Download a BEIR zip with ``curl`` and unzip it; point
``--source`` at the unzipped directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Make the tools package importable when run as ``python evals/tools/convert_beir.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.tools._common import (
    ConverterError,
    dump_dataset_yaml,
    dump_queries_yaml,
    ensure_clean_outdir,
    sanitize_stem,
    write_corpus_file,
)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ConverterError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ConverterError(
                    f"{path}:{lineno}: expected JSON object, got {type(obj).__name__}"
                )
            yield obj


def _load_qrels(path: Path) -> dict[str, list[str]]:
    """Return ``{query_id: [corpus_id, ...]}`` for rows with positive score."""
    by_query: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != ["query-id", "corpus-id", "score"]:
            raise ConverterError(
                f"{path}: expected header ['query-id', 'corpus-id', 'score'], "
                f"got {header!r}"
            )
        for row in reader:
            if len(row) != 3:
                continue
            qid, cid, score_str = row
            try:
                score = int(score_str)
            except ValueError as e:
                raise ConverterError(
                    f"{path}: non-integer score {score_str!r} for ({qid}, {cid})"
                ) from e
            if score > 0:
                by_query.setdefault(qid, []).append(cid)
    return by_query


def convert(
    source: Path,
    out: Path,
    *,
    qrels_split: str,
    name: str,
    description: str,
    published_baselines: dict[str, Any] | None,
) -> dict[str, int]:
    """Run the conversion. Returns a stats dict for logging."""
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

    # 1. Pass over corpus → write corpus files; remember stem mapping.
    cid_to_stem: dict[str, str] = {}
    corpus_dir = out / "corpus"
    n_corpus = 0
    for doc in _iter_jsonl(corpus_jsonl):
        cid = str(doc.get("_id") or "")
        title = doc.get("title")
        text = doc.get("text") or ""
        if not cid:
            raise ConverterError(f"corpus row missing _id: {doc!r}")
        stem = sanitize_stem(cid)
        if stem in cid_to_stem.values():
            raise ConverterError(
                f"sanitised stem collision: corpus id {cid!r} → {stem!r} (already used)"
            )
        write_corpus_file(corpus_dir, stem, title=title, body=text)
        cid_to_stem[cid] = stem
        n_corpus += 1

    # 2. Qrels grouped by query, restricted to positive judgements that
    #    point at a corpus row we actually kept.
    raw_qrels = _load_qrels(qrels_tsv)
    qrels: dict[str, list[str]] = {}
    n_qrels_dropped_unknown_cid = 0
    for qid, cids in raw_qrels.items():
        kept = [cid_to_stem[c] for c in cids if c in cid_to_stem]
        n_qrels_dropped_unknown_cid += len(cids) - len(kept)
        if kept:
            qrels[qid] = kept

    # 3. Queries → emit one entry per query that has positive qrels.
    queries: list[dict[str, Any]] = []
    n_queries_no_qrels = 0
    for q in _iter_jsonl(queries_jsonl):
        qid = str(q.get("_id") or "")
        text = q.get("text") or ""
        if not qid or not text:
            continue
        expect = qrels.get(qid)
        if not expect:
            n_queries_no_qrels += 1
            continue
        # Stable order — sorted stems are easier to diff on re-runs.
        queries.append({"q": text, "expect_any": sorted(expect)})

    if not queries:
        raise ConverterError(
            f"no queries survived after qrels join — "
            f"{n_queries_no_qrels} queries had no positive judgements"
        )

    dump_queries_yaml(out, queries)
    dump_dataset_yaml(
        out,
        name=name,
        description=description,
        thresholds={},
        published_baselines=published_baselines,
    )

    return {
        "corpus_files": n_corpus,
        "queries_kept": len(queries),
        "queries_no_qrels": n_queries_no_qrels,
        "qrels_dropped_unknown_cid": n_qrels_dropped_unknown_cid,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_beir.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an unzipped BEIR bundle (containing corpus.jsonl, "
        "queries.jsonl, qrels/<split>.tsv).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination directory under evals/datasets/.",
    )
    p.add_argument(
        "--qrels-split",
        default="test",
        choices=("train", "dev", "test"),
        help="Which qrels split to consume (default: test).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="dataset.yaml `name` field; defaults to the basename of --out.",
    )
    p.add_argument(
        "--description",
        default=None,
        help="dataset.yaml `description`; defaults to a generated string.",
    )
    p.add_argument(
        "--baseline-bm25-ndcg10",
        type=float,
        default=None,
        help="Optional published BEIR BM25 (Anserini) nDCG@10 baseline; "
        "rendered into dataset.yaml `published_baselines` for reference.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    name = args.name or args.out.name
    desc = args.description or (
        f"BEIR/{name} converted via evals/tools/convert_beir.py — "
        f"qrels split: {args.qrels_split}."
    )
    baselines: dict[str, Any] | None = None
    if args.baseline_bm25_ndcg10 is not None:
        baselines = {
            "source": "BEIR paper (Thakur et al., 2021)",
            "bm25_anserini": {"ndcg_at_10": args.baseline_bm25_ndcg10},
        }

    try:
        stats = convert(
            args.source,
            args.out,
            qrels_split=args.qrels_split,
            name=name,
            description=desc,
            published_baselines=baselines,
        )
    except ConverterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(
        f"OK — wrote {stats['corpus_files']} corpus files and "
        f"{stats['queries_kept']} queries to {args.out}"
    )
    if stats["queries_no_qrels"]:
        print(
            f"  ({stats['queries_no_qrels']} queries dropped — "
            "no positive qrels in the chosen split)"
        )
    if stats["qrels_dropped_unknown_cid"]:
        print(
            f"  ({stats['qrels_dropped_unknown_cid']} qrels rows dropped — "
            "corpus_id not in corpus.jsonl)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
