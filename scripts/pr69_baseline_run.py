"""Real-data A/B runner for synth existing-pages context (PR #69).

Wipe-and-rebuild path: assumes the caller has already removed
``wiki/`` and the ``.dikw/`` indexes (preserving codex OAuth tokens
under ``.dikw/auth.json``). Runs ingest -> synthesize -> lint against
the elon-musk-validation base and prints a JSON summary suitable for
dropping into the post-PR2 column of ``evals/BASELINES.md``.

Provider config + dim/model come from the base's ``dikw.yml``; the
script does not override them. Codex OAuth tokens live in
``<base>/.dikw/auth.json`` (per ``reference_openai_codex_setup.md``).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

from dikw_core import api
from dikw_core.config import load_config
from dikw_core.providers import build_embedder


def _page_counts_by_type(wiki_root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for md in wiki_root.rglob("*.md"):
        if md.name in {"index.md", "log.md"}:
            continue
        rel = md.relative_to(wiki_root).as_posix()
        bucket = rel.split("/", 1)[0] if "/" in rel else "(root)"
        counts[bucket] += 1
    return dict(sorted(counts.items()))


async def _run(base: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(base / ".env")

    cfg = load_config(base / "dikw.yml")
    embedder = build_embedder(cfg.provider)

    t0 = time.monotonic()
    ingest_report = await api.ingest(base, embedder=embedder)
    t1 = time.monotonic()
    synth_report = await api.synthesize(base, embedder=embedder)
    t2 = time.monotonic()
    lint_report = await api.lint(base)
    t3 = time.monotonic()

    wiki_root = base / "wiki"
    page_counts = _page_counts_by_type(wiki_root)

    summary = {
        "base": str(base),
        "wall_seconds": {
            "ingest": round(t1 - t0, 1),
            "synth": round(t2 - t1, 1),
            "lint": round(t3 - t2, 1),
            "total": round(t3 - t0, 1),
        },
        "ingest": {
            "scanned": ingest_report.scanned,
            "added": ingest_report.added,
            "chunks": ingest_report.chunks,
            "embedded": ingest_report.embedded,
        },
        "synth": {
            "candidates": synth_report.candidates,
            "sources_processed": synth_report.sources_processed,
            "groups_processed": synth_report.groups_processed,
            "created": synth_report.created,
            "updated": synth_report.updated,
            "skipped": synth_report.skipped,
            "errors": synth_report.errors,
            "unresolved_wikilinks": synth_report.unresolved_wikilinks,
        },
        "wiki_pages": {
            "total": sum(page_counts.values()),
            "by_type": page_counts,
        },
        "lint": {
            "total_issues": len(lint_report.issues),
            "by_kind": lint_report.by_kind(),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    base = Path(
        os.environ.get(
            "DIKW_PR69_BASELINE_BASE",
            r"C:\Users\HE LE\Project\opendikw\dikw-data\bases\elon-musk-validation",
        )
    ).expanduser()
    if not base.is_dir():
        print(f"base not found: {base}", file=sys.stderr)
        return 2
    asyncio.run(_run(base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
