"""Shared helpers for the public-benchmark converters.

The runner contract (see ``evals/README.md``) only cares about three
files per dataset:

* ``dataset.yaml`` — name, description, thresholds, optional
  ``published_baselines`` block.
* ``corpus/<stem>.md`` — one file per passage; the stem becomes the
  doc identity used in ``queries.yaml`` and surfaced in
  ``PerQueryRow.ranked``.
* ``queries.yaml`` — list of ``{q, expect_any: [stem, ...]}``.

Helpers in this module enforce safe filename stems (BEIR / CMTEB
passage IDs can contain slashes, dots, etc.) and emit YAML the runner
will round-trip.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

# Stem → filename. Anything outside ``[A-Za-z0-9_-]`` (the safe set we
# already use across the project for doc identifiers) collapses to ``_``.
# Empty / whitespace-only IDs are rejected — silent collapse-to-empty
# would mask real upstream data bugs.
_STEM_SAFE = re.compile(r"[^A-Za-z0-9_\-]")


class ConverterError(RuntimeError):
    """Raised when a public-benchmark bundle can't be converted cleanly."""


def sanitize_stem(passage_id: str) -> str:
    """Map an upstream passage ID to a filesystem-safe filename stem."""
    if not passage_id or not passage_id.strip():
        raise ConverterError("empty passage id")
    cleaned = _STEM_SAFE.sub("_", passage_id.strip())
    if not cleaned:
        raise ConverterError(f"passage id {passage_id!r} sanitised to empty")
    return cleaned


def write_corpus_file(
    corpus_dir: Path, stem: str, *, title: str | None, body: str
) -> Path:
    """Write one corpus file under ``corpus_dir/<stem>.md``.

    Adds a minimal YAML front-matter when a title is provided so dikw's
    markdown backend can pick it up; otherwise just writes the body.
    Raises if the destination already exists (sanitisation collisions
    must surface, not be silently merged).
    """
    path = corpus_dir / f"{stem}.md"
    if path.exists():
        raise ConverterError(
            f"corpus collision at {path} — two passages mapped to the same stem"
        )
    if title:
        # Escape quotes inside the title to keep YAML valid.
        safe_title = title.replace('"', "'")
        content = f'---\ntitle: "{safe_title}"\n---\n\n{body.strip()}\n'
    else:
        content = f"{body.strip()}\n"
    path.write_text(content, encoding="utf-8")
    return path


def dump_dataset_yaml(
    out_dir: Path,
    *,
    name: str,
    description: str,
    thresholds: Mapping[str, float] | None = None,
    published_baselines: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Render ``dataset.yaml`` to ``out_dir / 'dataset.yaml'``.

    ``thresholds`` defaults to an empty mapping (exploratory mode — the
    runner skips threshold gating on first runs). ``published_baselines``
    is an optional informational block; the loader ignores unknown
    top-level keys, so it round-trips safely.

    If ``out_dir/dataset.yaml`` already exists, its top-level keys are
    preserved — only keys passed via ``extra`` (the converter's
    run-fingerprint block, e.g. CMTEB's ``_sampling``) are refreshed.
    This keeps curated descriptions, calibrated thresholds, and
    published-baseline annotations from being wiped on re-conversion.
    """
    payload: dict[str, Any] = {
        "name": name,
        "description": description,
        "thresholds": dict(thresholds) if thresholds else {},
    }
    if published_baselines:
        payload["published_baselines"] = dict(published_baselines)
    if extra:
        payload.update(extra)
    path = out_dir / "dataset.yaml"
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(existing, Mapping):
            refresh_keys = set(extra.keys()) if extra else set()
            for key, value in existing.items():
                if key not in refresh_keys:
                    payload[key] = value
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def dump_queries_yaml(
    out_dir: Path,
    queries: Sequence[Mapping[str, Any]],
) -> Path:
    """Render ``queries.yaml`` to ``out_dir / 'queries.yaml'``.

    Each entry is passed through verbatim — the converter is responsible
    for shaping ``{q, expect_any: [...]}`` (positive) or
    ``{q, expect_none: true}`` (negative) per the runner's schema.
    """
    payload = {"queries": [dict(q) for q in queries]}
    path = out_dir / "queries.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def ensure_clean_outdir(out_dir: Path) -> None:
    """Create ``out_dir/corpus/`` and refuse to clobber existing data.

    Re-running the converter against a populated dataset would silently
    leave stale files; force the user to ``rm -rf`` first so the on-disk
    state matches the source bundle.
    """
    corpus = out_dir / "corpus"
    if corpus.exists() and any(corpus.iterdir()):
        raise ConverterError(
            f"{corpus} is non-empty — remove it before re-running the converter"
        )
    corpus.mkdir(parents=True, exist_ok=True)
