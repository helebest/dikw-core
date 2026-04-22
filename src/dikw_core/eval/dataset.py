"""Dataset loading — convention-over-configuration.

A dataset is a directory with exactly three things:

  <dataset>/
    dataset.yaml     — name, description, thresholds
    corpus/*.md      — documents to ingest (also *.html)
    queries.yaml     — {q, expect_any: [doc_stem, …]} pairs

``load_dataset(name_or_path)`` accepts either a registered name (looked up
under ``datasets_root()``) or a directory path (user-provided). No registry
file, no plugin system: the filesystem layout *is* the contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Metric keys accepted under ``thresholds:``. Mirrors what runner.py computes.
SUPPORTED_METRICS = frozenset({"hit_at_3", "hit_at_10", "mrr"})


class DatasetError(RuntimeError):
    """Raised when a dataset directory is malformed or missing pieces."""


class Query(BaseModel):
    """One entry in ``queries.yaml``.

    Polarity must be exactly one of:

    * ``expect_any: [doc_stem, …]`` — positive case; query is a hit at k if
      any listed stem appears in top-k.
    * ``expect_none: true`` — negative case; observed top-k is surfaced as
      diagnostic only and does NOT contribute to hit@k/MRR. Used for
      out-of-domain queries where the right behaviour is "no match".

    The two are mutually exclusive; giving neither is almost always a YAML typo.
    """

    model_config = ConfigDict(frozen=True)

    q: str
    expect_any: list[str] = Field(default_factory=list)
    expect_none: bool = False

    @model_validator(mode="after")
    def _polarity(self) -> Self:
        if self.expect_none and self.expect_any:
            raise ValueError("expect_none and expect_any are mutually exclusive")
        if not self.expect_none and not self.expect_any:
            raise ValueError(
                "query must provide either expect_any (positive) or expect_none=true (negative)"
            )
        return self


class DatasetSpec(BaseModel):
    """Validated view of a dataset directory on disk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    thresholds: dict[str, float]
    corpus_dir: Path
    queries: list[Query]


def datasets_root() -> Path:
    """Where packaged datasets live.

    Two resolutions, tried in order:

    1. **Installed wheel** — ``<package>/dikw_core/eval/datasets/``, populated
       by a hatch ``force-include`` mapping at build time.
    2. **Editable / source checkout** — walk up from this file to find
       ``<repo>/evals/datasets/``.

    Returning the first hit means test suites running from a source checkout
    see the committed data, while installed users see the bundled copy.
    """
    here = Path(__file__).resolve().parent  # src/dikw_core/eval/
    installed = here / "datasets"
    if installed.is_dir():
        return installed
    for parent in here.parents:
        candidate = parent / "evals" / "datasets"
        if candidate.is_dir():
            return candidate
    # No evals/ anywhere — still return the installed-path target so error
    # messages in load_dataset remain legible ("no such dataset under
    # /path/to/site-packages/dikw_core/eval/datasets").
    return installed


def load_dataset(name_or_path: str | Path) -> DatasetSpec:
    """Load and validate a dataset by registered name or filesystem path."""
    path = _resolve_path(name_or_path)
    if not path.is_dir():
        raise DatasetError(f"dataset directory not found: {path}")

    meta = _load_dataset_yaml(path)
    queries = _load_queries(path)
    corpus_dir = _require_corpus(path)

    thresholds = _parse_thresholds(meta.get("thresholds") or {})
    try:
        return DatasetSpec(
            name=str(meta.get("name") or path.name),
            description=str(meta.get("description") or ""),
            thresholds=thresholds,
            corpus_dir=corpus_dir,
            queries=queries,
        )
    except ValidationError as e:  # pragma: no cover — caught by earlier guards
        raise DatasetError(f"invalid dataset spec at {path}: {e}") from e


def _resolve_path(name_or_path: str | Path) -> Path:
    """Accept a path (existing dir) or a name (resolved under datasets_root)."""
    if isinstance(name_or_path, Path):
        return name_or_path
    # Heuristic: anything that looks like a path (contains os.sep, starts
    # with `.`/`/`, or is an existing directory) is treated as a path.
    as_path = Path(name_or_path)
    if as_path.is_dir() or "/" in name_or_path or name_or_path.startswith("."):
        return as_path
    # Bare name → under packaged/dev datasets root
    candidate = datasets_root() / name_or_path
    if not candidate.is_dir():
        raise DatasetError(
            f"no such dataset {name_or_path!r} under {datasets_root()}"
        )
    return candidate


def _load_dataset_yaml(path: Path) -> Mapping[str, Any]:
    meta_path = path / "dataset.yaml"
    if not meta_path.is_file():
        raise DatasetError(f"missing dataset.yaml at {meta_path}")
    raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DatasetError(f"{meta_path}: top-level YAML must be a mapping")
    return raw


def _require_corpus(path: Path) -> Path:
    corpus_dir = path / "corpus"
    if not corpus_dir.is_dir():
        raise DatasetError(f"missing corpus/ directory at {corpus_dir}")
    # Accept any SourceBackend-readable extension; a deeper check would
    # duplicate data/backends/__init__.py logic — for now require at least
    # one file to catch the "forgot to populate" case.
    if not any(corpus_dir.iterdir()):
        raise DatasetError(f"corpus/ at {corpus_dir} is empty")
    return corpus_dir


def _load_queries(path: Path) -> list[Query]:
    q_path = path / "queries.yaml"
    if not q_path.is_file():
        raise DatasetError(f"missing queries.yaml at {q_path}")
    raw = yaml.safe_load(q_path.read_text(encoding="utf-8")) or {}
    items = raw.get("queries") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise DatasetError(f"{q_path}: expected at least one query under 'queries:'")
    queries: list[Query] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise DatasetError(f"{q_path}: query #{i + 1} must be a mapping")
        try:
            queries.append(Query.model_validate(entry))
        except ValidationError as e:
            raise DatasetError(f"{q_path}: query #{i + 1} invalid: {e}") from e
    return queries


def _parse_thresholds(raw: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key not in SUPPORTED_METRICS:
            raise DatasetError(
                f"unknown threshold key {key!r}; supported: {sorted(SUPPORTED_METRICS)}"
            )
        try:
            out[key] = float(value)
        except (TypeError, ValueError) as e:
            raise DatasetError(f"threshold {key!r} is not a number: {value!r}") from e
    return out
