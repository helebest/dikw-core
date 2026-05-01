"""Dataset loading — convention-over-configuration.

A dataset is a directory with up to **four** entries:

  <dataset>/
    dataset.yaml     — name, description, thresholds (optionally namespaced
                       as ``<view>/<metric>``)
    corpus/*.md      — documents to ingest (also *.html); image files
                       referenced via ``![](path)`` live in ``corpus/`` too
                       (any sub-directory layout is fine).
    targets.yaml     — *(optional)* assets + chunks named-id catalog. When
                       present, queries can express chunk-level / asset-level
                       ground truth via ``expect_chunk_any`` /
                       ``expect_asset_any``. Absent → only doc-level eval.
    queries.yaml     — list of {q, expect_*_any | expect_none} entries.

``load_dataset(name_or_path)`` accepts either a registered name (looked up
under ``datasets_root()``) or a directory path (user-provided). No registry
file, no plugin system: the filesystem layout *is* the contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Metric keys accepted under ``thresholds:``. Mirrors what runner.py computes.
# nDCG@10 and Recall@100 are added for public-benchmark calibration (BEIR /
# CMTEB report nDCG@10 and Recall@100 as defaults); the dogfood mvp dataset
# does not have to set them.
SUPPORTED_METRICS = frozenset(
    {"hit_at_3", "hit_at_10", "mrr", "ndcg_at_10", "recall_at_100"}
)

# Granularity views the runner can score. ``doc`` is always available;
# ``chunk`` and ``asset`` require ``targets.yaml`` to define the named-id
# catalog and queries to opt in via ``expect_chunk_any`` / ``expect_asset_any``.
SUPPORTED_VIEWS = frozenset({"doc", "chunk", "asset"})

QueryType = Literal["doc", "text_chunk", "asset", "mixed"]


class DatasetError(RuntimeError):
    """Raised when a dataset directory is malformed or missing pieces."""


class AssetTarget(BaseModel):
    """One row in ``targets.yaml`` ``assets:`` — a named image/asset.

    ``id`` is the stable identity used in ``queries.yaml`` ``expect_asset_any``.
    ``path`` is relative to ``corpus/`` and points at the image file the loader
    will hash to derive the runtime ``asset_id`` (sha256 over file bytes).
    ``heading`` / ``anchor`` are advisory metadata for human readability and
    report rendering — they don't participate in matching.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    doc: str
    path: str
    heading: str = ""
    anchor: str = ""


class ChunkTarget(BaseModel):
    """One row in ``targets.yaml`` ``chunks:`` — a named text chunk.

    ``anchor`` is resolved against the markdown body to find the heading's
    char position; the runner then matches that position to the chunk
    produced by the chunker covering it. ``heading`` is advisory.

    ``asset_id`` (optional) declares the chunk ↔ asset binding the dataset
    author intended (one image per H2). The eval runner doesn't enforce
    this — ``Hit.asset_refs`` provides the runtime-observed binding — but
    it's preserved for diagnostic output.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    doc: str
    heading: str = ""
    anchor: str = ""
    asset_id: str | None = None


class TargetsSpec(BaseModel):
    """Parsed ``targets.yaml``.

    Two flat lists, both keyed by user-authored stable ``id``. Either may
    be empty; an empty file (or missing file) is equivalent to
    ``TargetsSpec(assets=[], chunks=[])`` — eval falls back to doc-level
    only.
    """

    model_config = ConfigDict(frozen=True)

    assets: list[AssetTarget] = Field(default_factory=list)
    chunks: list[ChunkTarget] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_dup_ids(self) -> Self:
        seen_a: set[str] = set()
        for a in self.assets:
            if a.id in seen_a:
                raise ValueError(f"duplicate asset target id: {a.id!r}")
            seen_a.add(a.id)
        seen_c: set[str] = set()
        for c in self.chunks:
            if c.id in seen_c:
                raise ValueError(f"duplicate chunk target id: {c.id!r}")
            seen_c.add(c.id)
        return self


class Query(BaseModel):
    """One entry in ``queries.yaml``.

    Polarity is exactly one of:

    * Positive — at least one of ``expect_any`` / ``expect_doc_any`` /
      ``expect_chunk_any`` / ``expect_asset_any`` is non-empty. The query
      is a hit at k (in the corresponding view) if any listed identity
      appears in top-k of that view's ranked list.
    * Negative — ``expect_none: true`` and all expect_* lists empty. The
      observed top-k surfaces as diagnostic only and does NOT contribute
      to hit@k/MRR.

    ``expect_any`` is the legacy doc-level field (kept for SciFact / CMTEB
    back-compat). It's an alias of ``expect_doc_any``; a query that sets
    ``expect_any`` cannot also set ``expect_doc_any`` (ambiguous).

    ``id`` and ``query_type`` are optional metadata: ``id`` lets the
    runner key per-query rows to a stable identity (otherwise an array
    index is used); ``query_type`` is informational.
    """

    model_config = ConfigDict(frozen=True)

    q: str
    id: str | None = None
    query_type: QueryType | None = None
    expect_any: list[str] = Field(default_factory=list)
    expect_doc_any: list[str] = Field(default_factory=list)
    expect_chunk_any: list[str] = Field(default_factory=list)
    expect_asset_any: list[str] = Field(default_factory=list)
    expect_none: bool = False

    @model_validator(mode="after")
    def _polarity(self) -> Self:
        # ``expect_any`` is the legacy alias of ``expect_doc_any``; both
        # set at once is ambiguous.
        if self.expect_any and self.expect_doc_any:
            raise ValueError(
                "cannot set both expect_any and expect_doc_any (expect_any "
                "is the legacy alias)"
            )
        positives = (
            list(self.expect_any)
            + list(self.expect_doc_any)
            + list(self.expect_chunk_any)
            + list(self.expect_asset_any)
        )
        if self.expect_none and positives:
            raise ValueError(
                "expect_none and any expect_*_any are mutually exclusive"
            )
        if not self.expect_none and not positives:
            raise ValueError(
                "query must provide at least one of expect_any / expect_doc_any /"
                " expect_chunk_any / expect_asset_any (positive), or "
                "expect_none=true (negative)"
            )
        return self

    @property
    def doc_positives(self) -> list[str]:
        """Doc-level positives: union of legacy ``expect_any`` and
        ``expect_doc_any``."""
        return list(self.expect_any) + list(self.expect_doc_any)


class DatasetSpec(BaseModel):
    """Validated view of a dataset directory on disk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    thresholds: dict[str, float]
    corpus_dir: Path
    queries: list[Query]
    targets: TargetsSpec = Field(default_factory=TargetsSpec)


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
    targets = _load_targets(path)
    _validate_query_targets(queries, targets)

    thresholds = _parse_thresholds(meta.get("thresholds") or {})
    try:
        return DatasetSpec(
            name=str(meta.get("name") or path.name),
            description=str(meta.get("description") or ""),
            thresholds=thresholds,
            corpus_dir=corpus_dir,
            queries=queries,
            targets=targets,
        )
    except ValidationError as e:  # pragma: no cover — caught by earlier guards
        raise DatasetError(f"invalid dataset spec at {path}: {e}") from e


def _validate_query_targets(
    queries: list[Query], targets: TargetsSpec
) -> None:
    """Cross-check query positives against ``targets.yaml``.

    A typo in ``expect_chunk_any`` / ``expect_asset_any`` would silently
    drop a query into "permanent miss" — same authoring failure as a
    bad anchor, which we already loud-fail on. Surface it at load time
    so the metric never quietly absorbs the bug.
    """
    chunk_ids = {c.id for c in targets.chunks}
    asset_ids = {a.id for a in targets.assets}
    for i, q in enumerate(queries, start=1):
        unknown_chunks = [x for x in q.expect_chunk_any if x not in chunk_ids]
        unknown_assets = [x for x in q.expect_asset_any if x not in asset_ids]
        if unknown_chunks:
            raise DatasetError(
                f"query #{i} ({q.id or q.q[:40]!r}): expect_chunk_any "
                f"references unknown chunk targets {unknown_chunks}; "
                f"available: {sorted(chunk_ids)}"
            )
        if unknown_assets:
            raise DatasetError(
                f"query #{i} ({q.id or q.q[:40]!r}): expect_asset_any "
                f"references unknown asset targets {unknown_assets}; "
                f"available: {sorted(asset_ids)}"
            )


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


def _load_targets(path: Path) -> TargetsSpec:
    """Load ``targets.yaml`` if present; missing file → empty TargetsSpec."""
    t_path = path / "targets.yaml"
    if not t_path.is_file():
        return TargetsSpec()
    raw = yaml.safe_load(t_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DatasetError(f"{t_path}: top-level YAML must be a mapping")
    try:
        return TargetsSpec.model_validate(raw)
    except ValidationError as e:
        raise DatasetError(f"{t_path}: invalid targets spec: {e}") from e


def _parse_thresholds(raw: Mapping[str, Any]) -> dict[str, float]:
    """Parse threshold dict; supports plain and ``<view>/<metric>`` keys.

    Plain keys (``hit_at_3``) gate on the canonical doc view (back-compat).
    Namespaced keys (``chunk/hit_at_3``, ``asset/hit_at_10``) gate on the
    corresponding granularity view; the runner mirrors the canonical doc
    view's plain key for both ``hit_at_3`` and ``doc/hit_at_3`` so existing
    dataset.yaml files keep gating without modification.
    """
    out: dict[str, float] = {}
    for raw_key, value in raw.items():
        key = str(raw_key)
        if "/" in key:
            view, _, metric = key.partition("/")
            if view not in SUPPORTED_VIEWS:
                raise DatasetError(
                    f"unknown threshold view {view!r} in key {key!r}; "
                    f"supported: {sorted(SUPPORTED_VIEWS)}"
                )
            if metric not in SUPPORTED_METRICS:
                raise DatasetError(
                    f"unknown threshold metric {metric!r} in key {key!r}; "
                    f"supported: {sorted(SUPPORTED_METRICS)}"
                )
        else:
            if key not in SUPPORTED_METRICS:
                raise DatasetError(
                    f"unknown threshold key {key!r}; supported: "
                    f"{sorted(SUPPORTED_METRICS)} or "
                    f"<view>/<metric> with view in {sorted(SUPPORTED_VIEWS)}"
                )
        try:
            out[key] = float(value)
        except (TypeError, ValueError) as e:
            raise DatasetError(f"threshold {key!r} is not a number: {value!r}") from e
    return out
