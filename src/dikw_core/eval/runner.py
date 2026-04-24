"""Runner — execute a ``DatasetSpec`` end-to-end and return an ``EvalReport``.

The contract is deliberately minimal:

- Input: a ``DatasetSpec`` (from ``load_dataset`` or constructed in code).
- Side effect: a temporary wiki is created, the corpus ingested, hybrid
  search run for each query, then everything is cleaned up.
- Output: ``EvalReport`` with per-metric values, the dataset's thresholds
  echoed for comparison, per-query diagnostic rows for the CLI's failure
  table, and a computed ``passed`` flag.

The runner is hermetic by default — ``FakeEmbeddings`` from
``dikw_core.eval.fake_embedder`` gives deterministic bag-of-words vectors
with no network or API-key dependency. Callers who want real-vector eval
pass their own embedder (e.g., via ``build_embedder(cfg.provider)``).

A single eval can also fan out to all three retrieval modes
(``mode="all"``) — the corpus is still ingested only once, and each
query is then run once per mode against the same storage connection.
This is the workflow used to compare bm25 vs vector vs hybrid against
public-benchmark baselines (BEIR, CMTEB).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .. import api
from ..config import CONFIG_FILENAME, ProviderConfig, RetrievalConfig, dump_config_yaml
from ..info.search import HybridSearcher, RetrievalMode
from ..providers import EmbeddingProvider
from ..storage import build_storage
from .dataset import DatasetSpec
from .fake_embedder import FakeEmbeddings
from .metrics import (
    mean_hit_at_k,
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
)

# Limit per query — large enough to compute recall@100. The same ranked
# list feeds hit@k / nDCG@10 (which slice from the top).
SEARCH_LIMIT = 100

EvalMode = RetrievalMode | Literal["all"]


class EvalError(RuntimeError):
    """Raised by the runner when a dataset can't be executed end-to-end."""


@dataclass(frozen=True)
class PerQueryRow:
    q: str
    expect_any: list[str]
    ranked: list[str]  # doc stems, top SEARCH_LIMIT

    def to_dict(self) -> dict[str, Any]:
        return {"q": self.q, "expect_any": self.expect_any, "ranked": self.ranked}


@dataclass(frozen=True)
class NegativeRow:
    """One ``expect_none=True`` query's observed retrieval. Diagnostic only —
    retrieval always returns something from a non-empty corpus, so there's no
    pass/fail here yet; scoring negatives requires a score threshold or an
    answer-level judge that we don't have in Phase A.
    """

    q: str
    ranked: list[str]  # doc stems, top SEARCH_LIMIT

    def to_dict(self) -> dict[str, Any]:
        return {"q": self.q, "ranked": self.ranked}


class EvalReport(BaseModel):
    """Result of a single ``run_eval`` invocation.

    In single-mode runs (``mode="hybrid"`` etc.) ``metrics`` keys are
    unprefixed (``hit_at_3``, …) — same shape as before this commit so
    existing dataset thresholds continue to bind.

    In ``mode="all"`` runs ``metrics`` carries both:

    * ``f"{mode}/{key}"`` for every (mode, metric) combination — the
      observational ablation surface.
    * Unprefixed ``key`` mirroring the **hybrid** mode's value, so a
      ``hit_at_10: 0.80`` threshold in ``dataset.yaml`` keeps gating on
      hybrid (the historical default), and existing tooling/CI doesn't
      break when a dataset is run with ``--retrieval all``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    per_query: list[dict[str, Any]] = Field(default_factory=list)
    negative_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    # Modes actually executed for this run. Single-element list for the
    # default; three elements for ``mode="all"``. Used by the CLI to pick
    # between the single-table and ablation-table renderings.
    modes: list[str] = Field(default_factory=lambda: ["hybrid"])

    @property
    def passed(self) -> bool:
        """True iff every threshold is met (or no thresholds are configured)."""
        for key, floor in self.thresholds.items():
            value = self.metrics.get(key)
            if value is None:
                # Threshold for a metric the runner didn't compute — fail loudly
                # rather than silently pass.
                return False
            if value < floor:
                return False
        return True

    def diagnostics_table(self) -> str:
        """Plain-text per-query breakdown for test-assertion messages."""
        lines = [
            f"Dataset: {self.dataset_name}",
            "metric         value    threshold   result",
            "-------------  -------  ----------  ------",
        ]
        for key in ("hit_at_3", "hit_at_10", "mrr", "ndcg_at_10", "recall_at_100"):
            if key not in self.metrics and key not in self.thresholds:
                continue
            val = self.metrics.get(key, float("nan"))
            thr = self.thresholds.get(key)
            verdict = "—" if thr is None else ("pass" if val >= thr else "FAIL")
            thr_str = f"{thr:.3f}" if thr is not None else "    -"
            lines.append(f"{key:13s}  {val:6.3f}  {thr_str:9s}  {verdict}")
        lines.append("")
        lines.append("per-query top-5:")
        for row in self.per_query:
            q_short = row["q"] if len(row["q"]) <= 60 else row["q"][:57] + "..."
            top5 = row["ranked"][:5]
            mark = "✓" if any(e in top5 for e in row["expect_any"]) else "✗"
            lines.append(f"  {mark} {q_short}")
            lines.append(f"       expected: {row['expect_any']}")
            lines.append(f"       top-5:    {top5}")
        return "\n".join(lines)


def _resolve_modes(mode: EvalMode) -> list[RetrievalMode]:
    if mode == "all":
        return list(get_args(RetrievalMode))
    return [mode]


def _compute_metrics(positives: list[PerQueryRow]) -> dict[str, float]:
    if not positives:
        return {}
    pairs = [(r.ranked, r.expect_any) for r in positives]
    return {
        "hit_at_3": mean_hit_at_k(pairs, 3),
        "hit_at_10": mean_hit_at_k(pairs, 10),
        "mrr": mean_reciprocal_rank(pairs),
        "ndcg_at_10": mean_ndcg_at_k(pairs, 10),
        "recall_at_100": mean_recall_at_k(pairs, 100),
    }


async def run_eval(
    spec: DatasetSpec,
    *,
    embedder: EmbeddingProvider | None = None,
    provider_config: ProviderConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    mode: EvalMode = "hybrid",
    raw_dump_path: Path | None = None,
) -> EvalReport:
    """Run a single dataset end-to-end; return metrics + diagnostics.

    ``embedder`` + ``provider_config`` are the two knobs:

    * Both ``None`` (the default) → hermetic: ``FakeEmbeddings`` + a
      ``ProviderConfig`` with ``embedding_model="fake"`` (everything else
      default). No network, no keys, <1s.
    * Both set → real-vector eval: the caller built an embedder from a
      wiki's ``ProviderConfig`` and hands both in. The runner serialises
      that config into the temp wiki's ``dikw.yml`` so ``api.ingest`` picks
      up vendor-specific ``embedding_batch_size`` / ``embedding_dimensions``
      exactly as the source wiki has them. Without this, a Gitee-configured
      provider would ingest at ``batch_size=64`` and get HTTP 400.

    ``embedder`` set but ``provider_config`` None (or vice-versa) falls
    back to the default half — useful for tests that want a custom embedder
    on the hermetic config.

    ``mode`` selects which retrieval leg(s) to score: ``"hybrid"``
    (default, BM25+vec via RRF), ``"bm25"``, ``"vector"``, or ``"all"``
    (run all three sequentially against the same ingested corpus).

    ``raw_dump_path`` — when set (and ``mode="all"``), appends one JSONL
    row per (query, mode) capturing the top-``SEARCH_LIMIT`` ranked doc
    stems plus ``expect_any`` / ``expect_none``. Callers supply a path
    they've already truncated (or never existed); the runner only
    appends. Downstream ``evals/tools/sweep_rrf.py`` reads this to
    re-fuse offline at arbitrary ``(rrf_k, weights)`` — avoiding a
    second expensive embedding pass. For single-mode runs the flag is
    silently a no-op since sweep needs both legs' rankings.
    """
    if not spec.corpus_dir.is_dir():
        raise EvalError(f"corpus directory not found: {spec.corpus_dir}")

    effective_embedder: EmbeddingProvider = embedder or FakeEmbeddings()
    effective_provider_cfg = provider_config or ProviderConfig(embedding_model="fake")
    effective_retrieval_cfg = retrieval_config or RetrievalConfig()
    modes = _resolve_modes(mode)

    with tempfile.TemporaryDirectory(prefix="dikw-eval-") as tmp:
        wiki = _materialise_wiki(
            Path(tmp),
            spec,
            provider_cfg=effective_provider_cfg,
            retrieval_cfg=effective_retrieval_cfg,
        )
        _copy_corpus(spec.corpus_dir, wiki / "sources")

        await api.ingest(wiki, embedder=effective_embedder)

        per_mode = await _run_queries(
            wiki,
            spec,
            embedder=effective_embedder,
            embedding_model=effective_provider_cfg.embedding_model,
            modes=modes,
        )

    # Metrics scored only over positive queries. Including negatives would
    # silently drag hit@k toward 0.0 (retrieval always returns something
    # from a non-empty corpus) and tie pass/fail to negative count.
    metrics: dict[str, float] = {}
    canonical_mode: RetrievalMode = (
        "hybrid" if "hybrid" in modes else modes[0]
    )
    canonical_positives = per_mode[canonical_mode][0]
    canonical_negatives = per_mode[canonical_mode][1]

    if len(modes) == 1:
        metrics.update(_compute_metrics(canonical_positives))
    else:
        # Multi-mode: prefix every metric, plus mirror the canonical mode
        # unprefixed so existing dataset thresholds keep gating.
        for m in modes:
            positives_m, _ = per_mode[m]
            for k, v in _compute_metrics(positives_m).items():
                metrics[f"{m}/{k}"] = v
        metrics.update(_compute_metrics(canonical_positives))

    if raw_dump_path is not None and len(modes) > 1:
        _dump_raw_ranked(raw_dump_path, spec.name, per_mode)

    return EvalReport(
        dataset_name=spec.name,
        metrics=metrics,
        thresholds=dict(spec.thresholds),
        per_query=[r.to_dict() for r in canonical_positives],
        negative_diagnostics=[r.to_dict() for r in canonical_negatives],
        modes=list(modes),
    )


def _materialise_wiki(
    tmp_root: Path,
    spec: DatasetSpec,
    *,
    provider_cfg: ProviderConfig,
    retrieval_cfg: RetrievalConfig,
) -> Path:
    """Scaffold a throwaway wiki + dikw.yml that matches ``spec``.

    ``provider_cfg`` and ``retrieval_cfg`` are copied verbatim into the
    written ``dikw.yml``. Downstream ``api.ingest`` reads the provider
    block; ``_run_queries`` re-loads the whole file to build
    ``HybridSearcher``, which picks up the retrieval block. This means
    eval reproducibly measures whatever fusion knobs the caller passed,
    without the runner having to thread them through a second path.
    """
    wiki = tmp_root / "wiki"
    api.init_wiki(wiki, description=f"eval/{spec.name}")

    from ..config import default_config  # local: keep module import light

    cfg = default_config(description=f"eval/{spec.name}")
    cfg.provider = provider_cfg
    cfg.retrieval = retrieval_cfg
    (wiki / CONFIG_FILENAME).write_text(dump_config_yaml(cfg), encoding="utf-8")
    return wiki


def _dump_raw_ranked(
    path: Path,
    dataset_name: str,
    per_mode: dict[RetrievalMode, tuple[list[PerQueryRow], list[NegativeRow]]],
) -> None:
    """Append per-mode ranked lists as JSONL.

    One row per (query, mode). Positive queries carry ``expect_any``;
    negatives carry ``expect_none: true`` with an empty ``expect_any``.
    A consumer groups by ``q`` (or ``q_id`` if we ever add one) to get
    each query's bm25/vector/hybrid rankings side-by-side.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for mode, (positives, negatives) in per_mode.items():
            for row in positives:
                f.write(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "mode": mode,
                            "q": row.q,
                            "expect_any": row.expect_any,
                            "expect_none": False,
                            "ranked": row.ranked,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for neg in negatives:
                f.write(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "mode": mode,
                            "q": neg.q,
                            "expect_any": [],
                            "expect_none": True,
                            "ranked": neg.ranked,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _copy_corpus(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
        elif item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)


@dataclass
class _DriverResult:
    per_query: list[PerQueryRow] = field(default_factory=list)


async def _run_queries(
    wiki: Path,
    spec: DatasetSpec,
    *,
    embedder: EmbeddingProvider,
    embedding_model: str,
    modes: list[RetrievalMode],
) -> dict[RetrievalMode, tuple[list[PerQueryRow], list[NegativeRow]]]:
    """Run every query in ``spec`` once per mode against a single storage
    connection. Returns a dict keyed by mode.
    """
    cfg, _root = api.load_wiki(wiki)
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        searcher = HybridSearcher(
            storage,
            embedder,
            embedding_model=embedding_model,
            rrf_k=cfg.retrieval.rrf_k,
            bm25_weight=cfg.retrieval.bm25_weight,
            vector_weight=cfg.retrieval.vector_weight,
            cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
        )
        results: dict[
            RetrievalMode, tuple[list[PerQueryRow], list[NegativeRow]]
        ] = {}
        for m in modes:
            positives: list[PerQueryRow] = []
            negatives: list[NegativeRow] = []
            for q in spec.queries:
                hits = await searcher.search(q.q, limit=SEARCH_LIMIT, mode=m)
                ranked_stems = [
                    Path(h.path).stem if h.path else h.doc_id for h in hits
                ]
                if q.expect_none:
                    negatives.append(NegativeRow(q=q.q, ranked=ranked_stems))
                else:
                    positives.append(
                        PerQueryRow(
                            q=q.q,
                            expect_any=list(q.expect_any),
                            ranked=ranked_stems,
                        )
                    )
            results[m] = (positives, negatives)
        return results
    finally:
        await storage.close()
