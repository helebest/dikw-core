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
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import api
from ..config import CONFIG_FILENAME, ProviderConfig, dump_config_yaml
from ..info.search import HybridSearcher
from ..providers import EmbeddingProvider
from ..storage import build_storage
from .dataset import DatasetSpec
from .fake_embedder import FakeEmbeddings
from .metrics import mean_hit_at_k, mean_reciprocal_rank


class EvalError(RuntimeError):
    """Raised by the runner when a dataset can't be executed end-to-end."""


@dataclass(frozen=True)
class PerQueryRow:
    q: str
    expect_any: list[str]
    ranked: list[str]  # doc stems, top 10

    def to_dict(self) -> dict[str, Any]:
        return {"q": self.q, "expect_any": self.expect_any, "ranked": self.ranked}


class EvalReport(BaseModel):
    """Result of a single ``run_eval`` invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    per_query: list[dict[str, Any]] = Field(default_factory=list)

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
            "metric      value    threshold   result",
            "----------  -------  ----------  ------",
        ]
        for key in ("hit_at_3", "hit_at_10", "mrr"):
            if key not in self.metrics and key not in self.thresholds:
                continue
            val = self.metrics.get(key, float("nan"))
            thr = self.thresholds.get(key)
            verdict = "—" if thr is None else ("pass" if val >= thr else "FAIL")
            thr_str = f"{thr:.3f}" if thr is not None else "    -"
            lines.append(f"{key:10s}  {val:6.3f}  {thr_str:9s}  {verdict}")
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


async def run_eval(
    spec: DatasetSpec,
    *,
    embedder: EmbeddingProvider | None = None,
    provider_config: ProviderConfig | None = None,
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
    """
    if not spec.corpus_dir.is_dir():
        raise EvalError(f"corpus directory not found: {spec.corpus_dir}")

    effective_embedder: EmbeddingProvider = embedder or FakeEmbeddings()
    effective_provider_cfg = provider_config or ProviderConfig(embedding_model="fake")

    with tempfile.TemporaryDirectory(prefix="dikw-eval-") as tmp:
        wiki = _materialise_wiki(Path(tmp), spec, provider_cfg=effective_provider_cfg)
        _copy_corpus(spec.corpus_dir, wiki / "sources")

        await api.ingest(wiki, embedder=effective_embedder)

        per_query = await _run_queries(
            wiki,
            spec,
            embedder=effective_embedder,
            embedding_model=effective_provider_cfg.embedding_model,
        )

    hit10_pairs = [(r.ranked, r.expect_any) for r in per_query]
    metrics = {
        "hit_at_3": mean_hit_at_k(hit10_pairs, 3),
        "hit_at_10": mean_hit_at_k(hit10_pairs, 10),
        "mrr": mean_reciprocal_rank(hit10_pairs),
    }
    return EvalReport(
        dataset_name=spec.name,
        metrics=metrics,
        thresholds=dict(spec.thresholds),
        per_query=[r.to_dict() for r in per_query],
    )


def _materialise_wiki(
    tmp_root: Path, spec: DatasetSpec, *, provider_cfg: ProviderConfig
) -> Path:
    """Scaffold a throwaway wiki + dikw.yml that matches ``spec``.

    The provider block is copied verbatim from ``provider_cfg`` (either the
    caller's real ``cfg.provider`` from a user wiki, or the fake-default).
    This is the single source of truth for downstream ``api.ingest`` — it
    reads ``embedding_model`` / ``embedding_batch_size`` / ``embedding_dimensions``
    from this file.
    """
    wiki = tmp_root / "wiki"
    api.init_wiki(wiki, description=f"eval/{spec.name}")

    from ..config import default_config  # local: keep module import light

    cfg = default_config(description=f"eval/{spec.name}")
    cfg.provider = provider_cfg
    (wiki / CONFIG_FILENAME).write_text(dump_config_yaml(cfg), encoding="utf-8")
    return wiki


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
) -> list[PerQueryRow]:
    cfg, _root = api.load_wiki(wiki)
    storage = build_storage(cfg.storage, root=wiki)
    await storage.connect()
    await storage.migrate()
    try:
        searcher = HybridSearcher(storage, embedder, embedding_model=embedding_model)
        rows: list[PerQueryRow] = []
        for q in spec.queries:
            hits = await searcher.search(q.q, limit=10)
            ranked_stems = [
                Path(h.path).stem if h.path else h.doc_id for h in hits
            ]
            rows.append(
                PerQueryRow(q=q.q, expect_any=list(q.expect_any), ranked=ranked_stems)
            )
        return rows
    finally:
        await storage.close()
