"""Synth + distill task wiring.

These ops mirror ``ingest_op.make_ingest_runner``: build a
``TaskRunner`` closure that ``TaskManager.submit`` schedules. Each runner
owns its own provider clients (LLM, optional embedder) for the duration
of the task and returns a JSON-serialisable result dict that the manager
folds into the ``final`` event.

Synth + distill share the same provider-failure mode: a missing
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` surfaces as a runner-level
``BadRequest``, so the task ends ``failed`` with a clear cause instead of
a partially-progressed terminal that hid the misconfiguration.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import api
from ..progress import ProgressReporter
from ..providers import build_embedder, build_llm
from .errors import BadRequest

logger = logging.getLogger(__name__)


def make_synth_runner(
    *,
    wiki_root: Path,
    cfg: Any,  # DikwConfig — typed Any to keep server modules thin
    force_all: bool,
    no_embed: bool,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.synthesize`` for one task."""

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        try:
            llm = build_llm(cfg.provider)
        except Exception as e:
            raise BadRequest(
                f"could not build LLM: {e}", code="llm_unavailable"
            ) from e

        embedder = None
        if not no_embed:
            try:
                embedder = build_embedder(cfg.provider)
            except Exception as e:
                # Synth without an embedder is a soft-degrade: pages
                # land on disk but new K-layer chunks miss the dense
                # leg. Surface this as a log so the operator sees it
                # but the task still succeeds.
                await reporter.log(
                    "WARN",
                    f"embedder unavailable, synth proceeds without dense indexing: {e}",
                )

        report = await api.synthesize(
            wiki_root,
            force_all=force_all,
            llm=llm,
            embedder=embedder,
            reporter=reporter,
        )
        return dataclasses.asdict(report)

    return _runner


def make_distill_runner(
    *,
    wiki_root: Path,
    cfg: Any,
    pages_per_call: int,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.distill`` for one task."""

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        try:
            llm = build_llm(cfg.provider)
        except Exception as e:
            raise BadRequest(
                f"could not build LLM: {e}", code="llm_unavailable"
            ) from e

        report = await api.distill(
            wiki_root,
            llm=llm,
            pages_per_call=pages_per_call,
            reporter=reporter,
        )
        return dataclasses.asdict(report)

    return _runner


def make_eval_runner(
    *,
    wiki_root: Path,
    cfg: Any,
    dataset: str,
    mode: str,
    cache_mode: str,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``eval.run_eval`` for one task.

    The runner builds an embedder + (optional) multimodal embedder from
    the server's wiki cfg so eval scores against the same vector space
    the live engine uses. ``dataset`` may be a registered name (resolved
    under the packaged datasets root) or an explicit path.
    """

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        # Defer the eval imports — the eval module pulls in dataset
        # validators + corpus walkers we don't need on a server that
        # never runs eval.
        from ..eval.dataset import DatasetError, load_dataset
        from ..eval.runner import EvalError, run_eval
        from ..providers import build_multimodal_embedder

        try:
            spec = load_dataset(dataset)
        except DatasetError as e:
            raise BadRequest(
                f"could not load dataset {dataset!r}: {e}",
                code="dataset_not_found",
            ) from e

        try:
            embedder = build_embedder(cfg.provider)
        except Exception as e:
            raise BadRequest(
                f"could not build embedder: {e}",
                code="embedder_unavailable",
            ) from e

        multimodal_embedder = None
        mm_cfg = cfg.assets.multimodal
        if mm_cfg is not None:
            try:
                multimodal_embedder = build_multimodal_embedder(
                    mm_cfg.provider,
                    base_url=mm_cfg.base_url,
                    batch=mm_cfg.batch,
                )
            except Exception as e:
                await reporter.log(
                    "WARN",
                    f"multimodal embedder unavailable, "
                    f"eval proceeds with text-only: {e}",
                )

        try:
            report = await run_eval(
                spec,
                embedder=embedder,
                provider_config=cfg.provider,
                retrieval_config=cfg.retrieval,
                assets_config=cfg.assets,
                multimodal_embedder=multimodal_embedder,
                mode=mode,  # type: ignore[arg-type]
                cache_mode=cache_mode,  # type: ignore[arg-type]
                reporter=reporter,
            )
        except EvalError as e:
            raise BadRequest(
                f"eval failed: {e}", code="eval_error"
            ) from e
        # ``EvalReport`` is a pydantic BaseModel — model_dump gives us
        # JSON-safe output the manager can drop straight into the final
        # event.
        _ = wiki_root  # eval owns its own throwaway wiki tree
        return report.model_dump(mode="json")

    return _runner


__all__ = [
    "make_distill_runner",
    "make_eval_runner",
    "make_synth_runner",
]
