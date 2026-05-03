"""Ingest task wiring — bridge between ``api.ingest`` and the TaskManager.

Lives in its own module (rather than inside ``routes_tasks.py``) so the
upload-commit logic + the TaskRunner closure are independently testable
and re-importable from Phase 4 ops that need the same staging-commit
helper (synth on a freshly uploaded source set, etc.).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import api
from ..config import CONFIG_FILENAME, load_config
from ..progress import ProgressReporter
from ..providers import build_embedder, build_multimodal_embedder
from .errors import BadRequest, NotFoundError
from .routes_upload import STAGING_DIRNAME

logger = logging.getLogger(__name__)


def commit_staging(wiki_root: Path, upload_id: str) -> dict[str, int]:
    """Move every file from a successful upload's staging tree into the
    wiki's ``sources/`` and ``assets/`` subtrees, then drop the staging
    dir. Returns counts so the caller can log.

    Same-filesystem ``os.replace`` keeps each rename atomic; cross-tree
    visibility (a half-committed staging is not observable) is the
    server's contract because we hold the upload lock for the duration.
    Raises ``NotFoundError`` if ``upload_id`` doesn't correspond to a
    staging directory — the most common operator error is calling
    ingest with a stale or wrong id.
    """
    staging = wiki_root / STAGING_DIRNAME / upload_id
    if not staging.is_dir():
        raise NotFoundError(
            f"unknown upload_id: {upload_id!r}",
            detail={"hint": "was the upload garbage-collected?"},
        )

    counts: dict[str, int] = {"sources": 0, "assets": 0}
    for top in ("sources", "assets"):
        src_root = staging / top
        if not src_root.is_dir():
            continue
        dst_root = wiki_root / top
        dst_root.mkdir(parents=True, exist_ok=True)
        for entry in src_root.rglob("*"):
            if entry.is_dir():
                continue
            rel = entry.relative_to(src_root)
            target = dst_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            # ``os.replace`` is atomic + overwrites — exactly the
            # semantics ingest needs (re-uploading the same file
            # overwrites the on-disk copy; ingest's content-hash check
            # then decides whether to re-process).
            os.replace(entry, target)
            counts[top] += 1

    shutil.rmtree(staging, ignore_errors=True)
    logger.info(
        "committed upload %s: sources=%d assets=%d → %s",
        upload_id,
        counts["sources"],
        counts["assets"],
        wiki_root,
    )
    return counts


def _ingest_report_to_dict(
    report: api.IngestReport,
    *,
    upload_commit: dict[str, int] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = dataclasses.asdict(report)
    if upload_commit is not None:
        out["upload_commit"] = upload_commit
    return out


def make_ingest_runner(
    *,
    wiki_root: Path,
    upload_id: str | None,
    no_embed: bool,
    lock: asyncio.Lock | None = None,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.ingest`` for one task.

    ``upload_id`` (when set) commits the staged tree onto the wiki BEFORE
    ingest runs, so the runner sees a coherent ``sources/`` tree. The
    commit happens inside the runner — not earlier — because the
    TaskManager-managed cancellation token covers the runner's whole
    lifetime, while a pre-submit commit would leave a "task created but
    nothing happened yet" gap visible to subscribers.

    ``lock`` (the runtime's ``ingest_lock``) serializes overlapping
    ingest tasks so two concurrent uploads can't interleave their
    staging-commit + on-disk writes. Tests that drive the runner in
    isolation may pass ``None``.
    """

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        # The lock is held for the whole runner — concurrent ingests on
        # the same wiki would also race on the storage row writes, not
        # just the staging commit, so single-threading the entire op is
        # the only safe story.
        guard = lock if lock is not None else asyncio.Lock()
        async with guard:
            # Reload cfg from disk INSIDE the runner so the provider
            # objects we build below stay aligned with whatever
            # ``api.ingest`` reads (it does its own ``_with_storage`` →
            # ``load_config`` reload). Otherwise an operator who edits
            # ``dikw.yml`` while the long-lived server is up gets
            # vectors tagged with the FRESH ``embed_version`` but
            # produced by the STALE provider URL/model — silent corruption.
            cfg = load_config(wiki_root / CONFIG_FILENAME)

            upload_commit: dict[str, int] | None = None
            if upload_id is not None:
                await reporter.progress(
                    phase="upload_commit",
                    current=0,
                    total=0,
                    detail={"upload_id": upload_id},
                )
                upload_commit = commit_staging(wiki_root, upload_id)
                await reporter.progress(
                    phase="upload_commit",
                    current=upload_commit["sources"] + upload_commit["assets"],
                    total=upload_commit["sources"] + upload_commit["assets"],
                    detail={"upload_id": upload_id, **upload_commit},
                )

            embedder = None
            multimodal_embedder = None
            if not no_embed:
                try:
                    embedder = build_embedder(cfg.provider)
                except Exception as e:
                    # Surface as a runner-level failure so the task ends
                    # ``failed`` with a clear cause (most often a missing API
                    # key); the engine itself can't differentiate config
                    # errors from network errors mid-run.
                    raise BadRequest(
                        f"could not build embedder: {e}",
                        code="embedder_unavailable",
                    ) from e
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
                            f"falling back to text-only ingest: {e}",
                        )

            report = await api.ingest(
                wiki_root,
                embedder=embedder,
                multimodal_embedder=multimodal_embedder,
                reporter=reporter,
            )
            return _ingest_report_to_dict(report, upload_commit=upload_commit)

    return _runner


__all__ = ["commit_staging", "make_ingest_runner"]
