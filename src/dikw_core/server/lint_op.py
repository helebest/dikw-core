"""Lint propose / apply task wiring.

Mirrors :mod:`synth_op`: each ``make_*_runner`` returns a ``TaskRunner``
closure. Apply's runner reads the source ``propose`` task's ``result``
out of :attr:`ServerRuntime.task_store` and reconstructs the
:class:`FixProposalReport` so the user can submit propose + apply as
two separate HTTP requests without round-tripping the (potentially
large) proposal payload through the wire.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import api
from ..domains.knowledge.lint import LintKind
from ..domains.knowledge.lint_fix import FixProposalReport
from ..progress import ProgressReporter
from .errors import BadRequest, NotFoundError
from .tasks import TaskStatus, TaskStore


def make_lint_propose_runner(
    *,
    wiki_root: Path,
    rule: LintKind | None,
    limit: int,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that drives ``api.lint_propose``."""

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        report = await api.lint_propose(
            wiki_root,
            rule=rule,
            limit=limit,
            reporter=reporter,
        )
        return report.model_dump(mode="json")

    return _runner


def make_lint_apply_runner(
    *,
    wiki_root: Path,
    proposal_task_id: str,
    task_store: TaskStore,
    pick: list[int] | None,
    skip: list[int] | None,
) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build a ``TaskRunner`` that reads a propose task's result and
    drives ``api.lint_apply``.

    The propose task must have terminated successfully; otherwise we
    raise :class:`BadRequest` so the apply task fails with a clear
    cause. Reading the proposal *inside* the runner (not at submit
    time) keeps the HTTP submit path fast and lets the runner emit
    progress events for the read step.
    """

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        await reporter.progress(
            phase="lint_apply",
            current=0,
            total=0,
            detail={"step": "loading proposal", "proposal_task_id": proposal_task_id},
        )
        row = await task_store.get(proposal_task_id)
        if row is None:
            raise NotFoundError(
                f"propose task {proposal_task_id!r} not found"
            )
        if row.status != TaskStatus.SUCCEEDED:
            raise BadRequest(
                f"propose task {proposal_task_id!r} is in status "
                f"{row.status.value!r}; only SUCCEEDED proposals can be applied",
                code="proposal_not_terminal",
            )
        if not row.result:
            raise BadRequest(
                f"propose task {proposal_task_id!r} has no result payload",
                code="proposal_empty",
            )
        try:
            proposal_report = FixProposalReport.model_validate(row.result)
        except Exception as e:
            raise BadRequest(
                f"propose task {proposal_task_id!r} result does not look like "
                f"a FixProposalReport: {e}",
                code="proposal_malformed",
            ) from e

        apply_report = await api.lint_apply(
            wiki_root,
            proposal_report=proposal_report,
            pick=pick,
            skip=skip,
            reporter=reporter,
        )
        return apply_report.model_dump(mode="json")

    return _runner


__all__ = ["make_lint_apply_runner", "make_lint_propose_runner"]
