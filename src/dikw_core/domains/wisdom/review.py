"""Candidate review state machine.

State transitions:

  candidate --approve--> approved
  candidate --reject ---> archived
  approved  --archive--> archived

On every transition we update the DB row, adjust on-disk artefacts
(candidate file and/or kind aggregate), and record a wiki_log entry so
reviews are auditable.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ...schemas import WikiLogEntry, WisdomEvidence, WisdomItem, WisdomKind, WisdomStatus
from ...storage.base import Storage
from .io import delete_candidate_file, regenerate_aggregate


class ReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewResult:
    item_id: str
    new_status: WisdomStatus


async def approve(
    storage: Storage, *, root: Path, item_id: str
) -> ReviewResult:
    item = await storage.get_wisdom(item_id)
    if item is None:
        raise ReviewError(f"no wisdom item with id {item_id!r}")
    if item.status is WisdomStatus.APPROVED:
        return ReviewResult(item_id=item_id, new_status=item.status)
    if item.status is WisdomStatus.ARCHIVED:
        raise ReviewError(f"{item_id} is archived; cannot approve")

    ts = time.time()
    await storage.set_wisdom_status(item_id, WisdomStatus.APPROVED, approved_ts=ts)
    delete_candidate_file(root, item.kind, item.title)
    await _regenerate_kind(storage, root=root, kind=item.kind)
    await storage.append_wiki_log(
        WikiLogEntry(ts=ts, action="review", src=item_id, dst=item.title, note="approved")
    )
    return ReviewResult(item_id=item_id, new_status=WisdomStatus.APPROVED)


async def reject(
    storage: Storage, *, root: Path, item_id: str
) -> ReviewResult:
    item = await storage.get_wisdom(item_id)
    if item is None:
        raise ReviewError(f"no wisdom item with id {item_id!r}")

    ts = time.time()
    await storage.set_wisdom_status(item_id, WisdomStatus.ARCHIVED)
    delete_candidate_file(root, item.kind, item.title)
    # archiving a previously-approved item: refresh the aggregate so it's removed
    if item.status is WisdomStatus.APPROVED:
        await _regenerate_kind(storage, root=root, kind=item.kind)
    await storage.append_wiki_log(
        WikiLogEntry(ts=ts, action="review", src=item_id, dst=item.title, note="rejected")
    )
    return ReviewResult(item_id=item_id, new_status=WisdomStatus.ARCHIVED)


async def regenerate_all_aggregates(storage: Storage, *, root: Path) -> None:
    for kind in WisdomKind:
        await _regenerate_kind(storage, root=root, kind=kind)


async def _regenerate_kind(
    storage: Storage, *, root: Path, kind: WisdomKind
) -> None:
    approved = await storage.list_wisdom(status=WisdomStatus.APPROVED, kind=kind)
    evidence_by_item: dict[str, list[WisdomEvidence]] = {}
    for item in approved:
        evidence_by_item[item.item_id] = await storage.get_wisdom_evidence(item.item_id)
    regenerate_aggregate(
        root, kind=kind, items=approved, evidence_by_item=evidence_by_item
    )


async def iter_candidates(storage: Storage) -> Iterable[WisdomItem]:
    return await storage.list_wisdom(status=WisdomStatus.CANDIDATE)
