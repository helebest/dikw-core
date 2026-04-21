from __future__ import annotations

import time

from dikw_core.schemas import WisdomItem, WisdomKind, WisdomStatus
from dikw_core.wisdom.apply import pick_applicable


def _item(title: str, body: str, kind: WisdomKind = WisdomKind.PRINCIPLE) -> WisdomItem:
    return WisdomItem(
        item_id=f"W-{title[:8]}",
        kind=kind,
        status=WisdomStatus.APPROVED,
        path=None,
        title=title,
        body=body,
        confidence=0.8,
        created_ts=time.time(),
        approved_ts=time.time(),
    )


def test_pick_applicable_scores_overlap() -> None:
    items = [
        _item("Prefer deterministic scoping", "Deterministic scoping beats probabilistic."),
        _item("Mock DB tests hide prod errors", "Mocking databases can hide real failures."),
    ]
    picked = pick_applicable("how should I scope retrieval deterministically?", items)
    assert picked and picked[0].item.title.startswith("Prefer deterministic")


def test_pick_applicable_respects_limit_and_threshold() -> None:
    items = [
        _item("Unrelated", "Completely unrelated words here foobar quux"),
    ]
    picked = pick_applicable("what's for lunch?", items)
    assert picked == []


def test_pick_applicable_empty_inputs() -> None:
    assert pick_applicable("anything", []) == []
    assert pick_applicable("", [_item("x", "y")]) == []
