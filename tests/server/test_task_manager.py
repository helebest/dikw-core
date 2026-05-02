"""TaskManager + ProgressBus + TaskBusReporter behavioural tests.

Drives the manager directly (no FastAPI) to cover:
  * happy-path submit → task_started → progress* → final{succeeded}
  * failure path → final{failed} with traceback in error
  * cancellation: pre-cancel + mid-flight cancel both end in final{cancelled}
  * multi-subscriber fanout via the bus
  * resume-by-seq replay through ``store.list_events``
  * restart_cleanup marks orphan rows as failed{server_restart}
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dikw_core.progress import ProgressReporter
from dikw_core.server.tasks import (
    ProgressBus,
    SqliteTaskStore,
    TaskManager,
    TaskStatus,
)


async def _wait_terminal(
    store: SqliteTaskStore, task_id: str, *, timeout: float = 5.0
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        row = await store.get(task_id)
        if row is not None and row.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"task {task_id} did not reach terminal state in {timeout}s")


@pytest.mark.asyncio
async def test_happy_path_emits_full_event_tape(
    manager_only: tuple[TaskManager, SqliteTaskStore, ProgressBus],
) -> None:
    manager, store, _bus = manager_only

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        await reporter.progress(phase="step", current=1, total=2)
        await reporter.progress(phase="step", current=2, total=2)
        await reporter.partial("hello", {"world": True})
        return {"ok": True}

    row = await manager.submit(op="echo", runner=_runner, params={"a": 1})
    await _wait_terminal(store, row.task_id)
    final_row = await store.get(row.task_id)
    assert final_row is not None
    assert final_row.status == TaskStatus.SUCCEEDED
    assert final_row.result == {"ok": True}
    assert final_row.params_digest, "params_digest should be a non-empty hash"

    events = await store.list_events(row.task_id)
    types = [e["type"] for e in events]
    assert types == [
        "task_started",
        "progress",
        "progress",
        "partial",
        "final",
    ]
    assert events[0]["task_id"] == row.task_id
    assert events[0]["op"] == "echo"
    assert events[-1]["status"] == "succeeded"
    # seq strictly monotonic
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert seqs[0] == 1


@pytest.mark.asyncio
async def test_failure_path_records_traceback(
    manager_only: tuple[TaskManager, SqliteTaskStore, ProgressBus],
) -> None:
    manager, store, _bus = manager_only

    async def _runner(_reporter: ProgressReporter) -> dict[str, Any]:
        raise ValueError("boom")

    row = await manager.submit(op="echo", runner=_runner)
    await _wait_terminal(store, row.task_id)
    final_row = await store.get(row.task_id)
    assert final_row is not None
    assert final_row.status == TaskStatus.FAILED
    assert final_row.error is not None
    assert final_row.error["code"] == "ValueError"
    assert final_row.error["message"] == "boom"
    assert "Traceback" in final_row.error["traceback"]


@pytest.mark.asyncio
async def test_pre_cancel_via_token(
    manager_only: tuple[TaskManager, SqliteTaskStore, ProgressBus],
) -> None:
    manager, store, _bus = manager_only

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        # Honour cancel token at the first checkpoint.
        for i in range(10):
            reporter.cancel_token().raise_if_cancelled()
            await reporter.progress(phase="x", current=i, total=10)
            await asyncio.sleep(0.01)
        return {"done": True}

    row = await manager.submit(op="echo", runner=_runner)
    # Race a cancel against the runner's first checkpoint.
    await asyncio.sleep(0)  # let the runner schedule
    await manager.cancel(row.task_id)
    await _wait_terminal(store, row.task_id)
    final_row = await store.get(row.task_id)
    assert final_row is not None
    assert final_row.status == TaskStatus.CANCELLED
    events = await store.list_events(row.task_id)
    assert events[-1]["type"] == "final"
    assert events[-1]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_unknown_task_returns_false(
    manager_only: tuple[TaskManager, SqliteTaskStore, ProgressBus],
) -> None:
    manager, _store, _bus = manager_only
    assert await manager.cancel("nope") is False


@pytest.mark.asyncio
async def test_bus_fanout_to_two_subscribers(
    manager_only: tuple[TaskManager, SqliteTaskStore, ProgressBus],
) -> None:
    manager, store, bus = manager_only

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        # Pause briefly so subscribers get a chance to attach before
        # the runner emits its first event.
        await asyncio.sleep(0.02)
        for i in range(3):
            await reporter.progress(phase="p", current=i, total=3)
        return {"ok": True}

    row = await manager.submit(op="echo", runner=_runner)

    async def _drain() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        sub = await bus.subscribe(row.task_id)
        async for ev in sub:
            out.append(ev)
        return out

    a, b = await asyncio.gather(_drain(), _drain())
    await _wait_terminal(store, row.task_id)
    # Both subscribers see the same event types in the same order.
    assert [e["type"] for e in a] == [e["type"] for e in b]
    # After the bus closes, no more events.
    assert a[-1]["type"] == "final"


@pytest.mark.asyncio
async def test_resume_by_seq_via_store(
    manager_only: tuple[TaskManager, SqliteTaskStore, ProgressBus],
) -> None:
    manager, store, _bus = manager_only

    async def _runner(reporter: ProgressReporter) -> dict[str, Any]:
        for i in range(5):
            await reporter.progress(phase="p", current=i, total=5)
        return {"ok": True}

    row = await manager.submit(op="echo", runner=_runner)
    await _wait_terminal(store, row.task_id)

    full = await store.list_events(row.task_id)
    assert len(full) >= 7  # task_started + 5 progress + final, possibly more
    tail = await store.list_events(row.task_id, from_seq=4)
    assert tail[0]["seq"] == 4
    assert {e["seq"] for e in tail} == {e["seq"] for e in full if e["seq"] >= 4}


@pytest.mark.asyncio
async def test_restart_cleanup_marks_orphans_failed(
    manager_only: tuple[TaskManager, SqliteTaskStore, ProgressBus],
) -> None:
    _manager, store, bus = manager_only
    # Simulate a leftover row from a prior process: PENDING in storage,
    # nothing in the in-memory manager.
    from dikw_core.server.tasks import TaskRow

    leftover = TaskRow(
        task_id="leftover-1",
        op="echo",
        status=TaskStatus.RUNNING,
        created_at="2026-05-02T11:00:00.000Z",
        started_at="2026-05-02T11:00:01.000Z",
    )
    await store.create(leftover)

    # New manager on the same store mimics a fresh server boot.
    new_manager = TaskManager(store=store, bus=bus)
    await new_manager.restart_cleanup()

    row = await store.get("leftover-1")
    assert row is not None
    assert row.status == TaskStatus.FAILED
    assert row.error == {"reason": "server_restart"}
    events = await store.list_events("leftover-1")
    # The synthetic final event lands on the tape so any latent NDJSON
    # subscriber sees a clean terminator.
    assert events[-1]["type"] == "final"
    assert events[-1]["status"] == "failed"
