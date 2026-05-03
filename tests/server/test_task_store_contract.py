"""Contract tests for TaskStore implementations.

Parametrised across the SQLite implementation (always runs) and the
Postgres implementation (runs only when ``DIKW_TEST_POSTGRES_TASKS_DSN``
is set, mirroring the wiki ``test_storage_contract.py`` pattern). Both
must pass the same behavioural assertions so the factory in
``server/tasks/__init__.py`` can swap them transparently.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dikw_core.server.tasks import (
    SqliteTaskStore,
    TaskNotFound,
    TaskRow,
    TaskStatus,
    TaskStore,
)
from dikw_core.server.tasks.store_postgres import PostgresTaskStore

POSTGRES_DSN_ENV = "DIKW_TEST_POSTGRES_TASKS_DSN"


def _now() -> str:
    return "2026-05-02T12:00:00.000Z"


def _row(task_id: str | None = None, *, op: str = "echo") -> TaskRow:
    return TaskRow(
        task_id=task_id or str(uuid.uuid4()),
        op=op,
        status=TaskStatus.PENDING,
        created_at=_now(),
    )


@pytest.fixture(params=["sqlite", "postgres"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[TaskStore]:
    if request.param == "sqlite":
        s: TaskStore = SqliteTaskStore(path=tmp_path / "tasks.db")
        await s.init()
        try:
            yield s
        finally:
            await s.close()
        return

    dsn = os.environ.get(POSTGRES_DSN_ENV)
    if not dsn:
        pytest.skip(f"Postgres TaskStore tests require {POSTGRES_DSN_ENV}")
    schema = f"dikw_test_tasks_{uuid.uuid4().hex[:8]}"
    s = PostgresTaskStore(dsn=dsn, schema=schema)
    await s.init()
    try:
        yield s
    finally:
        await s.close()
        # Best-effort teardown of the per-test schema.
        import psycopg

        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


# ---- task rows ----------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_get_roundtrip(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    fetched = await store.get(row.task_id)
    assert fetched is not None
    assert fetched.task_id == row.task_id
    assert fetched.op == "echo"
    assert fetched.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_get_unknown_returns_none(store: TaskStore) -> None:
    assert await store.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_update_status_and_terminal_payload(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    await store.update_status(
        row.task_id,
        TaskStatus.RUNNING,
        started_at="2026-05-02T12:00:01.000Z",
    )
    await store.update_status(
        row.task_id,
        TaskStatus.SUCCEEDED,
        finished_at="2026-05-02T12:00:02.000Z",
        result={"echoed": 5},
    )
    fetched = await store.get(row.task_id)
    assert fetched is not None
    assert fetched.status == TaskStatus.SUCCEEDED
    assert fetched.started_at == "2026-05-02T12:00:01.000Z"
    assert fetched.finished_at == "2026-05-02T12:00:02.000Z"
    assert fetched.result == {"echoed": 5}


@pytest.mark.asyncio
async def test_update_unknown_raises(store: TaskStore) -> None:
    with pytest.raises(TaskNotFound):
        await store.update_status("nope", TaskStatus.SUCCEEDED)


@pytest.mark.asyncio
async def test_list_filters_and_orders(store: TaskStore) -> None:
    a = _row(op="echo")
    b = _row(op="ingest")
    await store.create(a)
    await store.create(b)
    await store.update_status(a.task_id, TaskStatus.SUCCEEDED)

    by_status = await store.list_tasks(status=TaskStatus.PENDING)
    assert {r.task_id for r in by_status} == {b.task_id}

    by_op = await store.list_tasks(op="echo")
    assert {r.task_id for r in by_op} == {a.task_id}

    every = await store.list_tasks()
    # created_at is identical in tests, so ordering by it is stable but
    # not deterministic across rows; we only assert membership.
    assert {r.task_id for r in every} == {a.task_id, b.task_id}


@pytest.mark.asyncio
async def test_list_running_excludes_terminal(store: TaskStore) -> None:
    pending = _row()
    running = _row()
    done = _row()
    for r in (pending, running, done):
        await store.create(r)
    await store.update_status(running.task_id, TaskStatus.RUNNING)
    await store.update_status(done.task_id, TaskStatus.SUCCEEDED)
    rows = await store.list_running()
    assert {r.task_id for r in rows} == {pending.task_id, running.task_id}


# ---- event tape ---------------------------------------------------------


@pytest.mark.asyncio
async def test_append_event_assigns_monotonic_seq(store: TaskStore) -> None:
    row = _row()
    await store.create(row)

    seq1 = await store.append_event(row.task_id, {"type": "task_started"})
    seq2 = await store.append_event(
        row.task_id, {"type": "progress", "phase": "x"}
    )
    seq3 = await store.append_event(row.task_id, {"type": "final"})
    assert (seq1, seq2, seq3) == (1, 2, 3)


@pytest.mark.asyncio
async def test_list_events_replays_in_seq_order(store: TaskStore) -> None:
    row = _row()
    await store.create(row)

    await store.append_event(row.task_id, {"type": "task_started"})
    await store.append_event(
        row.task_id, {"type": "progress", "phase": "p", "current": 1}
    )
    await store.append_event(row.task_id, {"type": "final"})

    events = await store.list_events(row.task_id)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["type"] == "task_started"
    assert events[1]["phase"] == "p"
    assert "ts" in events[0]


@pytest.mark.asyncio
async def test_list_events_from_seq_truncates(store: TaskStore) -> None:
    row = _row()
    await store.create(row)
    for i in range(5):
        await store.append_event(row.task_id, {"type": "progress", "i": i})

    tail = await store.list_events(row.task_id, from_seq=3)
    assert [e["seq"] for e in tail] == [3, 4, 5]


@pytest.mark.asyncio
async def test_event_isolation_across_tasks(store: TaskStore) -> None:
    a = _row()
    b = _row()
    await store.create(a)
    await store.create(b)
    await store.append_event(a.task_id, {"type": "x"})
    await store.append_event(a.task_id, {"type": "y"})
    await store.append_event(b.task_id, {"type": "z"})
    a_events = await store.list_events(a.task_id)
    b_events = await store.list_events(b.task_id)
    assert [e["seq"] for e in a_events] == [1, 2]
    assert [e["seq"] for e in b_events] == [1]
