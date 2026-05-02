"""Async task endpoints: submit, query, cancel, follow.

Phase 2 ships the *plumbing* — the only registered op is ``echo``, a
five-event mock that lets tests exercise the full submit → events →
final → result loop end-to-end. Phase 3 wires real ops (ingest, synth,
distill, eval) on top of these same routes.

URL shape:
  POST   /v1/{op}                 → submit, returns {task_id, op, links}
  GET    /v1/tasks                → list with status / op filters
  GET    /v1/tasks/{task_id}      → row snapshot
  GET    /v1/tasks/{task_id}/result    → terminal result (404 until terminal)
  GET    /v1/tasks/{task_id}/events    → NDJSON stream (?from_seq=N to resume)
  POST   /v1/tasks/{task_id}/cancel    → request cancellation
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .errors import BadRequest, NotFoundError
from .ndjson import ndjson_lines, stream_response
from .runtime import ServerRuntime, get_runtime
from .tasks import (
    TaskRow,
    TaskRunner,
    TaskStatus,
)

# ---- op runners ---------------------------------------------------------


async def _echo_runner(reporter: Any, *, count: int = 5) -> dict[str, Any]:
    """Mock op for Phase 2 end-to-end testing.

    Emits ``count`` progress events with a simulated phase plus a single
    partial, then returns a result dict the manager wraps in the final
    event. Honours cancellation between iterations.
    """
    for i in range(1, count + 1):
        reporter.cancel_token().raise_if_cancelled()
        await reporter.progress(
            phase="echo",
            current=i,
            total=count,
            detail={"step": i},
        )
        # Sleep long enough that a test can race a cancel against an
        # in-flight task without flakiness, but short enough that the
        # happy-path test finishes quickly. The scale comes from the
        # caller via params.
        await asyncio.sleep(0.0)
    await reporter.partial("echo_done", {"count": count})
    return {"echoed": count}


# ---- request bodies -----------------------------------------------------


class EchoSubmit(BaseModel):
    """Phase 2 mock submit body. Real ops will define their own."""

    count: int = 5


class TaskHandle(BaseModel):
    task_id: str
    op: str
    status: TaskStatus
    created_at: str
    links: dict[str, str]


def _handle(row: TaskRow) -> TaskHandle:
    base = f"/v1/tasks/{row.task_id}"
    return TaskHandle(
        task_id=row.task_id,
        op=row.op,
        status=row.status,
        created_at=row.created_at,
        links={
            "self": base,
            "events": f"{base}/events",
            "result": f"{base}/result",
            "cancel": f"{base}/cancel",
        },
    )


# ---- router -------------------------------------------------------------


class TaskResultBody(BaseModel):
    task_id: str
    status: TaskStatus
    started_at: str | None
    finished_at: str | None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class CancelResponse(BaseModel):
    task_id: str
    cancelled: bool
    already_terminal: bool


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    # ---- submit echo (Phase 2 only) -----------------------------------

    @router.post("/echo", response_model=TaskHandle)
    async def submit_echo(
        request: Request,
        body: EchoSubmit = Body(default_factory=EchoSubmit),
    ) -> TaskHandle:
        rt: ServerRuntime = get_runtime(request.app)
        count = max(1, int(body.count))

        async def _runner(reporter: Any) -> dict[str, Any]:
            return await _echo_runner(reporter, count=count)

        runner: TaskRunner = _runner
        row = await rt.manager.submit(
            op="echo", runner=runner, params={"count": count}
        )
        return _handle(row)

    # ---- task lifecycle endpoints -------------------------------------

    @router.get("/tasks", response_model=list[TaskRow])
    async def list_tasks(
        request: Request,
        status: TaskStatus | None = Query(default=None),
        op: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[TaskRow]:
        rt: ServerRuntime = get_runtime(request.app)
        return await rt.task_store.list_tasks(
            status=status, op=op, limit=limit
        )

    @router.get("/tasks/{task_id}", response_model=TaskRow)
    async def get_task(request: Request, task_id: str) -> TaskRow:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")
        return row

    @router.get(
        "/tasks/{task_id}/result", response_model=TaskResultBody
    )
    async def get_task_result(
        request: Request, task_id: str
    ) -> TaskResultBody:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")
        if row.status not in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            raise BadRequest(
                f"task {task_id!r} is still {row.status.value}; "
                "subscribe to /events to wait for the final state",
                code="task_not_terminal",
                detail={"current_status": row.status.value},
            )
        return TaskResultBody(
            task_id=row.task_id,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            result=row.result,
            error=row.error,
        )

    @router.get("/tasks/{task_id}/events")
    async def follow_task(
        request: Request,
        task_id: str,
        from_seq: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")

        stream = ndjson_lines(
            task_id=task_id,
            store=rt.task_store,
            bus=rt.bus,
            from_seq=from_seq,
        )
        return stream_response(stream)

    @router.post(
        "/tasks/{task_id}/cancel", response_model=CancelResponse
    )
    async def cancel_task(
        request: Request, task_id: str
    ) -> CancelResponse:
        rt: ServerRuntime = get_runtime(request.app)
        row = await rt.task_store.get(task_id)
        if row is None:
            raise NotFoundError(f"task_id {task_id!r} not found")
        was_running = await rt.manager.cancel(task_id)
        terminal = row.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )
        return CancelResponse(
            task_id=task_id,
            cancelled=was_running,
            already_terminal=terminal,
        )

    return router


__all__ = ["make_router"]
