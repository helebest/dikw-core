"""``follow_to_terminal`` — the client-side long-poll loop that drives
every ``--wait`` flag on op commands and the ``dikw client tasks wait`` UX.

Contract under test:

* Returns ``(TaskStatus, result|error|None)``.
* Renders progress through an optional ``TaskProgressRenderer`` (we
  drive it through a fake renderer here so the test asserts on event
  delivery, not on terminal output).
* Maps total-budget exhaustion to ``TimeoutError`` (the CLI layer
  translates this to exit 124).
* Pages through the full event tape — backlog plus tail — without
  duplicates or gaps.

Tests use the in-memory ASGI ``client_transport`` fixture plus the
echo runner's ``delay_ms`` knob (per Step 5) to exercise the live
wake-up path deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from dikw_core.client.task_follow import follow_to_terminal
from dikw_core.client.transport import Transport


class _CollectingRenderer:
    """Stand-in for ``TaskProgressRenderer`` — records every event so
    tests can verify the helper actually pumps progress through it."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def render(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))


@pytest.mark.asyncio
async def test_follow_to_terminal_returns_succeeded(
    client_transport: Transport,
) -> None:
    handle = await client_transport.post_json(
        "/v1/echo", json_body={"count": 3}
    )
    task_id = str(handle["task_id"])

    status, payload = await follow_to_terminal(
        client_transport, task_id, poll_wait=5
    )
    assert status == "succeeded"
    assert payload is not None and payload.get("echoed") == 3


@pytest.mark.asyncio
async def test_follow_to_terminal_pumps_events_to_renderer(
    client_transport: Transport,
) -> None:
    handle = await client_transport.post_json(
        "/v1/echo", json_body={"count": 4}
    )
    task_id = str(handle["task_id"])
    renderer = _CollectingRenderer()

    status, _ = await follow_to_terminal(
        client_transport, task_id, renderer=renderer, poll_wait=5
    )
    assert status == "succeeded"
    types = [e.get("type") for e in renderer.events]
    # task_started + per-iteration progress + a partial + final all flow.
    assert types[0] == "task_started"
    assert types[-1] == "final"
    assert types.count("progress") >= 4


@pytest.mark.asyncio
async def test_follow_to_terminal_pages_through_backlog(
    client_transport: Transport,
) -> None:
    """A small ``page_limit`` forces multiple paged GETs; every event
    must surface exactly once, in seq order."""
    handle = await client_transport.post_json(
        "/v1/echo", json_body={"count": 12}
    )
    task_id = str(handle["task_id"])
    renderer = _CollectingRenderer()

    status, _ = await follow_to_terminal(
        client_transport,
        task_id,
        renderer=renderer,
        poll_wait=5,
        page_limit=3,
    )
    assert status == "succeeded"
    # Every event has a unique seq; seqs are monotonically increasing.
    seqs = [int(e["seq"]) for e in renderer.events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


@pytest.mark.asyncio
async def test_follow_to_terminal_cancelled_returns_cancelled(
    client_transport: Transport,
) -> None:
    handle = await client_transport.post_json(
        "/v1/echo", json_body={"count": 50, "delay_ms": 80}
    )
    task_id = str(handle["task_id"])

    async def _cancel_soon() -> None:
        await asyncio.sleep(0.1)
        await client_transport.post_json(f"/v1/tasks/{task_id}/cancel")

    cancel_task = asyncio.create_task(_cancel_soon())
    try:
        status, _ = await follow_to_terminal(
            client_transport, task_id, poll_wait=5
        )
    finally:
        await cancel_task
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_follow_to_terminal_total_timeout_raises(
    client_transport: Transport,
) -> None:
    """``total_timeout`` is a client-side budget — when it fires we
    raise ``TimeoutError`` and the CLI maps it to exit 124. The task
    itself is left running (not auto-cancelled)."""
    handle = await client_transport.post_json(
        "/v1/echo", json_body={"count": 30, "delay_ms": 200}
    )
    task_id = str(handle["task_id"])

    with pytest.raises(TimeoutError):
        await follow_to_terminal(
            client_transport,
            task_id,
            poll_wait=1,
            total_timeout=0.4,
        )
    # Cleanup so the runner doesn't leak past the test.
    await client_transport.post_json(f"/v1/tasks/{task_id}/cancel")
