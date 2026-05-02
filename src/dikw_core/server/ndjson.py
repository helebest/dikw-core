"""NDJSON streaming helper for task event endpoints.

The server emits one JSON object per line (``application/x-ndjson``) so
clients can incrementally parse without HTTP/2 server-sent-events
ceremony. This module provides:

  * ``ndjson_lines`` — joins a historical replay (from ``TaskStore``)
    with a live tail (from ``ProgressBus``) and injects a heartbeat
    every ``HEARTBEAT_INTERVAL`` seconds so reverse proxies don't close
    the long-poll.
  * ``stream_response`` — wraps the iterator in a ``StreamingResponse``
    with the right Content-Type and a no-buffering hint header.

Heartbeats are NOT persisted in the store; they carry ``seq=0`` and
clients drop them on parse.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi.responses import StreamingResponse

from .tasks import ProgressBus, TaskStore

# Tuned to undercut typical proxy idle timeouts (60s on most CDNs / nginx
# defaults) by 4x; bumping it doesn't hurt the engine but does increase
# the time a client takes to notice a fully stalled task.
HEARTBEAT_INTERVAL = 15.0


def _isoformat() -> str:
    return (
        datetime.fromtimestamp(time.time(), tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


async def ndjson_lines(
    *,
    task_id: str,
    store: TaskStore,
    bus: ProgressBus,
    from_seq: int = 0,
    heartbeat_interval: float = HEARTBEAT_INTERVAL,
) -> AsyncIterator[bytes]:
    """Yield bytes for one full NDJSON stream.

    Strategy:
      1. Replay every event with ``seq >= from_seq`` from the store.
      2. If the bus is already closed for this task (terminal state
         persisted), stop — the replay above contains the full tape.
      3. Otherwise subscribe to the bus, BUT first re-replay any events
         that landed in the store between step 1 and the subscription
         (otherwise a client that resumes mid-stream can miss events
         that flushed between the read and the subscribe). De-duplicate
         by seq when forwarding live events.
      4. Inject a heartbeat every ``heartbeat_interval`` seconds while
         the bus stream is active.
    """
    seen_seq = max(0, from_seq - 1)

    # Step 1 — historical replay.
    historical = await store.list_events(task_id, from_seq=from_seq)
    for event in historical:
        yield _serialise(event)
        seen_seq = max(seen_seq, int(event.get("seq", 0)))

    # Step 2 — short-circuit if the task is already terminal.
    if bus.is_closed(task_id):
        return

    # Step 3 — subscribe + catch-up gap-fill.
    sub = await bus.subscribe(task_id)
    catchup = await store.list_events(task_id, from_seq=seen_seq + 1)
    for event in catchup:
        yield _serialise(event)
        seen_seq = max(seen_seq, int(event.get("seq", 0)))

    # Step 4 — interleave live events and heartbeats.
    sub_iter = sub.__aiter__()
    next_event_task = asyncio.ensure_future(sub_iter.__anext__())
    try:
        while True:
            try:
                done, _pending = await asyncio.wait(
                    {next_event_task},
                    timeout=heartbeat_interval,
                )
            except asyncio.CancelledError:
                next_event_task.cancel()
                raise
            if not done:
                # Heartbeat — ephemeral, seq=0.
                yield _serialise(
                    {"type": "heartbeat", "seq": 0, "ts": _isoformat()}
                )
                continue
            try:
                event = next_event_task.result()
            except StopAsyncIteration:
                return
            seq = int(event.get("seq", 0))
            if seq <= seen_seq:
                # Already delivered via replay/catchup — drop.
                next_event_task = asyncio.ensure_future(sub_iter.__anext__())
                continue
            yield _serialise(event)
            seen_seq = seq
            next_event_task = asyncio.ensure_future(sub_iter.__anext__())
    finally:
        if not next_event_task.done():
            next_event_task.cancel()


def stream_response(stream: AsyncIterator[bytes]) -> StreamingResponse:
    """Wrap a byte stream as an NDJSON ``StreamingResponse``."""
    return StreamingResponse(
        stream,
        media_type="application/x-ndjson",
        # Hint to nginx + similar proxies that this stream must not be
        # buffered. Document this in docs/server.md so deployers know
        # to pair with `proxy_buffering off`.
        headers={"X-Accel-Buffering": "no"},
    )


def _serialise(event: dict[str, Any]) -> bytes:
    """Encode a single event as one NDJSON line (trailing ``\\n``)."""
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


__all__ = ["HEARTBEAT_INTERVAL", "ndjson_lines", "stream_response"]
