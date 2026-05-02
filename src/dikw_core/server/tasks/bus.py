"""In-memory ProgressBus: per-task event fanout to live subscribers.

Live subscribers (one per ``GET /v1/tasks/{id}/events`` HTTP connection)
get events as they're produced. Resume-by-seq for clients that reconnect
mid-stream is the responsibility of the caller — the route handler reads
historical events from ``TaskStore.list_events(from_seq=...)`` first and
*then* subscribes here for the live tail. The bus itself doesn't replay.

Tasks are explicitly ``register``ed before any event is published. Once
``close``d, every subscriber's iterator drains the remaining queued
events and terminates. Late subscribers (after close) get an empty
iterator and must fall back to ``GET /v1/tasks/{id}/result``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from .store import TaskStoreError

# Sentinel posted on each subscriber queue when the task is closed.
# Coroutines that pull from the queue treat ``None`` as end-of-stream.
_CLOSE_SENTINEL: object = object()


class ProgressBus:
    """One bus instance per server process; tasks are namespaced by id.

    Subscriber bookkeeping uses a per-task list of unbounded
    ``asyncio.Queue`` instances. Unbounded is intentional: the producer
    (the engine task itself) must never block on a slow subscriber, and
    progress events are bounded in size and frequency. Worst case a
    single rogue subscriber holds memory until the task ends + the bus
    closes its queue.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[Any]]] = {}
        self._closed: set[str] = set()
        self._lock = asyncio.Lock()

    async def register(self, task_id: str) -> None:
        async with self._lock:
            if task_id in self._queues or task_id in self._closed:
                raise TaskStoreError(
                    f"task_id {task_id!r} already registered on the bus"
                )
            self._queues[task_id] = []

    async def publish(self, task_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._queues.get(task_id, ()))
        # Outside the lock: queue.put on unbounded queue is non-blocking.
        for q in queues:
            await q.put(event)

    async def close(self, task_id: str) -> None:
        async with self._lock:
            if task_id in self._closed:
                return
            self._closed.add(task_id)
            queues = list(self._queues.get(task_id, ()))
        for q in queues:
            await q.put(_CLOSE_SENTINEL)

    def is_closed(self, task_id: str) -> bool:
        return task_id in self._closed

    async def subscribe(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """Async iterator over live events for ``task_id``.

        Caller is expected to also have read historical events from the
        store before subscribing — events seen here are *only* those
        published after this call. Subscribers added after ``close`` get
        an empty stream.
        """
        q: asyncio.Queue[Any] = asyncio.Queue()
        async with self._lock:
            if task_id in self._closed or task_id not in self._queues:
                # Closed or never-registered: return immediately.
                async def _empty() -> AsyncIterator[dict[str, Any]]:
                    if False:  # pragma: no cover
                        yield {}

                return _empty()
            self._queues[task_id].append(q)

        async def _iter() -> AsyncIterator[dict[str, Any]]:
            try:
                while True:
                    item = await q.get()
                    if item is _CLOSE_SENTINEL:
                        return
                    yield item
            finally:
                async with self._lock:
                    queues = self._queues.get(task_id)
                    if queues is not None and q in queues:
                        queues.remove(q)

        return _iter()


__all__ = ["ProgressBus"]
