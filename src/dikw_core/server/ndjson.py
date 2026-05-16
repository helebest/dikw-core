"""NDJSON streaming helpers for tier-2 request/response routes.

The server emits one JSON object per line (``application/x-ndjson``) so
clients can incrementally parse without HTTP/2 server-sent-events
ceremony. This module provides:

  * ``stream_response`` — wraps an async byte iterator in a
    ``StreamingResponse`` with the right Content-Type and a no-buffering
    hint header.
  * ``serialise_event`` — encode one dict as one NDJSON line.
  * ``StreamReporterBase`` — shared ``ProgressReporter`` plumbing for
    short-lived streaming routes (``/v1/retrieve``, ``/v1/import``,
    ``/v1/sync``) that own their own NDJSON response cycle.

The tier-1 task subsystem (``/v1/tasks/{id}/events``) has moved to a
cursor-paged JSON endpoint and no longer uses these helpers; the prior
``ndjson_lines`` orchestrator + ``ProgressBus`` integration has been
removed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from ..progress import CancelToken
from ._time import isoformat_utc_ms

# Tuned to undercut typical proxy idle timeouts (60s on most CDNs / nginx
# defaults) by 4x; bumping it doesn't hurt the engine but does increase
# the time a client takes to notice a fully stalled stream.
HEARTBEAT_INTERVAL = 15.0


def isoformat_now() -> str:
    """Backwards-compatible alias for :func:`isoformat_utc_ms` — kept so
    existing imports of ``isoformat_now`` keep working."""
    return isoformat_utc_ms()


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


def serialise_event(event: dict[str, Any]) -> bytes:
    """Encode a single event as one NDJSON line (trailing ``\\n``)."""
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


class StreamReporterBase:
    """Common ``ProgressReporter`` plumbing for short-lived streaming routes.

    Subclasses override ``partial`` to translate engine-specific
    ``partial(kind, payload)`` calls into the route's wire-event vocabulary;
    progress / log / cancel handling is shared because every streaming
    route handles them identically.
    """

    def __init__(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._queue = queue
        self._token = CancelToken()

    async def progress(
        self,
        *,
        phase: str,
        current: int = 0,
        total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "type": "progress",
            "ts": isoformat_now(),
            "phase": phase,
            "current": current,
            "total": total,
        }
        if detail is not None:
            event["detail"] = detail
        await self._queue.put(event)

    async def log(self, level: str, message: str) -> None:
        await self._queue.put(
            {
                "type": "log",
                "ts": isoformat_now(),
                "level": level,
                "message": message,
            }
        )

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        # Default fallthrough — generic envelope so future engine events
        # surface without route plumbing. Subclasses override for
        # op-specific wire shapes.
        await self._queue.put(
            {
                "type": "partial",
                "ts": isoformat_now(),
                "kind": kind,
                "payload": payload,
            }
        )

    def cancel_token(self) -> CancelToken:
        return self._token


__all__ = [
    "HEARTBEAT_INTERVAL",
    "StreamReporterBase",
    "isoformat_now",
    "serialise_event",
    "stream_response",
]
