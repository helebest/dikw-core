"""Query NDJSON streaming.

``POST /v1/query`` is the one short-lived op that streams instead of going
through the TaskManager: the response shape is a sequence of NDJSON
events the client renders as it arrives. Wire shape:

  {"type":"query_started","q":"...","limit":5}
  {"type":"retrieval_done","hits":[...]}
  {"type":"llm_token","delta":"..."}            # zero or more
  {"type":"final","status":"succeeded","result":{...QueryResult...}}

On error, the final event has ``status:"failed"`` and an ``error`` body.
The route runs ``api.query`` in a sibling task, plumbs progress through a
``_QueryStreamReporter`` into an asyncio queue, and yields each queued
event out the response. Cancellation: when the client disconnects, the
StreamingResponse generator's ``finally`` cancels the worker task — the
engine's ``CancelToken`` then short-circuits the next retrieval/LLM step.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel

from .. import api
from ..progress import CancelToken
from .errors import ApiError
from .ndjson import HEARTBEAT_INTERVAL, stream_response
from .runtime import ServerRuntime, get_runtime

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    q: str
    limit: int = 5


def _isoformat() -> str:
    return (
        datetime.fromtimestamp(time.time(), tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _serialise(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


class _QueryStreamReporter:
    """Adapter from ``ProgressReporter`` to an NDJSON event queue.

    ``api.query`` emits two ``partial`` kinds: ``retrieval_done`` and
    ``llm_token`` (Phase 4 streaming) plus a final ``llm_done`` we drop —
    the route reconstructs the full ``QueryResult`` from the engine's
    return value, so ``llm_done`` would be redundant on the wire. Other
    ``partial`` kinds pass through under a generic ``partial`` envelope
    so future engine events surface without route plumbing.
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
            "ts": _isoformat(),
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
                "ts": _isoformat(),
                "level": level,
                "message": message,
            }
        )

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "retrieval_done":
            await self._queue.put(
                {
                    "type": "retrieval_done",
                    "ts": _isoformat(),
                    "hits": payload.get("hits", []),
                }
            )
        elif kind == "llm_token":
            await self._queue.put(
                {
                    "type": "llm_token",
                    "ts": _isoformat(),
                    "delta": payload.get("delta", ""),
                }
            )
        elif kind == "llm_done":
            # Skipped — the engine's return value drives ``final``.
            return
        else:
            await self._queue.put(
                {
                    "type": "partial",
                    "ts": _isoformat(),
                    "kind": kind,
                    "payload": payload,
                }
            )

    def cancel_token(self) -> CancelToken:
        return self._token


def make_router(*, auth_dep: Any) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    @router.post("/query")
    async def post_query(
        request: Request,
        body: QueryRequest = Body(...),
    ) -> Any:
        rt: ServerRuntime = get_runtime(request.app)
        # Validate up front so a bad input fails the HTTP request rather
        # than landing as a ``final{failed}`` event mid-stream — clients
        # branching on status code stay simple.
        if not body.q.strip():
            raise ApiError(
                "query 'q' must not be empty",
                code="bad_request",
                status_code=400,
            )
        if body.limit < 1 or body.limit > 100:
            raise ApiError(
                f"limit must be in [1, 100], got {body.limit}",
                code="bad_request",
                status_code=400,
            )

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        reporter = _QueryStreamReporter(queue)

        async def _run() -> None:
            try:
                result = await api.query(
                    body.q, rt.root, limit=body.limit, reporter=reporter
                )
                await queue.put(
                    {
                        "type": "final",
                        "ts": _isoformat(),
                        "status": "succeeded",
                        "result": result.model_dump(mode="json"),
                    }
                )
            except asyncio.CancelledError:
                await queue.put(
                    {
                        "type": "final",
                        "ts": _isoformat(),
                        "status": "cancelled",
                    }
                )
                raise
            except ApiError as e:
                await queue.put(
                    {
                        "type": "final",
                        "ts": _isoformat(),
                        "status": "failed",
                        "error": {
                            "code": e.code,
                            "message": e.message,
                            **({"detail": e.detail} if e.detail else {}),
                        },
                    }
                )
            except Exception as e:
                logger.exception("query stream worker failed")
                await queue.put(
                    {
                        "type": "final",
                        "ts": _isoformat(),
                        "status": "failed",
                        "error": {
                            "code": "engine_error",
                            "message": f"{type(e).__name__}: {e}",
                        },
                    }
                )
            finally:
                # Sentinel — tells the response generator to stop pulling.
                await queue.put(None)

        worker = asyncio.create_task(_run())

        async def _gen() -> AsyncIterator[bytes]:
            yield _serialise(
                {
                    "type": "query_started",
                    "ts": _isoformat(),
                    "q": body.q,
                    "limit": body.limit,
                }
            )
            try:
                # Heartbeat-or-event loop: a non-streaming LLM
                # (``complete()`` fallback) or a slow first token would
                # otherwise leave the wire silent past the client's
                # 60s read timeout. Mirrors ``ndjson_lines``'s 15s
                # heartbeat cadence so reverse proxies don't close the
                # long-poll either.
                while True:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=HEARTBEAT_INTERVAL
                        )
                    except TimeoutError:
                        yield _serialise(
                            {"type": "heartbeat", "ts": _isoformat()}
                        )
                        continue
                    if event is None:
                        break
                    yield _serialise(event)
            finally:
                # Client disconnect: ask the engine to bail and wait.
                if not worker.done():
                    reporter.cancel_token().cancel()
                    worker.cancel()
                # Surface worker failures by awaiting; suppress both
                # CancelledError (disconnect path) and any worker
                # exception — worker errors already landed as
                # ``final{failed}`` in the stream, so re-raising here
                # would replace the streamed body with a 500.
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await worker

        return stream_response(_gen())

    return router


__all__ = ["make_router"]
