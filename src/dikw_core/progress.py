"""Progress reporting + cancellation primitives for engine long-running tasks.

The engine talks only to ``ProgressReporter``; concrete reporters live in
the server's task subsystem (the NDJSON event bus, landing in Phase 2) and
in tests (in-memory ``ListReporter``). Keeping this module dependency-free
preserves the engine's transport-agnostic contract — ``server/`` may import
``progress`` but ``progress`` never imports back.

Design notes
------------
* All sink methods are awaitable so a remote reporter can flush over the
  wire without blocking the engine.
* Long-running engine loops should call :pymeth:`CancelToken.raise_if_cancelled`
  between iterations so a controlling task can request a cooperative bail
  via :pymeth:`CancelToken.cancel`. Cancellation surfaces as the standard
  ``asyncio.CancelledError`` so existing ``try/finally`` / ``TaskGroup``
  unwinders behave normally.
* The ``NoopReporter`` is the default for in-process callers (CLI today,
  tests). The CLI still emits its own ``rich.progress`` bar via
  ``api._embedding_progress`` independently — the two channels coexist
  until Phase 5 collapses the CLI rendering into a remote-driven reporter.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable


class CancelToken:
    """Cooperative cancellation signal shared between a controller and an
    engine main loop.

    The token is independent of any specific ``asyncio.Task``: a server's
    TaskManager can flip it from outside the engine call, and the engine
    polls :pyattr:`raised` (or calls :pymeth:`raise_if_cancelled`) at safe
    points between work units. ``CancelledError`` is the chosen signal so
    standard async unwinding and ``finally`` blocks fire as usual.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def raised(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise asyncio.CancelledError("engine task cancelled by reporter")


@runtime_checkable
class ProgressReporter(Protocol):
    """Sink for engine progress, log lines, and partial intermediate results.

    Concrete reporters:
      * ``NoopReporter`` — discards everything; the in-process default.
      * ``ListReporter`` (tests) — appends events to a list for assertions.
      * ``TaskBusReporter`` (Phase 2) — fans events out to NDJSON subscribers
        via the per-task ``ProgressBus`` and persists them via ``TaskStore``.
    """

    async def progress(
        self,
        *,
        phase: str,
        current: int = 0,
        total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Report a phase advance. ``total == 0`` means "unknown total"."""

    async def log(self, level: str, message: str) -> None:
        """Emit a structured log line. ``level`` is a logging-style string
        (``"INFO"`` / ``"WARN"`` / ``"ERROR"``)."""

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        """Push an intermediate domain result (e.g. retrieval hits before
        the LLM step, per-batch ingest counters). ``kind`` is the
        consumer-visible discriminator; ``payload`` must be JSON-serialisable."""

    def cancel_token(self) -> CancelToken:
        """Return the token the engine polls between work units."""


class NoopReporter:
    """Default reporter when the engine is called without a server task
    wrapping it. Discards everything; never raises cancellation."""

    def __init__(self) -> None:
        self._token = CancelToken()

    async def progress(
        self,
        *,
        phase: str,
        current: int = 0,
        total: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def log(self, level: str, message: str) -> None:
        return None

    async def partial(self, kind: str, payload: dict[str, Any]) -> None:
        return None

    def cancel_token(self) -> CancelToken:
        return self._token


__all__ = ["CancelToken", "NoopReporter", "ProgressReporter"]
