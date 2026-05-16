"""Client-side long-poll loop that drives every ``--wait`` UX.

Used by:

* ``--wait`` on ``dikw client {ingest, synth, distill, eval, lint
  propose, lint apply}`` — submit + block + render + exit-code map.
* ``dikw client tasks wait <id>`` — same loop, applied to an existing
  task_id.

Replaces the NDJSON ``stream_ndjson(/events)`` loop deleted in PR4.
The wire contract is now the cursor JSON ``EventsPage`` (one HTTP call
per page; server holds for up to ``poll_wait`` seconds), so this
module owns the cursor advance, terminal short-circuit, and total-
budget logic that used to live inline at each op command.

Design notes:

* No retries on transport errors — :class:`ClientError` bubbles up to
  the CLI layer, which renders a clean message + exits.
* The renderer is optional and **stateless from this module's point of
  view** — we just call ``renderer.render(event)`` for each event,
  including ``final``. The caller controls the live widget lifecycle.
* ``total_timeout`` is a *client-side* budget; expiry raises
  :class:`TimeoutError` and the CLI maps it to exit 124. The task
  itself is *not* auto-cancelled — agents that wanted that chain a
  ``dikw tasks cancel`` themselves.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .transport import Transport

# Default per-HTTP-call hold time. 30s keeps round-trips within the
# server's 60s hard cap while amortising request overhead. Tests pass
# smaller values to keep wall-clock short.
_DEFAULT_POLL_WAIT = 30
# Default page size. Big enough that a typical task's whole tape fits
# in one page (~hundreds of events), small enough that the JSON
# payload stays well under any reasonable proxy buffer.
_DEFAULT_PAGE_LIMIT = 200
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@runtime_checkable
class _EventRenderer(Protocol):
    """Minimal contract for the optional progress renderer.

    Decoupled from :class:`TaskProgressRenderer` so this module stays
    importable without rich — the test suite drives it through a
    one-method fake.
    """

    def render(self, event: Mapping[str, Any]) -> None: ...


async def follow_to_terminal(
    transport: Transport,
    task_id: str,
    *,
    renderer: _EventRenderer | None = None,
    poll_wait: int = _DEFAULT_POLL_WAIT,
    page_limit: int = _DEFAULT_PAGE_LIMIT,
    total_timeout: float | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Page through ``/v1/tasks/{task_id}/events`` until terminal.

    Returns ``(status, payload)`` where ``status`` is one of
    ``succeeded`` / ``failed`` / ``cancelled`` (matching the server's
    ``TaskStatus`` enum string values) and ``payload`` is the final
    event's ``result`` dict on success, or the ``error`` dict on
    failure, or ``None`` if the server emitted neither.

    Raises :class:`TimeoutError` when ``total_timeout`` elapses before
    the task reaches a terminal state. The task is *not* auto-cancelled
    on timeout — that's the caller's choice.
    """
    deadline = time.monotonic() + total_timeout if total_timeout else None
    cursor = 0
    final_status: str | None = None
    final_payload: dict[str, Any] | None = None
    # Consecutive pages where the server reported a terminal
    # ``task_status`` but the tape had no ``final`` event. The manager's
    # ``finally`` block makes this window narrow in the happy path, but a
    # server crash between ``update_status(terminal)`` and
    # ``emit_raw(final)``, or any ``append_event`` failure after the row
    # update, would leave the tape permanently missing its final and the
    # loop would otherwise spin forever. After this many quiet
    # confirmations we fall back to the row's terminal status so the
    # caller still gets a deterministic exit instead of hanging.
    terminal_without_final_polls = 0
    _MAX_TERMINAL_WITHOUT_FINAL_POLLS = 3

    while True:
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise TimeoutError(
                f"task {task_id} did not finish within total_timeout"
            )
        # Cap the per-call wait so a short total_timeout doesn't
        # over-shoot. Server caps independently at 60s.
        call_wait = poll_wait
        if remaining is not None:
            call_wait = max(0, min(poll_wait, int(remaining)))

        page = await transport.get_task_events_page(
            task_id,
            from_seq=cursor,
            limit=page_limit,
            wait=call_wait,
        )
        events = page.get("events") or []
        for event in events:
            if renderer is not None:
                renderer.render(event)
            if event.get("type") == "final":
                final_status = str(event.get("status") or "failed")
                result = event.get("result")
                error = event.get("error")
                if isinstance(result, dict):
                    final_payload = result
                elif isinstance(error, dict):
                    final_payload = error
        # Final-event arrival is the primary exit signal —
        # ``page.task_status`` flips to terminal *before* the final
        # event lands on the tape (manager writes the row, then emits
        # ``final``), so exiting on terminal-without-final naively
        # would silently swallow the result payload. The server's
        # belt-and-braces ``_notify_task`` in the runner's ``finally``
        # block guarantees the next paged read picks up the final in
        # the happy path. The fallback below covers crash windows
        # where the final never lands.
        if final_status is not None:
            return final_status, final_payload
        cursor = int(page.get("next_from_seq") or cursor)
        task_status = str(page.get("task_status") or "")
        if (
            task_status in _TERMINAL_STATUSES
            and not page.get("has_more")
        ):
            terminal_without_final_polls += 1
            if terminal_without_final_polls >= _MAX_TERMINAL_WITHOUT_FINAL_POLLS:
                # Tape is permanently missing the final event (server
                # crash, append_event failure). Surface the row's
                # terminal status with no payload — caller can fetch
                # ``/v1/tasks/{id}/result`` if it wants the recovered
                # result/error.
                return task_status, None
        else:
            terminal_without_final_polls = 0


__all__ = ["follow_to_terminal"]
