"""HTTP server for dikw-core (FastAPI + NDJSON).

Wraps the in-process engine (``dikw_core.api``) behind a FastAPI app, exposing
sync RPC endpoints, async task endpoints (NDJSON event streams), and a
multipart sources upload endpoint. ``server/*`` may import the engine; the
reverse direction is forbidden so the engine remains transport-agnostic.

Modules land in subsequent migration phases (see plan
`dikw-core-client-server-eventual-clarke`):

  * Phase 2 — ``app``, ``runtime``, ``auth``, ``routes_sync``, ``ndjson``,
    ``errors``, and the ``tasks/`` subpackage (manager, store, bus, events).
  * Phase 3 — ``routes_upload`` and the ingest task wiring in ``routes_tasks``.
  * Phase 4 — ``routes_query`` (NDJSON streaming) plus synth/distill/eval ops.
"""
