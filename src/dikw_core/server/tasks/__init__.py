"""Async task subsystem: TaskManager, TaskStore, ProgressBus, event schemas.

Long-running engine operations (ingest, synth, distill, eval) are dispatched
through a TaskManager that persists task rows + an append-only event log to a
TaskStore (independent of wiki storage), and fans events out to subscribers
via an in-memory ProgressBus. NDJSON event streams support resume-by-seq.

Implemented in Phase 2 of the client/server migration.
"""
