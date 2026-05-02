"""Async task subsystem: TaskManager, TaskStore, ProgressBus, event schemas.

Long-running engine operations (ingest, synth, distill, eval) are dispatched
through a TaskManager that persists task rows + an append-only event log to a
TaskStore (independent of wiki storage), and fans events out to subscribers
via an in-memory ProgressBus. NDJSON event streams support resume-by-seq.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...config import (
    DikwConfig,
    FilesystemStorageConfig,
    PostgresStorageConfig,
    SQLiteStorageConfig,
)
from .bus import ProgressBus
from .events import (
    ErrorEvent,
    FinalEvent,
    HeartbeatEvent,
    LogEvent,
    PartialEvent,
    ProgressEvent,
    TaskStartedEvent,
)
from .manager import TaskBusReporter, TaskManager, TaskRunner
from .store import (
    TERMINAL_STATUSES,
    TaskCounters,
    TaskNotFound,
    TaskRow,
    TaskStatus,
    TaskStore,
    TaskStoreError,
)
from .store_sqlite import SqliteTaskStore

_TASKS_DSN_ENV = "DIKW_SERVER_TASKS_DSN"
_DEFAULT_TASKS_DB_PATH = ".dikw/server-tasks.db"


def build_task_store(cfg: DikwConfig, root: Path) -> TaskStore:
    """Pick a TaskStore matching the wiki's storage backend.

    Per the migration plan's Phase-0 decision:
      * SQLite or Filesystem wiki backends → ``<root>/.dikw/server-tasks.db``
        (a SQLite file regardless of which the wiki uses).
      * Postgres wiki backend → a separate Postgres database whose DSN
        comes from ``DIKW_SERVER_TASKS_DSN`` (intentionally distinct from
        the wiki DSN so a wiki schema wipe doesn't lose task history).
    """
    storage_cfg = cfg.storage
    if isinstance(storage_cfg, PostgresStorageConfig):
        dsn = os.environ.get(_TASKS_DSN_ENV)
        if not dsn:
            raise TaskStoreError(
                f"wiki storage is postgres but {_TASKS_DSN_ENV} is not set; "
                "the server task store must live in a separate database. "
                "Export DIKW_SERVER_TASKS_DSN pointing at a Postgres DB you "
                "control."
            )
        # Lazy import keeps `import dikw_core.server.tasks` working without
        # the [postgres] extra installed (e.g. when only sqlite is configured).
        from .store_postgres import PostgresTaskStore

        return PostgresTaskStore(dsn=dsn)

    if isinstance(storage_cfg, SQLiteStorageConfig | FilesystemStorageConfig):
        return SqliteTaskStore(path=root / _DEFAULT_TASKS_DB_PATH)

    raise TaskStoreError(
        f"unrecognised storage backend: {type(storage_cfg).__name__}"
    )


__all__ = [
    "TERMINAL_STATUSES",
    "ErrorEvent",
    "FinalEvent",
    "HeartbeatEvent",
    "LogEvent",
    "PartialEvent",
    "ProgressBus",
    "ProgressEvent",
    "SqliteTaskStore",
    "TaskBusReporter",
    "TaskCounters",
    "TaskManager",
    "TaskNotFound",
    "TaskRow",
    "TaskRunner",
    "TaskStartedEvent",
    "TaskStatus",
    "TaskStore",
    "TaskStoreError",
    "build_task_store",
]
