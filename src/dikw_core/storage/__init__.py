"""Storage factory: resolves a backend instance from a ``StorageConfig``."""

from __future__ import annotations

from pathlib import Path

from ..config import (
    FilesystemStorageConfig,
    PostgresStorageConfig,
    SQLiteStorageConfig,
    StorageConfig,
)
from .base import NotSupported, Storage, StorageError
from .sqlite import SQLiteStorage


def build_storage(config: StorageConfig, *, root: str | Path | None = None) -> Storage:
    """Instantiate a Storage adapter. Paths are resolved relative to ``root`` if given."""
    if isinstance(config, SQLiteStorageConfig):
        path = Path(config.path)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        return SQLiteStorage(path)
    if isinstance(config, PostgresStorageConfig):
        raise NotSupported("Postgres storage adapter ships in Phase 5")
    if isinstance(config, FilesystemStorageConfig):
        raise NotSupported("Filesystem storage adapter ships in Phase 5")
    raise StorageError(f"unknown storage backend: {config!r}")


__all__ = [
    "NotSupported",
    "SQLiteStorage",
    "Storage",
    "StorageError",
    "build_storage",
]
