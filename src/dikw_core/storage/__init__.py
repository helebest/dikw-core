"""Storage factory: resolves a backend instance from a ``StorageConfig``."""

from __future__ import annotations

from pathlib import Path

from ..config import (
    FilesystemStorageConfig,
    PostgresStorageConfig,
    SQLiteStorageConfig,
    StorageConfig,
)
from ..info.tokenize import CjkTokenizer
from .base import NotSupported, Storage, StorageError
from .filesystem import FilesystemStorage
from .sqlite import SQLiteStorage


def build_storage(
    config: StorageConfig,
    *,
    root: str | Path | None = None,
    cjk_tokenizer: CjkTokenizer = "none",
) -> Storage:
    """Instantiate a Storage adapter. Paths are resolved relative to ``root`` if given.

    ``cjk_tokenizer`` controls how CJK text is segmented before being
    written to the backend's full-text index. SQLite and filesystem
    both honour it via ``preprocess_for_fts`` on ingest + query;
    Postgres ignores the knob because its tokenization happens inside
    ``to_tsvector('simple', …)`` and isn't preprocessed in Python.
    See ``RetrievalConfig.cjk_tokenizer`` for the wiki-level surface.
    """
    if isinstance(config, SQLiteStorageConfig):
        path = Path(config.path)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        return SQLiteStorage(path, cjk_tokenizer=cjk_tokenizer)
    if isinstance(config, FilesystemStorageConfig):
        fs_root = Path(config.root)
        if not fs_root.is_absolute() and root is not None:
            fs_root = Path(root) / fs_root
        return FilesystemStorage(fs_root, cjk_tokenizer=cjk_tokenizer)
    if isinstance(config, PostgresStorageConfig):
        try:
            from .postgres import PostgresStorage
        except ImportError as e:  # pragma: no cover - only without the extra
            raise StorageError(
                "Postgres adapter requires the `postgres` extra — "
                "install via `uv pip install dikw-core[postgres]`"
            ) from e
        return PostgresStorage(
            config.dsn, schema=config.schema_, pool_size=config.pool_size
        )
    raise StorageError(f"unknown storage backend: {config!r}")


__all__ = [
    "FilesystemStorage",
    "NotSupported",
    "SQLiteStorage",
    "Storage",
    "StorageError",
    "build_storage",
]
