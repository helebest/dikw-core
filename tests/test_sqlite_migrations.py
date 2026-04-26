"""SQLite-specific migration guards.

Lives outside the cross-backend ``test_storage_contract.py`` because these
tests pre-create a chunks table with the legacy ``start``/``"end"`` columns
to verify ``migrate()`` rejects pre-rename schemas — there's no analogous
"old shape" on Postgres or Filesystem.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dikw_core.storage.base import StorageError
from dikw_core.storage.sqlite import SQLiteStorage


async def test_legacy_chunks_offset_columns_rejected(tmp_path: Path) -> None:
    """``CREATE TABLE IF NOT EXISTS`` in 001_init.sql is a no-op against the
    existing table, so without the guard an upgraded user would silently keep
    the old shape and break every chunk SQL with a "no such column" error
    deep in the request path.
    """
    db_path = tmp_path / "legacy.sqlite"
    legacy_conn = sqlite3.connect(db_path)
    try:
        legacy_conn.executescript(
            """
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT,
                hash TEXT NOT NULL,
                mtime REAL,
                layer TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                seq INTEGER NOT NULL,
                start INTEGER NOT NULL,
                "end" INTEGER NOT NULL,
                text TEXT NOT NULL,
                UNIQUE (doc_id, seq)
            );
            """
        )
        legacy_conn.commit()
    finally:
        legacy_conn.close()

    storage = SQLiteStorage(db_path)
    await storage.connect()
    try:
        with pytest.raises(StorageError) as exc_info:
            await storage.migrate()
        msg = str(exc_info.value)
        assert "start" in msg and "end" in msg, (
            f"error must name the legacy columns: {msg!r}"
        )
        assert "rm" in msg or "delete" in msg.lower(), (
            f"error must instruct user to rebuild the SQLite file: {msg!r}"
        )
    finally:
        await storage.close()
