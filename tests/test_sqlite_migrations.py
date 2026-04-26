"""SQLite-specific migration paths.

Lives outside the cross-backend ``test_storage_contract.py`` because these
tests pre-create a chunks table with the legacy ``start``/``"end"`` columns
to verify ``migrate()`` rewrites the shape in place — there's no analogous
"old shape" on Postgres or Filesystem.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dikw_core.storage.sqlite import SQLiteStorage


async def test_legacy_chunks_offset_columns_migrated(tmp_path: Path) -> None:
    """``CREATE TABLE IF NOT EXISTS`` in 001_init.sql is a no-op against the
    existing table, so the rename must happen via ``ALTER TABLE`` inside
    ``migrate()`` — otherwise an upgraded user would silently keep the old
    shape and break every chunk SQL with a "no such column" error.

    Pinned here (not a guard test): chunk rows must survive the rename, so
    embedding metadata and the ``wiki_log`` audit stream stay intact across
    the bump.
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
            INSERT INTO documents(doc_id, path, hash, layer)
                VALUES ('source:legacy.md', 'legacy.md', 'h1', 'source');
            INSERT INTO chunks(doc_id, seq, start, "end", text)
                VALUES ('source:legacy.md', 0, 0, 12, 'hello world.');
            """
        )
        legacy_conn.commit()
    finally:
        legacy_conn.close()

    storage = SQLiteStorage(db_path)
    await storage.connect()
    try:
        await storage.migrate()
        conn = storage._conn
        assert conn is not None
        cols = {r["name"] for r in conn.execute("PRAGMA table_info('chunks')")}
        assert "start_off" in cols and "end_off" in cols, (
            f"rename did not run, got {sorted(cols)}"
        )
        assert "start" not in cols and "end" not in cols, (
            f"legacy columns still present, got {sorted(cols)}"
        )
        row = conn.execute(
            "SELECT seq, start_off, end_off, text FROM chunks "
            "WHERE doc_id = 'source:legacy.md'"
        ).fetchone()
        assert row is not None
        assert (row["seq"], row["start_off"], row["end_off"], row["text"]) == (
            0,
            0,
            12,
            "hello world.",
        )
    finally:
        await storage.close()
