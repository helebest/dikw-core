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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info('{table}')")}


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
        cols = _table_columns(conn, "chunks")
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


async def test_legacy_assets_columns_dropped_on_migrate(tmp_path: Path) -> None:
    """An older DB has assets(width, height, caption, caption_model) — the
    media_meta JSON refactor must drop those four columns, add media_meta,
    and **backfill** any captured width/height into media_meta JSON so
    upgraded users don't silently lose dimensions. caption/caption_model
    were never populated in production, so they're dropped without
    backfill."""
    db_path = tmp_path / "legacy_assets.sqlite"
    legacy_conn = sqlite3.connect(db_path)
    try:
        legacy_conn.executescript(
            """
            CREATE TABLE assets (
                asset_id       TEXT PRIMARY KEY,
                hash           TEXT UNIQUE NOT NULL,
                kind           TEXT NOT NULL,
                mime           TEXT NOT NULL,
                stored_path    TEXT NOT NULL,
                original_paths TEXT NOT NULL,
                bytes          INTEGER NOT NULL,
                width          INTEGER,
                height         INTEGER,
                caption        TEXT,
                caption_model  TEXT,
                created_ts     REAL NOT NULL
            );
            INSERT INTO assets(asset_id, hash, kind, mime, stored_path,
                               original_paths, bytes, width, height,
                               caption, caption_model, created_ts)
                VALUES
                    -- Both dims set — full ImageMediaMeta backfill.
                    ('aaa', 'aaa', 'image', 'image/png',
                     'assets/aa/x.png', '["x.png"]', 42, 640, 480,
                     NULL, NULL, 1700000000.0),
                    -- Only width set — partial backfill.
                    ('bbb', 'bbb', 'image', 'image/png',
                     'assets/bb/y.png', '["y.png"]', 1, 120, NULL,
                     NULL, NULL, 1700000001.0),
                    -- Both dims NULL (e.g. SVG legacy row) — stays NULL.
                    ('ccc', 'ccc', 'image', 'image/svg+xml',
                     'assets/cc/z.svg', '["z.svg"]', 1, NULL, NULL,
                     NULL, NULL, 1700000002.0);
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
        cols = _table_columns(conn, "assets")
        for legacy in ("width", "height", "caption", "caption_model"):
            assert legacy not in cols, f"{legacy!r} survived migration: {sorted(cols)}"
        assert "media_meta" in cols, f"media_meta missing: {sorted(cols)}"

        rows = {
            r["asset_id"]: r
            for r in conn.execute(
                "SELECT asset_id, hash, mime, stored_path, original_paths, "
                "bytes, media_meta, created_ts FROM assets ORDER BY asset_id"
            )
        }
        assert set(rows) == {"aaa", "bbb", "ccc"}, "asset rows lost during migration"

        # Full payload survives + dims backfilled into media_meta JSON.
        full = rows["aaa"]
        assert (
            full["hash"],
            full["mime"],
            full["stored_path"],
            full["original_paths"],
            full["bytes"],
            full["created_ts"],
        ) == ("aaa", "image/png", "assets/aa/x.png", '["x.png"]', 42, 1700000000.0)
        assert full["media_meta"] == '{"kind":"image","width":640,"height":480}'

        # Partial dims preserve the missing side as null in JSON.
        partial = rows["bbb"]
        assert partial["media_meta"] == '{"kind":"image","width":120,"height":null}'

        # Rows that had no dims to begin with stay NULL — no synthetic blob.
        no_dims = rows["ccc"]
        assert no_dims["media_meta"] is None
    finally:
        await storage.close()


async def test_half_migrated_assets_table_recovers(tmp_path: Path) -> None:
    """Crash-safety: if a prior ``migrate()`` got partway through (e.g.
    ``width`` dropped, ``height`` still present, ``media_meta`` not yet
    added), the next run must finish cleanly instead of bricking on
    ``no such column: width`` from a hard-coded WHERE clause."""
    db_path = tmp_path / "half_migrated.sqlite"
    legacy_conn = sqlite3.connect(db_path)
    try:
        legacy_conn.executescript(
            """
            CREATE TABLE assets (
                asset_id       TEXT PRIMARY KEY,
                hash           TEXT UNIQUE NOT NULL,
                kind           TEXT NOT NULL,
                mime           TEXT NOT NULL,
                stored_path    TEXT NOT NULL,
                original_paths TEXT NOT NULL,
                bytes          INTEGER NOT NULL,
                height         INTEGER,
                created_ts     REAL NOT NULL
            );
            INSERT INTO assets(asset_id, hash, kind, mime, stored_path,
                               original_paths, bytes, height, created_ts)
                VALUES ('aaa', 'aaa', 'image', 'image/png',
                        'assets/aa/x.png', '["x.png"]', 42, 480, 1700000000.0);
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
        cols = _table_columns(conn, "assets")
        assert "height" not in cols, f"height not dropped: {sorted(cols)}"
        assert "media_meta" in cols
        row = conn.execute(
            "SELECT media_meta FROM assets WHERE asset_id = 'aaa'"
        ).fetchone()
        assert row is not None
        assert row["media_meta"] == '{"kind":"image","width":null,"height":480}'
    finally:
        await storage.close()


async def test_fresh_install_assets_schema(tmp_path: Path) -> None:
    """A fresh DB skips the legacy migration path entirely and lands on the
    new column layout directly from 002_assets.sql."""
    storage = SQLiteStorage(tmp_path / "fresh.sqlite")
    await storage.connect()
    try:
        await storage.migrate()
        conn = storage._conn
        assert conn is not None
        cols = _table_columns(conn, "assets")
        for legacy in ("width", "height", "caption", "caption_model"):
            assert legacy not in cols, f"fresh DB grew {legacy!r}: {sorted(cols)}"
        assert "media_meta" in cols
        # Spot-check that the rest of the table didn't change shape.
        for required in (
            "asset_id",
            "hash",
            "kind",
            "mime",
            "stored_path",
            "original_paths",
            "bytes",
            "created_ts",
        ):
            assert required in cols, f"missing {required!r}: {sorted(cols)}"
    finally:
        await storage.close()


async def test_migrate_assets_idempotent(tmp_path: Path) -> None:
    """Running migrate() twice must not throw "no such column" once the
    legacy columns have already been dropped."""
    storage = SQLiteStorage(tmp_path / "idem.sqlite")
    await storage.connect()
    try:
        await storage.migrate()
        await storage.migrate()
        conn = storage._conn
        assert conn is not None
        cols = _table_columns(conn, "assets")
        assert "media_meta" in cols
        for legacy in ("width", "height", "caption", "caption_model"):
            assert legacy not in cols
    finally:
        await storage.close()
