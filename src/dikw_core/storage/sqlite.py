"""SQLite + sqlite-vec + FTS5 storage adapter (MVP).

Uses stdlib ``sqlite3`` synchronously under the hood and exposes an ``async``
surface on the Storage Protocol via ``asyncio.to_thread``. This is intentional:
SQLite's pattern is one-writer-at-a-time; wrapping it in a worker thread lets
the rest of the engine stay ``async`` without pulling in aiosqlite.
"""

from __future__ import annotations

import asyncio
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import sqlite_vec

from ..schemas import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    FTSHit,
    Layer,
    LinkRecord,
    LinkType,
    StorageCounts,
    VecHit,
    WikiLogEntry,
    WisdomEvidence,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
)
from .base import NotSupported, StorageError

MIGRATIONS_PACKAGE = "dikw_core.storage.migrations.sqlite"


def _serialize_vec(values: list[float]) -> bytes:
    """Pack a float32 vector for sqlite-vec."""
    return struct.pack(f"{len(values)}f", *values)


class SQLiteStorage:
    """SQLite-backed Storage implementation.

    The embedding dimension is set on first ``upsert_embeddings`` call (or can
    be pre-set explicitly) so that the ``chunks_vec`` virtual table can be
    created lazily. This matches how ``qmd`` and ``mineru-doc-explorer`` handle
    variable embedding sizes across providers.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._embedding_dim: int | None = None

    # ---- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        def _open() -> sqlite3.Connection:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return conn

        self._conn = await asyncio.to_thread(_open)

    async def close(self) -> None:
        conn = self._conn
        if conn is None:
            return
        self._conn = None
        await asyncio.to_thread(conn.close)

    async def migrate(self) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta_kv (key TEXT PRIMARY KEY, value TEXT)"
                )
                for name in sorted(
                    r.name
                    for r in resources.files(MIGRATIONS_PACKAGE).iterdir()
                    if r.is_file() and r.name.endswith(".sql")
                ):
                    sql = (
                        resources.files(MIGRATIONS_PACKAGE)
                        .joinpath(name)
                        .read_text(encoding="utf-8")
                    )
                    conn.executescript(sql)
            # Restore embedding dimension from a previous run, if any.
            meta = conn.execute(
                "SELECT value FROM meta_kv WHERE key = 'embedding_dim'"
            ).fetchone()
            if meta is not None:
                self._embedding_dim = int(meta[0])

        await asyncio.to_thread(_run)

    # ---- D layer ---------------------------------------------------------

    async def put_content(self, hash_: str, body: str) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO content(hash, body) VALUES (?, ?)",
                    (hash_, body),
                )

        await asyncio.to_thread(_run)

    async def upsert_document(self, doc: DocumentRecord) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO documents(doc_id, path, title, hash, mtime, layer, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        path = excluded.path,
                        title = excluded.title,
                        hash = excluded.hash,
                        mtime = excluded.mtime,
                        layer = excluded.layer,
                        active = excluded.active
                    """,
                    (
                        doc.doc_id,
                        doc.path,
                        doc.title,
                        doc.hash,
                        doc.mtime,
                        doc.layer.value,
                        int(doc.active),
                    ),
                )

        await asyncio.to_thread(_run)

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        def _run() -> DocumentRecord | None:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return _row_to_document(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def list_documents(
        self,
        *,
        layer: Layer | None = None,
        active: bool | None = True,
        since_ts: float | None = None,
    ) -> Iterable[DocumentRecord]:
        def _run() -> list[DocumentRecord]:
            conn = self._require_conn()
            q = "SELECT * FROM documents WHERE 1=1"
            params: list[Any] = []
            if layer is not None:
                q += " AND layer = ?"
                params.append(layer.value)
            if active is not None:
                q += " AND active = ?"
                params.append(int(active))
            if since_ts is not None:
                q += " AND mtime >= ?"
                params.append(since_ts)
            rows = conn.execute(q, params).fetchall()
            return [_row_to_document(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def deactivate_document(self, doc_id: str) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "UPDATE documents SET active = 0 WHERE doc_id = ?", (doc_id,)
                )

        await asyncio.to_thread(_run)

    # ---- I layer ---------------------------------------------------------

    async def replace_chunks(
        self, doc_id: str, chunks: Sequence[ChunkRecord]
    ) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                for c in chunks:
                    conn.execute(
                        'INSERT INTO chunks(doc_id, seq, start, "end", text) '
                        "VALUES (?, ?, ?, ?, ?)",
                        (c.doc_id, c.seq, c.start, c.end, c.text),
                    )
                # refresh FTS rows for this document: delete stale, add new
                conn.execute(
                    "DELETE FROM documents_fts WHERE path = "
                    "(SELECT path FROM documents WHERE doc_id = ?)",
                    (doc_id,),
                )
                doc_row = conn.execute(
                    "SELECT path, title, layer FROM documents WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
                if doc_row is not None:
                    for c in chunks:
                        conn.execute(
                            "INSERT INTO documents_fts(path, title, body, layer) "
                            "VALUES (?, ?, ?, ?)",
                            (doc_row["path"], doc_row["title"], c.text, doc_row["layer"]),
                        )

        await asyncio.to_thread(_run)

    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        if not rows:
            return

        def _run() -> None:
            conn = self._require_conn()
            dim = len(rows[0].embedding)
            self._ensure_vec_table(conn, dim)
            with conn:
                for r in rows:
                    if len(r.embedding) != dim:
                        raise StorageError(
                            f"embedding dim mismatch: expected {dim}, got {len(r.embedding)}"
                        )
                    conn.execute(
                        "INSERT OR REPLACE INTO embed_meta(chunk_id, model) VALUES (?, ?)",
                        (r.chunk_id, r.model),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                        (r.chunk_id, _serialize_vec(r.embedding)),
                    )

        await asyncio.to_thread(_run)

    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None:
        def _run() -> ChunkRecord | None:
            conn = self._require_conn()
            row = conn.execute(
                'SELECT chunk_id, doc_id, seq, start, "end", text '
                "FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                return None
            return ChunkRecord(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                seq=row["seq"],
                start=row["start"],
                end=row["end"],
                text=row["text"],
            )

        return await asyncio.to_thread(_run)

    async def fts_search(
        self, q: str, *, limit: int = 20, layer: Layer | None = None
    ) -> list[FTSHit]:
        def _run() -> list[FTSHit]:
            conn = self._require_conn()
            sql = (
                "SELECT path, snippet(documents_fts, 2, '<mark>', '</mark>', '…', 10) AS snip, "
                "bm25(documents_fts) AS score "
                "FROM documents_fts WHERE documents_fts MATCH ?"
            )
            params: list[Any] = [q]
            if layer is not None:
                sql += " AND layer = ?"
                params.append(layer.value)
            sql += " ORDER BY score LIMIT ?"
            params.append(limit)
            hits: list[FTSHit] = []
            for row in conn.execute(sql, params).fetchall():
                doc_row = conn.execute(
                    "SELECT doc_id FROM documents WHERE path = ?", (row["path"],)
                ).fetchone()
                if doc_row is None:
                    continue
                # bm25 returns lower-is-better; invert so higher = better
                hits.append(
                    FTSHit(doc_id=doc_row["doc_id"], score=-float(row["score"]), snippet=row["snip"])
                )
            return hits

        return await asyncio.to_thread(_run)

    async def vec_search(
        self, embedding: list[float], *, limit: int = 20, layer: Layer | None = None
    ) -> list[VecHit]:
        def _run() -> list[VecHit]:
            conn = self._require_conn()
            if self._embedding_dim is None:
                raise NotSupported("no embeddings indexed yet")
            if len(embedding) != self._embedding_dim:
                raise StorageError(
                    f"query embedding dim {len(embedding)} != index dim {self._embedding_dim}"
                )
            sql = (
                "SELECT cv.rowid AS chunk_id, c.doc_id AS doc_id, "
                "vec_distance_cosine(cv.embedding, ?) AS dist "
                "FROM chunks_vec cv JOIN chunks c ON c.chunk_id = cv.rowid "
                "JOIN documents d ON d.doc_id = c.doc_id"
            )
            params: list[Any] = [_serialize_vec(embedding)]
            if layer is not None:
                sql += " WHERE d.layer = ?"
                params.append(layer.value)
            sql += " ORDER BY dist LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [
                VecHit(doc_id=r["doc_id"], chunk_id=int(r["chunk_id"]), distance=float(r["dist"]))
                for r in rows
            ]

        return await asyncio.to_thread(_run)

    # ---- K layer ---------------------------------------------------------

    async def upsert_link(self, link: LinkRecord) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO links(src_doc_id, dst_path, link_type, anchor, line) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (link.src_doc_id, link.dst_path, link.link_type.value, link.anchor, link.line),
                )

        await asyncio.to_thread(_run)

    async def links_from(self, src_doc_id: str) -> list[LinkRecord]:
        def _run() -> list[LinkRecord]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT * FROM links WHERE src_doc_id = ?", (src_doc_id,)
            ).fetchall()
            return [_row_to_link(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def links_to(self, dst_path: str) -> list[LinkRecord]:
        def _run() -> list[LinkRecord]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT * FROM links WHERE dst_path = ?", (dst_path,)
            ).fetchall()
            return [_row_to_link(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def append_wiki_log(self, entry: WikiLogEntry) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "INSERT INTO wiki_log(ts, action, src, dst, note) VALUES (?, ?, ?, ?, ?)",
                    (entry.ts, entry.action, entry.src, entry.dst, entry.note),
                )

        await asyncio.to_thread(_run)

    # ---- W layer ---------------------------------------------------------

    async def put_wisdom(
        self, item: WisdomItem, evidence: Sequence[WisdomEvidence]
    ) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO wisdom_items(
                        item_id, kind, status, path, title, body,
                        confidence, created_ts, approved_ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        kind = excluded.kind,
                        status = excluded.status,
                        path = excluded.path,
                        title = excluded.title,
                        body = excluded.body,
                        confidence = excluded.confidence,
                        approved_ts = excluded.approved_ts
                    """,
                    (
                        item.item_id,
                        item.kind.value,
                        item.status.value,
                        item.path,
                        item.title,
                        item.body,
                        item.confidence,
                        item.created_ts,
                        item.approved_ts,
                    ),
                )
                conn.execute("DELETE FROM wisdom_evidence WHERE item_id = ?", (item.item_id,))
                for e in evidence:
                    conn.execute(
                        "INSERT INTO wisdom_evidence(item_id, doc_id, excerpt, line) "
                        "VALUES (?, ?, ?, ?)",
                        (item.item_id, e.doc_id, e.excerpt, e.line),
                    )

        await asyncio.to_thread(_run)

    async def list_wisdom(
        self,
        *,
        status: WisdomStatus | None = None,
        kind: WisdomKind | None = None,
    ) -> list[WisdomItem]:
        def _run() -> list[WisdomItem]:
            conn = self._require_conn()
            sql = "SELECT * FROM wisdom_items WHERE 1=1"
            params: list[Any] = []
            if status is not None:
                sql += " AND status = ?"
                params.append(status.value)
            if kind is not None:
                sql += " AND kind = ?"
                params.append(kind.value)
            sql += " ORDER BY created_ts DESC"
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_wisdom(r) for r in rows]

        return await asyncio.to_thread(_run)

    async def set_wisdom_status(self, item_id: str, status: WisdomStatus) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "UPDATE wisdom_items SET status = ? WHERE item_id = ?",
                    (status.value, item_id),
                )

        await asyncio.to_thread(_run)

    # ---- diagnostics -----------------------------------------------------

    async def counts(self) -> StorageCounts:
        def _run() -> StorageCounts:
            conn = self._require_conn()
            by_layer: dict[str, int] = {}
            for row in conn.execute(
                "SELECT layer, COUNT(*) AS n FROM documents WHERE active = 1 GROUP BY layer"
            ).fetchall():
                by_layer[row["layer"]] = int(row["n"])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            embeddings = int(conn.execute("SELECT COUNT(*) FROM embed_meta").fetchone()[0])
            links = int(conn.execute("SELECT COUNT(*) FROM links").fetchone()[0])
            by_status: dict[str, int] = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM wisdom_items GROUP BY status"
            ).fetchall():
                by_status[row["status"]] = int(row["n"])
            last_log = conn.execute("SELECT MAX(ts) AS ts FROM wiki_log").fetchone()
            return StorageCounts(
                documents_by_layer=by_layer,
                chunks=chunks,
                embeddings=embeddings,
                links=links,
                wisdom_by_status=by_status,
                last_wiki_log_ts=float(last_log["ts"]) if last_log and last_log["ts"] else None,
            )

        return await asyncio.to_thread(_run)

    # ---- internals -------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("SQLiteStorage is not connected; call `connect()` first")
        return self._conn

    def _ensure_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        if self._embedding_dim is None:
            # Persist the dim so subsequent startups can restore it without an insert.
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{dim}])"
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta_kv(key, value) VALUES ('embedding_dim', ?)",
                (str(dim),),
            )
            self._embedding_dim = dim
            return
        if self._embedding_dim != dim:
            raise StorageError(
                f"embedding dim mismatch: index uses {self._embedding_dim}, got {dim}; "
                "rebuild the vector index to change dimensions"
            )


# ---- row → DTO helpers ---------------------------------------------------


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        path=row["path"],
        title=row["title"],
        hash=row["hash"],
        mtime=float(row["mtime"] or 0.0),
        layer=Layer(row["layer"]),
        active=bool(row["active"]),
    )


def _row_to_link(row: sqlite3.Row) -> LinkRecord:
    return LinkRecord(
        src_doc_id=row["src_doc_id"],
        dst_path=row["dst_path"],
        link_type=LinkType(row["link_type"]),
        anchor=row["anchor"],
        line=int(row["line"]),
    )


def _row_to_wisdom(row: sqlite3.Row) -> WisdomItem:
    return WisdomItem(
        item_id=row["item_id"],
        kind=WisdomKind(row["kind"]),
        status=WisdomStatus(row["status"]),
        path=row["path"],
        title=row["title"],
        body=row["body"],
        confidence=float(row["confidence"]),
        created_ts=float(row["created_ts"]),
        approved_ts=float(row["approved_ts"]) if row["approved_ts"] is not None else None,
    )
