"""SQLite + sqlite-vec + FTS5 storage adapter (MVP).

Uses stdlib ``sqlite3`` synchronously under the hood and exposes an ``async``
surface on the Storage Protocol via ``asyncio.to_thread``. This is intentional:
SQLite's pattern is one-writer-at-a-time; wrapping it in a worker thread lets
the rest of the engine stay ``async`` without pulling in aiosqlite.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
import time
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import sqlite_vec

from ..info.tokenize import CjkTokenizer, preprocess_for_fts
from ..schemas import (
    AssetEmbeddingRow,
    AssetKind,
    AssetRecord,
    AssetVecHit,
    ChunkAssetRef,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    EmbeddingVersion,
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

    def __init__(
        self,
        path: str | Path,
        *,
        cjk_tokenizer: CjkTokenizer = "none",
    ) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._embedding_dim: int | None = None
        # Applied to title + chunk.text before INSERT into documents_fts.
        # Must match what `_sanitize_fts` does on query text at read time
        # — flipping this post-ingest mismatches index vs query and
        # silently drops hits. `RetrievalConfig.cjk_tokenizer` is marked
        # "locked at first ingest" for exactly this reason.
        self._cjk_tokenizer: CjkTokenizer = cjk_tokenizer

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
    ) -> list[int]:
        def _run() -> list[int]:
            conn = self._require_conn()
            ids: list[int] = []
            with conn:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                for c in chunks:
                    cur = conn.execute(
                        'INSERT INTO chunks(doc_id, seq, start, "end", text) '
                        "VALUES (?, ?, ?, ?, ?)",
                        (c.doc_id, c.seq, c.start, c.end, c.text),
                    )
                    ids.append(int(cur.lastrowid or 0))
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
                    # Use explicit rowid = chunk_id so FTS rowids stay
                    # aligned with chunks.chunk_id — that lets fts_search
                    # return chunk_id and the hybrid searcher attach
                    # chunk_asset_refs to the right Hit even when only
                    # the FTS leg fired.
                    #
                    # Title/body also get preprocessed (via jieba when
                    # cjk_tokenizer="jieba") so BM25's title and body
                    # columns tokenize consistently with the query-side
                    # _sanitize_fts path. Both columns feed the BM25
                    # score FTS5 returns.
                    title = preprocess_for_fts(
                        doc_row["title"] or "", tokenizer=self._cjk_tokenizer
                    )
                    for cid, c in zip(ids, chunks, strict=True):
                        body = preprocess_for_fts(
                            c.text, tokenizer=self._cjk_tokenizer
                        )
                        conn.execute(
                            "INSERT INTO documents_fts(rowid, path, title, body, layer) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (
                                cid,
                                doc_row["path"],
                                title,
                                body,
                                doc_row["layer"],
                            ),
                        )
            return ids

        return await asyncio.to_thread(_run)

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
                "SELECT documents_fts.rowid AS chunk_id, path, "
                "snippet(documents_fts, 2, '<mark>', '</mark>', '…', 10) AS snip, "
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
                    FTSHit(
                        doc_id=doc_row["doc_id"],
                        chunk_id=int(row["chunk_id"]),
                        score=-float(row["score"]),
                        snippet=row["snip"],
                    )
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

    async def list_wiki_log(
        self, *, since_ts: float | None = None, limit: int | None = None
    ) -> list[WikiLogEntry]:
        def _run() -> list[WikiLogEntry]:
            conn = self._require_conn()
            sql = "SELECT ts, action, src, dst, note FROM wiki_log"
            params: list[Any] = []
            if since_ts is not None:
                sql += " WHERE ts >= ?"
                params.append(since_ts)
            sql += " ORDER BY ts ASC"
            if limit is not None:
                sql += " LIMIT ?"
                params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            return [
                WikiLogEntry(
                    ts=float(r["ts"]),
                    action=r["action"],
                    src=r["src"],
                    dst=r["dst"],
                    note=r["note"],
                )
                for r in rows
            ]

        return await asyncio.to_thread(_run)

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

    async def set_wisdom_status(
        self,
        item_id: str,
        status: WisdomStatus,
        *,
        approved_ts: float | None = None,
    ) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                if approved_ts is None:
                    conn.execute(
                        "UPDATE wisdom_items SET status = ? WHERE item_id = ?",
                        (status.value, item_id),
                    )
                else:
                    conn.execute(
                        "UPDATE wisdom_items SET status = ?, approved_ts = ? "
                        "WHERE item_id = ?",
                        (status.value, approved_ts, item_id),
                    )

        await asyncio.to_thread(_run)

    async def get_wisdom(self, item_id: str) -> WisdomItem | None:
        def _run() -> WisdomItem | None:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM wisdom_items WHERE item_id = ?", (item_id,)
            ).fetchone()
            return _row_to_wisdom(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def get_wisdom_evidence(self, item_id: str) -> list[WisdomEvidence]:
        def _run() -> list[WisdomEvidence]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT doc_id, excerpt, line FROM wisdom_evidence "
                "WHERE item_id = ? ORDER BY rowid ASC",
                (item_id,),
            ).fetchall()
            return [
                WisdomEvidence(
                    doc_id=r["doc_id"],
                    excerpt=r["excerpt"],
                    line=int(r["line"]) if r["line"] is not None else None,
                )
                for r in rows
            ]

        return await asyncio.to_thread(_run)

    # ---- D layer: multimedia assets --------------------------------------

    async def upsert_asset(self, asset: AssetRecord) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO assets(
                        asset_id, hash, kind, mime, stored_path, original_paths,
                        bytes, width, height, caption, caption_model, created_ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_id) DO UPDATE SET
                        hash = excluded.hash,
                        kind = excluded.kind,
                        mime = excluded.mime,
                        stored_path = excluded.stored_path,
                        original_paths = excluded.original_paths,
                        bytes = excluded.bytes,
                        width = excluded.width,
                        height = excluded.height,
                        caption = excluded.caption,
                        caption_model = excluded.caption_model
                    """,
                    (
                        asset.asset_id,
                        asset.hash,
                        asset.kind.value,
                        asset.mime,
                        asset.stored_path,
                        json.dumps(asset.original_paths),
                        asset.bytes,
                        asset.width,
                        asset.height,
                        asset.caption,
                        asset.caption_model,
                        asset.created_ts,
                    ),
                )

        await asyncio.to_thread(_run)

    async def get_asset(self, asset_id: str) -> AssetRecord | None:
        def _run() -> AssetRecord | None:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            return _row_to_asset(row) if row is not None else None

        return await asyncio.to_thread(_run)

    # ---- I layer: chunk ↔ asset bridge -----------------------------------

    async def replace_chunk_asset_refs(
        self, chunk_id: int, refs: Sequence[ChunkAssetRef]
    ) -> None:
        def _run() -> None:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    "DELETE FROM chunk_asset_refs WHERE chunk_id = ?", (chunk_id,)
                )
                for r in refs:
                    if r.chunk_id != chunk_id:
                        raise StorageError(
                            f"ChunkAssetRef.chunk_id={r.chunk_id} doesn't match "
                            f"target chunk_id={chunk_id}"
                        )
                    conn.execute(
                        """
                        INSERT INTO chunk_asset_refs(
                            chunk_id, asset_id, ord, alt,
                            start_in_chunk, end_in_chunk
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            r.chunk_id,
                            r.asset_id,
                            r.ord,
                            r.alt,
                            r.start_in_chunk,
                            r.end_in_chunk,
                        ),
                    )

        await asyncio.to_thread(_run)

    async def chunk_asset_refs_for_chunks(
        self, chunk_ids: Sequence[int]
    ) -> dict[int, list[ChunkAssetRef]]:
        def _run() -> dict[int, list[ChunkAssetRef]]:
            conn = self._require_conn()
            out: dict[int, list[ChunkAssetRef]] = {cid: [] for cid in chunk_ids}
            if not chunk_ids:
                return out
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = conn.execute(
                f"""
                SELECT chunk_id, asset_id, ord, alt, start_in_chunk, end_in_chunk
                FROM chunk_asset_refs
                WHERE chunk_id IN ({placeholders})
                ORDER BY chunk_id, ord
                """,
                tuple(chunk_ids),
            ).fetchall()
            for r in rows:
                out[int(r["chunk_id"])].append(
                    ChunkAssetRef(
                        chunk_id=int(r["chunk_id"]),
                        asset_id=r["asset_id"],
                        ord=int(r["ord"]),
                        alt=r["alt"],
                        start_in_chunk=int(r["start_in_chunk"]),
                        end_in_chunk=int(r["end_in_chunk"]),
                    )
                )
            return out

        return await asyncio.to_thread(_run)

    async def chunks_referencing_assets(
        self, asset_ids: Sequence[str]
    ) -> dict[str, list[int]]:
        def _run() -> dict[str, list[int]]:
            conn = self._require_conn()
            out: dict[str, list[int]] = {aid: [] for aid in asset_ids}
            if not asset_ids:
                return out
            placeholders = ",".join("?" for _ in asset_ids)
            rows = conn.execute(
                f"""
                SELECT asset_id, chunk_id
                FROM chunk_asset_refs
                WHERE asset_id IN ({placeholders})
                ORDER BY asset_id, chunk_id
                """,
                tuple(asset_ids),
            ).fetchall()
            for r in rows:
                out[r["asset_id"]].append(int(r["chunk_id"]))
            return out

        return await asyncio.to_thread(_run)

    # ---- I layer: asset embeddings (multimodal) --------------------------

    async def upsert_asset_embeddings(
        self, rows: Sequence[AssetEmbeddingRow]
    ) -> None:
        if not rows:
            return

        def _run() -> None:
            conn = self._require_conn()
            by_version: dict[int, list[AssetEmbeddingRow]] = {}
            for r in rows:
                by_version.setdefault(r.version_id, []).append(r)
            for version_id, batch in by_version.items():
                version = _fetch_version(conn, version_id)
                if version is None:
                    raise StorageError(
                        f"unknown embed version_id={version_id}; "
                        "call upsert_embed_version first"
                    )
                for r in batch:
                    if len(r.embedding) != version.dim:
                        raise StorageError(
                            f"asset embedding dim {len(r.embedding)} != "
                            f"version {version_id} dim {version.dim}"
                        )
                self._ensure_asset_vec_table(conn, version_id, version.dim)
                rowid_table = f"asset_vec_rowid_v{version_id}"
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {rowid_table}("
                    "rowid INTEGER PRIMARY KEY, "
                    "asset_id TEXT NOT NULL UNIQUE)"
                )
                vec_table = f"vec_assets_v{version_id}"
                with conn:
                    for r in batch:
                        # sqlite-vec needs an integer rowid; derive a stable
                        # 60-bit int from the sha256 asset_id. Birthday
                        # collisions are ~2^-30 even at 10^9 assets, but
                        # detect them explicitly so a hit corrupts no data.
                        rowid = _asset_id_to_rowid(r.asset_id)
                        existing = conn.execute(
                            f"SELECT asset_id FROM {rowid_table} WHERE rowid = ?",
                            (rowid,),
                        ).fetchone()
                        if existing is not None and existing["asset_id"] != r.asset_id:
                            raise StorageError(
                                f"asset rowid collision in version {version_id}: "
                                f"rowid {rowid} already used by "
                                f"{existing['asset_id']!r}, refused for "
                                f"{r.asset_id!r}"
                            )
                        conn.execute(
                            "INSERT OR REPLACE INTO asset_embed_meta"
                            "(asset_id, version_id) VALUES (?, ?)",
                            (r.asset_id, r.version_id),
                        )
                        conn.execute(
                            f"INSERT OR REPLACE INTO {vec_table}(rowid, embedding) "
                            "VALUES (?, ?)",
                            (rowid, _serialize_vec(r.embedding)),
                        )
                        conn.execute(
                            f"INSERT OR REPLACE INTO {rowid_table}"
                            "(rowid, asset_id) VALUES (?, ?)",
                            (rowid, r.asset_id),
                        )

        await asyncio.to_thread(_run)

    async def vec_search_assets(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[AssetVecHit]:
        # `layer` is reserved for future use (e.g. when assets get layer
        # attribution); v1 assets are always D-layer adjacent.
        del layer

        def _run() -> list[AssetVecHit]:
            conn = self._require_conn()
            version = _fetch_version(conn, version_id)
            if version is None:
                raise NotSupported(
                    f"no embed version_id={version_id} registered"
                )
            if len(embedding) != version.dim:
                raise StorageError(
                    f"query embedding dim {len(embedding)} != "
                    f"version {version_id} dim {version.dim}"
                )
            table = f"vec_assets_v{version_id}"
            row_table = f"asset_vec_rowid_v{version_id}"
            # Empty index → no hits. Skip the SQL to avoid sqlite-vec errors
            # on a non-existent virtual table.
            tbl_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if tbl_exists is None:
                return []
            rows = conn.execute(
                f"""
                SELECT m.asset_id AS asset_id,
                       vec_distance_cosine(v.embedding, ?) AS dist
                FROM {table} v
                JOIN {row_table} m ON m.rowid = v.rowid
                ORDER BY dist
                LIMIT ?
                """,
                (_serialize_vec(embedding), limit),
            ).fetchall()
            return [
                AssetVecHit(asset_id=r["asset_id"], distance=float(r["dist"]))
                for r in rows
            ]

        return await asyncio.to_thread(_run)

    # ---- Embedding version registry --------------------------------------

    async def upsert_embed_version(self, v: EmbeddingVersion) -> int:
        def _run() -> int:
            conn = self._require_conn()
            with conn:
                # Match key: (provider, model, revision, dim, normalize, distance).
                row = conn.execute(
                    """
                    SELECT version_id FROM embed_versions
                    WHERE provider = ? AND model = ? AND revision = ?
                      AND dim = ? AND normalize = ? AND distance = ?
                    """,
                    (
                        v.provider,
                        v.model,
                        v.revision,
                        v.dim,
                        int(v.normalize),
                        v.distance,
                    ),
                ).fetchone()
                if row is not None:
                    return int(row["version_id"])
                # New version: insert and demote prior active versions of
                # the same modality.
                conn.execute(
                    "UPDATE embed_versions SET is_active = 0 WHERE modality = ?",
                    (v.modality,),
                )
                created_ts = (
                    v.created_ts if v.created_ts is not None else time.time()
                )
                cur = conn.execute(
                    """
                    INSERT INTO embed_versions(
                        provider, model, revision, dim, normalize, distance,
                        modality, created_ts, is_active
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        v.provider,
                        v.model,
                        v.revision,
                        v.dim,
                        int(v.normalize),
                        v.distance,
                        v.modality,
                        created_ts,
                    ),
                )
                return int(cur.lastrowid or 0)

        return await asyncio.to_thread(_run)

    async def get_active_embed_version(
        self, *, modality: Literal["text", "multimodal"]
    ) -> EmbeddingVersion | None:
        def _run() -> EmbeddingVersion | None:
            conn = self._require_conn()
            row = conn.execute(
                """
                SELECT * FROM embed_versions
                WHERE modality = ? AND is_active = 1
                ORDER BY version_id DESC LIMIT 1
                """,
                (modality,),
            ).fetchone()
            return _row_to_embed_version(row) if row is not None else None

        return await asyncio.to_thread(_run)

    async def list_embed_versions(self) -> list[EmbeddingVersion]:
        def _run() -> list[EmbeddingVersion]:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT * FROM embed_versions ORDER BY version_id"
            ).fetchall()
            return [_row_to_embed_version(r) for r in rows]

        return await asyncio.to_thread(_run)

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
            assets_count = int(conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
            asset_emb_count = int(
                conn.execute("SELECT COUNT(*) FROM asset_embed_meta").fetchone()[0]
            )
            return StorageCounts(
                documents_by_layer=by_layer,
                chunks=chunks,
                embeddings=embeddings,
                links=links,
                wisdom_by_status=by_status,
                last_wiki_log_ts=float(last_log["ts"]) if last_log and last_log["ts"] else None,
                assets=assets_count,
                asset_embeddings=asset_emb_count,
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

    def _ensure_asset_vec_table(
        self, conn: sqlite3.Connection, version_id: int, dim: int
    ) -> None:
        """Create ``vec_assets_v<version_id>`` lazily. sqlite-vec needs the
        embedding dim baked into the table at CREATE time, so each version
        gets its own dim-locked virtual table — switching multimodal model
        creates a new version + new table, leaving prior data intact."""
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_assets_v{version_id} "
            f"USING vec0(embedding float[{dim}])"
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


def _row_to_asset(row: sqlite3.Row) -> AssetRecord:
    return AssetRecord(
        asset_id=row["asset_id"],
        hash=row["hash"],
        kind=AssetKind(row["kind"]),
        mime=row["mime"],
        stored_path=row["stored_path"],
        original_paths=json.loads(row["original_paths"]),
        bytes=int(row["bytes"]),
        width=int(row["width"]) if row["width"] is not None else None,
        height=int(row["height"]) if row["height"] is not None else None,
        caption=row["caption"],
        caption_model=row["caption_model"],
        created_ts=float(row["created_ts"]),
    )


def _row_to_embed_version(row: sqlite3.Row) -> EmbeddingVersion:
    return EmbeddingVersion(
        version_id=int(row["version_id"]),
        provider=row["provider"],
        model=row["model"],
        revision=row["revision"],
        dim=int(row["dim"]),
        normalize=bool(row["normalize"]),
        distance=row["distance"],
        modality=row["modality"],
        created_ts=float(row["created_ts"]),
        is_active=bool(row["is_active"]),
    )


def _fetch_version(
    conn: sqlite3.Connection, version_id: int
) -> EmbeddingVersion | None:
    row = conn.execute(
        "SELECT * FROM embed_versions WHERE version_id = ?", (version_id,)
    ).fetchone()
    return _row_to_embed_version(row) if row is not None else None


def _asset_id_to_rowid(asset_id: str) -> int:
    """Stable, signed-INT64-safe rowid derived from the sha256 asset_id.

    sqlite-vec uses INTEGER rowids; we need a deterministic mapping from
    the hex asset_id to a Python int that fits in sqlite3's signed 64-bit
    rowid space. Take the first 15 hex chars (60 bits) so the result is
    always non-negative and safely below 2**63.
    """
    return int(asset_id[:15], 16)
