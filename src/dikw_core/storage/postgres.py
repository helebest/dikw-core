"""Postgres + pgvector storage adapter (Phase 5, optional extra).

Uses ``psycopg[binary,pool]`` v3 asynchronously and the ``pgvector`` Python
bindings to mirror the SQLite adapter's contract against a multi-writer
database. Install via ``uv pip install dikw-core[postgres]``.

Schema lives in ``storage/migrations/postgres/001_init.sql``; the
``chunks_vec`` table is created lazily at first embedding insert so the
vector dimension matches the active embedding model, exactly like the
SQLite adapter.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

from ..schemas import (
    AssetEmbeddingRow,
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

if TYPE_CHECKING:  # imports happen in connect() so base install works without pg deps
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool


MIGRATIONS_PACKAGE = "dikw_core.storage.migrations.postgres"


class PostgresStorage:
    """Storage Protocol impl on top of Postgres + pgvector."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "dikw",
        pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._schema = schema
        self._pool_size = pool_size
        self._pool: AsyncConnectionPool | None = None
        self._embedding_dim: int | None = None

    # ---- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        try:
            import psycopg
            from psycopg_pool import AsyncConnectionPool
        except ImportError as e:  # pragma: no cover - exercised only without extras
            raise StorageError(
                "Postgres adapter requires the `postgres` extra — "
                "install via `uv pip install dikw-core[postgres]`"
            ) from e

        # Extensions must exist before the pool hands out connections,
        # because ``configure`` below registers pgvector types on each
        # new connection — which needs the ``vector`` type present.
        boot = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        try:
            async with boot.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
        finally:
            await boot.close()

        # pgvector types flow through explicit ``::vector`` casts in the SQL
        # so we don't need a type-registration configure hook here — the
        # extension only needs to exist (which we guaranteed above).

        self._pool = AsyncConnectionPool(
            conninfo=self._dsn,
            min_size=1,
            max_size=self._pool_size,
            kwargs={"autocommit": False},
            open=False,
        )
        await self._pool.open()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> None:
        # Extensions + schema are created in ``connect()`` so the pool can
        # register pgvector. Migrations here just apply tables/indexes.
        sql_text = self._load_migration_sql()
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql_text)
            await conn.commit()
        # Restore the vector dim (if we've indexed before).
        await self._load_embedding_dim()

    # ---- D layer ---------------------------------------------------------

    async def put_content(self, hash_: str, body: str) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO content(hash, body) VALUES (%s, %s) "
                    "ON CONFLICT (hash) DO NOTHING",
                    (hash_, body),
                )
            await conn.commit()

    async def upsert_document(self, doc: DocumentRecord) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO documents(doc_id, path, title, hash, mtime, layer, active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        path = EXCLUDED.path,
                        title = EXCLUDED.title,
                        hash = EXCLUDED.hash,
                        mtime = EXCLUDED.mtime,
                        layer = EXCLUDED.layer,
                        active = EXCLUDED.active
                    """,
                    (
                        doc.doc_id,
                        doc.path,
                        doc.title,
                        doc.hash,
                        doc.mtime,
                        doc.layer.value,
                        doc.active,
                    ),
                )
            await conn.commit()

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT doc_id, path, title, hash, mtime, layer, active "
                "FROM documents WHERE doc_id = %s",
                (doc_id,),
            )
            row = await cur.fetchone()
        return _row_to_document(row) if row else None

    async def get_documents(
        self, doc_ids: Iterable[str]
    ) -> list[DocumentRecord]:
        ids = list(doc_ids)
        if not ids:
            return []
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT doc_id, path, title, hash, mtime, layer, active "
                "FROM documents WHERE doc_id = ANY(%s)",
                (ids,),
            )
            rows = await cur.fetchall()
        return [_row_to_document(r) for r in rows]

    async def list_documents(
        self,
        *,
        layer: Layer | None = None,
        active: bool | None = True,
        since_ts: float | None = None,
    ) -> Iterable[DocumentRecord]:
        sql = (
            "SELECT doc_id, path, title, hash, mtime, layer, active "
            "FROM documents WHERE TRUE"
        )
        params: list[Any] = []
        if layer is not None:
            sql += " AND layer = %s"
            params.append(layer.value)
        if active is not None:
            sql += " AND active = %s"
            params.append(bool(active))
        if since_ts is not None:
            sql += " AND mtime >= %s"
            params.append(since_ts)
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [_row_to_document(r) for r in rows]

    async def deactivate_document(self, doc_id: str) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET active = FALSE WHERE doc_id = %s",
                    (doc_id,),
                )
            await conn.commit()

    # ---- I layer ---------------------------------------------------------

    async def replace_chunks(
        self, doc_id: str, chunks: Sequence[ChunkRecord]
    ) -> list[int]:
        async with self._acquire() as conn:
            ids: list[int] = []
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                for chunk in chunks:
                    await cur.execute(
                        "INSERT INTO chunks(doc_id, seq, start_off, end_off, text) "
                        "VALUES (%s, %s, %s, %s, %s) RETURNING chunk_id",
                        (doc_id, chunk.seq, chunk.start, chunk.end, chunk.text),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise StorageError("failed to insert chunk")
                    ids.append(int(row[0]))
            await conn.commit()
            return ids

    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        if not rows:
            return
        dim = len(rows[0].embedding)
        async with self._acquire() as conn:
            await self._ensure_vec_table(conn, dim)
            async with conn.cursor() as cur:
                for row in rows:
                    if len(row.embedding) != dim:
                        raise StorageError(
                            f"embedding dim mismatch: expected {dim}, got {len(row.embedding)}"
                        )
                    await cur.execute(
                        "INSERT INTO embed_meta(chunk_id, model) VALUES (%s, %s) "
                        "ON CONFLICT (chunk_id, model) DO NOTHING",
                        (row.chunk_id, row.model),
                    )
                    await cur.execute(
                        "INSERT INTO chunks_vec(chunk_id, embedding) "
                        "VALUES (%s, %s::vector) "
                        "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                        (row.chunk_id, list(row.embedding)),
                    )
            await conn.commit()

    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks WHERE chunk_id = %s",
                (chunk_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return ChunkRecord(
            chunk_id=int(row[0]),
            doc_id=row[1],
            seq=int(row[2]),
            start=int(row[3]),
            end=int(row[4]),
            text=row[5],
        )

    async def get_chunks(self, chunk_ids: Iterable[int]) -> list[ChunkRecord]:
        ids = list(chunk_ids)
        if not ids:
            return []
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, seq, start_off, end_off, text "
                "FROM chunks WHERE chunk_id = ANY(%s)",
                (ids,),
            )
            rows = await cur.fetchall()
        return [
            ChunkRecord(
                chunk_id=int(r[0]),
                doc_id=r[1],
                seq=int(r[2]),
                start=int(r[3]),
                end=int(r[4]),
                text=r[5],
            )
            for r in rows
        ]

    async def fts_search(
        self, q: str, *, limit: int = 20, layer: Layer | None = None
    ) -> list[FTSHit]:
        sql = (
            "SELECT c.doc_id, c.chunk_id, "
            "ts_rank(c.fts, plainto_tsquery('simple', %s)) AS score, "
            "ts_headline('simple', c.text, plainto_tsquery('simple', %s), "
            "  'StartSel=<mark>,StopSel=</mark>,ShortWord=2,MaxWords=25,MinWords=5') AS snip "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE d.active = TRUE AND c.fts @@ plainto_tsquery('simple', %s)"
        )
        params: list[Any] = [q, q, q]
        if layer is not None:
            sql += " AND d.layer = %s"
            params.append(layer.value)
        sql += " ORDER BY score DESC LIMIT %s"
        params.append(limit)

        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        best_by_doc: dict[str, FTSHit] = {}
        for row in rows:
            doc_id = row[0]
            if doc_id in best_by_doc:
                continue
            best_by_doc[doc_id] = FTSHit(
                doc_id=doc_id,
                chunk_id=int(row[1]),
                score=float(row[2]),
                snippet=row[3],
            )
        return list(best_by_doc.values())[:limit]

    async def vec_search(
        self, embedding: list[float], *, limit: int = 20, layer: Layer | None = None
    ) -> list[VecHit]:
        if self._embedding_dim is None:
            raise NotSupported("no embeddings indexed yet")
        if len(embedding) != self._embedding_dim:
            raise StorageError(
                f"query embedding dim {len(embedding)} != index dim {self._embedding_dim}"
            )

        # Cosine distance is undefined for the zero vector. pgvector's
        # ``<=>`` operator does NOT return NULL on zero vectors — it
        # returns NaN (sqlite-vec returns NULL, hence the original guard
        # below). NaN slips past ``IS NOT NULL`` and ``ORDER BY ASC``
        # places NaN somewhere implementation-defined. Drop both at the
        # Python layer with ``math.isnan`` so degenerate rows never
        # surface as hits regardless of which backend produced them.
        vec = list(embedding)
        sql = (
            "SELECT cv.chunk_id, c.doc_id, (cv.embedding <=> %s::vector) AS dist "
            "FROM chunks_vec cv JOIN chunks c ON c.chunk_id = cv.chunk_id "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE d.active = TRUE AND (cv.embedding <=> %s::vector) IS NOT NULL"
        )
        params: list[Any] = [vec, vec]
        if layer is not None:
            sql += " AND d.layer = %s"
            params.append(layer.value)
        sql += " ORDER BY dist ASC LIMIT %s"
        params.append(limit)

        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [
            VecHit(chunk_id=int(r[0]), doc_id=r[1], distance=float(r[2]))
            for r in rows
            if r[2] is not None and not math.isnan(r[2])
        ]

    # ---- K layer ---------------------------------------------------------

    async def upsert_link(self, link: LinkRecord) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO links(src_doc_id, dst_path, link_type, anchor, line)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (src_doc_id, dst_path, line) DO UPDATE SET
                        link_type = EXCLUDED.link_type,
                        anchor = EXCLUDED.anchor
                    """,
                    (link.src_doc_id, link.dst_path, link.link_type.value, link.anchor, link.line),
                )
            await conn.commit()

    async def links_from(self, src_doc_id: str) -> list[LinkRecord]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT src_doc_id, dst_path, link_type, anchor, line "
                "FROM links WHERE src_doc_id = %s",
                (src_doc_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_link(r) for r in rows]

    async def links_to(self, dst_path: str) -> list[LinkRecord]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT src_doc_id, dst_path, link_type, anchor, line "
                "FROM links WHERE dst_path = %s",
                (dst_path,),
            )
            rows = await cur.fetchall()
        return [_row_to_link(r) for r in rows]

    async def append_wiki_log(self, entry: WikiLogEntry) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO wiki_log(ts, action, src, dst, note) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (entry.ts, entry.action, entry.src, entry.dst, entry.note),
                )
            await conn.commit()

    async def list_wiki_log(
        self, *, since_ts: float | None = None, limit: int | None = None
    ) -> list[WikiLogEntry]:
        sql = "SELECT ts, action, src, dst, note FROM wiki_log"
        params: list[Any] = []
        if since_ts is not None:
            sql += " WHERE ts >= %s"
            params.append(since_ts)
        sql += " ORDER BY ts ASC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [
            WikiLogEntry(
                ts=float(r[0]),
                action=r[1],
                src=r[2],
                dst=r[3],
                note=r[4],
            )
            for r in rows
        ]

    # ---- W layer ---------------------------------------------------------

    async def put_wisdom(
        self, item: WisdomItem, evidence: Sequence[WisdomEvidence]
    ) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO wisdom_items(
                        item_id, kind, status, path, title, body,
                        confidence, created_ts, approved_ts
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (item_id) DO UPDATE SET
                        kind = EXCLUDED.kind,
                        status = EXCLUDED.status,
                        path = EXCLUDED.path,
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        confidence = EXCLUDED.confidence,
                        approved_ts = EXCLUDED.approved_ts
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
                await cur.execute(
                    "DELETE FROM wisdom_evidence WHERE item_id = %s", (item.item_id,)
                )
                for ev in evidence:
                    await cur.execute(
                        "INSERT INTO wisdom_evidence(item_id, doc_id, excerpt, line) "
                        "VALUES (%s, %s, %s, %s)",
                        (item.item_id, ev.doc_id, ev.excerpt, ev.line),
                    )
            await conn.commit()

    async def list_wisdom(
        self,
        *,
        status: WisdomStatus | None = None,
        kind: WisdomKind | None = None,
    ) -> list[WisdomItem]:
        sql = (
            "SELECT item_id, kind, status, path, title, body, confidence, "
            "       created_ts, approved_ts FROM wisdom_items WHERE TRUE"
        )
        params: list[Any] = []
        if status is not None:
            sql += " AND status = %s"
            params.append(status.value)
        if kind is not None:
            sql += " AND kind = %s"
            params.append(kind.value)
        sql += " ORDER BY created_ts DESC"
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [_row_to_wisdom(r) for r in rows]

    async def set_wisdom_status(
        self,
        item_id: str,
        status: WisdomStatus,
        *,
        approved_ts: float | None = None,
    ) -> None:
        async with self._acquire() as conn:
            async with conn.cursor() as cur:
                if approved_ts is None:
                    await cur.execute(
                        "UPDATE wisdom_items SET status = %s WHERE item_id = %s",
                        (status.value, item_id),
                    )
                else:
                    await cur.execute(
                        "UPDATE wisdom_items SET status = %s, approved_ts = %s "
                        "WHERE item_id = %s",
                        (status.value, approved_ts, item_id),
                    )
            await conn.commit()

    async def get_wisdom(self, item_id: str) -> WisdomItem | None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT item_id, kind, status, path, title, body, confidence, "
                "       created_ts, approved_ts FROM wisdom_items WHERE item_id = %s",
                (item_id,),
            )
            row = await cur.fetchone()
        return _row_to_wisdom(row) if row else None

    async def get_wisdom_evidence(self, item_id: str) -> list[WisdomEvidence]:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT doc_id, excerpt, line FROM wisdom_evidence "
                "WHERE item_id = %s ORDER BY id ASC",
                (item_id,),
            )
            rows = await cur.fetchall()
        return [
            WisdomEvidence(
                doc_id=r[0],
                excerpt=r[1],
                line=int(r[2]) if r[2] is not None else None,
            )
            for r in rows
        ]

    # ---- diagnostics -----------------------------------------------------

    async def counts(self) -> StorageCounts:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT layer, COUNT(*) FROM documents "
                "WHERE active = TRUE GROUP BY layer"
            )
            by_layer = {row[0]: int(row[1]) for row in await cur.fetchall()}
            await cur.execute("SELECT COUNT(*) FROM chunks")
            chunks = int((await cur.fetchone())[0])
            await cur.execute("SELECT COUNT(*) FROM embed_meta")
            embeddings = int((await cur.fetchone())[0])
            await cur.execute("SELECT COUNT(*) FROM links")
            links = int((await cur.fetchone())[0])
            await cur.execute(
                "SELECT status, COUNT(*) FROM wisdom_items GROUP BY status"
            )
            by_status = {row[0]: int(row[1]) for row in await cur.fetchall()}
            await cur.execute("SELECT MAX(ts) FROM wiki_log")
            last = (await cur.fetchone())[0]

        return StorageCounts(
            documents_by_layer=by_layer,
            chunks=chunks,
            embeddings=embeddings,
            links=links,
            wisdom_by_status=by_status,
            last_wiki_log_ts=float(last) if last is not None else None,
        )

    # ---- multimedia assets (Phase 5: not yet implemented) ----------------
    #
    # Postgres adapter's asset / version-aware embedding support lands in a
    # follow-up phase. Until then every new method raises NotSupported so
    # callers (and contract tests) can detect and skip cleanly.

    async def upsert_asset(self, asset: AssetRecord) -> None:
        raise NotSupported("postgres adapter: assets not implemented yet")

    async def get_asset(self, asset_id: str) -> AssetRecord | None:
        raise NotSupported("postgres adapter: assets not implemented yet")

    async def replace_chunk_asset_refs(
        self, chunk_id: int, refs: Sequence[ChunkAssetRef]
    ) -> None:
        raise NotSupported("postgres adapter: chunk_asset_refs not implemented yet")

    async def chunk_asset_refs_for_chunks(
        self, chunk_ids: Sequence[int]
    ) -> dict[int, list[ChunkAssetRef]]:
        raise NotSupported("postgres adapter: chunk_asset_refs not implemented yet")

    async def chunks_referencing_assets(
        self, asset_ids: Sequence[str]
    ) -> dict[str, list[int]]:
        raise NotSupported("postgres adapter: chunk_asset_refs not implemented yet")

    async def upsert_asset_embeddings(
        self, rows: Sequence[AssetEmbeddingRow]
    ) -> None:
        raise NotSupported("postgres adapter: asset embeddings not implemented yet")

    async def vec_search_assets(
        self,
        embedding: list[float],
        *,
        version_id: int,
        limit: int = 20,
        layer: Layer | None = None,
    ) -> list[AssetVecHit]:
        raise NotSupported("postgres adapter: asset embeddings not implemented yet")

    async def upsert_embed_version(self, v: EmbeddingVersion) -> int:
        raise NotSupported("postgres adapter: embed versioning not implemented yet")

    async def get_active_embed_version(
        self, *, modality: Literal["text", "multimodal"]
    ) -> EmbeddingVersion | None:
        raise NotSupported("postgres adapter: embed versioning not implemented yet")

    async def list_embed_versions(self) -> list[EmbeddingVersion]:
        raise NotSupported("postgres adapter: embed versioning not implemented yet")

    # ---- internals -------------------------------------------------------

    def _acquire(self) -> Any:
        if self._pool is None:
            raise StorageError("PostgresStorage is not connected; call `connect()` first")
        return _ConnectionContext(self._pool, self._schema)

    def _load_migration_sql(self) -> str:
        return (
            resources.files(MIGRATIONS_PACKAGE)
            .joinpath("001_init.sql")
            .read_text(encoding="utf-8")
        )

    async def _ensure_vec_table(self, conn: AsyncConnection, dim: int) -> None:
        if self._embedding_dim is None:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"CREATE TABLE IF NOT EXISTS chunks_vec ("
                    f"  chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,"
                    f"  embedding vector({dim}) NOT NULL"
                    f")"
                )
                await cur.execute(
                    "INSERT INTO meta_kv(key, value) VALUES ('embedding_dim', %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (str(dim),),
                )
            await conn.commit()
            self._embedding_dim = dim
            return
        if self._embedding_dim != dim:
            raise StorageError(
                f"embedding dim mismatch: index uses {self._embedding_dim}, got {dim}"
            )

    async def _load_embedding_dim(self) -> None:
        async with self._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT value FROM meta_kv WHERE key = 'embedding_dim'"
            )
            row = await cur.fetchone()
        if row is not None:
            try:
                self._embedding_dim = int(row[0])
            except (ValueError, TypeError):
                self._embedding_dim = None


class _ConnectionContext:
    """``async with``-friendly wrapper that sets the schema search path per conn."""

    def __init__(self, pool: AsyncConnectionPool, schema: str) -> None:
        self._pool = pool
        self._schema = schema
        self._cm: Any = None

    async def __aenter__(self) -> AsyncConnection:
        self._cm = self._pool.connection()
        conn: AsyncConnection = await self._cm.__aenter__()
        async with conn.cursor() as cur:
            await cur.execute(f"SET search_path TO {self._schema}, public")
        return conn

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._cm.__aexit__(*exc_info)


# ---- row → DTO helpers ---------------------------------------------------


def _row_to_document(row: Any) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row[0],
        path=row[1],
        title=row[2],
        hash=row[3],
        mtime=float(row[4] or 0.0),
        layer=Layer(row[5]),
        active=bool(row[6]),
    )


def _row_to_link(row: Any) -> LinkRecord:
    return LinkRecord(
        src_doc_id=row[0],
        dst_path=row[1],
        link_type=LinkType(row[2]),
        anchor=row[3],
        line=int(row[4]),
    )


def _row_to_wisdom(row: Any) -> WisdomItem:
    return WisdomItem(
        item_id=row[0],
        kind=WisdomKind(row[1]),
        status=WisdomStatus(row[2]),
        path=row[3],
        title=row[4],
        body=row[5],
        confidence=float(row[6]),
        created_ts=float(row[7]),
        approved_ts=float(row[8]) if row[8] is not None else None,
    )
