"""Storage contract tests.

Parameterized over every backend so the MVP can't grow SQLite-only assumptions
before Phase 5 lands the Postgres and Filesystem adapters. Phase 0 only
exercises the SQLite adapter; Postgres/Filesystem parameterizations are
declared but skip until their adapters exist.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from dikw_core.schemas import (
    AssetEmbeddingRow,
    AssetKind,
    AssetRecord,
    CachedEmbeddingRow,
    ChunkAssetRef,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    EmbeddingVersion,
    ImageMediaMeta,
    Layer,
    LinkRecord,
    LinkType,
    WikiLogEntry,
    WisdomEvidence,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
)
from dikw_core.storage.base import NotSupported, Storage
from dikw_core.storage.filesystem import FilesystemStorage
from dikw_core.storage.sqlite import SQLiteStorage


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param(
            "postgres",
            id="postgres",
            marks=pytest.mark.skipif(
                not os.environ.get("DIKW_TEST_POSTGRES_DSN"),
                reason="Postgres adapter tests require DIKW_TEST_POSTGRES_DSN",
            ),
        ),
        pytest.param("filesystem", id="filesystem"),
    ]
)
async def storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Storage]:
    backend = request.param
    if backend == "sqlite":
        s: Storage = SQLiteStorage(tmp_path / "index.sqlite")
    elif backend == "filesystem":
        s = FilesystemStorage(tmp_path / ".dikw" / "fs", embed=True)
    elif backend == "postgres":
        from dikw_core.storage.postgres import PostgresStorage

        dsn = os.environ["DIKW_TEST_POSTGRES_DSN"]
        # Use a schema derived from the test tmpdir so parallel runs don't collide.
        schema = f"dikw_test_{abs(hash(str(tmp_path))) % 10_000_000:07d}"
        s = PostgresStorage(dsn, schema=schema, pool_size=2)
    else:
        raise RuntimeError(f"unreachable: adapter {backend}")

    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
        # Drop the Postgres schema to keep the test DB clean between runs.
        if backend == "postgres":
            from psycopg import AsyncConnection

            conn = await AsyncConnection.connect(os.environ["DIKW_TEST_POSTGRES_DSN"])
            try:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                await conn.commit()
            finally:
                await conn.close()
        await s.close()


def _make_doc(path: str, layer: Layer = Layer.SOURCE) -> DocumentRecord:
    body_hash = f"hash-{path}"
    return DocumentRecord(
        doc_id=f"doc::{path}",
        path=path,
        title=path.rsplit("/", 1)[-1],
        hash=body_hash,
        mtime=time.time(),
        layer=layer,
        active=True,
    )


async def test_migrate_is_idempotent(storage: Storage) -> None:
    await storage.migrate()
    await storage.migrate()
    counts = await storage.counts()
    assert counts.chunks == 0
    assert counts.documents_by_layer == {}


async def test_document_roundtrip(storage: Storage) -> None:
    doc = _make_doc("sources/a.md")
    await storage.upsert_document(doc)

    fetched = await storage.get_document(doc.doc_id)
    assert fetched is not None
    assert fetched.path == "sources/a.md"
    assert fetched.layer == Layer.SOURCE

    docs = list(await storage.list_documents(layer=Layer.SOURCE))
    assert len(docs) == 1

    await storage.deactivate_document(doc.doc_id)
    active_docs = list(await storage.list_documents(layer=Layer.SOURCE, active=True))
    assert active_docs == []


async def test_storage_has_no_put_content(storage: Storage) -> None:
    """The legacy ``put_content`` Protocol method is gone — D-layer schema
    no longer carries a write-only ``content`` table. Sentinel that catches
    accidental re-introduction of the API."""
    assert not hasattr(storage, "put_content")


async def test_upsert_document_without_put_content(storage: Storage) -> None:
    """``upsert_document`` must succeed without any prior content-table write.

    Pre-refactor this failed on SQLite/Postgres because ``documents.hash``
    was an FK to ``content(hash)``. After the refactor the FK is gone and
    ``documents.hash`` is just an indexed text column, so the call stands
    on its own.
    """
    doc = _make_doc("sources/no-content.md")
    await storage.upsert_document(doc)
    fetched = await storage.get_document(doc.doc_id)
    assert fetched is not None
    assert fetched.hash == doc.hash


async def test_documents_hash_indexed(storage: Storage) -> None:
    """``documents.hash`` must be indexed so content-addressed lookups
    (``WHERE hash = ?``) don't fall back to a sequential scan once the
    legacy ``content`` table is gone.

    Filesystem skips: no SQL indexes, the in-memory dicts already hash by
    doc_id and the documents file is small enough to scan.
    """
    if isinstance(storage, FilesystemStorage):
        pytest.skip("filesystem adapter has no SQL indexes")

    # Schema-introspection assertion — by design the Storage Protocol exposes no
    # raw connection, so this contract test reaches into adapter internals to
    # verify the migration actually created the index. Adding a Protocol method
    # just to dedupe one assertion would leak SQL up to the engine.
    if isinstance(storage, SQLiteStorage):
        conn = storage._conn
        assert conn is not None
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='documents'"
        ).fetchall()
        names = {r[0] for r in rows}
    else:  # PostgresStorage
        async with storage._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'documents' "
                "AND schemaname = current_schema()"
            )
            rows = await cur.fetchall()
        names = {r[0] for r in rows}

    assert "documents_hash_idx" in names, (
        f"expected documents_hash_idx among {sorted(names)}"
    )


async def test_chunks_offset_columns_renamed(storage: Storage) -> None:
    """The DTO field names ``ChunkRecord.start``/``.end`` are unaffected —
    adapters translate at the SQL boundary.

    Filesystem skips: no SQL columns.
    """
    if isinstance(storage, FilesystemStorage):
        pytest.skip("filesystem adapter has no SQL columns")

    if isinstance(storage, SQLiteStorage):
        conn = storage._conn
        assert conn is not None
        cols = {r["name"] for r in conn.execute("PRAGMA table_info('chunks')")}
    else:  # PostgresStorage
        async with storage._acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'chunks' "
                "AND table_schema = current_schema()"
            )
            rows = await cur.fetchall()
        cols = {r[0] for r in rows}

    assert "start_off" in cols and "end_off" in cols, (
        f"expected start_off/end_off in chunks columns, got {sorted(cols)}"
    )
    assert "start" not in cols and "end" not in cols, (
        f"legacy start/\"end\" columns must be gone, got {sorted(cols)}"
    )


async def test_get_documents_batch(storage: Storage) -> None:
    """Batch fetch is the N+1 fix used by chunk-level retrieval — every
    adapter must satisfy it (missing ids are dropped silently, not raised).
    """
    a = _make_doc("sources/batch_a.md")
    b = _make_doc("sources/batch_b.md")
    await storage.upsert_document(a)
    await storage.upsert_document(b)

    fetched = await storage.get_documents([a.doc_id, b.doc_id, "missing:nope"])
    by_id = {d.doc_id: d for d in fetched}
    assert set(by_id.keys()) == {a.doc_id, b.doc_id}
    assert by_id[a.doc_id].path == "sources/batch_a.md"
    assert by_id[b.doc_id].path == "sources/batch_b.md"

    # Empty input → empty output, no DB hit needed.
    assert await storage.get_documents([]) == []


async def test_get_chunks_batch(storage: Storage) -> None:
    """Batch chunk fetch — same contract as ``get_documents`` for the
    chunk side. Used by the search path to avoid N+1 over retrieved
    chunk_ids."""
    doc = _make_doc("sources/chunks_batch.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=4, text="aaaa"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=4, end=8, text="bbbb"),
            ChunkRecord(doc_id=doc.doc_id, seq=2, start=8, end=12, text="cccc"),
        ],
    )
    fetched = await storage.get_chunks([ids[0], ids[2], 999_999])
    by_id = {c.chunk_id: c for c in fetched}
    assert set(by_id.keys()) == {ids[0], ids[2]}
    assert by_id[ids[0]].text == "aaaa"
    assert by_id[ids[2]].text == "cccc"
    assert await storage.get_chunks([]) == []


async def test_chunks_and_fts_search(storage: Storage) -> None:
    doc = _make_doc("sources/chunked.md")
    await storage.upsert_document(doc)

    await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=18, text="The quick brown fox"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=18, end=36, text="jumps over the fence"),
        ],
    )
    hits = await storage.fts_search("brown", limit=5)
    assert any(h.doc_id == doc.doc_id for h in hits)
    # snippet highlighting is available
    assert any(h.snippet and "brown" in h.snippet.lower() for h in hits)


async def test_embeddings_and_vec_search(storage: Storage) -> None:
    doc = _make_doc("sources/vec.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=5, text="alpha")],
    )
    assert len(ids) == 1
    cid = ids[0]

    await storage.upsert_embeddings(
        [EmbeddingRow(chunk_id=cid, model="test-embed", embedding=[1.0, 0.0, 0.0, 0.0])]
    )
    hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=3)
    assert hits and hits[0].chunk_id == cid
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


async def test_vec_search_skips_zero_vector_embeddings(storage: Storage) -> None:
    """Zero-vector indexed embeddings have undefined cosine distance.

    Surfaces in practice when a hashed bag-of-words embedder hits text
    outside its alphabet (e.g., FakeEmbeddings on CJK), or when an
    upstream provider returns degenerate output. Adapters must skip
    such rows rather than crash on a NULL/NaN distance.
    """
    a = _make_doc("sources/normal.md")
    b = _make_doc("sources/zero.md")
    for d in (a, b):
        await storage.upsert_document(d)
    a_ids = await storage.replace_chunks(
        a.doc_id, [ChunkRecord(doc_id=a.doc_id, seq=0, start=0, end=5, text="alpha")]
    )
    b_ids = await storage.replace_chunks(
        b.doc_id, [ChunkRecord(doc_id=b.doc_id, seq=0, start=0, end=5, text="zero")]
    )
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=a_ids[0], model="test-embed", embedding=[1.0, 0.0, 0.0, 0.0]
            ),
            EmbeddingRow(
                chunk_id=b_ids[0], model="test-embed", embedding=[0.0, 0.0, 0.0, 0.0]
            ),
        ]
    )

    hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=10)
    chunk_ids = {h.chunk_id for h in hits}
    assert a_ids[0] in chunk_ids, "non-zero indexed embedding must surface"
    assert b_ids[0] not in chunk_ids, (
        "zero-vector indexed embedding must be skipped (cosine distance is undefined)"
    )
    # No hit should carry NaN/None — the float() coercion would crash.
    for h in hits:
        assert h.distance == h.distance, f"NaN distance: {h}"  # NaN != NaN


# ---- embed_cache (chunk-level content-hash cache) ------------------------
#
# Skip cleanly on backends that don't implement the cache (filesystem,
# pre-alpha). The cache is keyed by (sha256(chunk.text), model) and
# decouples embedding reuse from chunks.chunk_id, so re-ingest under
# replace_chunks's delete-and-reinsert semantics doesn't lose API spend
# on byte-identical chunks.


async def test_get_cached_embeddings_empty_input_no_roundtrip(
    storage: Storage,
) -> None:
    """Empty input list must short-circuit before touching the DB."""
    try:
        result = await storage.get_cached_embeddings([], model="any-model")
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert result == {}


async def test_cache_then_get_roundtrip(storage: Storage) -> None:
    """cache_embeddings → get_cached_embeddings returns the same vectors."""
    rows = [
        CachedEmbeddingRow(
            content_hash="a" * 64, model="m1", dim=4, embedding=[1.0, 0.0, 0.0, 0.0]
        ),
        CachedEmbeddingRow(
            content_hash="b" * 64, model="m1", dim=4, embedding=[0.0, 1.0, 0.0, 0.0]
        ),
        CachedEmbeddingRow(
            content_hash="c" * 64, model="m1", dim=4, embedding=[0.0, 0.0, 1.0, 0.0]
        ),
    ]
    try:
        await storage.cache_embeddings(rows)
        got = await storage.get_cached_embeddings(
            ["a" * 64, "b" * 64, "c" * 64, "z" * 64], model="m1"
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert set(got.keys()) == {"a" * 64, "b" * 64, "c" * 64}
    assert got["a" * 64] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert got["b" * 64] == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert got["c" * 64] == pytest.approx([0.0, 0.0, 1.0, 0.0])


async def test_cache_embeddings_idempotent(storage: Storage) -> None:
    """Inserting the same (content_hash, model) twice is a no-op.

    Vectors for the same content + model are deterministic by definition,
    so the second insert must NOT raise and MUST NOT clobber the first
    (the cache is a content-addressed lookup table, not a version log).
    """
    row = CachedEmbeddingRow(
        content_hash="d" * 64, model="m1", dim=4, embedding=[0.5, 0.5, 0.5, 0.5]
    )
    try:
        await storage.cache_embeddings([row])
        await storage.cache_embeddings([row])  # second call: no-op, no exception
        got = await storage.get_cached_embeddings(["d" * 64], model="m1")
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert got["d" * 64] == pytest.approx([0.5, 0.5, 0.5, 0.5])


async def test_cache_partitioned_by_model(storage: Storage) -> None:
    """Same content_hash under two different models = two independent rows.

    Lookup must filter by model so a content-hash can carry vectors for
    multiple embedding models in parallel without collision.
    """
    h = "e" * 64
    try:
        await storage.cache_embeddings(
            [
                CachedEmbeddingRow(
                    content_hash=h, model="m1", dim=4, embedding=[1.0, 0.0, 0.0, 0.0]
                ),
                CachedEmbeddingRow(
                    content_hash=h, model="m2", dim=4, embedding=[0.0, 1.0, 0.0, 0.0]
                ),
            ]
        )
        got_m1 = await storage.get_cached_embeddings([h], model="m1")
        got_m2 = await storage.get_cached_embeddings([h], model="m2")
    except NotSupported:
        pytest.skip("backend doesn't implement embed cache")
    assert got_m1[h] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert got_m2[h] == pytest.approx([0.0, 1.0, 0.0, 0.0])


async def test_list_chunks_missing_embedding(storage: Storage) -> None:
    """Resume-scan: returns chunks without an embed_meta row for ``model``.

    A chunk that's been embedded under a DIFFERENT model still counts
    as "missing for model X" — model is part of the dedup key, the same
    chunk text could legitimately be re-embedded under a new model.
    """
    doc = _make_doc("sources/scan.md")
    await storage.upsert_document(doc)
    chunk_ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="aaa"),
            ChunkRecord(doc_id=doc.doc_id, seq=1, start=10, end=20, text="bbb"),
            ChunkRecord(doc_id=doc.doc_id, seq=2, start=20, end=30, text="ccc"),
        ],
    )
    # Embed chunk 0 under model "m1", chunk 1 under "m2"; chunk 2 unembedded.
    try:
        await storage.upsert_embeddings(
            [
                EmbeddingRow(
                    chunk_id=chunk_ids[0], model="m1", embedding=[1.0, 0.0, 0.0, 0.0]
                ),
                EmbeddingRow(
                    chunk_id=chunk_ids[1], model="m2", embedding=[0.0, 1.0, 0.0, 0.0]
                ),
            ]
        )
    except NotSupported:
        pytest.skip("backend doesn't implement embeddings")
    missing = await storage.list_chunks_missing_embedding(model="m1")
    missing_ids = {c.chunk_id for c in missing}
    # Under m1: chunk 0 has it; chunks 1 + 2 don't.
    assert missing_ids == {chunk_ids[1], chunk_ids[2]}
    # Returned ChunkRecords carry the original text so the caller can
    # re-embed without a separate fetch.
    by_id = {c.chunk_id: c for c in missing}
    assert by_id[chunk_ids[1]].text == "bbb"
    assert by_id[chunk_ids[2]].text == "ccc"


async def test_filesystem_cache_methods_raise_notsupported(
    tmp_path: Path,
) -> None:
    """Filesystem adapter declines the embed cache (pre-alpha).

    Documents the contract explicitly, separate from the parametrized
    cache tests above (which skip rather than fail).
    """
    fs = FilesystemStorage(tmp_path / ".dikw" / "fs", embed=True)
    await fs.connect()
    await fs.migrate()
    try:
        with pytest.raises(NotSupported):
            await fs.get_cached_embeddings(["a" * 64], model="m1")
        with pytest.raises(NotSupported):
            await fs.cache_embeddings(
                [
                    CachedEmbeddingRow(
                        content_hash="a" * 64,
                        model="m1",
                        dim=4,
                        embedding=[1.0, 0.0, 0.0, 0.0],
                    )
                ]
            )
    finally:
        await fs.close()


async def test_vec_search_returns_in_distance_order(storage: Storage) -> None:
    """KNN MATCH must return rows ordered by ascending cosine distance.

    Builds a deterministic 5-chunk corpus where each chunk's embedding is
    a known angular distance from the query vector, then asserts the
    returned ranking matches that distance order.
    """
    doc = _make_doc("sources/order.md")
    await storage.upsert_document(doc)
    # 5 chunks, each with a 4-dim vector at increasing angle from
    # [1, 0, 0, 0]. Closest first.
    embeddings = [
        [1.00, 0.00, 0.00, 0.00],  # cosine dist = 0.0
        [0.99, 0.10, 0.00, 0.00],  # ~0.005
        [0.80, 0.60, 0.00, 0.00],  # ~0.20
        [0.50, 0.866, 0.00, 0.00],  # ~0.50
        [0.00, 1.00, 0.00, 0.00],  # 1.0
    ]
    chunks = [
        ChunkRecord(doc_id=doc.doc_id, seq=i, start=i * 5, end=i * 5 + 5, text=f"c{i}")
        for i in range(len(embeddings))
    ]
    chunk_ids = await storage.replace_chunks(doc.doc_id, chunks)
    await storage.upsert_embeddings(
        [
            EmbeddingRow(chunk_id=cid, model="test-embed", embedding=emb)
            for cid, emb in zip(chunk_ids, embeddings, strict=True)
        ]
    )
    hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=5)
    assert len(hits) == 5
    # Returned ranking must follow the embedding insertion order (which
    # is also the distance order by construction).
    assert [h.chunk_id for h in hits] == chunk_ids
    # Distances must be monotone non-decreasing.
    distances = [h.distance for h in hits]
    assert distances == sorted(distances), f"distances not in order: {distances}"


async def test_vec_search_layer_filter(storage: Storage) -> None:
    """Layer filter applied after KNN must return only the requested layer.

    Inserts a SOURCE-layer doc and a WIKI-layer doc whose embeddings
    bracket the query; without the filter both surface, with
    ``layer=WIKI`` only the wiki chunk surfaces.
    """
    src_doc = _make_doc("sources/src.md", layer=Layer.SOURCE)
    wiki_doc = _make_doc("wiki/page.md", layer=Layer.WIKI)
    for d in (src_doc, wiki_doc):
        await storage.upsert_document(d)
    src_ids = await storage.replace_chunks(
        src_doc.doc_id,
        [ChunkRecord(doc_id=src_doc.doc_id, seq=0, start=0, end=5, text="src")],
    )
    wiki_ids = await storage.replace_chunks(
        wiki_doc.doc_id,
        [ChunkRecord(doc_id=wiki_doc.doc_id, seq=0, start=0, end=5, text="wiki")],
    )
    # Source chunk is the closer match; wiki chunk is the farther one.
    await storage.upsert_embeddings(
        [
            EmbeddingRow(
                chunk_id=src_ids[0], model="test-embed", embedding=[1.0, 0.0, 0.0, 0.0]
            ),
            EmbeddingRow(
                chunk_id=wiki_ids[0], model="test-embed", embedding=[0.0, 1.0, 0.0, 0.0]
            ),
        ]
    )
    # Unfiltered: both come back, source first.
    all_hits = await storage.vec_search([1.0, 0.0, 0.0, 0.0], limit=10)
    assert {h.chunk_id for h in all_hits} == {src_ids[0], wiki_ids[0]}
    # Layer-filtered to WIKI: only the wiki chunk surfaces, even though
    # the over-fetch would have pulled the source chunk into the KNN
    # candidate set.
    wiki_hits = await storage.vec_search(
        [1.0, 0.0, 0.0, 0.0], limit=10, layer=Layer.WIKI
    )
    assert [h.chunk_id for h in wiki_hits] == [wiki_ids[0]]
    # Symmetric check.
    src_hits = await storage.vec_search(
        [1.0, 0.0, 0.0, 0.0], limit=10, layer=Layer.SOURCE
    )
    assert [h.chunk_id for h in src_hits] == [src_ids[0]]


async def test_link_graph(storage: Storage) -> None:
    src_doc = _make_doc("wiki/a.md", layer=Layer.WIKI)
    dst_doc = _make_doc("wiki/b.md", layer=Layer.WIKI)
    for d in (src_doc, dst_doc):
        await storage.upsert_document(d)

    link = LinkRecord(
        src_doc_id=src_doc.doc_id,
        dst_path="wiki/b.md",
        link_type=LinkType.WIKILINK,
        anchor=None,
        line=3,
    )
    await storage.upsert_link(link)

    out = await storage.links_from(src_doc.doc_id)
    inb = await storage.links_to("wiki/b.md")
    assert len(out) == 1 and len(inb) == 1
    assert out[0].dst_path == "wiki/b.md"


async def test_wiki_log_append(storage: Storage) -> None:
    ts = time.time()
    await storage.append_wiki_log(
        WikiLogEntry(ts=ts, action="ingest", src="sources/a.md", dst=None, note="hello")
    )
    counts = await storage.counts()
    assert counts.last_wiki_log_ts is not None
    assert counts.last_wiki_log_ts == pytest.approx(ts, abs=1.0)


async def test_wisdom_lifecycle(storage: Storage) -> None:
    doc = _make_doc("wiki/concept.md", layer=Layer.WIKI)
    await storage.upsert_document(doc)

    ts = time.time()
    item = WisdomItem(
        item_id="W-000001",
        kind=WisdomKind.PRINCIPLE,
        status=WisdomStatus.CANDIDATE,
        path="wisdom/_candidates/prefer-determinism.md",
        title="Prefer deterministic scoping",
        body="Use deterministic structure before invoking an LLM.",
        confidence=0.8,
        created_ts=ts,
        approved_ts=None,
    )
    evidence = [
        WisdomEvidence(doc_id=doc.doc_id, excerpt="Karpathy argues...", line=12),
        WisdomEvidence(doc_id=doc.doc_id, excerpt="...", line=18),
    ]
    await storage.put_wisdom(item, evidence)

    candidates = await storage.list_wisdom(status=WisdomStatus.CANDIDATE)
    assert len(candidates) == 1 and candidates[0].item_id == "W-000001"

    await storage.set_wisdom_status("W-000001", WisdomStatus.APPROVED)
    approved = await storage.list_wisdom(status=WisdomStatus.APPROVED)
    assert len(approved) == 1


# ---- Multimedia assets + chunk_asset_refs (Phase F) ----------------------


def _make_asset(
    asset_id: str,
    *,
    original_path: str = "img.png",
    width: int | None = 100,
    height: int | None = 80,
) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        hash=asset_id,
        kind=AssetKind.IMAGE,
        mime="image/png",
        stored_path=f"assets/{asset_id[:2]}/{asset_id[:8]}-img.png",
        original_paths=[original_path],
        bytes=42,
        media_meta=ImageMediaMeta(width=width, height=height),
        created_ts=time.time(),
    )


async def test_asset_upsert_and_get_roundtrip(storage: Storage) -> None:
    a = _make_asset("ab3f12ef" + "0" * 56)
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert fetched.asset_id == a.asset_id
    assert fetched.mime == "image/png"
    assert fetched.original_paths == ["img.png"]
    assert isinstance(fetched.media_meta, ImageMediaMeta)
    assert fetched.media_meta.width == 100
    assert fetched.media_meta.height == 80


async def test_asset_media_meta_none_round_trips(storage: Storage) -> None:
    """SVG / WebP / probe-miss path stores media_meta=None — the round-trip
    must preserve None rather than coerce to an empty ImageMediaMeta()."""
    a = _make_asset("dead0001" + "0" * 56).model_copy(update={"media_meta": None})
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert fetched.media_meta is None


async def test_asset_media_meta_partial_dimensions(storage: Storage) -> None:
    """Some image probes recover only one dimension — the JSON round-trip
    must keep the missing side as None (not 0, not absent-as-error)."""
    a = _make_asset("dead0002" + "0" * 56, width=120, height=None)
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert isinstance(fetched.media_meta, ImageMediaMeta)
    assert fetched.media_meta.width == 120
    assert fetched.media_meta.height is None


async def test_asset_upsert_replaces_with_new_metadata(storage: Storage) -> None:
    a = _make_asset("cafe1234" + "0" * 56, original_path="a.png")
    try:
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement assets yet")

    a2 = a.model_copy(update={"original_paths": ["a.png", "b.png"]})
    await storage.upsert_asset(a2)
    fetched = await storage.get_asset(a.asset_id)
    assert fetched is not None
    assert fetched.original_paths == ["a.png", "b.png"]


async def test_chunk_asset_refs_roundtrip(storage: Storage) -> None:
    doc = _make_doc("sources/with-image.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(
                doc_id=doc.doc_id, seq=0, start=0, end=20, text="Has ![](x.png) inline"
            ),
        ],
    )
    cid = ids[0]
    a1 = _make_asset("aa" + "0" * 62, original_path="x.png")
    a2 = _make_asset("bb" + "0" * 62, original_path="y.png")
    try:
        await storage.upsert_asset(a1)
        await storage.upsert_asset(a2)
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a1.asset_id,
                    ord=0,
                    alt="x",
                    start_in_chunk=4,
                    end_in_chunk=14,
                ),
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a2.asset_id,
                    ord=1,
                    alt="y",
                    start_in_chunk=14,
                    end_in_chunk=20,
                ),
            ],
        )
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    refs = await storage.chunk_asset_refs_for_chunks([cid])
    assert len(refs[cid]) == 2
    assert [r.asset_id for r in refs[cid]] == [a1.asset_id, a2.asset_id]
    assert [r.ord for r in refs[cid]] == [0, 1]

    # Reverse lookup
    by_asset = await storage.chunks_referencing_assets([a1.asset_id, a2.asset_id])
    assert by_asset[a1.asset_id] == [cid]
    assert by_asset[a2.asset_id] == [cid]


async def test_chunk_asset_refs_replaced_on_reupsert(storage: Storage) -> None:
    """Calling replace_chunk_asset_refs again must wipe the previous set."""
    doc = _make_doc("sources/replace.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="hello")],
    )
    cid = ids[0]
    a = _make_asset("dd" + "0" * 62)
    try:
        await storage.upsert_asset(a)
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a.asset_id,
                    ord=0,
                    alt="",
                    start_in_chunk=0,
                    end_in_chunk=5,
                )
            ],
        )
        # Replace with empty list — refs should disappear.
        await storage.replace_chunk_asset_refs(cid, [])
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    assert await storage.chunk_asset_refs_for_chunks([cid]) == {cid: []}


async def test_chunk_asset_refs_cascade_on_chunk_delete(storage: Storage) -> None:
    """When the parent chunk is replaced (DELETE-then-INSERT pattern), the
    chunk_asset_refs rows must cascade so they don't dangle."""
    doc = _make_doc("sources/cascade.md")
    await storage.upsert_document(doc)
    ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="hello")],
    )
    cid = ids[0]
    a = _make_asset("ee" + "0" * 62)
    try:
        await storage.upsert_asset(a)
        await storage.replace_chunk_asset_refs(
            cid,
            [
                ChunkAssetRef(
                    chunk_id=cid,
                    asset_id=a.asset_id,
                    ord=0,
                    alt="",
                    start_in_chunk=0,
                    end_in_chunk=5,
                )
            ],
        )
    except NotSupported:
        pytest.skip("backend doesn't implement chunk_asset_refs yet")

    # Replace chunks for this doc — old chunk row is gone, refs should follow.
    new_ids = await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=10, text="rewritten")],
    )
    new_cid = new_ids[0]
    refs = await storage.chunk_asset_refs_for_chunks([cid, new_cid])
    assert refs == {cid: [], new_cid: []}
    # Asset row still exists (only chunk_asset_refs cascade, not assets).
    assert await storage.get_asset(a.asset_id) is not None


# ---- Embedding versioning + asset embeddings (Phase F) -------------------


async def test_embed_version_upsert_idempotent(storage: Storage) -> None:
    v = EmbeddingVersion(
        provider="gitee_multimodal",
        model="jina-clip-v2",
        revision="",
        dim=768,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    try:
        vid_a = await storage.upsert_embed_version(v)
        vid_b = await storage.upsert_embed_version(v)
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")
    assert vid_a == vid_b
    versions = await storage.list_embed_versions()
    assert sum(1 for x in versions if x.version_id == vid_a) == 1


async def test_embed_version_new_demotes_prior_active(storage: Storage) -> None:
    """A new version with the same modality must mark prior versions of
    that modality as is_active=0; other modalities are untouched."""
    text_v1 = EmbeddingVersion(
        provider="legacy",
        model="text-embedding-3-small",
        dim=1536,
        normalize=True,
        distance="cosine",
        modality="text",
    )
    mm_v1 = EmbeddingVersion(
        provider="gitee_multimodal",
        model="jina-clip-v2",
        dim=768,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    mm_v2 = mm_v1.model_copy(update={"dim": 1024})
    try:
        await storage.upsert_embed_version(text_v1)
        await storage.upsert_embed_version(mm_v1)
        await storage.upsert_embed_version(mm_v2)
    except NotSupported:
        pytest.skip("backend doesn't implement embed versioning yet")

    active_text = await storage.get_active_embed_version(modality="text")
    active_mm = await storage.get_active_embed_version(modality="multimodal")
    assert active_text is not None
    assert active_text.dim == 1536  # text untouched by mm bump
    assert active_mm is not None
    assert active_mm.dim == 1024  # latest multimodal wins


async def test_asset_embeddings_upsert_and_search(storage: Storage) -> None:
    """End-to-end: register version → upsert asset embedding → vec_search
    finds it."""
    v = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v1",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    a = _make_asset("ff" + "0" * 62)
    try:
        version_id = await storage.upsert_embed_version(v)
        await storage.upsert_asset(a)
        await storage.upsert_asset_embeddings(
            [
                AssetEmbeddingRow(
                    asset_id=a.asset_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                )
            ]
        )
        hits = await storage.vec_search_assets(
            [1.0, 0.0, 0.0, 0.0], version_id=version_id, limit=5
        )
    except NotSupported:
        pytest.skip("backend doesn't implement asset embeddings yet")

    assert len(hits) == 1
    assert hits[0].asset_id == a.asset_id
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


async def test_asset_embeddings_dim_mismatch_raises(storage: Storage) -> None:
    v = EmbeddingVersion(
        provider="fake_mm",
        model="fake-mm-v1-dim4",
        dim=4,
        normalize=True,
        distance="cosine",
        modality="multimodal",
    )
    a = _make_asset("12" + "0" * 62)
    try:
        version_id = await storage.upsert_embed_version(v)
        await storage.upsert_asset(a)
    except NotSupported:
        pytest.skip("backend doesn't implement asset embeddings yet")

    from dikw_core.storage.base import StorageError

    with pytest.raises(StorageError):
        await storage.upsert_asset_embeddings(
            [
                AssetEmbeddingRow(
                    asset_id=a.asset_id,
                    version_id=version_id,
                    embedding=[1.0, 0.0],  # wrong dim
                )
            ]
        )
