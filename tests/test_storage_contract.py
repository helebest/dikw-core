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
    ChunkAssetRef,
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    EmbeddingVersion,
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
    await storage.put_content(doc.hash, "hello world")
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


async def test_chunks_and_fts_search(storage: Storage) -> None:
    doc = _make_doc("sources/chunked.md")
    await storage.put_content(doc.hash, "x" * 10)
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
    await storage.put_content(doc.hash, "body")
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


async def test_link_graph(storage: Storage) -> None:
    src_doc = _make_doc("wiki/a.md", layer=Layer.WIKI)
    dst_doc = _make_doc("wiki/b.md", layer=Layer.WIKI)
    for d in (src_doc, dst_doc):
        await storage.put_content(d.hash, "x")
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
    await storage.put_content(doc.hash, "body")
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


def _make_asset(asset_id: str, *, original_path: str = "img.png") -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        hash=asset_id,
        kind=AssetKind.IMAGE,
        mime="image/png",
        stored_path=f"assets/{asset_id[:2]}/{asset_id[:8]}-img.png",
        original_paths=[original_path],
        bytes=42,
        width=100,
        height=80,
        caption=None,
        caption_model=None,
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
    assert fetched.width == 100


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
    await storage.put_content(doc.hash, "x")
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
    await storage.put_content(doc.hash, "x")
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
    await storage.put_content(doc.hash, "x")
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
