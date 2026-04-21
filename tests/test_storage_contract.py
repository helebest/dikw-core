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
    ChunkRecord,
    DocumentRecord,
    EmbeddingRow,
    Layer,
    LinkRecord,
    LinkType,
    WikiLogEntry,
    WisdomEvidence,
    WisdomItem,
    WisdomKind,
    WisdomStatus,
)
from dikw_core.storage.base import Storage
from dikw_core.storage.sqlite import SQLiteStorage


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param(
            "postgres",
            id="postgres",
            marks=pytest.mark.skipif(
                not os.environ.get("DIKW_TEST_POSTGRES_DSN"),
                reason="Phase 5: Postgres adapter not implemented yet",
            ),
        ),
        pytest.param(
            "filesystem",
            id="filesystem",
            marks=pytest.mark.skip(reason="Phase 5: filesystem adapter not implemented"),
        ),
    ]
)
async def storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Storage]:
    backend = request.param
    if backend == "sqlite":
        s: Storage = SQLiteStorage(tmp_path / "index.sqlite")
    else:
        raise RuntimeError(f"unreachable: adapter {backend} should be skipped")
    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
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
