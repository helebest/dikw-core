"""Tests for the asset channel in hybrid search.

The v1 HybridSearcher fuses three retrieval legs into one chunk-level
ranking via RRF (k=60):

  * BM25 over chunk text (existing)
  * Vector search over chunk vectors (existing)
  * Vector search over asset vectors → parent chunks (NEW)

The asset leg lets a text query like "transformer architecture diagram"
promote chunks that *reference* a matching image, even when the chunk
text itself doesn't say "transformer". Each returned Hit carries the
``asset_refs`` it touches so downstream consumers (CLI, MCP, LLM
synthesis) can render or cite them.

Backward-compat invariant: searches that don't pass a multimodal
embedder behave identically to before — no new failures, no new
requests against the asset leg.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dikw_core.info.search import HybridSearcher, MultimodalSearch
from dikw_core.schemas import (
    AssetEmbeddingRow,
    AssetKind,
    AssetRecord,
    ChunkAssetRef,
    ChunkRecord,
    DocumentRecord,
    EmbeddingVersion,
    Layer,
    MultimodalInput,
)
from dikw_core.storage.sqlite import SQLiteStorage

# ---- Fixtures ------------------------------------------------------------


@pytest.fixture
async def storage(tmp_path: Path) -> AsyncIterator[SQLiteStorage]:
    s = SQLiteStorage(tmp_path / "search.sqlite")
    await s.connect()
    await s.migrate()
    try:
        yield s
    finally:
        await s.close()


@dataclass
class FixedVectorMM:
    """Deterministic multimodal provider that always returns the same
    vector regardless of input — for tests that need the query vector to
    match a pre-set asset embedding."""

    vector: list[float]
    last_inputs: list[MultimodalInput] = field(default_factory=list, init=False)

    async def embed(
        self, inputs: list[MultimodalInput], *, model: str
    ) -> list[list[float]]:
        _ = model
        self.last_inputs = list(inputs)
        return [list(self.vector) for _ in inputs]


def _doc(path: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=f"doc::{path}",
        path=path,
        title=path.rsplit("/", 1)[-1],
        hash=f"hash-{path}",
        mtime=time.time(),
        layer=Layer.SOURCE,
        active=True,
    )


def _asset(asset_id: str) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        hash=asset_id,
        kind=AssetKind.IMAGE,
        mime="image/png",
        stored_path=f"assets/{asset_id[:2]}/{asset_id[:8]}-x.png",
        original_paths=["x.png"],
        bytes=1,
        width=None,
        height=None,
        caption=None,
        caption_model=None,
        created_ts=time.time(),
    )


# ---- Tests ---------------------------------------------------------------


async def test_search_without_mm_embedder_unchanged(
    storage: SQLiteStorage,
) -> None:
    """Backward-compat: passing no multimodal config means no asset
    channel; existing 2-leg behavior is preserved exactly."""
    doc = _doc("sources/a.md")
    await storage.put_content(doc.hash, "x")
    await storage.upsert_document(doc)
    await storage.replace_chunks(
        doc.doc_id,
        [ChunkRecord(doc_id=doc.doc_id, seq=0, start=0, end=20, text="lemma alpha beta")],
    )
    searcher = HybridSearcher(storage, embedder=None)
    hits = await searcher.search("alpha", limit=5)
    assert any(h.doc_id == doc.doc_id for h in hits)
    # No asset_refs because no multimodal config.
    assert all(getattr(h, "asset_refs", []) == [] for h in hits)


async def test_asset_hit_promotes_parent_chunk(
    storage: SQLiteStorage,
) -> None:
    """The headline cross-modal case: a chunk whose body has *no* matching
    text gets retrieved because the image it references matches the
    query vector."""
    doc = _doc("sources/visual.md")
    await storage.put_content(doc.hash, "x")
    await storage.upsert_document(doc)
    chunk_ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(
                doc_id=doc.doc_id,
                seq=0,
                start=0,
                end=30,
                text="See diagram below ![](arch.png)",
            )
        ],
    )
    cid = chunk_ids[0]
    a = _asset("aa" + "0" * 62)
    await storage.upsert_asset(a)
    await storage.replace_chunk_asset_refs(
        cid,
        [
            ChunkAssetRef(
                chunk_id=cid,
                asset_id=a.asset_id,
                ord=0,
                alt="",
                start_in_chunk=18,
                end_in_chunk=30,
            )
        ],
    )
    version_id = await storage.upsert_embed_version(
        EmbeddingVersion(
            provider="fake",
            model="fake-mm",
            dim=4,
            normalize=True,
            distance="cosine",
            modality="multimodal",
        )
    )
    target_vec = [1.0, 0.0, 0.0, 0.0]
    await storage.upsert_asset_embeddings(
        [
            AssetEmbeddingRow(
                asset_id=a.asset_id, version_id=version_id, embedding=target_vec
            )
        ]
    )

    mm = FixedVectorMM(vector=target_vec)
    searcher = HybridSearcher(
        storage,
        embedder=None,
        multimodal=MultimodalSearch(
            embedder=mm, model="fake-mm", asset_version_id=version_id
        ),
    )
    # Query text doesn't include any of the chunk's tokens; only the
    # asset vector match should pull the chunk in.
    hits = await searcher.search("transformer attention pattern", limit=5)
    assert any(h.chunk_id == cid for h in hits), (
        f"asset-promoted chunk should appear in results, got {hits}"
    )
    promoted = next(h for h in hits if h.chunk_id == cid)
    assert len(promoted.asset_refs) == 1
    assert promoted.asset_refs[0].asset_id == a.asset_id
    # And the multimodal provider was actually called for query embedding.
    assert mm.last_inputs and mm.last_inputs[0].text == "transformer attention pattern"


async def test_asset_refs_attached_for_text_match_too(
    storage: SQLiteStorage,
) -> None:
    """Even when the chunk is retrieved by FTS / vec, any assets it
    references must come back attached to the Hit so the consumer can
    render them."""
    doc = _doc("sources/text.md")
    await storage.put_content(doc.hash, "x")
    await storage.upsert_document(doc)
    chunk_ids = await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(
                doc_id=doc.doc_id,
                seq=0,
                start=0,
                end=30,
                text="alpha beta gamma ![](x.png)",
            )
        ],
    )
    cid = chunk_ids[0]
    a = _asset("bb" + "0" * 62)
    await storage.upsert_asset(a)
    await storage.replace_chunk_asset_refs(
        cid,
        [
            ChunkAssetRef(
                chunk_id=cid,
                asset_id=a.asset_id,
                ord=0,
                alt="",
                start_in_chunk=17,
                end_in_chunk=30,
            )
        ],
    )
    version_id = await storage.upsert_embed_version(
        EmbeddingVersion(
            provider="fake",
            model="fake-mm",
            dim=4,
            normalize=True,
            distance="cosine",
            modality="multimodal",
        )
    )
    mm = FixedVectorMM(vector=[0.0, 0.0, 0.0, 1.0])
    searcher = HybridSearcher(
        storage,
        embedder=None,
        multimodal=MultimodalSearch(
            embedder=mm, model="fake-mm", asset_version_id=version_id
        ),
    )
    hits = await searcher.search("alpha", limit=5)
    assert hits, "FTS should find the chunk via 'alpha'"
    h = next(x for x in hits if x.chunk_id == cid)
    assert len(h.asset_refs) == 1
    assert h.asset_refs[0].asset_id == a.asset_id


async def test_search_handles_no_asset_index(storage: SQLiteStorage) -> None:
    """When asset indexing is configured but no asset embeddings exist
    yet, search should still succeed (degrade to BM25 + chunk-vec only)."""
    doc = _doc("sources/no-assets.md")
    await storage.put_content(doc.hash, "x")
    await storage.upsert_document(doc)
    await storage.replace_chunks(
        doc.doc_id,
        [
            ChunkRecord(
                doc_id=doc.doc_id, seq=0, start=0, end=20, text="hello world content"
            )
        ],
    )
    version_id = await storage.upsert_embed_version(
        EmbeddingVersion(
            provider="fake",
            model="fake-mm",
            dim=4,
            normalize=True,
            distance="cosine",
            modality="multimodal",
        )
    )
    mm = FixedVectorMM(vector=[0.5, 0.5, 0.5, 0.5])
    searcher = HybridSearcher(
        storage,
        embedder=None,
        multimodal=MultimodalSearch(
            embedder=mm, model="fake-mm", asset_version_id=version_id
        ),
    )
    hits = await searcher.search("hello", limit=5)
    assert any(h.doc_id == doc.doc_id for h in hits)
    # No asset refs since none exist for this chunk.
    assert all(h.asset_refs == [] for h in hits)
