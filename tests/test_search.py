from __future__ import annotations

import pytest

from dikw_core.info.search import HybridSearcher, reciprocal_rank_fusion
from dikw_core.storage.sqlite import SQLiteStorage

from .fakes import FakeEmbeddings


def test_rrf_favors_consensus() -> None:
    # Doc "a" is #1 in both lists → top score
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert max(fused, key=lambda k: fused[k]) == "a"
    # "b" and "c" tied on consensus weight; ordering should be stable-equal-ish
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]


def test_rrf_handles_disjoint_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
    assert set(fused) == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_hybrid_search_returns_hits(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()

    # Load the fixture corpus into the storage engine.
    import time
    from pathlib import Path

    from dikw_core.data.backends.markdown import parse_file
    from dikw_core.info.chunk import chunk_markdown
    from dikw_core.info.embed import ChunkToEmbed, embed_chunks
    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    fixtures = Path(__file__).parent / "fixtures" / "notes"
    to_embed: list[ChunkToEmbed] = []
    for md_path in sorted(fixtures.glob("*.md")):
        parsed = parse_file(md_path, rel_path=f"sources/notes/{md_path.name}")
        doc_id = f"source:sources/notes/{md_path.name}"
        await storage.put_content(parsed.hash, parsed.body)
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path=parsed.path,
                title=parsed.title,
                hash=parsed.hash,
                mtime=parsed.mtime or time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        chunks = chunk_markdown(parsed.body)
        records = [
            ChunkRecord(doc_id=doc_id, seq=c.seq, start=c.start, end=c.end, text=c.text)
            for c in chunks
        ]
        ids = await storage.replace_chunks(doc_id, records)
        to_embed.extend(ChunkToEmbed(chunk_id=cid, text=r.text) for cid, r in zip(ids, records, strict=True))

    embedder = FakeEmbeddings()
    rows = await embed_chunks(embedder, to_embed, model="fake")
    await storage.upsert_embeddings(rows)

    searcher = HybridSearcher(storage, embedder, embedding_model="fake")
    hits = await searcher.search("reciprocal rank fusion", limit=3)
    assert hits, "no hits returned"
    assert any("retrieval.md" in (h.path or "") for h in hits)

    # the question about DIKW pyramid should surface dikw.md
    hits = await searcher.search("DIKW pyramid", limit=3)
    assert any("dikw.md" in (h.path or "") for h in hits)

    await storage.close()
