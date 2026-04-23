from __future__ import annotations

import pytest

from dikw_core.info.search import HybridSearcher, _sanitize_fts, reciprocal_rank_fusion
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


# ---- _sanitize_fts ----------------------------------------------------------


def test_sanitize_fts_or_joins_word_tokens() -> None:
    # Natural-language query → bag-of-words OR. The Phase 1 phrase-quote
    # implementation returned `"foo bar baz"` and matched ~nothing in real
    # corpora; this is the regression guard.
    assert _sanitize_fts("foo bar baz") == '"foo" OR "bar" OR "baz"'


def test_sanitize_fts_strips_punctuation() -> None:
    # Hyphens, question marks, parentheses → whitespace; tokens preserved.
    assert _sanitize_fts("DIKW-core (what is it?)") == (
        '"DIKW" OR "core" OR "what" OR "is" OR "it"'
    )


def test_sanitize_fts_drops_fts5_reserved_tokens() -> None:
    # AND / OR / NOT / NEAR are FTS5 operators — silently dropping them
    # avoids syntax errors when a user query happens to contain them.
    assert _sanitize_fts("cats AND dogs OR birds") == '"cats" OR "dogs" OR "birds"'
    assert _sanitize_fts("just NOT a phrase") == '"just" OR "a" OR "phrase"'


def test_sanitize_fts_preserves_underscores() -> None:
    # Identifier-like tokens (e.g., "expect_any") survive intact because
    # \w matches underscore.
    assert _sanitize_fts("test expect_any field") == (
        '"test" OR "expect_any" OR "field"'
    )


def test_sanitize_fts_preserves_cjk() -> None:
    # CJK Unified Ideographs survive the punctuation strip. Whitespace-
    # separated CJK tokens are still tokenized at whitespace; running text
    # without spaces becomes a single token (a known FTS5 limitation, not
    # something this sanitizer can fix without a real CJK tokenizer).
    assert _sanitize_fts("机器 学习 入门") == '"机器" OR "学习" OR "入门"'


def test_sanitize_fts_empty_or_punctuation_only_returns_empty() -> None:
    assert _sanitize_fts("") == ""
    assert _sanitize_fts("   ") == ""
    assert _sanitize_fts("???") == ""


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


# ---- retrieval mode (bm25 / vector / hybrid) --------------------------------


async def _populate_fixture_corpus(storage):
    """Helper: load tests/fixtures/notes/ into ``storage`` (FakeEmbeddings)."""
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
        to_embed.extend(
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(ids, records, strict=True)
        )

    embedder = FakeEmbeddings()
    rows = await embed_chunks(embedder, to_embed, model="fake")
    await storage.upsert_embeddings(rows)
    return embedder


@pytest.mark.asyncio
async def test_mode_bm25_skips_vector_leg(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder = await _populate_fixture_corpus(storage)

    # Spy on vec_search to verify it never fires in bm25 mode.
    real_vec = storage.vec_search
    calls = {"vec": 0}

    async def spy_vec(*args, **kwargs):
        calls["vec"] += 1
        return await real_vec(*args, **kwargs)

    storage.vec_search = spy_vec  # type: ignore[method-assign]

    searcher = HybridSearcher(storage, embedder, embedding_model="fake")
    hits = await searcher.search("DIKW pyramid", limit=3, mode="bm25")
    assert hits, "bm25 mode returned no hits"
    assert calls["vec"] == 0, "vec_search must not run in bm25 mode"

    await storage.close()


@pytest.mark.asyncio
async def test_mode_vector_skips_fts_leg(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder = await _populate_fixture_corpus(storage)

    real_fts = storage.fts_search
    calls = {"fts": 0}

    async def spy_fts(*args, **kwargs):
        calls["fts"] += 1
        return await real_fts(*args, **kwargs)

    storage.fts_search = spy_fts  # type: ignore[method-assign]

    searcher = HybridSearcher(storage, embedder, embedding_model="fake")
    hits = await searcher.search("DIKW pyramid", limit=3, mode="vector")
    assert hits, "vector mode returned no hits"
    assert calls["fts"] == 0, "fts_search must not run in vector mode"

    await storage.close()


@pytest.mark.asyncio
async def test_mode_hybrid_runs_both_legs(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder = await _populate_fixture_corpus(storage)

    real_fts = storage.fts_search
    real_vec = storage.vec_search
    calls = {"fts": 0, "vec": 0}

    async def spy_fts(*args, **kwargs):
        calls["fts"] += 1
        return await real_fts(*args, **kwargs)

    async def spy_vec(*args, **kwargs):
        calls["vec"] += 1
        return await real_vec(*args, **kwargs)

    storage.fts_search = spy_fts  # type: ignore[method-assign]
    storage.vec_search = spy_vec  # type: ignore[method-assign]

    searcher = HybridSearcher(storage, embedder, embedding_model="fake")
    hits = await searcher.search("DIKW pyramid", limit=3, mode="hybrid")
    assert hits
    assert calls["fts"] == 1 and calls["vec"] == 1

    await storage.close()


@pytest.mark.asyncio
async def test_mode_bm25_works_without_embedder(tmp_path) -> None:
    """bm25 mode should not require an embedder at all."""
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    await _populate_fixture_corpus(storage)

    searcher = HybridSearcher(storage, embedder=None, embedding_model=None)
    hits = await searcher.search("DIKW pyramid", limit=3, mode="bm25")
    assert hits

    await storage.close()


@pytest.mark.asyncio
async def test_mode_vector_requires_embedder(tmp_path) -> None:
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    await _populate_fixture_corpus(storage)

    searcher = HybridSearcher(storage, embedder=None, embedding_model=None)
    with pytest.raises(ValueError, match="vector"):
        await searcher.search("anything", limit=3, mode="vector")

    await storage.close()
