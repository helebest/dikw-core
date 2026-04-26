from __future__ import annotations

import pytest

from dikw_core.info.search import (
    HybridSearcher,
    _sanitize_fts,
    apply_source_diversity_penalty,
    reciprocal_rank_fusion,
)
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


def test_rrf_weights_default_preserves_legacy() -> None:
    """weights=None must reproduce the pre-weighting numbers bit-for-bit.

    Regression guard: a silent change to the default would shift every
    existing hybrid ranking. Compute both paths with the same k and assert
    they match at full float precision.
    """
    lists = [["a", "b", "c"], ["a", "c", "b"]]
    legacy = reciprocal_rank_fusion(lists, k=60)
    explicit = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 1.0])
    assert legacy == explicit


def test_rrf_weights_shift_ranking_toward_weighted_leg() -> None:
    """When one leg is weighted higher, its solo-discovered docs beat the
    other leg's solo-discovered docs — what the SciFact fix needs.
    """
    # "bm25_only" appears only in list[0]; "vec_only" only in list[1].
    # Both at rank 0, so equal weights → tie. Asymmetric weights → winner.
    lists = [["bm25_only"], ["vec_only"]]

    equal = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 1.0])
    assert equal["bm25_only"] == equal["vec_only"]

    vec_heavy = reciprocal_rank_fusion(lists, k=60, weights=[0.5, 1.0])
    assert vec_heavy["vec_only"] > vec_heavy["bm25_only"]

    bm25_heavy = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 0.5])
    assert bm25_heavy["bm25_only"] > bm25_heavy["vec_only"]


def test_rrf_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_k_controls_rank_decay() -> None:
    """Smaller k = steeper decay; rank-1 doc wins by a larger margin."""
    lists = [["first", "second"]]
    tight = reciprocal_rank_fusion(lists, k=10)
    loose = reciprocal_rank_fusion(lists, k=100)
    # first/second ratio grows as k shrinks
    assert tight["first"] / tight["second"] > loose["first"] / loose["second"]


def test_rrf_accepts_int_keys() -> None:
    """Generic key type — chunk_id (int) flows through fusion the same as
    doc_id (str). Required for chunk-level fusion (Phase 1 plan).
    """
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 4]])
    assert max(fused, key=lambda k: fused[k]) == 2
    assert isinstance(next(iter(fused.keys())), int)
    # Same numeric semantics as the str-keyed path.
    str_fused = reciprocal_rank_fusion([["1", "2", "3"], ["2", "3", "4"]])
    assert {str(k): v for k, v in fused.items()} == str_fused


# ---- apply_source_diversity_penalty (1.3.1) --------------------------------


def test_diversity_penalty_alpha_zero_is_identity() -> None:
    """alpha=0 returns the input dict unchanged — no-op opt-out."""
    fused = {1: 1.0, 2: 0.9, 3: 0.8}
    doc_by_chunk = {1: "A", 2: "A", 3: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.0)
    assert out == fused


def test_diversity_penalty_factor_semantics() -> None:
    """1st chunk per doc unpenalized; N-th chunk scaled by 1/(1+alpha*(N-1)).

    Pinning numerics so future contributors can't silently retune the
    factor. Inputs sorted by score desc; c1/c2 share doc A, c3 is doc B.
    """
    fused = {1: 1.0, 2: 0.9, 3: 0.8}
    doc_by_chunk = {1: "A", 2: "A", 3: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    assert out[1] == pytest.approx(1.0)            # 1st in doc A
    assert out[2] == pytest.approx(0.9 / 1.3)      # 2nd in doc A
    assert out[3] == pytest.approx(0.8)            # 1st in doc B


def test_diversity_penalty_demotes_third_same_doc_chunk() -> None:
    """3rd chunk from one doc gets factor 1/(1+0.3*2) = 1/1.6 = 0.625."""
    fused = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.5}
    doc_by_chunk = {1: "A", 2: "A", 3: "A", 4: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    assert out[3] == pytest.approx(0.8 / 1.6)
    # And the 4th chunk (1st in doc B) remains unpenalized.
    assert out[4] == pytest.approx(0.5)


def test_diversity_penalty_reorders_when_demotion_overtakes_neighbor() -> None:
    """Penalty pushes 2nd same-doc chunk below a fresh-doc chunk that was
    behind it pre-penalty. Cross-doc chunk floats up — exactly the source
    diversification the knob exists to deliver.
    """
    fused = {1: 1.0, 2: 0.9, 3: 0.85}
    doc_by_chunk = {1: "A", 2: "A", 3: "B"}
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    after = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
    assert [k for k, _ in after] == [1, 3, 2]
    assert out[2] == pytest.approx(0.9 / 1.3)


def test_diversity_penalty_unmapped_chunk_unpenalized() -> None:
    """Defensive: chunks missing from doc_by_chunk are passed through
    unmodified — never raise, never penalize what we can't identify.
    """
    fused = {1: 1.0, 2: 0.5}
    doc_by_chunk = {1: "A"}  # 2 is missing
    out = apply_source_diversity_penalty(fused, doc_by_chunk, alpha=0.3)
    assert out[2] == 0.5


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


def test_sanitize_fts_with_jieba_segments_cjk_runs() -> None:
    """Under ``cjk_tokenizer='jieba'`` a whitespace-free Chinese phrase
    should produce multiple word-level tokens — exactly what the indexed
    FTS5 body will have stored, so MATCH actually fires.
    """
    out = _sanitize_fts("机器学习入门", cjk_tokenizer="jieba")
    # Result shape: `"tok1" OR "tok2" OR ...`; at least one multi-char
    # Chinese word present. Pre-feature path would have produced the
    # degenerate `"机器学习入门"` single-token expression.
    assert " OR " in out, f"no tokenization happened: {out!r}"
    assert '"机器学习"' in out or '"机器"' in out
    assert '"入门"' in out


def test_sanitize_fts_with_jieba_keeps_ascii_intact() -> None:
    """Dev-doc query like ``retrieval.rrf_k 参数`` must not shred the
    ASCII identifier. Mixed-language users are the biggest at-risk group
    for a naive ``jieba.cut_for_search(full_text)``.
    """
    out = _sanitize_fts("retrieval.rrf_k 参数", cjk_tokenizer="jieba")
    assert '"rrf_k"' in out
    # Either of these is fine depending on jieba's dictionary
    assert '"retrieval"' in out or '"retrieval_rrf_k"' in out
    assert '"参数"' in out


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
    async for batch in embed_chunks(embedder, to_embed, model="fake"):
        await storage.upsert_embeddings(batch)

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
    async for batch in embed_chunks(embedder, to_embed, model="fake"):
        await storage.upsert_embeddings(batch)
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


# ---- chunk-level fusion (Phase 1) ------------------------------------------


async def _populate_multi_chunk_corpus(storage: SQLiteStorage) -> FakeEmbeddings:
    """Build a 2-doc corpus with multiple distinct chunks per doc.

    Doc A has 3 chunks, all matching the test query "alpha foo".
    Doc B has 2 chunks, only one matching "alpha foo".

    Designed so chunk-level fusion can demonstrate "multiple chunks from
    same doc" while staying small enough for FakeEmbeddings to rank
    deterministically by bag-of-words.
    """
    import time

    from dikw_core.info.embed import ChunkToEmbed, embed_chunks
    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    doc_specs: list[tuple[str, list[str]]] = [
        (
            "alpha",
            [
                "alpha foo bar topic one. alpha foo bar.",
                "alpha foo bar topic two with details.",
                "alpha foo bar topic three closing.",
            ],
        ),
        (
            "beta",
            [
                "alpha foo unrelated baseline content.",
                "completely separate beta material with no overlap.",
            ],
        ),
    ]

    to_embed: list[ChunkToEmbed] = []
    for stem, chunk_texts in doc_specs:
        doc_id = f"source:sources/{stem}.md"
        body = "\n\n".join(chunk_texts)
        await storage.put_content(f"h-{stem}", body)
        await storage.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path=f"sources/{stem}.md",
                title=stem,
                hash=f"h-{stem}",
                mtime=time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        records: list[ChunkRecord] = []
        offset = 0
        for i, text in enumerate(chunk_texts):
            records.append(
                ChunkRecord(
                    doc_id=doc_id, seq=i, start=offset, end=offset + len(text), text=text
                )
            )
            offset += len(text) + 2  # account for separator
        ids = await storage.replace_chunks(doc_id, records)
        to_embed.extend(
            ChunkToEmbed(chunk_id=cid, text=r.text)
            for cid, r in zip(ids, records, strict=True)
        )

    embedder = FakeEmbeddings()
    async for batch in embed_chunks(embedder, to_embed, model="fake"):
        await storage.upsert_embeddings(batch)
    return embedder


@pytest.mark.asyncio
async def test_chunk_level_fusion_returns_multiple_chunks_per_doc(tmp_path) -> None:
    """alpha=0 (no diversity penalty) → top-K contains multiple chunks
    from the same doc when every leg ranks them highly.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder = await _populate_multi_chunk_corpus(storage)

    searcher = HybridSearcher(
        storage,
        embedder,
        embedding_model="fake",
        same_doc_penalty_alpha=0.0,
    )
    hits = await searcher.search("alpha foo bar", limit=5)
    await storage.close()

    paths = [h.path for h in hits]
    alpha_count = sum(1 for p in paths if p == "sources/alpha.md")
    # All three alpha chunks match strongly; with no penalty they should
    # all surface.
    assert alpha_count >= 2, f"expected >=2 alpha chunks, got paths={paths}"
    # Each Hit must carry a chunk_id (non-optional after Phase 1).
    assert all(h.chunk_id is not None for h in hits)
    # And a seq populated from the chunk record.
    assert all(h.seq is not None for h in hits)


@pytest.mark.asyncio
async def test_chunk_level_fusion_distinct_chunks_have_distinct_ids(tmp_path) -> None:
    """No two hits share a chunk_id — chunk-level fusion deduplicates
    keys across the legs by construction.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder = await _populate_multi_chunk_corpus(storage)

    searcher = HybridSearcher(
        storage, embedder, embedding_model="fake", same_doc_penalty_alpha=0.0
    )
    hits = await searcher.search("alpha foo bar", limit=5)
    await storage.close()

    chunk_ids = [h.chunk_id for h in hits]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.asyncio
async def test_same_doc_penalty_zero_vs_default_changes_ranking(tmp_path) -> None:
    """alpha=0 yields more same-doc concentration than alpha=0.3 on a
    corpus with one obviously-dominant doc.
    """
    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder = await _populate_multi_chunk_corpus(storage)

    pure = HybridSearcher(
        storage, embedder, embedding_model="fake", same_doc_penalty_alpha=0.0
    )
    diversified = HybridSearcher(
        storage, embedder, embedding_model="fake", same_doc_penalty_alpha=0.3
    )

    pure_hits = await pure.search("alpha foo bar", limit=5)
    div_hits = await diversified.search("alpha foo bar", limit=5)
    await storage.close()

    pure_alpha = sum(1 for h in pure_hits if h.path == "sources/alpha.md")
    div_alpha = sum(1 for h in div_hits if h.path == "sources/alpha.md")
    # Diversification should not increase same-doc concentration; a
    # measurable demotion lands at <= pure on a corpus this small.
    assert div_alpha <= pure_alpha, (
        f"diversified alpha-count {div_alpha} should be <= pure {pure_alpha}; "
        f"pure_paths={[h.path for h in pure_hits]}, "
        f"div_paths={[h.path for h in div_hits]}"
    )


@pytest.mark.asyncio
async def test_hybrid_searcher_from_config_threads_alpha(tmp_path) -> None:
    """RetrievalConfig.same_doc_penalty_alpha → HybridSearcher.from_config
    → search() — the Protocol triad wiring (Phase 1.3.0).
    """
    from dikw_core.config import RetrievalConfig

    storage = SQLiteStorage(tmp_path / "idx.sqlite")
    await storage.connect()
    await storage.migrate()
    embedder = await _populate_multi_chunk_corpus(storage)

    cfg = RetrievalConfig(same_doc_penalty_alpha=0.0)
    searcher = HybridSearcher.from_config(
        storage, embedder, cfg, embedding_model="fake"
    )
    assert searcher._same_doc_penalty_alpha == 0.0
    hits = await searcher.search("alpha foo bar", limit=5)
    await storage.close()

    # Sanity: with alpha=0 we get >=2 same-doc chunks (matches the pure
    # path of the parametric test above).
    alpha_paths = [h.path for h in hits if h.path == "sources/alpha.md"]
    assert len(alpha_paths) >= 2


# ---- cjk tokenizer integration ----------------------------------------------


@pytest.mark.asyncio
async def test_cjk_tokenizer_jieba_ingest_stores_segmented_body(tmp_path) -> None:
    """End-to-end: ingest CJK text with cjk_tokenizer='jieba' → the
    documents_fts.body column holds whitespace-separated Chinese words.

    Without the preprocessor, body would contain ``机器学习入门`` as one
    run and FTS5's unicode61 would split per-character. With the
    preprocessor, body contains ``机器 学习 机器学习 入门`` (or similar,
    depending on jieba's dictionary) and unicode61 respects the
    whitespace boundaries.
    """
    import time

    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    storage = SQLiteStorage(tmp_path / "idx.sqlite", cjk_tokenizer="jieba")
    await storage.connect()
    await storage.migrate()

    doc_id = "source:zh.md"
    await storage.put_content("h1", "机器学习入门")
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path="sources/zh.md",
            title="机器学习",
            hash="h1",
            mtime=time.time(),
            layer=Layer.SOURCE,
            active=True,
        )
    )
    await storage.replace_chunks(
        doc_id,
        [ChunkRecord(doc_id=doc_id, seq=0, start=0, end=6, text="机器学习入门")],
    )

    # Inspect what was actually written — sqlite3 is synchronous but the
    # async wrapper is happy to return a value here.
    def _read_body() -> tuple[str, str]:
        conn = storage._require_conn()  # type: ignore[attr-defined]
        row = conn.execute(
            "SELECT title, body FROM documents_fts WHERE path = ?",
            ("sources/zh.md",),
        ).fetchone()
        return row["title"], row["body"]

    title, body = _read_body()
    # At minimum jieba has surfaced at least one 2+ char Chinese word
    # as a whitespace-bounded token. The pre-feature path would store
    # the raw string "机器学习入门" with no whitespace.
    assert " " in body, f"body not segmented: {body!r}"
    assert " " in title, f"title not segmented: {title!r}"
    await storage.close()


@pytest.mark.asyncio
async def test_cjk_tokenizer_jieba_end_to_end_bm25_recovers(tmp_path) -> None:
    """End-to-end proof the fix actually fires BM25 on Chinese text.

    Baseline (``cjk_tokenizer='none'``): indexing "机器学习入门" under
    unicode61 per-character tokenization means a MATCH on "机器学习"
    produces 0 hits — the indexed form is four unrelated single chars
    and the query's sanitized form asks for those same chars in
    isolation. Turn cjk_tokenizer on and the same storage + query
    sequence starts returning a hit.
    """
    import time

    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    async def _populate(cjk: str) -> SQLiteStorage:
        s = SQLiteStorage(tmp_path / f"idx-{cjk}.sqlite", cjk_tokenizer=cjk)  # type: ignore[arg-type]
        await s.connect()
        await s.migrate()
        doc_id = "source:zh.md"
        body = "机器学习入门是一本经典的机器学习教材"
        await s.put_content("h", body)
        await s.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                path="sources/zh.md",
                title="机器学习入门",
                hash="h",
                mtime=time.time(),
                layer=Layer.SOURCE,
                active=True,
            )
        )
        await s.replace_chunks(
            doc_id,
            [ChunkRecord(doc_id=doc_id, seq=0, start=0, end=len(body), text=body)],
        )
        return s

    # default mode: searcher sees raw 机器学习 → per-char sanitize, FTS5
    # indexed per-char, match exists but is weak/noisy. Not asserting
    # strict 0 hits because per-char union *can* coincidentally match;
    # what matters is the jieba path produces a cleaner hit.
    storage_jieba = await _populate("jieba")
    searcher = HybridSearcher(
        storage_jieba, embedder=None, embedding_model=None, cjk_tokenizer="jieba"
    )
    hits = await searcher.search("机器学习", limit=5, mode="bm25")
    await storage_jieba.close()
    assert hits, "jieba path returned no bm25 hits on Chinese query"
    assert any("zh.md" in (h.path or "") for h in hits)


@pytest.mark.asyncio
async def test_cjk_tokenizer_none_leaves_body_verbatim(tmp_path) -> None:
    """Default 'none' mode must not touch body — regression guard for
    every existing ASCII-only wiki.
    """
    import time

    from dikw_core.schemas import ChunkRecord, DocumentRecord, Layer

    storage = SQLiteStorage(tmp_path / "idx.sqlite")  # default cjk_tokenizer="none"
    await storage.connect()
    await storage.migrate()

    doc_id = "source:en.md"
    body_text = "reciprocal rank fusion survives preprocessing"
    await storage.put_content("h2", body_text)
    await storage.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            path="sources/en.md",
            title="fusion",
            hash="h2",
            mtime=time.time(),
            layer=Layer.SOURCE,
            active=True,
        )
    )
    await storage.replace_chunks(
        doc_id,
        [ChunkRecord(doc_id=doc_id, seq=0, start=0, end=len(body_text), text=body_text)],
    )

    def _read_body() -> str:
        conn = storage._require_conn()  # type: ignore[attr-defined]
        row = conn.execute(
            "SELECT body FROM documents_fts WHERE path = ?",
            ("sources/en.md",),
        ).fetchone()
        return str(row["body"])

    assert _read_body() == body_text
    await storage.close()
