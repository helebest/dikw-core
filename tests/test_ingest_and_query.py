from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api

from .fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent / "fixtures" / "notes"


@pytest.fixture()
def wiki_with_fixtures(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    api.init_wiki(wiki, description="ingest-query test wiki")
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_ingest_is_idempotent_and_fills_storage(wiki_with_fixtures: Path) -> None:
    embedder = FakeEmbeddings()
    report = await api.ingest(wiki_with_fixtures, embedder=embedder)
    assert report.scanned == 3
    assert report.added == 3
    assert report.updated == 0
    assert report.unchanged == 0
    assert report.chunks >= 3
    assert report.embedded >= 3

    # re-run: all files should now be unchanged
    report2 = await api.ingest(wiki_with_fixtures, embedder=embedder)
    assert report2.scanned == 3
    assert report2.unchanged == 3
    assert report2.added == 0
    assert report2.chunks == 0


@pytest.mark.asyncio
async def test_query_returns_answer_with_citations(wiki_with_fixtures: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    llm = FakeLLM(response_text="Deterministic scoping matters [#1].")
    result = await api.query(
        "what does Karpathy say about scoping?",
        wiki_with_fixtures,
        limit=3,
        llm=llm,
        embedder=embedder,
    )
    assert result.answer.startswith("Deterministic")
    assert result.citations, "expected at least one citation"
    assert llm.last_user is not None and "QUESTION" in llm.last_user
    assert any("karpathy" in c.path.lower() for c in result.citations)


@pytest.mark.asyncio
async def test_query_returns_no_citations_when_corpus_empty(tmp_path: Path) -> None:
    wiki = tmp_path / "empty"
    api.init_wiki(wiki)
    llm = FakeLLM()
    embedder = FakeEmbeddings()
    result = await api.query("anything", wiki, llm=llm, embedder=embedder)
    assert result.citations == []
    assert "ingest sources" in result.answer or "rephrase" in result.answer
    # LLM shouldn't even be asked when no excerpts exist
    assert llm.last_user is None
