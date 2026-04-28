"""MCP server handler tests.

Drives the ``doc.read`` resolver directly via its module-level helper —
spinning up the MCP transport just to call one tool would add 50 lines
of asyncio plumbing for zero coverage gain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import dump_config_yaml, load_config
from dikw_core.mcp_server import _doc_read
from dikw_core.schemas import Layer
from dikw_core.storage import build_storage

from .fakes import FakeEmbeddings, init_test_wiki


@pytest.fixture()
async def wiki_with_chunks(tmp_path: Path) -> tuple[Path, list[int], str]:
    """A wiki with one source doc ingested. Returns (wiki, chunk_ids, doc_text)."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="mcp test wiki")
    src_dir = wiki / "sources"
    src_dir.mkdir(exist_ok=True)
    body = "# Title\n\nFirst paragraph about alpha and beta.\n"
    (src_dir / "doc.md").write_text(body, encoding="utf-8")

    embedder = FakeEmbeddings()
    await api.ingest(wiki, embedder=embedder)

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    try:
        docs = list(await storage.list_documents(layer=Layer.SOURCE, active=True))
        assert docs, "ingest should have produced at least one source doc"
        # FTS lookup is the public surface for getting chunk_ids back out
        # of an ingested wiki — mirrors the doc.search → doc.read flow.
        hits = await storage.fts_search("alpha", limit=5, layer=Layer.SOURCE)
    finally:
        await storage.close()
    chunk_ids = [h.chunk_id for h in hits if h.chunk_id is not None]
    assert chunk_ids, "FTS should surface at least one chunk_id"
    return wiki, chunk_ids, body


@pytest.mark.asyncio
async def test_doc_read_by_chunk_id_returns_chunk_text(
    wiki_with_chunks: tuple[Path, list[int], str],
) -> None:
    wiki, chunk_ids, _body = wiki_with_chunks

    text = await _doc_read({"chunk_id": chunk_ids[0], "wiki_path": str(wiki)})

    assert "alpha" in text.lower() or "title" in text.lower()


@pytest.mark.asyncio
async def test_doc_read_missing_chunk_id_raises(
    wiki_with_chunks: tuple[Path, list[int], str],
) -> None:
    wiki, chunk_ids, _ = wiki_with_chunks
    bogus = max(chunk_ids) + 999

    with pytest.raises(ValueError, match="chunk_id"):
        await _doc_read({"chunk_id": bogus, "wiki_path": str(wiki)})


@pytest.mark.asyncio
async def test_doc_read_requires_path_or_chunk_id(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="mcp empty wiki")

    with pytest.raises(ValueError, match=r"path.*chunk_id"):
        await _doc_read({"wiki_path": str(wiki)})


@pytest.mark.asyncio
async def test_doc_read_by_path_unchanged(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="mcp path wiki")
    body = "# Hello\n\nplain body content.\n"
    (wiki / "sources").mkdir(exist_ok=True)
    (wiki / "sources" / "hello.md").write_text(body, encoding="utf-8")
    # ensure dikw.yml stays consistent for load_wiki
    cfg = load_config(wiki / "dikw.yml")
    (wiki / "dikw.yml").write_text(dump_config_yaml(cfg), encoding="utf-8")

    text = await _doc_read({"path": "sources/hello.md", "wiki_path": str(wiki)})

    assert text == body
