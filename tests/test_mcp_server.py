"""MCP server handler tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import load_config
from dikw_core.mcp_server import _doc_read
from dikw_core.schemas import Layer
from dikw_core.storage import build_storage

from .fakes import FakeEmbeddings, init_test_wiki


@pytest.fixture()
async def wiki_with_chunks(tmp_path: Path) -> tuple[Path, int, str]:
    """A wiki with one source doc ingested. Returns (wiki, chunk_id, chunk_text)."""
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="mcp test wiki")
    src_dir = wiki / "sources"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "doc.md").write_text(
        "# Title\n\nFirst paragraph about alpha and beta.\n", encoding="utf-8"
    )

    await api.ingest(wiki, embedder=FakeEmbeddings())

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    try:
        hits = await storage.fts_search("alpha", limit=5, layer=Layer.SOURCE)
        assert hits and hits[0].chunk_id is not None
        chunk = await storage.get_chunk(hits[0].chunk_id)
        assert chunk is not None
    finally:
        await storage.close()
    return wiki, chunk.chunk_id, chunk.text  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_doc_read_by_chunk_id_returns_chunk_text(
    wiki_with_chunks: tuple[Path, int, str],
) -> None:
    wiki, chunk_id, expected_text = wiki_with_chunks

    text = await _doc_read({"chunk_id": chunk_id, "wiki_path": str(wiki)})

    assert text == expected_text


@pytest.mark.asyncio
async def test_doc_read_missing_chunk_id_raises(
    wiki_with_chunks: tuple[Path, int, str],
) -> None:
    wiki, chunk_id, _ = wiki_with_chunks
    bogus = chunk_id + 999

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

    text = await _doc_read({"path": "sources/hello.md", "wiki_path": str(wiki)})

    assert text == body
