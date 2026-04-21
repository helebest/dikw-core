from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api

from .fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent / "fixtures" / "notes"

_SCRIPT = {
    "sources/notes/dikw.md": (
        "<page path=\"wiki/concepts/dikw-pyramid.md\" type=\"concept\">\n"
        "---\ntags: [dikw, pyramid]\n---\n\n"
        "# DIKW pyramid\n\n"
        "The DIKW pyramid organises raw data into four layers. "
        "See [[Karpathy LLM Wiki]] for a related pattern.\n"
        "</page>"
    ),
    "sources/notes/karpathy-wiki.md": (
        "<page path=\"wiki/concepts/karpathy-llm-wiki.md\" type=\"concept\">\n"
        "---\ntags: [pattern, llm]\n---\n\n"
        "# Karpathy LLM Wiki\n\n"
        "Karpathy's pattern defines a wiki built from source documents. "
        "It complements the [[DIKW pyramid]] model.\n"
        "</page>"
    ),
    "sources/notes/retrieval.md": (
        "<page path=\"wiki/concepts/hybrid-retrieval.md\" type=\"concept\">\n"
        "---\ntags: [search]\n---\n\n"
        "# Hybrid retrieval\n\n"
        "BM25 + dense vectors fused with RRF. Useful background for the "
        "[[DIKW pyramid]] engine.\n"
        "</page>"
    ),
}


class ScriptedLLM:
    """Returns a canned <page> response keyed by which source appears in the prompt."""

    def __init__(self, script: dict[str, str]) -> None:
        self._script = script
        self.last_user: str | None = None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list | None = None,
    ):
        from dikw_core.providers import LLMResponse

        self.last_user = user
        for src_path, resp in self._script.items():
            if src_path in user:
                return LLMResponse(text=resp, finish_reason="end_turn")
        raise AssertionError(f"no script entry matched prompt: {user[:200]}")


@pytest.fixture()
def wiki_with_fixtures(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    api.init_wiki(wiki)
    dest = wiki / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_synth_creates_linked_wiki_pages_and_clean_lint(wiki_with_fixtures: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    llm = ScriptedLLM(_SCRIPT)
    report = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert report.candidates == 3
    assert report.created == 3
    assert report.skipped == 0
    assert report.errors == 0

    # on-disk artefacts
    assert (wiki_with_fixtures / "wiki" / "concepts" / "dikw-pyramid.md").is_file()
    assert (wiki_with_fixtures / "wiki" / "concepts" / "karpathy-llm-wiki.md").is_file()
    assert (wiki_with_fixtures / "wiki" / "concepts" / "hybrid-retrieval.md").is_file()
    index_text = (wiki_with_fixtures / "wiki" / "index.md").read_text(encoding="utf-8")
    assert "DIKW pyramid" in index_text
    assert "Karpathy LLM Wiki" in index_text
    log_text = (wiki_with_fixtures / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "synth" in log_text

    # Lint expectations:
    # - Each page references [[DIKW pyramid]] or [[Karpathy LLM Wiki]] which exist
    #   → no broken_wikilink.
    # - Hybrid retrieval page has no inbound wikilinks → should be reported as orphan.
    lint_report = await api.lint(wiki_with_fixtures)
    kinds = lint_report.by_kind()
    assert kinds.get("broken_wikilink", 0) == 0
    assert kinds.get("duplicate_title", 0) == 0
    assert kinds.get("orphan_page", 0) >= 1


@pytest.mark.asyncio
async def test_synth_is_idempotent_without_force_all(wiki_with_fixtures: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)
    llm = ScriptedLLM(_SCRIPT)

    first = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert first.created == 3

    second = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert second.created == 0
    assert second.skipped == 3


@pytest.mark.asyncio
async def test_synth_handles_llm_parse_errors(wiki_with_fixtures: Path) -> None:
    embedder = FakeEmbeddings()
    await api.ingest(wiki_with_fixtures, embedder=embedder)

    class BadLLM(FakeLLM):
        pass

    llm = BadLLM(response_text="this is not a page block")
    report = await api.synthesize(wiki_with_fixtures, llm=llm, embedder=embedder)
    assert report.errors == 3
    assert report.created == 0
