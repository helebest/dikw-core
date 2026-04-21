"""Live end-to-end smoke test against a real MiniMax endpoint.

Gated by ``MINIMAX_API_KEY`` and supporting config env vars — when any
required variable is missing the test is skipped, so CI runs it only on
jobs that explicitly inject the secret.

The test ingests the MVP dogfood corpus (reused from
``tests/fixtures/mvp_corpus/``), then asks one Karpathy question and
asserts the returned citations include the karpathy source. It does not
assert a specific answer string — production LLM output is
non-deterministic — only that the grounding pipeline round-trips end to
end against a real provider.

Run manually either by exporting the env vars:

    export MINIMAX_API_KEY=...
    export MINIMAX_LLM_MODEL=<anthropic-compatible model name>
    export MINIMAX_LLM_BASE_URL=<MiniMax Anthropic endpoint>
    export MINIMAX_EMBEDDING_MODEL=<embedding model>
    export MINIMAX_EMBEDDING_BASE_URL=<MiniMax OpenAI-compatible endpoint>
    uv run pytest tests/test_mvp_e2e.py -v -s

…or by copying ``.env.example`` to ``.env`` (gitignored) and filling it
in — ``pytest-dotenv`` auto-loads it at collection time.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from dikw_core import api

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mvp_corpus"

REQUIRED_ENV = (
    "MINIMAX_API_KEY",
    "MINIMAX_LLM_MODEL",
    "MINIMAX_LLM_BASE_URL",
    "MINIMAX_EMBEDDING_MODEL",
    "MINIMAX_EMBEDDING_BASE_URL",
)


def _missing_env() -> list[str]:
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


pytestmark = pytest.mark.skipif(
    bool(_missing_env()),
    reason=(
        "live MiniMax smoke test — set all of: "
        + ", ".join(REQUIRED_ENV)
        + " to run (missing: "
        + ", ".join(_missing_env() or ["(none)"])
        + ")"
    ),
)


def _write_minimax_config(wiki: Path) -> None:
    """Overwrite the scaffolded dikw.yml with MiniMax provider settings."""
    cfg_text = f"""\
provider:
  llm: anthropic
  llm_model: {os.environ["MINIMAX_LLM_MODEL"]}
  llm_base_url: {os.environ["MINIMAX_LLM_BASE_URL"]}
  embedding: openai_compat
  embedding_model: {os.environ["MINIMAX_EMBEDDING_MODEL"]}
  embedding_base_url: {os.environ["MINIMAX_EMBEDDING_BASE_URL"]}
storage:
  backend: sqlite
  path: .dikw/index.sqlite
schema:
  description: "mvp e2e smoke test wiki"
sources:
  - path: ./sources
    pattern: "**/*.md"
"""
    (wiki / "dikw.yml").write_text(cfg_text, encoding="utf-8")


@pytest.fixture()
def mvp_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Both SDKs read their own env var; MiniMax uses a single key for both surfaces.
    key = os.environ["MINIMAX_API_KEY"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    monkeypatch.setenv("OPENAI_API_KEY", key)

    wiki = tmp_path / "wiki"
    api.init_wiki(wiki, description="mvp e2e smoke test wiki")
    _write_minimax_config(wiki)

    dest = wiki / "sources" / "corpus"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES_DIR.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_check_connectivity_roundtrips(mvp_wiki: Path) -> None:
    """Verify provider config before we ingest — fail fast on bad credentials."""
    report = await api.check_providers(mvp_wiki)
    assert report.llm.ok, f"LLM probe failed: {report.llm.detail}"
    assert report.embed.ok, f"Embedding probe failed: {report.embed.detail}"


@pytest.mark.asyncio
async def test_ingest_then_query_returns_karpathy_citation(
    mvp_wiki: Path,
) -> None:
    ingest_report = await api.ingest(mvp_wiki)
    assert ingest_report.added > 0, "expected some docs to be added"
    assert ingest_report.embedded > 0, "expected some chunks to be embedded"

    result = await api.query(
        "What does Karpathy mean by deterministic scoping versus probabilistic reasoning?",
        mvp_wiki,
        limit=5,
    )
    assert result.citations, "expected at least one citation from a live query"
    assert any("karpathy" in c.path.lower() for c in result.citations), (
        f"expected a karpathy citation, got: {[c.path for c in result.citations]}"
    )
