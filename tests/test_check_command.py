"""``dikw check`` — verifiable config tool.

Connects to the configured LLM + embedding providers with one tiny call
each and reports whether each leg roundtripped. Used by operators after
editing ``dikw.yml`` to confirm credentials + endpoints before running
ingest/query against a real budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from dikw_core import api
from dikw_core.cli import app
from dikw_core.providers import LLMResponse, ToolSpec
from tests.fakes import FakeEmbeddings, FakeLLM


class BrokenLLM:
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        _ = (system, user, model, max_tokens, temperature, tools)
        raise RuntimeError("llm backend refused connection")


class BrokenEmbedder:
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        _ = (texts, model)
        raise RuntimeError("embedding backend refused connection")


@pytest.fixture()
def wiki(tmp_path: Path) -> Path:
    w = tmp_path / "wiki"
    api.init_wiki(w, description="check-command test wiki")
    return w


@pytest.mark.asyncio
async def test_check_passes_when_both_providers_roundtrip(wiki: Path) -> None:
    report = await api.check_providers(wiki, llm=FakeLLM(), embedder=FakeEmbeddings())
    assert report.ok
    assert report.llm.ok
    assert report.embed.ok


@pytest.mark.asyncio
async def test_check_reports_llm_failure_distinctly_from_embed_failure(
    wiki: Path,
) -> None:
    report = await api.check_providers(wiki, llm=BrokenLLM(), embedder=FakeEmbeddings())
    assert not report.ok
    assert not report.llm.ok
    assert "refused" in report.llm.detail.lower()
    # Embed leg still reports its own result — LLM failure doesn't short-circuit it.
    assert report.embed.ok


@pytest.mark.asyncio
async def test_check_reports_embed_failure_distinctly_from_llm_failure(
    wiki: Path,
) -> None:
    report = await api.check_providers(wiki, llm=FakeLLM(), embedder=BrokenEmbedder())
    assert not report.ok
    assert report.llm.ok
    assert not report.embed.ok
    assert "refused" in report.embed.detail.lower()


def test_cli_check_exits_nonzero_on_failure(
    wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dikw_core.api.build_llm", lambda cfg: BrokenLLM())
    monkeypatch.setattr("dikw_core.api.build_embedder", lambda cfg: FakeEmbeddings())

    runner = CliRunner()
    result = runner.invoke(app, ["check", "--path", str(wiki)])
    assert result.exit_code == 1, result.stdout


def test_cli_check_exits_zero_on_success(
    wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_llm(_cfg: Any) -> Any:
        return FakeLLM()

    def _fake_embed(_cfg: Any) -> Any:
        return FakeEmbeddings()

    monkeypatch.setattr("dikw_core.api.build_llm", _fake_llm)
    monkeypatch.setattr("dikw_core.api.build_embedder", _fake_embed)

    runner = CliRunner()
    result = runner.invoke(app, ["check", "--path", str(wiki)])
    assert result.exit_code == 0, result.stdout
