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
    assert report.llm is not None
    assert report.embed is not None
    assert report.llm.ok
    assert report.embed.ok


@pytest.mark.asyncio
async def test_check_reports_llm_failure_distinctly_from_embed_failure(
    wiki: Path,
) -> None:
    report = await api.check_providers(wiki, llm=BrokenLLM(), embedder=FakeEmbeddings())
    assert not report.ok
    assert report.llm is not None
    assert report.embed is not None
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
    assert report.llm is not None
    assert report.embed is not None
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


# ---- Per-leg verification (--llm-only / --embed-only) -----------------


@pytest.mark.asyncio
async def test_check_llm_only_skips_embedding_probe(wiki: Path) -> None:
    """``llm_only=True`` must not build or call the embedding provider.

    Even a broken embedder should not fail the report — it's simply never
    consulted. This is the whole point of single-leg verification: a user
    can validate the LLM before the embedding side is configured at all.
    """
    report = await api.check_providers(
        wiki, llm=FakeLLM(), embedder=BrokenEmbedder(), llm_only=True
    )
    assert report.ok
    assert report.llm is not None and report.llm.ok
    assert report.embed is None


@pytest.mark.asyncio
async def test_check_embed_only_skips_llm_probe(wiki: Path) -> None:
    report = await api.check_providers(
        wiki, llm=BrokenLLM(), embedder=FakeEmbeddings(), embed_only=True
    )
    assert report.ok
    assert report.embed is not None and report.embed.ok
    assert report.llm is None


@pytest.mark.asyncio
async def test_check_rejects_mutually_exclusive_flags(wiki: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        await api.check_providers(
            wiki, llm=FakeLLM(), embedder=FakeEmbeddings(),
            llm_only=True, embed_only=True,
        )
    assert "mutually exclusive" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_check_llm_only_does_not_build_embedder(
    wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--llm-only`` must skip build_embedder entirely.

    A config with a broken embedder factory (e.g., missing API key) should
    still let the user verify the LLM leg. If we call build_embedder first
    and it raises, the whole probe dies — which defeats the feature.
    """
    def _boom(_cfg: Any) -> Any:
        raise RuntimeError("embedder factory should not have been called")

    monkeypatch.setattr("dikw_core.api.build_embedder", _boom)
    report = await api.check_providers(wiki, llm=FakeLLM(), llm_only=True)
    assert report.ok
    assert report.llm is not None and report.llm.ok
    assert report.embed is None


def test_cli_check_llm_only_exits_zero_when_embed_would_fail(
    wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_llm(_cfg: Any) -> Any:
        return FakeLLM()

    def _boom(_cfg: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("dikw_core.api.build_llm", _fake_llm)
    monkeypatch.setattr("dikw_core.api.build_embedder", _boom)

    runner = CliRunner()
    result = runner.invoke(app, ["check", "--path", str(wiki), "--llm-only"])
    assert result.exit_code == 0, result.stdout


def test_cli_check_rejects_both_only_flags(wiki: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["check", "--path", str(wiki), "--llm-only", "--embed-only"]
    )
    assert result.exit_code == 2, result.stdout


def test_cli_check_embed_only_shows_provider_label_when_yaml_set(
    wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``provider.embedding_provider_label`` becomes a free-form tag in probe detail.

    Display-only — it doesn't change any behaviour. Users running
    MiniMax LLM + Gitee AI embeddings want the output to make that split
    obvious at a glance. The label lives in ``dikw.yml`` so it travels
    with the wiki instead of needing a shell env var every time.
    """
    import yaml

    def _fake_embed(_cfg: Any) -> Any:
        return FakeEmbeddings()

    # Post-edit the scaffolded dikw.yml to include the label.
    cfg_path = wiki / "dikw.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("provider", {})["embedding_provider_label"] = "gitee-ai"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr("dikw_core.api.build_embedder", _fake_embed)

    runner = CliRunner()
    result = runner.invoke(app, ["check", "--path", str(wiki), "--embed-only"])
    assert result.exit_code == 0, result.stdout
    assert "gitee-ai" in result.stdout
