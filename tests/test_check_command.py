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


# ---- Multimodal probe ----------------------------------------------------
#
# When ``assets.multimodal`` is configured, the engine ingests chunks AND
# assets through the multimodal embedder, never the text-only one. The
# check must follow that route, otherwise it greenlights a wiki whose
# real ingest path will fail. The probe sends one text + one image input
# in a single batched request — exactly the shape the production pipeline
# uses, no RTT stacking.


@pytest.fixture()
def multimodal_wiki(tmp_path: Path) -> Path:
    """A wiki whose dikw.yml carries an ``assets.multimodal`` block."""
    import yaml

    w = tmp_path / "wiki-mm"
    api.init_wiki(w, description="multimodal probe test wiki")
    cfg_path = w / "dikw.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("assets", {})["multimodal"] = {
        "provider": "gitee_multimodal",
        "model": "Qwen3-VL-Embedding-8B",
        "dim": 4096,
        "batch": 16,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return w


@pytest.mark.asyncio
async def test_check_embed_probe_routes_through_multimodal_when_configured(
    multimodal_wiki: Path,
) -> None:
    """When ``assets.multimodal`` is configured, --embed-only must probe the
    multimodal embedder, not the text-only one. The probe sends one text +
    one image input in a single batched request (no RTT stacking)."""
    from tests.fakes import FakeMultimodalEmbedding

    fake_mm = FakeMultimodalEmbedding(dim=4096)
    report = await api.check_providers(
        multimodal_wiki,
        llm=FakeLLM(),
        multimodal_embedder=fake_mm,
        embed_only=True,
    )
    assert report.ok
    assert report.embed is not None and report.embed.ok
    # Detail must surface the dim probed AND that both modalities were sent.
    assert "dim=4096" in report.embed.detail
    assert "text+image" in report.embed.detail
    # Single batch call: 2 inputs (1 text + 1 image), not 2 separate calls.
    assert len(fake_mm.last_inputs) == 2
    text_inputs = [i for i in fake_mm.last_inputs if i.text]
    image_inputs = [i for i in fake_mm.last_inputs if i.images]
    assert len(text_inputs) == 1
    assert len(image_inputs) == 1
    # Probe targets the multimodal model name, not provider.embedding_model.
    assert fake_mm.last_model == "Qwen3-VL-Embedding-8B"


@pytest.mark.asyncio
async def test_check_embed_probe_uses_text_path_when_no_multimodal(
    wiki: Path,
) -> None:
    """The legacy text-embed probe must keep working when ``assets.multimodal``
    is absent — regression guard against routing every wiki through the
    multimodal path."""
    fake_text = FakeEmbeddings()
    report = await api.check_providers(
        wiki, llm=FakeLLM(), embedder=fake_text, embed_only=True
    )
    assert report.ok
    assert report.embed is not None and report.embed.ok
    # The text-only path doesn't probe images, so its detail line must
    # not advertise a "modalities=" tag.
    assert "modalities" not in report.embed.detail


@pytest.mark.asyncio
async def test_check_multimodal_probe_target_reflects_mm_base_url(
    tmp_path: Path,
) -> None:
    """The probe detail's target must point at the multimodal endpoint the
    request is actually sent to — ``assets.multimodal.base_url`` — not at
    ``provider.embedding_base_url`` (which is the text leg's URL).

    In split-vendor setups (text leg one vendor, multimodal another) the
    two URLs differ; reporting the wrong one makes a green check appear to
    validate one endpoint while it actually exercised another."""
    import yaml

    from tests.fakes import FakeMultimodalEmbedding

    w = tmp_path / "wiki-mm-split"
    api.init_wiki(w, description="split-base-url test wiki")
    cfg_path = w / "dikw.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("provider", {})["embedding_base_url"] = (
        "https://text-embed.example.com/v1"
    )
    raw.setdefault("assets", {})["multimodal"] = {
        "provider": "gitee_multimodal",
        "model": "Qwen3-VL-Embedding-8B",
        "dim": 4096,
        "batch": 16,
        "base_url": "https://multimodal.example.com/v1",
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    fake_mm = FakeMultimodalEmbedding(dim=4096)
    report = await api.check_providers(
        w, llm=FakeLLM(), multimodal_embedder=fake_mm, embed_only=True
    )
    assert report.ok
    assert report.embed is not None
    assert report.embed.target == "https://multimodal.example.com/v1"
    # The text-leg URL must NOT appear on the multimodal probe row — that
    # was the bug the codex review surfaced.
    assert "text-embed.example.com" not in report.embed.target


@pytest.mark.asyncio
async def test_check_falls_back_to_text_on_storage_without_multimodal_support(
    tmp_path: Path,
) -> None:
    """When the storage backend doesn't support multimodal versioning,
    ``ingest()`` silently degrades to the text-only path (api.py: NotSupported
    → ``mm_cfg = None``). The check must mirror that, otherwise it can fail
    on a misconfigured multimodal endpoint that real ingest never reaches —
    the check would no longer describe the behavior operators actually get.

    Filesystem and postgres backends raise ``NotSupported`` from
    ``upsert_embed_version`` today; sqlite is the only backend with
    multimodal versioning."""
    import yaml

    w = tmp_path / "wiki-fs-mm"
    api.init_wiki(w, description="filesystem + mm fallback test wiki")
    cfg_path = w / "dikw.yml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw["storage"] = {"backend": "filesystem"}
    raw.setdefault("assets", {})["multimodal"] = {
        "provider": "gitee_multimodal",
        "model": "Qwen3-VL-Embedding-8B",
        "dim": 4096,
        "batch": 16,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    fake_text = FakeEmbeddings()
    report = await api.check_providers(
        w, llm=FakeLLM(), embedder=fake_text, embed_only=True
    )
    assert report.ok
    assert report.embed is not None and report.embed.ok
    # Fallback path is the text leg — no "modalities=" tag.
    assert "modalities" not in report.embed.detail
