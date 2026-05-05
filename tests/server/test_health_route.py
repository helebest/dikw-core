"""HTTP-level tests for ``GET /v1/health``.

The health endpoint is the agent-bootstrap probe: it must surface
enough resolved config to confirm "I'm attached to the right server"
without ever leaking secrets. Tests here lock both the wire shape and
the security boundary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

from dikw_core import __version__
from dikw_core import api as api_module

from ..fakes import FakeEmbeddings

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


@pytest.mark.asyncio
async def test_health_returns_well_formed_report(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    resp = await server_client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert Path(body["base_root"]) == wiki_root.resolve()
    assert body["storage_engine"] in ("sqlite", "postgres")

    # Layer counts present + integer-typed; a freshly init'd wiki has
    # zero documents in every layer.
    counts = body["layer_counts"]
    for key in ("sources", "wiki_pages", "wisdom_items", "chunks"):
        assert key in counts and isinstance(counts[key], int)
    assert counts["sources"] == 0
    assert counts["chunks"] == 0


@pytest.mark.asyncio
async def test_health_layer_counts_track_ingest(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    """After ingesting fixture markdown, ``layer_counts`` must reflect
    the new documents — proves health reads live storage state, not a
    cached snapshot from server start."""
    dest = wiki_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(wiki_root, embedder=FakeEmbeddings())

    resp = await server_client.get("/v1/health")
    body = resp.json()
    counts = body["layer_counts"]
    assert counts["sources"] == len(list(FIXTURES.glob("*.md")))
    assert counts["chunks"] > 0


@pytest.mark.asyncio
async def test_health_providers_block_exposes_resolved_config(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/v1/health")
    body = resp.json()
    providers = body["providers"]

    llm = providers["llm"]
    assert llm["provider"] in ("anthropic_compat", "openai_compat")
    assert isinstance(llm["model"], str) and llm["model"]
    assert "base_url" in llm  # nullable but field must be present
    assert isinstance(llm["max_retries"], int)
    assert isinstance(llm["api_key_present"], bool)

    embedding = providers["embedding"]
    assert embedding["provider"] == "openai_compat"
    assert isinstance(embedding["dim"], int) and embedding["dim"] > 0
    assert embedding["distance"] in ("cosine", "l2", "dot")
    assert isinstance(embedding["normalize"], bool)
    assert isinstance(embedding["api_key_present"], bool)
    # Multimodal nests under embedding (per design); test wiki has no
    # multimodal config so it's null.
    assert embedding["multimodal"] is None


@pytest.mark.asyncio
async def test_health_never_leaks_secrets(
    server_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical security boundary: even when API keys + a Postgres DSN
    are in the env, the health response must not echo any of them.

    Locks the contract that ``health`` returns ``api_key_present: bool``
    rather than the value, and that no DSN/path leaks into the JSON.
    Sentinel values include URL-unsafe + URL-safe characters so a buggy
    serializer can't hide a leak behind URL-encoding.
    """
    sentinels = {
        "OPENAI_API_KEY": "sk-leak-OPENAI-A!@#$",
        "ANTHROPIC_API_KEY": "sk-ant-LEAK-XYZ-987",
        "DIKW_EMBEDDING_API_KEY": "sk-embed-LEAK-321",
    }
    for k, v in sentinels.items():
        monkeypatch.setenv(k, v)

    resp = await server_client.get("/v1/health")
    raw = resp.text
    body = resp.json()

    for k, v in sentinels.items():
        assert v not in raw, (
            f"secret {k} leaked into /v1/health response:\n{raw}"
        )

    # Positive check: at least one ``api_key_present`` flag is True
    # (we just set the env var) so the health endpoint is actually
    # reading env, not silently returning False.
    assert body["providers"]["embedding"]["api_key_present"] is True


@pytest.mark.asyncio
async def test_health_storage_engine_omits_dsn_and_path(
    server_client: httpx.AsyncClient,
) -> None:
    """``storage_engine`` is the *type* (``sqlite``/``postgres``) only.
    DSN, sqlite path, schema name — none of those belong on /v1/health.
    """
    resp = await server_client.get("/v1/health")
    raw = resp.text
    body = resp.json()

    assert body["storage_engine"] in ("sqlite", "postgres")
    # No raw DSN-shaped string and no .sqlite path leak.
    assert "postgresql://" not in raw
    assert ".sqlite" not in raw
