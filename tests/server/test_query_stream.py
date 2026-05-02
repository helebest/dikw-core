"""HTTP-level tests for ``POST /v1/query``.

Covers the NDJSON wire shape end-to-end via the in-memory ASGI
transport. The engine's LLM + embedder are monkeypatched onto the
factory entry points so the route exercises a hermetic pipeline (no
API keys, no network) but still drives the real ``api.query`` codepath.

Asserts:
  * Event order: query_started → retrieval_done → llm_token* → final.
  * Streaming token emission lands in real time (one event per chunk).
  * Provider that doesn't implement streaming falls back to a single
    ``final`` (no llm_token events) — no NotImplementedError leaks.
  * Validation errors land as 4xx, not as a partial stream.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

from dikw_core import api as api_module
from dikw_core.providers import build_embedder

from ..fakes import FakeEmbeddings, FakeLLM

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes"


@pytest.fixture()
async def ingested_wiki(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> Path:
    """Wiki with three fixture markdown files ingested + dense vectors.

    Uses the engine directly (not the HTTP ingest task) so the fixture
    stays cheap and isolated from task-pipeline noise — this fixture is
    about query, not about ingest plumbing.
    """
    dest = wiki_root / "sources" / "notes"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.glob("*.md"):
        shutil.copy2(src, dest / src.name)
    await api_module.ingest(wiki_root, embedder=FakeEmbeddings())
    _ = server_client  # ensure runtime lifespan is up before we ingest
    return wiki_root


def _patch_providers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm: FakeLLM,
) -> None:
    """Stub ``build_llm`` (every callsite) and reuse ``FakeEmbeddings``
    for the embedder build path so the route's hermetic pipeline doesn't
    need API keys."""

    def _llm_factory(_cfg: object) -> FakeLLM:
        return llm

    monkeypatch.setattr(api_module, "build_llm", _llm_factory)

    real_build = build_embedder

    def _embed_factory(cfg: object, *, dim_override: int | None = None) -> object:
        # Honour the dim_override pathway for completeness, even though
        # FakeEmbeddings always emits its native dim.
        _ = real_build, dim_override, cfg
        return FakeEmbeddings()

    monkeypatch.setattr(api_module, "build_embedder", _embed_factory)


@pytest.mark.asyncio
async def test_query_stream_emits_token_then_final(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM(
        response_text="answer [#1]",
        stream_chunks=["ans", "wer ", "[#1]"],
    )
    _patch_providers(monkeypatch, llm=llm)

    async with server_client.stream(
        "POST",
        "/v1/query",
        json={"q": "what about scoping?", "limit": 3},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]

    types = [e["type"] for e in events]
    assert types[0] == "query_started"
    assert "retrieval_done" in types
    # Three stream chunks → three llm_token events.
    token_events = [e for e in events if e["type"] == "llm_token"]
    assert [e["delta"] for e in token_events] == ["ans", "wer ", "[#1]"]
    assert types[-1] == "final"
    final = events[-1]
    assert final["status"] == "succeeded"
    # Final result mirrors QueryResult exactly.
    assert "answer" in final["result"]
    assert isinstance(final["result"]["citations"], list)
    # Server reassembled the answer from the streamed parts (or the
    # done event's text — either is authoritative).
    assert final["result"]["answer"].strip() == "answer [#1]"


@pytest.mark.asyncio
async def test_query_stream_falls_back_when_provider_lacks_streaming(
    server_client: httpx.AsyncClient,
    ingested_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``stream_chunks=None`` → FakeLLM.complete_stream raises
    # NotImplementedError; api.query falls back to ``complete``.
    llm = FakeLLM(response_text="non-streamed answer")
    _patch_providers(monkeypatch, llm=llm)

    async with server_client.stream(
        "POST",
        "/v1/query",
        json={"q": "scoping?", "limit": 3},
    ) as resp:
        events = [
            json.loads(line)
            for line in [ln async for ln in resp.aiter_lines()]
            if line.strip()
        ]

    token_events = [e for e in events if e["type"] == "llm_token"]
    assert token_events == []  # no tokens — provider couldn't stream
    final = events[-1]
    assert final["type"] == "final"
    assert final["status"] == "succeeded"
    assert final["result"]["answer"].strip() == "non-streamed answer"


@pytest.mark.asyncio
async def test_query_stream_rejects_empty_q(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post(
        "/v1/query", json={"q": "   ", "limit": 5}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_query_stream_rejects_out_of_range_limit(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post(
        "/v1/query", json={"q": "x", "limit": 0}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"
