"""HTTP-level tests for ``POST /v1/init``.

The bound wiki is already initialised at server startup (``runtime``
fixture scaffolds it via ``init_test_wiki``), so the endpoint's primary
responsibility is to refuse with ``409 Conflict`` and let the client
branch — exactly what the bootstrap-from-client flow needs to detect
"already wired up, skip".
"""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_init_on_already_initialised_wiki_returns_409(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/init", json={})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "wiki_already_initialised"


@pytest.mark.asyncio
async def test_init_disabled_via_env_returns_409(
    server_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIKW_SERVER_DISABLE_INIT", "1")
    resp = await server_client.post(
        "/v1/init", json={"description": "ignored"}
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "init_disabled"


@pytest.mark.asyncio
async def test_init_default_body_is_optional(
    server_client: httpx.AsyncClient,
) -> None:
    # Empty body should still be parsed (description is optional).
    resp = await server_client.post("/v1/init")
    assert resp.status_code == 409  # already-initialised dominates
    assert resp.json()["error"]["code"] == "wiki_already_initialised"
