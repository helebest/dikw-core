"""Removal-verification tests for the deleted ``/v1/query`` endpoint.

PR-1 removed the in-engine query verb. dikw-core now stops at retrieve;
LLM synthesis is the agent's job. These tests guard against accidental
reintroduction of the endpoint via a regression or a stale router
registration somewhere in ``server/app.py``.

Asserts:
  * ``POST /v1/query`` returns 404 (route truly not registered, not just
    405 Method Not Allowed — a stub registration with the wrong verb
    would manifest as 405 and silently leak the surface).
  * No FastAPI route advertises any ``/v1/query`` family path in the
    OpenAPI schema (catches reintroduction via a path with a different
    HTTP method that still shows up in the auto-generated schema).
"""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_query_route_returns_404(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.post("/v1/query", json={"q": "ping", "limit": 1})
    assert resp.status_code == 404, (
        f"/v1/query should be removed entirely, got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_query_route_absent_from_openapi(
    server_client: httpx.AsyncClient,
) -> None:
    resp = await server_client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = list(schema.get("paths", {}).keys())
    query_paths = [p for p in paths if "/v1/query" in p]
    assert query_paths == [], (
        f"/v1/query family must be absent from OpenAPI, found: {query_paths}"
    )
