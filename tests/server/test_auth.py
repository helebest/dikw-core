"""Bearer-token auth: localhost-default + token-required + 0.0.0.0 invariant."""

from __future__ import annotations

import httpx
import pytest

from dikw_core.server.auth import (
    AuthConfig,
    ensure_auth_invariant,
    load_auth_config,
)


def test_loopback_no_token_skips_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIKW_SERVER_TOKEN", raising=False)
    cfg = load_auth_config(host="127.0.0.1")
    assert cfg.required is False
    # Should not raise on loopback without a token.
    ensure_auth_invariant(cfg)


def test_loopback_with_token_requires_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIKW_SERVER_TOKEN", "abc")
    cfg = load_auth_config(host="127.0.0.1")
    assert cfg.required is True


def test_token_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_SERVER_TOKEN", "env-token")
    cfg = load_auth_config(host="127.0.0.1", token_override="cli-token")
    assert cfg.token == "cli-token"


def test_zero_zero_zero_zero_without_token_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DIKW_SERVER_TOKEN", raising=False)
    cfg = load_auth_config(host="0.0.0.0")
    with pytest.raises(RuntimeError, match="without an auth token"):
        ensure_auth_invariant(cfg)


def test_zero_zero_zero_zero_with_token_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIKW_SERVER_TOKEN", "abc")
    cfg = load_auth_config(host="0.0.0.0")
    ensure_auth_invariant(cfg)
    assert cfg.required is True


def test_empty_string_env_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shell quoting accidents (``DIKW_SERVER_TOKEN=`` exports an empty
    string) must not silently arm the required path with an empty
    token."""
    monkeypatch.setenv("DIKW_SERVER_TOKEN", "")
    cfg = load_auth_config(host="127.0.0.1")
    assert cfg.token is None
    assert cfg.required is False


def test_auth_config_required_property() -> None:
    assert AuthConfig(host="127.0.0.1", token=None).required is False
    assert AuthConfig(host="127.0.0.1", token="x").required is True
    assert AuthConfig(host="0.0.0.0", token=None).required is True
    assert AuthConfig(host="0.0.0.0", token="x").required is True


# ---- HTTP-level token enforcement ---------------------------------------


@pytest.mark.asyncio
async def test_protected_endpoint_rejects_missing_token(
    server_client_with_token: tuple[httpx.AsyncClient, str],
) -> None:
    client, _token = server_client_with_token
    resp = await client.get("/v1/status")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_accepts_valid_token(
    server_client_with_token: tuple[httpx.AsyncClient, str],
) -> None:
    client, token = server_client_with_token
    resp = await client.get(
        "/v1/status", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_protected_endpoint_rejects_wrong_token(
    server_client_with_token: tuple[httpx.AsyncClient, str],
) -> None:
    client, _token = server_client_with_token
    resp = await client.get(
        "/v1/status", headers={"Authorization": "Bearer wrong"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_skips_auth_check(
    server_client_with_token: tuple[httpx.AsyncClient, str],
) -> None:
    """``/v1/healthz`` is part of the protected router today (auth_dep
    runs on every endpoint of the router) so it MUST require the token
    when one is configured. The test pins that contract — flipping
    healthz to be unauthenticated would be a change worth code review."""
    client, _token = server_client_with_token
    resp = await client.get("/v1/healthz")
    assert resp.status_code == 401
