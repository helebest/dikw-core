"""Bearer-token auth for the dikw server.

Two modes:
  * **localhost-default** (``host == "127.0.0.1"`` and no token configured):
    Requests skip the auth check entirely. The startup banner warns once
    so the operator knows the server is open on the loopback.
  * **token-required** (any other case): every protected request must
    carry ``Authorization: Bearer <token>`` matching ``DIKW_SERVER_TOKEN``;
    mismatches return 401.

The startup-time check (``ensure_auth_invariant``) refuses to bind
``0.0.0.0`` without a token — the most common foot-gun is "I just want
to expose this on my LAN" and silently shipping no auth there is the
worst-case default.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Header

from .errors import Unauthorized

_TOKEN_ENV = "DIKW_SERVER_TOKEN"


class AuthConfig:
    """Resolved at server bootstrap; injected as a FastAPI dependency."""

    def __init__(self, *, host: str, token: str | None) -> None:
        self.host = host
        self.token = token

    @property
    def required(self) -> bool:
        """Token is required unless we're on localhost AND none was set."""
        if self.token:
            return True
        return not _is_loopback(self.host)


def load_auth_config(*, host: str, token_override: str | None = None) -> AuthConfig:
    """Build an ``AuthConfig`` from explicit args + environment.

    ``token_override`` (e.g. from ``--token``) wins over the env var so a
    one-off ``dikw serve --token …`` doesn't need to mutate the shell env.
    """
    token = token_override if token_override is not None else os.environ.get(_TOKEN_ENV)
    if token == "":  # treat empty string as unset to match shell quoting accidents
        token = None
    return AuthConfig(host=host, token=token)


def ensure_auth_invariant(cfg: AuthConfig) -> None:
    """Reject configurations that would expose the server with no auth.

    Called from the CLI entry point before binding the socket — earlier is
    cheaper than letting uvicorn open the listener and then refusing the
    first request.
    """
    if cfg.token is None and not _is_loopback(cfg.host):
        raise RuntimeError(
            f"refusing to bind {cfg.host} without an auth token. "
            f"Set the {_TOKEN_ENV} environment variable (or pass "
            "--token) before binding any non-loopback interface."
        )


def make_dependency(
    cfg: AuthConfig,
) -> Callable[[str | None], Awaitable[None]]:
    """Return a FastAPI dependency that validates the bearer token.

    Closing over ``cfg`` lets the route handlers stay agnostic of the
    server's auth posture; the dependency knows whether a token is
    required and what it is.
    """

    async def _check(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not cfg.required:
            return
        expected = cfg.token
        if expected is None:  # cfg.required True ⇒ token must be set
            raise Unauthorized(
                "server is misconfigured: token required but not set",
                code="server_misconfigured",
                status_code=500,
            )
        if not authorization:
            raise Unauthorized("missing Authorization header")
        scheme, _, presented = authorization.partition(" ")
        if scheme.lower() != "bearer" or presented.strip() != expected:
            raise Unauthorized("invalid bearer token")

    return _check


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


__all__ = [
    "AuthConfig",
    "ensure_auth_invariant",
    "load_auth_config",
    "make_dependency",
]
