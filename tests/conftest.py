"""Pytest session bootstrap + cross-suite fixtures.

Windows ships ``asyncio.ProactorEventLoop`` as the default, which
psycopg's async client refuses to run under (raises ``InterfaceError``
at connect). Switch the policy to ``WindowsSelectorEventLoopPolicy`` so
the Postgres storage-contract tests run locally; Linux / macOS / CI
are unaffected.

The ``asgi_client``, ``client_transport``, and ``patch_transport_factory``
fixtures live here (rather than in ``tests/client/conftest.py``) so the
top-level ``tests/test_eval_cli.py`` — which is a CLI-routed test, not
a pure client unit test — can pull them in too.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from dikw_core.client.config import ClientConfig
from dikw_core.client.transport import Transport
from dikw_core.server.app import build_app
from dikw_core.server.auth import AuthConfig
from dikw_core.server.runtime import ServerRuntime, build_runtime, teardown_runtime

from .fakes import init_test_wiki

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture()
def client_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, description="client-test wiki")
    return wiki


@pytest.fixture()
async def asgi_client(
    client_wiki: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, ServerRuntime]]:
    """Build the FastAPI app + an ``httpx.AsyncClient`` bound to it.

    The runtime is built once per test from a fresh tmp wiki and torn
    down on exit. The fixture yields a (client, runtime) pair so tests
    can poke runtime state directly when needed (engine ingest before
    a CLI command, etc.).
    """
    auth = AuthConfig(host="127.0.0.1", token=None)
    rt = await build_runtime(root=client_wiki, auth=auth)

    async def _factory() -> ServerRuntime:
        return rt

    app = build_app(runtime_factory=_factory, auth=auth)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            async with app.router.lifespan_context(app):
                yield client, rt
    finally:
        await teardown_runtime(rt)


@pytest.fixture()
async def client_transport(
    asgi_client: tuple[httpx.AsyncClient, ServerRuntime],
) -> AsyncIterator[Transport]:
    httpx_client, _rt = asgi_client
    cfg = ClientConfig(server_url="http://test", token=None)
    t = Transport.from_config(cfg, client=httpx_client)
    try:
        yield t
    finally:
        # The asgi_client fixture owns the underlying httpx client's
        # lifecycle; closing it here would race the lifespan teardown.
        pass


@pytest.fixture()
def patch_transport_factory(
    asgi_client: tuple[httpx.AsyncClient, ServerRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], None]:
    """Returns a hook that monkeypatches ``Transport.from_config`` to
    always return a transport bound to the in-memory ASGI client.

    Tests using the Typer ``CliRunner`` call this once at the top of
    the test body. Without it, ``dikw client *`` commands try to open a
    real httpx client against ``http://127.0.0.1:8765`` and the test
    hangs.
    """

    httpx_client, _rt = asgi_client

    def _hook() -> None:
        def fake_from_config(
            _cfg: ClientConfig, *, client: httpx.AsyncClient | None = None
        ) -> Transport:
            del client
            return Transport(client=httpx_client, token=None)

        monkeypatch.setattr(
            "dikw_core.client.cli_app.Transport.from_config",
            staticmethod(fake_from_config),
        )

        async def _no_aexit(self: Transport, *_: Any) -> None:
            del self
            return None

        monkeypatch.setattr(Transport, "__aexit__", _no_aexit)

    return _hook
