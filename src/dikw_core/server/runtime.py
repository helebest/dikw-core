"""Per-process engine handle: cfg / wiki root / storage / task subsystem.

One ``ServerRuntime`` is built at server startup and torn down at shutdown;
route handlers reach it via FastAPI dependencies. We keep this small —
just the long-lived state — because the engine itself is largely stateless
(LLM / embedding providers are built per-request as today).

Lifecycle:
  startup  →  load cfg, build storage, connect, migrate,
              build task store + manager, restart_cleanup
  shutdown →  manager.shutdown, store.close, task_store.close
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI

from ..config import CONFIG_FILENAME, DikwConfig, load_config
from ..storage import Storage, build_storage
from .auth import AuthConfig
from .tasks import (
    ProgressBus,
    TaskManager,
    TaskStore,
    build_task_store,
)

logger = logging.getLogger(__name__)


def _server_instance_id(root: Path) -> str:
    """Stable per-server identity. Two ``dikw serve`` processes pointed
    at the same wiki on the same host get the same id; different hosts
    or different wiki paths get different ids.

    Used by the task store to scope ``restart_cleanup`` so a server
    only ever reaps its own leftover rows from a previous run, even
    when a shared Postgres task DB is in use across multiple servers.
    """
    raw = f"{socket.gethostname()}::{root.resolve()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ServerRuntime:
    """All server-wide state. Held under ``app.state.runtime``."""

    cfg: DikwConfig
    root: Path
    storage: Storage
    task_store: TaskStore
    bus: ProgressBus
    manager: TaskManager
    auth: AuthConfig
    # Serializes wiki-mutating ops (currently ingest) so two concurrent
    # tasks can't interleave their staging-commit + on-disk writes and
    # leave the sources/ tree as a mix of both. Held for the entire ingest
    # runner — concurrent ingests on the same wiki are a degenerate case.
    ingest_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def build_runtime(
    *, root: Path, auth: AuthConfig
) -> ServerRuntime:
    """Resolve cfg + open every long-lived handle. Caller owns teardown."""
    root = root.resolve()
    cfg_path = root / CONFIG_FILENAME
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"no {CONFIG_FILENAME} at {root} — initialise the wiki first "
            "or point `dikw serve --wiki` at an existing dikw directory"
        )
    cfg = load_config(cfg_path)

    storage = build_storage(
        cfg.storage,
        root=root,
        cjk_tokenizer=cfg.retrieval.cjk_tokenizer,
    )
    await storage.connect()
    await storage.migrate()

    task_store = build_task_store(
        cfg, root=root, instance_id=_server_instance_id(root)
    )
    await task_store.init()

    bus = ProgressBus()
    manager = TaskManager(store=task_store, bus=bus)
    # Safe for both stores: ``list_running()`` filters by the store's
    # ``instance_id``, so a shared Postgres task DB only surfaces rows
    # this exact server (host + wiki path) submitted in a previous run.
    # Other live ``dikw serve`` processes pointed at the same DB stay
    # untouched.
    await manager.restart_cleanup()

    return ServerRuntime(
        cfg=cfg,
        root=root,
        storage=storage,
        task_store=task_store,
        bus=bus,
        manager=manager,
        auth=auth,
    )


async def teardown_runtime(rt: ServerRuntime) -> None:
    await rt.manager.shutdown()
    await rt.storage.close()
    await rt.task_store.close()


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """FastAPI ``lifespan`` hook. Pulls a pre-prepared ``ServerRuntime`` off
    ``app.state.runtime_factory`` (a callable set by the app builder so
    tests can inject without spinning a real engine)."""
    factory = getattr(app.state, "runtime_factory", None)
    if factory is None:
        raise RuntimeError(
            "app.state.runtime_factory is unset; build the app via "
            "server.app.build_app(...)"
        )
    rt: ServerRuntime = await factory()
    app.state.runtime = rt
    logger.info(
        "dikw server ready  wiki=%s storage=%s auth=%s",
        rt.root,
        rt.cfg.storage.backend,
        "token" if rt.auth.required else "off",
    )
    try:
        yield
    finally:
        await teardown_runtime(rt)


def get_runtime(app: FastAPI) -> ServerRuntime:
    """Access the runtime from any route handler via ``request.app``."""
    rt = getattr(app.state, "runtime", None)
    if rt is None:
        raise RuntimeError("server runtime is not initialised")
    return rt  # type: ignore[no-any-return]


__all__ = [
    "ServerRuntime",
    "build_runtime",
    "get_runtime",
    "lifespan",
    "teardown_runtime",
]
