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

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
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

    task_store = build_task_store(cfg, root=root)
    await task_store.init()

    bus = ProgressBus()
    manager = TaskManager(store=task_store, bus=bus)
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
