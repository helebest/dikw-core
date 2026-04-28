"""Coverage for ``evals/tools/*`` recovery scripts.

The Phase 1.5 replay tool's active-version pinning is the bit most
likely to silently drift if ``api.query()`` ever changes its embedding-
space contract. A unit test on the helper guarantees the snapshot
replay path stays in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dikw_core.storage.base import NotSupported, Storage

# evals/ isn't on sys.path by default — the script is a CLI entry, not
# a package. Add it so we can import the helper under test.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "evals" / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from run_phase15_from_snapshot import resolve_active_text_version  # noqa: E402

from .fakes import init_test_wiki, register_text_version  # noqa: E402


@pytest.mark.asyncio
async def test_resolve_active_version_uses_active_row_dim(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki, dim=128)

    from dikw_core.config import load_config
    from dikw_core.storage import build_storage

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        version_id = await register_text_version(
            storage, dim=128, model="qwen3-test", provider="gitee"
        )
        result = await resolve_active_text_version(
            storage, default_model="cfg-default-model"
        )
    finally:
        await storage.close()

    assert result == (version_id, "qwen3-test", 128)


@pytest.mark.asyncio
async def test_resolve_active_version_falls_back_to_cfg_when_none(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    init_test_wiki(wiki)

    from dikw_core.config import load_config
    from dikw_core.storage import build_storage

    cfg = load_config(wiki / "dikw.yml")
    storage = build_storage(
        cfg.storage, root=wiki, cjk_tokenizer=cfg.retrieval.cjk_tokenizer
    )
    await storage.connect()
    await storage.migrate()
    try:
        result = await resolve_active_text_version(
            storage, default_model="fallback-model"
        )
    finally:
        await storage.close()

    assert result == (None, "fallback-model", None)


class _NotSupportedStorage:
    """Fake storage that raises ``NotSupported`` on the version read.

    Mirrors what filesystem-style backends do when they don't implement
    embed versioning. Only the one method the helper calls is needed.
    """

    async def get_active_embed_version(self, *, modality: str) -> object:
        raise NotSupported(f"embed versions not supported (modality={modality})")


@pytest.mark.asyncio
async def test_resolve_active_version_handles_notsupported() -> None:
    storage: Storage = _NotSupportedStorage()  # type: ignore[assignment]

    result = await resolve_active_text_version(
        storage, default_model="fallback-model"
    )

    assert result == (None, "fallback-model", None)
