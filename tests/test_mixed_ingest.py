from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dikw_core import api
from dikw_core.config import SourceConfig, dump_config_yaml, load_config

from .fakes import FakeEmbeddings

FIXTURES = Path(__file__).parent / "fixtures" / "mixed"


@pytest.fixture()
def wiki_mixed(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    api.init_wiki(wiki)
    # Two source entries — one glob per extension — because fnmatch doesn't
    # expand brace groups. Realistic for users with mixed corpora.
    cfg_path = wiki / "dikw.yml"
    cfg = load_config(cfg_path)
    cfg.sources = [
        SourceConfig(path="./sources", pattern="**/*.md"),
        SourceConfig(path="./sources", pattern="**/*.html"),
    ]
    cfg_path.write_text(dump_config_yaml(cfg), encoding="utf-8")

    dest = wiki / "sources" / "mixed"
    dest.mkdir(parents=True, exist_ok=True)
    for src in FIXTURES.iterdir():
        if src.is_file():
            shutil.copy2(src, dest / src.name)
    return wiki


@pytest.mark.asyncio
async def test_mixed_ingest_parses_md_and_html(wiki_mixed: Path) -> None:
    report = await api.ingest(wiki_mixed, embedder=FakeEmbeddings())
    assert report.scanned == 2
    assert report.added == 2
    assert report.chunks >= 2
    assert report.embedded >= 2

    # Re-running is idempotent across backends too.
    again = await api.ingest(wiki_mixed, embedder=FakeEmbeddings())
    assert again.unchanged == 2
    assert again.added == 0
