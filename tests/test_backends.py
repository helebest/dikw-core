from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.domains.data.backends import (
    MarkdownBackend,
    UnsupportedFormat,
    get_backend,
    parse_any,
    supported_extensions,
)


def test_registry_covers_markdown_only() -> None:
    exts = supported_extensions()
    for ext in (".md", ".markdown"):
        assert ext in exts
    assert isinstance(get_backend(Path("note.md")), MarkdownBackend)


def test_parse_any_raises_for_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.7\n")
    with pytest.raises(UnsupportedFormat):
        parse_any(f, rel_path="x.pdf")


def test_markdown_backend_still_works(tmp_path: Path) -> None:
    page = tmp_path / "note.md"
    page.write_text("---\ntitle: Hello\n---\n\nBody text.", encoding="utf-8")
    parsed = parse_any(page, rel_path="note.md")
    assert parsed.title == "Hello"
    assert "Body text" in parsed.body
