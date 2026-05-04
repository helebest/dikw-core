from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.domains.data.backends import (
    HtmlBackend,
    MarkdownBackend,
    UnsupportedFormat,
    get_backend,
    parse_any,
    supported_extensions,
)


def test_registry_covers_markdown_and_html() -> None:
    exts = supported_extensions()
    for ext in (".md", ".markdown", ".html", ".htm"):
        assert ext in exts
    assert isinstance(get_backend(Path("note.md")), MarkdownBackend)
    assert isinstance(get_backend(Path("PAGE.HTML")), HtmlBackend)


def test_parse_any_raises_for_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.7\n")
    with pytest.raises(UnsupportedFormat):
        parse_any(f, rel_path="x.pdf")


def test_html_backend_extracts_title_and_body(tmp_path: Path) -> None:
    page = tmp_path / "sample.html"
    page.write_text(
        "<html><head><title>  DIKW sample  </title></head>"
        "<body>"
        "<script>alert('hi')</script>"
        "<h1>DIKW sample</h1>"
        "<p>This page exercises the HTML backend registry.</p>"
        "<ul><li>First bullet</li><li>Second bullet</li></ul>"
        "</body></html>",
        encoding="utf-8",
    )
    parsed = parse_any(page, rel_path="sources/sample.html")
    assert parsed.path == "sources/sample.html"
    assert parsed.title == "DIKW sample"
    # script contents must have been dropped
    assert "alert" not in parsed.body
    # body should be plain text with bullet conversion
    assert "First bullet" in parsed.body
    assert "Second bullet" in parsed.body
    # hash is stable and non-empty
    assert parsed.hash and len(parsed.hash) == 64


def test_html_backend_title_falls_back_to_h1_then_stem(tmp_path: Path) -> None:
    page = tmp_path / "no-title.html"
    page.write_text("<html><body><h1>From H1</h1><p>body</p></body></html>", encoding="utf-8")
    parsed = parse_any(page, rel_path="no-title.html")
    assert parsed.title == "From H1"

    bare = tmp_path / "bare-filename.html"
    bare.write_text("<html><body><p>only paragraph</p></body></html>", encoding="utf-8")
    parsed = parse_any(bare, rel_path="bare-filename.html")
    assert parsed.title == "bare-filename"


def test_markdown_backend_still_works(tmp_path: Path) -> None:
    page = tmp_path / "note.md"
    page.write_text("---\ntitle: Hello\n---\n\nBody text.", encoding="utf-8")
    parsed = parse_any(page, rel_path="note.md")
    assert parsed.title == "Hello"
    assert "Body text" in parsed.body
