"""Acceptance tests for the HTML-backend removal.

Invariant: dikw-core's D layer accepts Markdown only. These tests fail
loudly if anyone re-introduces an HtmlBackend or registers .html / .htm.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.domains.data.backends import (
    UnsupportedFormat,
    parse_any,
    supported_extensions,
)


def test_html_backend_symbol_gone() -> None:
    import dikw_core.domains.data.backends as backends_pkg

    assert not hasattr(backends_pkg, "HtmlBackend")


def test_supported_extensions_are_markdown_only() -> None:
    assert set(supported_extensions()) == {".md", ".markdown"}


def test_parse_any_rejects_html(tmp_path: Path) -> None:
    page = tmp_path / "x.html"
    page.write_text("<html><body><p>hi</p></body></html>", encoding="utf-8")
    with pytest.raises(UnsupportedFormat):
        parse_any(page, rel_path="x.html")
