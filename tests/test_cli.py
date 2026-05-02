"""Top-level CLI tests for the local-only commands.

After Phase 5 of the client/server migration the only commands that run
in-process are ``version``, ``init``, and ``serve``; everything else is
a thin wrapper around an HTTP call to a running ``dikw serve``.

The remote command surface (``status``, ``query``, ``ingest`` …) is
exercised end-to-end against an in-memory ASGI server in
``tests/client/test_cli_e2e.py``. This file's job is to keep the
local-only commands honest.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dikw_core.cli import app

runner = CliRunner()


def test_version_prints_non_empty() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_init_scaffolds_expected_tree(tmp_path: Path) -> None:
    wiki = tmp_path / "my-wiki"
    result = runner.invoke(
        app, ["init", str(wiki), "--description", "phase-5 cli test"]
    )
    assert result.exit_code == 0, result.stdout

    assert (wiki / "dikw.yml").is_file()
    assert (wiki / "sources").is_dir()
    assert (wiki / "wiki" / "index.md").is_file()
    assert (wiki / "wiki" / "log.md").is_file()
    assert (wiki / "wisdom" / "principles.md").is_file()
    assert (wiki / ".dikw").is_dir()
    assert (wiki / ".gitignore").read_text().strip() == ".dikw/"


def test_init_refuses_to_overwrite_existing_wiki(tmp_path: Path) -> None:
    wiki = tmp_path / "my-wiki"
    first = runner.invoke(app, ["init", str(wiki)])
    assert first.exit_code == 0

    second = runner.invoke(app, ["init", str(wiki)])
    assert second.exit_code == 1


def test_serve_help_lists_options() -> None:
    """``dikw serve --help`` should at least mention the bind-host
    and token flags so the operator can configure auth posture
    without reading source code."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--host" in out
    assert "--token" in out
    assert "--port" in out


def test_top_level_status_alias_present() -> None:
    """Top-level ``dikw status`` should exist as an alias for
    ``dikw client status`` — its presence in the command list is what
    keeps muscle memory working post-migration."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "status" in result.stdout
    assert "client" in result.stdout
