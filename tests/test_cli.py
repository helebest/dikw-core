"""Top-level CLI tests for the local-only commands.

The only top-level commands that run in-process are ``version``,
``init``, ``serve`` and the ``auth`` subgroup. Every HTTP-bound command
lives under ``dikw client *`` — there are no top-level aliases.

The remote command surface (``dikw client status``, ``dikw client
ingest`` …) is exercised end-to-end against an in-memory ASGI server in
``tests/client/test_cli_e2e.py``. This file's job is to keep the
local-only commands honest and to guard against splice regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dikw_core.cli import app

from .conftest import removed_top_level_short_names

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


def test_serve_help_lists_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dikw serve --help`` should at least mention the bind-host
    and token flags so the operator can configure auth posture
    without reading source code.

    Forces a wide terminal so rich/typer doesn't wrap option names
    across visual lines (CI's narrow default broke ``--host`` apart).
    """
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERM", "dumb")  # disable colour escapes
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--host" in out
    assert "--token" in out
    assert "--port" in out


def test_top_level_help_lists_only_local_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dikw --help`` must list only the local-only commands plus the
    ``client`` subgroup — never any of the HTTP-bound short names.

    Forces a wide terminal so rich/typer doesn't visually wrap command
    names across lines (a narrow shell would split ``serve-and-run``
    and a naïve substring search would miss it)."""
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("TERM", "dumb")
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for expected in ("version", "init", "serve", "auth", "client"):
        assert expected in out, f"missing local command {expected!r} in --help"
    # First whitespace-delimited token on each help row is a command name
    # (or a section header / option flag — both safe). Forbidden names
    # appearing as a row's leading token mean the splice resurrected.
    cmd_starts = {
        stripped.split(" ", 1)[0]
        for line in out.splitlines()
        if (stripped := line.strip())
    }
    for forbidden in removed_top_level_short_names():
        assert forbidden not in cmd_starts, (
            f"HTTP-bound short name {forbidden!r} leaked into top-level "
            f"--help; got command-row tokens: {sorted(cmd_starts)}"
        )
