from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dikw_core.cli import app

runner = CliRunner()


def test_version_prints_non_empty() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_init_then_status_golden_path(tmp_path: Path, monkeypatch) -> None:
    wiki = tmp_path / "my-wiki"
    result = runner.invoke(app, ["init", str(wiki), "--description", "golden path"])
    assert result.exit_code == 0, result.stdout

    # scaffolded tree
    assert (wiki / "dikw.yml").is_file()
    assert (wiki / "sources").is_dir()
    assert (wiki / "wiki" / "index.md").is_file()
    assert (wiki / "wiki" / "log.md").is_file()
    assert (wiki / "wisdom" / "principles.md").is_file()
    assert (wiki / ".dikw").is_dir()
    assert (wiki / ".gitignore").read_text().strip() == ".dikw/"

    # re-running init should fail (never overwrite)
    second = runner.invoke(app, ["init", str(wiki)])
    assert second.exit_code == 1

    monkeypatch.chdir(wiki)
    status_result = runner.invoke(app, ["status"])
    assert status_result.exit_code == 0, status_result.stdout
    # Empty wiki: zero rows across all layers
    assert "source" in status_result.stdout
    assert "wisdom candidate" in status_result.stdout


def test_status_errors_outside_wiki(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "no dikw.yml" in result.stdout
