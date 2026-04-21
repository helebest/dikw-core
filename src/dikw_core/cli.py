"""``dikw`` CLI — thin wrapper around ``dikw_core.api``.

Phase 0 commands: ``init``, ``status``, ``version``. More land with subsequent
phases. All commands validate arguments with typer + Pydantic and delegate
real work to the ``api`` module so the MCP server can reuse the same logic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, api

app = typer.Typer(
    name="dikw",
    help="AI-native knowledge engine — Data · Information · Knowledge · Wisdom",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command("version")
def version_cmd() -> None:
    """Print the dikw-core package version."""
    console.print(__version__)


@app.command("init")
def init_cmd(
    path: Annotated[
        Path,
        typer.Argument(help="Directory to scaffold the wiki into. Created if it doesn't exist."),
    ] = Path("."),
    description: Annotated[
        str,
        typer.Option("--description", "-d", help="One-line description for dikw.yml."),
    ] = "",
) -> None:
    """Scaffold a new dikw wiki at PATH."""
    try:
        root = api.init_wiki(path, description=description or None)
    except FileExistsError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    console.print(f"[green]initialized[/green] wiki at [bold]{root}[/bold]")
    console.print("Next: add markdown under [cyan]sources/[/cyan] and run [cyan]dikw status[/cyan].")


@app.command("status")
def status_cmd(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="A path inside the wiki; dikw walks up to find dikw.yml.",
        ),
    ] = Path("."),
) -> None:
    """Show storage-backend counts for the nearest wiki."""
    try:
        counts = asyncio.run(api.status(path))
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    table = Table(title="dikw status", show_header=True, header_style="bold")
    table.add_column("layer", justify="left")
    table.add_column("count", justify="right")
    for layer in ("source", "wiki", "wisdom"):
        table.add_row(layer, str(counts.documents_by_layer.get(layer, 0)))
    table.add_row("chunks (I)", str(counts.chunks))
    table.add_row("embeddings (I)", str(counts.embeddings))
    table.add_row("links (K)", str(counts.links))
    for status_name in ("candidate", "approved", "archived"):
        table.add_row(
            f"wisdom {status_name}", str(counts.wisdom_by_status.get(status_name, 0))
        )
    console.print(table)
    if counts.last_wiki_log_ts is not None:
        console.print(f"last wiki_log ts: [dim]{counts.last_wiki_log_ts}[/dim]")


def main() -> None:  # pragma: no cover - entry point shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
