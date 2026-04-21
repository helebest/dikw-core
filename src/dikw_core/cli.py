"""``dikw`` CLI — thin wrapper around ``dikw_core.api``.

Phase 0-2 commands: ``version``, ``init``, ``status``, ``ingest``, ``query``,
``synth``, ``lint``, ``mcp``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
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


@app.command("ingest")
def ingest_cmd(
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="A path inside the wiki."),
    ] = Path("."),
    no_embed: Annotated[
        bool,
        typer.Option(
            "--no-embed",
            help="Skip embedding (only FTS-indexed). Useful offline or before API keys are set.",
        ),
    ] = False,
) -> None:
    """Scan configured sources and update the D + I layers."""
    from .providers import build_embedder

    async def _run() -> api.IngestReport:
        embedder = None
        if not no_embed:
            cfg, _ = api.load_wiki(path)
            embedder = build_embedder(cfg.provider)
        return await api.ingest(path, embedder=embedder)

    try:
        report = asyncio.run(_run())
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    table = Table(title="dikw ingest", show_header=True, header_style="bold")
    table.add_column("metric", justify="left")
    table.add_column("count", justify="right")
    table.add_row("scanned", str(report.scanned))
    table.add_row("added", str(report.added))
    table.add_row("updated", str(report.updated))
    table.add_row("unchanged", str(report.unchanged))
    table.add_row("chunks", str(report.chunks))
    table.add_row("embeddings", str(report.embedded))
    console.print(table)


@app.command("query")
def query_cmd(
    question: Annotated[str, typer.Argument(help="Natural-language question.")],
    path: Annotated[
        Path, typer.Option("--path", "-p", help="A path inside the wiki.")
    ] = Path("."),
    limit: Annotated[
        int, typer.Option("--limit", "-k", help="Number of excerpts to retrieve.")
    ] = 5,
) -> None:
    """Answer QUESTION using the wiki as context, citing sources."""
    try:
        result = asyncio.run(api.query(question, path, limit=limit))
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print(Panel(result.answer, title="answer", border_style="cyan"))
    if not result.citations:
        console.print("[dim]no citations[/dim]")
        return

    table = Table(title="citations", show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("layer")
    table.add_column("path")
    table.add_column("excerpt")
    for c in result.citations:
        table.add_row(str(c.n), c.layer, c.path, c.excerpt[:120] + ("…" if len(c.excerpt) > 120 else ""))
    console.print(table)


@app.command("synth")
def synth_cmd(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="A path inside the wiki.")
    ] = Path("."),
    force_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Re-synthesise every source, even those already turned into wiki pages.",
        ),
    ] = False,
    no_embed: Annotated[
        bool,
        typer.Option(
            "--no-embed",
            help="Skip embedding of generated wiki pages (still written to disk and FTS).",
        ),
    ] = False,
) -> None:
    """Turn source docs into K-layer wiki pages via the configured LLM."""
    from .providers import build_embedder

    async def _run() -> api.SynthReport:
        embedder = None
        if not no_embed:
            cfg, _ = api.load_wiki(path)
            embedder = build_embedder(cfg.provider)
        return await api.synthesize(path, force_all=force_all, embedder=embedder)

    try:
        report = asyncio.run(_run())
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    table = Table(title="dikw synth", show_header=True, header_style="bold")
    table.add_column("metric", justify="left")
    table.add_column("count", justify="right")
    table.add_row("candidates", str(report.candidates))
    table.add_row("created", str(report.created))
    table.add_row("updated", str(report.updated))
    table.add_row("skipped", str(report.skipped))
    table.add_row("errors", str(report.errors))
    console.print(table)


@app.command("lint")
def lint_cmd(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="A path inside the wiki.")
    ] = Path("."),
) -> None:
    """Report broken wikilinks, orphan pages, and duplicate titles."""
    try:
        report = asyncio.run(api.lint(path))
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    if report.ok:
        console.print("[green]lint clean[/green] — 0 issues")
        return

    summary = " · ".join(f"{kind}: {n}" for kind, n in sorted(report.by_kind().items()))
    console.print(f"[yellow]lint issues[/yellow] — {summary}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("kind")
    table.add_column("path")
    table.add_column("line", justify="right")
    table.add_column("detail")
    for issue in report.issues:
        table.add_row(
            issue.kind,
            issue.path,
            str(issue.line) if issue.line is not None else "",
            issue.detail,
        )
    console.print(table)
    raise typer.Exit(code=1)


@app.command("mcp")
def mcp_cmd(
    stdio: Annotated[bool, typer.Option("--stdio", help="Use stdio transport.")] = True,
) -> None:
    """Launch the MCP server over the chosen transport."""
    if not stdio:
        console.print("[yellow]HTTP transport lands in Phase 4; running stdio.[/yellow]")
    from .mcp_server import run_stdio

    run_stdio()


def main() -> None:  # pragma: no cover - entry point shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
