"""``dikw`` CLI — thin wrapper around ``dikw_core.api``.

Phase 0-3 commands: ``version``, ``init``, ``status``, ``ingest``, ``query``,
``synth``, ``lint``, ``distill``, ``review``, ``mcp``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal

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


@app.command("check")
def check_cmd(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="A path inside the wiki; dikw walks up to find dikw.yml.",
        ),
    ] = Path("."),
    llm_only: Annotated[
        bool,
        typer.Option(
            "--llm-only",
            help="Probe only the LLM leg; skip embedding config and ping.",
        ),
    ] = False,
    embed_only: Annotated[
        bool,
        typer.Option(
            "--embed-only",
            help="Probe only the embedding leg; skip LLM config and ping.",
        ),
    ] = False,
) -> None:
    """Verify the configured LLM + embedding providers by pinging each endpoint."""
    if llm_only and embed_only:
        console.print("[red]error:[/red] --llm-only and --embed-only are mutually exclusive")
        raise typer.Exit(code=2)
    try:
        report = asyncio.run(
            api.check_providers(path, llm_only=llm_only, embed_only=embed_only)
        )
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    table = Table(title="dikw check", show_header=True, header_style="bold")
    table.add_column("provider", justify="left")
    table.add_column("target", justify="left")
    table.add_column("status", justify="left")
    table.add_column("detail", justify="left")
    for label, probe in (("LLM", report.llm), ("Embedding", report.embed)):
        if probe is None:
            continue
        status = "[green]✓ OK[/green]" if probe.ok else "[red]✗ FAIL[/red]"
        table.add_row(label, probe.target, status, probe.detail)
    console.print(table)

    if not report.ok:
        raise typer.Exit(code=1)


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
    table.add_column("seq", justify="right")
    table.add_column("excerpt")
    for c in result.citations:
        table.add_row(
            str(c.n),
            c.layer,
            c.path,
            "" if c.seq is None else str(c.seq),
            c.excerpt[:120] + ("…" if len(c.excerpt) > 120 else ""),
        )
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


@app.command("distill")
def distill_cmd(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="A path inside the wiki.")
    ] = Path("."),
    pages_per_call: Annotated[
        int,
        typer.Option(
            "--batch",
            help="How many K-layer pages to pack into one distill LLM call.",
        ),
    ] = 8,
) -> None:
    """Propose W-layer candidates from the current K-layer wiki."""
    try:
        report = asyncio.run(api.distill(path, pages_per_call=pages_per_call))
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    table = Table(title="dikw distill", show_header=True, header_style="bold")
    table.add_column("metric", justify="left")
    table.add_column("count", justify="right")
    table.add_row("K pages read", str(report.pages_read))
    table.add_row("candidates added", str(report.candidates_added))
    table.add_row("rejected (invariant)", str(report.rejected))
    table.add_row("errors", str(report.errors))
    console.print(table)
    if report.candidates_added:
        console.print(
            "Review with [cyan]dikw review list[/cyan] / "
            "[cyan]dikw review approve <id>[/cyan]."
        )


review_app = typer.Typer(help="Review wisdom candidates.", no_args_is_help=True)
app.add_typer(review_app, name="review")


@review_app.command("list")
def review_list_cmd(
    path: Annotated[
        Path, typer.Option("--path", "-p", help="A path inside the wiki.")
    ] = Path("."),
) -> None:
    """List candidate W-layer items awaiting review."""
    try:
        items = asyncio.run(api.list_candidates(path))
    except FileNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    if not items:
        console.print("[green]no candidates[/green]")
        return

    table = Table(title="wisdom candidates", show_header=True, header_style="bold")
    table.add_column("id")
    table.add_column("kind")
    table.add_column("conf", justify="right")
    table.add_column("title")
    for item in items:
        table.add_row(
            item.item_id, item.kind.value, f"{item.confidence:.2f}", item.title
        )
    console.print(table)


@review_app.command("approve")
def review_approve_cmd(
    item_id: Annotated[str, typer.Argument(help="Wisdom item id (W-xxxxxx).")],
    path: Annotated[
        Path, typer.Option("--path", "-p", help="A path inside the wiki.")
    ] = Path("."),
) -> None:
    """Approve a candidate — promote it to approved and refresh the aggregate."""
    try:
        result = asyncio.run(api.approve_wisdom(item_id, path))
    except (FileNotFoundError, api.ReviewError) as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    console.print(f"[green]{result.item_id} -> {result.new_status.value}[/green]")


@review_app.command("reject")
def review_reject_cmd(
    item_id: Annotated[str, typer.Argument(help="Wisdom item id (W-xxxxxx).")],
    path: Annotated[
        Path, typer.Option("--path", "-p", help="A path inside the wiki.")
    ] = Path("."),
) -> None:
    """Reject a candidate — archive it and drop the candidate file."""
    try:
        result = asyncio.run(api.reject_wisdom(item_id, path))
    except (FileNotFoundError, api.ReviewError) as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    console.print(f"[yellow]{result.item_id} -> {result.new_status.value}[/yellow]")


@app.command("eval")
def eval_cmd(
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help=(
                "Dataset name (resolved under the packaged datasets root) or "
                "a filesystem path to a dataset directory. Omit to run every "
                "packaged dataset."
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Stream ingest progress (per-batch embed markers) to stderr.",
        ),
    ] = False,
    embedder_mode: Annotated[
        str,
        typer.Option(
            "--embedder",
            help="'fake' (hermetic, default) or 'provider' (use wiki's configured provider).",
        ),
    ] = "fake",
    wiki_path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="A path inside a wiki; required when --embedder provider.",
        ),
    ] = Path("."),
    retrieval: Annotated[
        str,
        typer.Option(
            "--retrieval",
            help=(
                "Retrieval mode to score: 'hybrid' (default), 'bm25', "
                "'vector', or 'all' (run all three for ablation)."
            ),
        ),
    ] = "hybrid",
    dump_raw: Annotated[
        Path | None,
        typer.Option(
            "--dump-raw",
            help=(
                "Append per-(query, mode) top-100 ranked lists to this "
                "JSONL file. Only meaningful with --retrieval all — "
                "``evals/tools/sweep_rrf.py`` consumes this to re-fuse "
                "offline without re-embedding."
            ),
        ),
    ] = None,
    rebuild_cache: Annotated[
        bool,
        typer.Option(
            "--rebuild-cache",
            help=(
                "Force a cold rebuild of the eval-snapshot cache "
                "(evals/.cache/snapshots/) for each dataset. Mutually "
                "exclusive with --no-cache."
            ),
        ),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help=(
                "Disable the eval-snapshot cache for this run; ingest "
                "into a tempdir and discard. Mutually exclusive with "
                "--rebuild-cache."
            ),
        ),
    ] = False,
) -> None:
    """Run retrieval-quality evaluation against one or every packaged dataset."""
    # Lazy imports: keep top-of-module `from . import api` light for the
    # non-eval commands and avoid circular imports with dikw_core.eval.
    import logging
    import sys

    from .eval.dataset import DatasetError
    from .eval.runner import run_eval

    if verbose:
        # Per-batch embed markers + any other INFO-level progress logs go
        # to stderr; the rich tables stay on stdout untouched.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(message)s",
            stream=sys.stderr,
        )

    valid_modes = {"bm25", "vector", "hybrid", "all"}
    if retrieval not in valid_modes:
        console.print(
            f"[red]error:[/red] --retrieval must be one of {sorted(valid_modes)}, "
            f"got {retrieval!r}"
        )
        raise typer.Exit(code=2)

    if rebuild_cache and no_cache:
        console.print(
            "[red]error:[/red] --rebuild-cache and --no-cache are mutually exclusive"
        )
        raise typer.Exit(code=2)
    cache_mode: Literal["read_write", "rebuild", "off"] = (
        "rebuild" if rebuild_cache else "off" if no_cache else "read_write"
    )

    # --dump-raw needs both legs' rankings to be useful; warn-and-ignore
    # in single-mode runs rather than silently writing a file the sweep
    # tool would reject anyway.
    if dump_raw is not None and retrieval != "all":
        console.print(
            "[yellow]warning:[/yellow] --dump-raw is ignored unless "
            "--retrieval all; skipping."
        )
        dump_raw = None

    if dump_raw is not None:
        # Truncate before the first run_eval — each dataset then appends
        # its rows. Avoids silent contamination from an old sweep.
        dump_raw.parent.mkdir(parents=True, exist_ok=True)
        dump_raw.write_text("", encoding="utf-8")

    # Resolve which datasets to run.
    try:
        specs = _collect_eval_specs(dataset)
    except DatasetError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    if not specs:
        from .eval.dataset import datasets_root

        root = datasets_root()
        console.print(
            f"[red]error:[/red] no datasets found under {root}. "
            "Create one as a `<name>/` subdirectory with dataset.yaml, "
            "corpus/, queries.yaml."
        )
        raise typer.Exit(code=2)

    # Build the embedder + provider config once if --embedder provider.
    embedder = None
    provider_cfg = None
    retrieval_cfg = None
    assets_cfg = None
    multimodal_embedder = None
    if embedder_mode == "provider":
        from .providers import build_embedder, build_multimodal_embedder

        try:
            cfg, _root = api.load_wiki(wiki_path)
        except FileNotFoundError as e:
            console.print(f"[red]error:[/red] {e}")
            raise typer.Exit(code=2) from e
        embedder = build_embedder(cfg.provider)
        # Forward all three config blocks — runner otherwise picks up
        # RetrievalConfig() / AssetsConfig() defaults and silently
        # disables per-wiki cjk_tokenizer / weight / multimodal overrides.
        provider_cfg = cfg.provider
        retrieval_cfg = cfg.retrieval
        assets_cfg = cfg.assets
        if cfg.assets.multimodal is not None:
            multimodal_embedder = build_multimodal_embedder(
                cfg.assets.multimodal.provider,
                base_url=cfg.assets.multimodal.base_url,
                batch=cfg.assets.multimodal.batch,
            )
    elif embedder_mode != "fake":
        console.print(
            f"[red]error:[/red] --embedder must be 'fake' or 'provider', got {embedder_mode!r}"
        )
        raise typer.Exit(code=2)

    all_passed = True
    for spec in specs:
        try:
            report = asyncio.run(
                run_eval(
                    spec,
                    embedder=embedder,
                    provider_config=provider_cfg,
                    retrieval_config=retrieval_cfg,
                    assets_config=assets_cfg,
                    multimodal_embedder=multimodal_embedder,
                    mode=retrieval,  # type: ignore[arg-type]
                    raw_dump_path=dump_raw,
                    cache_mode=cache_mode,
                )
            )
        except Exception as e:  # runner-level error — report, fail this dataset
            console.print(f"[red]error in {spec.name}:[/red] {e}")
            all_passed = False
            continue
        _print_eval_report(report)
        if not report.passed:
            all_passed = False

    raise typer.Exit(code=0 if all_passed else 1)


def _collect_eval_specs(dataset: str | None) -> list[Any]:
    """Resolve the CLI ``--dataset`` option to a list of loaded ``DatasetSpec``.

    - ``None`` → every subdirectory of ``datasets_root()`` that contains a
      ``dataset.yaml``. Incomplete stubs (missing ``corpus/`` or
      ``queries.yaml``) are *skipped with a warning*, not a hard failure
      — public-benchmark datasets ship as a committed ``dataset.yaml``
      stub plus a converter the user runs locally to materialise the
      corpus, and a stub-only ``scifact/`` directory shouldn't break
      ``dikw eval`` for someone who hasn't downloaded it yet.
    - ``"<name>"`` or ``"<path>"`` → a single loaded spec; missing pieces
      raise ``DatasetError`` so the user sees the exact problem.
    """
    from .eval.dataset import DatasetError, DatasetSpec, datasets_root, load_dataset

    if dataset is not None:
        return [load_dataset(dataset)]

    root = datasets_root()
    if not root.is_dir():
        return []
    specs: list[DatasetSpec] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "dataset.yaml").is_file():
            continue
        try:
            specs.append(load_dataset(child))
        except DatasetError as e:
            console.print(
                f"[yellow]skipping {child.name}: {e}[/yellow]"
            )
    return specs


def _print_eval_report(report: Any) -> None:
    """Render an ``EvalReport`` as a rich table with per-metric verdict.

    Two layouts depending on how many retrieval modes were exercised:
    - 1 mode (the default): a single metric table with threshold gating.
    - 3 modes ("all"): an ablation table with one row per mode and one
      column per metric. Threshold gating still applies to the canonical
      (hybrid) mode via the unprefixed metric mirror.
    """
    from .eval.runner import METRIC_KEYS

    title = f"dikw eval — {report.dataset_name}"
    modes = list(report.modes)

    if len(modes) > 1:
        ablation = Table(
            title=f"{title}  (retrieval ablation: {' / '.join(modes)})",
            show_header=True,
            header_style="bold",
        )
        ablation.add_column("mode")
        for k in METRIC_KEYS:
            ablation.add_column(k, justify="right")
        for m in modes:
            cells: list[str] = [m]
            for k in METRIC_KEYS:
                v = report.metrics.get(f"{m}/{k}")
                cells.append(f"{v:.3f}" if v is not None else "-")
            ablation.add_row(*cells)
        console.print(ablation)

    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("result")
    for key, val, thr in report.iter_metric_rows():
        val_str = f"{val:.3f}" if val is not None else "-"
        thr_str = f"{thr:.3f}" if thr is not None else "-"
        if thr is None or val is None:
            verdict = "[dim]—[/dim]"
        elif val >= thr:
            verdict = "[green]✓ pass[/green]"
        else:
            verdict = "[red]✗ FAIL[/red]"
        table.add_row(key, val_str, thr_str, verdict)
    console.print(table)

    if not report.passed:
        console.print("[yellow]per-query diagnostic (top-5):[/yellow]")
        for row in report.per_query:
            q_short = row["q"] if len(row["q"]) <= 60 else row["q"][:57] + "..."
            top5 = row["ranked"][:5]
            mark = "✓" if any(e in top5 for e in row["expect_any"]) else "✗"
            console.print(f"  {mark} {q_short}")
            console.print(f"       expected: {row['expect_any']}")
            console.print(f"       top-5:    {top5}")

    negatives = getattr(report, "negative_diagnostics", []) or []
    if negatives:
        neg_table = Table(
            title="negative queries (top-3 observed — diagnostic only)",
            show_header=True,
            header_style="bold",
        )
        neg_table.add_column("#", justify="right")
        neg_table.add_column("query")
        neg_table.add_column("top-3")
        for i, row in enumerate(negatives, start=1):
            q_short = row["q"] if len(row["q"]) <= 60 else row["q"][:57] + "..."
            top3 = ", ".join(row["ranked"][:3])
            neg_table.add_row(str(i), q_short, top3)
        console.print(neg_table)


@app.command("mcp")
def mcp_cmd(
    stdio: Annotated[bool, typer.Option("--stdio", help="Use stdio transport.")] = True,
) -> None:
    """[DEPRECATED] MCP support is being removed; use `dikw serve` instead."""
    del stdio  # unused — kept for option-compat until the command is dropped
    console.print(
        "[red]error:[/red] `dikw mcp` is deprecated and will be removed in a "
        "future release.\n"
        "The `mcp` runtime dependency has already been dropped; run "
        "[cyan]dikw serve[/cyan] (HTTP + NDJSON) instead."
    )
    raise typer.Exit(code=1)


@app.command("serve")
def serve_cmd(
    wiki: Annotated[
        Path,
        typer.Option(
            "--wiki",
            "-w",
            help="Path to the wiki root (must contain dikw.yml). Defaults to cwd.",
        ),
    ] = Path("."),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-H",
            help="Interface to bind. 0.0.0.0 requires DIKW_SERVER_TOKEN to be set.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="TCP port to bind."),
    ] = 8765,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help=(
                "Bearer token clients must present. Overrides "
                "DIKW_SERVER_TOKEN. Required when --host is non-loopback."
            ),
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="uvicorn log level."),
    ] = "info",
) -> None:
    """Start the dikw HTTP server (FastAPI + NDJSON) on this host."""
    import uvicorn

    from .server.app import build_app_from_disk
    from .server.auth import (
        ensure_auth_invariant,
        load_auth_config,
    )

    wiki_root = wiki.resolve()
    auth_cfg = load_auth_config(host=host, token_override=token)
    try:
        ensure_auth_invariant(auth_cfg)
    except RuntimeError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=2) from e

    fastapi_app = build_app_from_disk(
        wiki_root=wiki_root,
        host=host,
        token_override=token,
    )
    # Print a banner to stderr so `serve-and-run` (Phase 6) can probe
    # readiness without parsing uvicorn's own output. Logged before
    # uvicorn binds the socket — gives the operator the auth posture
    # explicitly so silent localhost-no-token isn't surprising.
    posture = "token" if auth_cfg.required else "open (localhost only, no token)"
    console.print(
        f"[green]dikw serve[/green]  wiki=[cyan]{wiki_root}[/cyan]  "
        f"bind=[cyan]http://{host}:{port}[/cyan]  auth=[cyan]{posture}[/cyan]",
        highlight=False,
    )
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level=log_level,
    )


def main() -> None:  # pragma: no cover - entry point shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
