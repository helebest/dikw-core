"""``dikw client *`` Typer app.

Every command resolves a :class:`ClientConfig`, opens a single
:class:`Transport`, calls the matching HTTP endpoint, and renders the
response. Long ops (ingest / synth / distill / eval) submit a task,
follow its NDJSON event stream, and dispatch to the op-specific final
renderer; sync ops just decode the JSON body and render directly.

Each command body sits inside an ``async def`` and is driven by
``asyncio.run`` from the Typer wrapper — this keeps the transport pool
lifecycle bounded to a single command invocation. We deliberately don't
share a transport across commands: a CLI run is short, the
``AsyncClient`` constructor is cheap, and per-command lifetime makes
cancel-on-Ctrl-C work without extra plumbing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from . import serve_and_run as _sar
from .config import ClientConfig, resolve
from .progress import (
    QueryStreamRenderer,
    TaskProgressRenderer,
    render_check_report,
    render_distill_report,
    render_eval_report,
    render_ingest_report,
    render_lint_report,
    render_status,
    render_synth_report,
)
from .transport import ClientError, Transport
from .upload import UploadError, build_upload

app = typer.Typer(
    name="client",
    help="Talk to a running ``dikw serve`` instance.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# ---- shared options ----------------------------------------------------


def _server_option() -> Any:
    # Inside ``Annotated[…, typer.Option(...)]`` Typer expects the
    # *param decls* only — the default value is supplied by the
    # parameter's ``= None`` assignment. Passing ``None`` as the first
    # argument here is the legacy non-Annotated form and trips a
    # confusing ``isidentifier`` AttributeError deep in click.
    return typer.Option(
        "--server",
        help="Server URL. Default: env $DIKW_SERVER_URL or http://127.0.0.1:8765.",
    )


def _token_option() -> Any:
    return typer.Option(
        "--token",
        help="Bearer token. Default: env $DIKW_SERVER_TOKEN or client.toml.",
    )


def _resolve(server: str | None, token: str | None) -> ClientConfig:
    return resolve(server_url=server, token=token)


def _on_error(err: ClientError) -> None:
    """Translate a transport-layer error into a terse stderr line + exit.

    ``cancelled`` is the one expected non-zero status — surface it as a
    yellow notice rather than red so users don't think their cancel
    command failed.
    """
    if err.status == 0:
        console.print(f"[red]network error:[/red] {err.message}")
    else:
        console.print(
            f"[red]error[/red] [{err.status} {err.code}]: {err.message}"
        )
    if err.detail:
        console.print(f"[dim]detail: {err.detail}[/dim]")


def _run(coro: Any) -> Any:
    """Run an async command with a uniform error → exit-code mapping."""
    try:
        return asyncio.run(coro)
    except ClientError as e:
        _on_error(e)
        raise typer.Exit(code=1) from e
    except UploadError as e:
        console.print(f"[red]upload error:[/red] {e}")
        raise typer.Exit(code=1) from e


# ---- meta + sync commands ---------------------------------------------


@app.command("info")
def info_cmd(
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Print the server's ``GET /v1/info`` response."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.get_json("/v1/info")
        console.print(json.dumps(payload, indent=2))

    _run(_go())


@app.command("status")
def status_cmd(
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Show storage-backend counts."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            counts = await t.get_json("/v1/status")
        render_status(console, counts)

    _run(_go())


@app.command("check")
def check_cmd(
    llm_only: Annotated[
        bool, typer.Option("--llm-only", help="Probe only the LLM leg.")
    ] = False,
    embed_only: Annotated[
        bool,
        typer.Option("--embed-only", help="Probe only the embedding leg."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Verify configured providers via the server."""
    if llm_only and embed_only:
        console.print(
            "[red]error:[/red] --llm-only and --embed-only are mutually exclusive"
        )
        raise typer.Exit(code=2)

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            report = await t.post_json(
                "/v1/check",
                json_body={"llm_only": llm_only, "embed_only": embed_only},
            )
        render_check_report(console, report)
        # ``CheckReport.ok`` is a ``@property`` that pydantic drops on
        # serialization; recompute here from the per-leg probe results
        # so the exit code matches the engine's intent.
        legs = [
            report.get("llm"),
            report.get("embed"),
        ]
        present = [leg for leg in legs if isinstance(leg, dict)]
        if not present or not all(bool(leg.get("ok")) for leg in present):
            raise typer.Exit(code=1)

    _run(_go())


@app.command("init")
def init_cmd(
    description: Annotated[
        str | None,
        typer.Option(
            "--description",
            "-d",
            help="One-line description for dikw.yml on the server.",
        ),
    ] = None,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Ask the server to scaffold its bound wiki (no-op when already scaffolded).

    For starting a fresh wiki *locally*, use the top-level ``dikw init``
    command — it works without a running server.
    """

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.post_json(
                "/v1/init",
                json_body={"description": description}
                if description is not None
                else {},
            )
        console.print(f"[green]initialized[/green] wiki at {payload.get('root')}")

    _run(_go())


@app.command("lint")
def lint_cmd(
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Run lint against the server's wiki."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            report = await t.post_json("/v1/lint")
        render_lint_report(console, report)
        # ``LintReport`` is a dataclass with ``ok`` defined as a
        # ``@property``; pydantic's response serializer drops properties
        # so the wire shape is just ``{"issues": [...]}``. Compute
        # ``ok`` from issue presence here so CI can still gate on the
        # exit code.
        issues = report.get("issues") or []
        if isinstance(issues, list) and issues:
            raise typer.Exit(code=1)

    _run(_go())


# ---- query (NDJSON stream) --------------------------------------------


@app.command("query")
def query_cmd(
    question: Annotated[str, typer.Argument(help="Natural-language question.")],
    limit: Annotated[
        int, typer.Option("--limit", "-k", help="Excerpts to retrieve.")
    ] = 5,
    show_hits: Annotated[
        bool,
        typer.Option(
            "--show-hits",
            help="Also dump raw retrieval hits after the citations table.",
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option(
            "--plain",
            help="Disable rich rendering (useful for piping into other tools).",
        ),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Answer ``question`` via streaming NDJSON."""

    async def _go() -> None:
        renderer = QueryStreamRenderer(
            console, plain=plain, show_hits=show_hits
        )
        async with (
            Transport.from_config(_resolve(server, token)) as t,
            t.stream_ndjson(
                "POST",
                "/v1/query",
                json_body={"q": question, "limit": limit},
            ) as events,
        ):
            final = await renderer.run(events)
        if final.status != "succeeded":
            console.print(f"[red]query {final.status}[/red]")
            if final.error:
                console.print(f"[dim]{final.error}[/dim]")
            raise typer.Exit(code=1)

    _run(_go())


# ---- async task commands ----------------------------------------------


async def _follow_task(
    t: Transport,
    *,
    submit_path: str,
    body: dict[str, Any],
    plain: bool,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    handle = await t.post_json(submit_path, json_body=body)
    task_id = str(handle["task_id"])
    renderer = TaskProgressRenderer(console, plain=plain)
    with renderer.live():
        async with t.stream_ndjson(
            "GET", f"/v1/tasks/{task_id}/events"
        ) as events:
            final = await renderer.run(events)
    return final.status, final.result, final.error


def _exit_on_failure(
    status: str, error: dict[str, Any] | None
) -> None:
    if status == "succeeded":
        return
    console.print(f"[red]task {status}[/red]")
    if error:
        console.print(f"[dim]{error}[/dim]")
    raise typer.Exit(code=1)


@app.command("ingest")
def ingest_cmd(
    from_dir: Annotated[
        Path | None,
        typer.Option(
            "--from",
            "-f",
            help=(
                "Local directory to upload as the wiki's sources before "
                "ingest. If omitted, the server ingests whatever is "
                "already on its disk."
            ),
        ),
    ] = None,
    no_embed: Annotated[
        bool,
        typer.Option("--no-embed", help="Skip the dense embedding pass."),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Upload sources (optional) + run ingest, streaming progress."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            upload_id: str | None = None
            if from_dir is not None:
                with build_upload(from_dir) as bundle:
                    response = await t.post_multipart(
                        "/v1/upload/sources",
                        files={
                            "payload": (
                                "payload.tar.gz",
                                bundle.payload,
                                "application/gzip",
                            )
                        },
                        data={"manifest": bundle.manifest_json},
                    )
                upload_id = str(response["upload_id"])
                console.print(
                    f"[green]uploaded[/green] {response.get('files_count')} "
                    f"file(s), {response.get('bytes')} bytes "
                    f"(id={upload_id})"
                )
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/ingest",
                body={"upload_id": upload_id, "no_embed": no_embed},
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            render_ingest_report(console, result)
        _exit_on_failure(status, error)

    _run(_go())


@app.command("synth")
def synth_cmd(
    force_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Re-synthesise every source, even ones already wiki-ified.",
        ),
    ] = False,
    no_embed: Annotated[
        bool,
        typer.Option(
            "--no-embed",
            help="Skip embedding the generated K-layer pages.",
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Synthesise K-layer wiki pages from D-layer sources."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/synth",
                body={"force_all": force_all, "no_embed": no_embed},
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            render_synth_report(console, result)
        _exit_on_failure(status, error)

    _run(_go())


@app.command("distill")
def distill_cmd(
    pages_per_call: Annotated[
        int,
        typer.Option(
            "--batch",
            help="K-layer pages packed into one distill LLM call.",
        ),
    ] = 8,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Propose W-layer candidates from current K-layer pages."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/distill",
                body={"pages_per_call": pages_per_call},
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            render_distill_report(console, result)
            if int(result.get("candidates_added") or 0):
                console.print(
                    "Review with [cyan]dikw client review list[/cyan] / "
                    "[cyan]dikw client review approve <id>[/cyan]."
                )
        _exit_on_failure(status, error)

    _run(_go())


@app.command("eval")
def eval_cmd(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help=(
                "Dataset name (resolved on the server) or path on the "
                "server. The client doesn't ship dataset bytes — the "
                "server reads them from its packaged datasets root."
            ),
        ),
    ],
    mode: Annotated[
        str,
        typer.Option(
            "--retrieval",
            help="Retrieval mode: hybrid|bm25|vector|all.",
        ),
    ] = "hybrid",
    cache_mode: Annotated[
        str,
        typer.Option(
            "--cache",
            help="Eval-snapshot cache: read_write|rebuild|off.",
        ),
    ] = "read_write",
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Disable progress widget."),
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Run a packaged retrieval-eval dataset on the server."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            status, result, error = await _follow_task(
                t,
                submit_path="/v1/eval",
                body={
                    "dataset": dataset,
                    "mode": mode,
                    "cache_mode": cache_mode,
                },
                plain=plain,
            )
        if status == "succeeded" and result is not None:
            render_eval_report(console, result)
            if not bool(result.get("passed", True)):
                raise typer.Exit(code=1)
        _exit_on_failure(status, error)

    _run(_go())


# ---- review subcommands -----------------------------------------------

review_app = typer.Typer(
    help="Review wisdom candidates.", no_args_is_help=True
)
app.add_typer(review_app, name="review")


@review_app.command("list")
def review_list_cmd(
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """List candidate W-layer items awaiting review."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            items = await t.get_json(
                "/v1/wisdom", params={"status": "candidate"}
            )
        if not items:
            console.print("[green]no candidates[/green]")
            return
        table = Table(
            title="wisdom candidates", show_header=True, header_style="bold"
        )
        table.add_column("id")
        table.add_column("kind")
        table.add_column("conf", justify="right")
        table.add_column("title")
        for item in items:
            if not isinstance(item, dict):
                continue
            conf_val = item.get("confidence")
            conf_str = (
                f"{float(conf_val):.2f}" if isinstance(conf_val, int | float) else "-"
            )
            table.add_row(
                str(item.get("item_id") or ""),
                str(item.get("kind") or ""),
                conf_str,
                str(item.get("title") or ""),
            )
        console.print(table)

    _run(_go())


@review_app.command("approve")
def review_approve_cmd(
    item_id: Annotated[str, typer.Argument(help="Wisdom item id (W-xxxxxx).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Approve a candidate."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            result = await t.post_json(f"/v1/wisdom/{item_id}/approve")
        console.print(
            f"[green]{result.get('item_id')} -> "
            f"{result.get('new_status')}[/green]"
        )

    _run(_go())


@review_app.command("reject")
def review_reject_cmd(
    item_id: Annotated[str, typer.Argument(help="Wisdom item id (W-xxxxxx).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Reject a candidate."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            result = await t.post_json(f"/v1/wisdom/{item_id}/reject")
        console.print(
            f"[yellow]{result.get('item_id')} -> "
            f"{result.get('new_status')}[/yellow]"
        )

    _run(_go())


# ---- tasks subcommands ------------------------------------------------

tasks_app = typer.Typer(
    help="Inspect server-side async tasks.", no_args_is_help=True
)
app.add_typer(tasks_app, name="tasks")


@tasks_app.command("list")
def tasks_list_cmd(
    op: Annotated[
        str | None, typer.Option("--op", help="Filter by op name.")
    ] = None,
    status_filter: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Filter by status (pending|running|succeeded|failed|cancelled).",
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max rows to return.")
    ] = 100,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """List server-side tasks."""

    async def _go() -> None:
        params: dict[str, Any] = {"limit": limit}
        if op is not None:
            params["op"] = op
        if status_filter is not None:
            params["status"] = status_filter
        async with Transport.from_config(_resolve(server, token)) as t:
            rows = await t.get_json("/v1/tasks", params=params)
        if not rows:
            console.print("[dim]no tasks[/dim]")
            return
        table = Table(title="tasks", show_header=True, header_style="bold")
        table.add_column("task_id")
        table.add_column("op")
        table.add_column("status")
        table.add_column("created_at")
        for row in rows:
            if not isinstance(row, dict):
                continue
            table.add_row(
                str(row.get("task_id") or ""),
                str(row.get("op") or ""),
                str(row.get("status") or ""),
                str(row.get("created_at") or ""),
            )
        console.print(table)

    _run(_go())


@tasks_app.command("show")
def tasks_show_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Print the JSON snapshot of a task row."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            row = await t.get_json(f"/v1/tasks/{task_id}")
        console.print(json.dumps(row, indent=2))

    _run(_go())


@tasks_app.command("follow")
def tasks_follow_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    from_seq: Annotated[
        int, typer.Option("--from-seq", help="Resume from this seq number.")
    ] = 0,
    plain: Annotated[
        bool, typer.Option("--plain", help="Disable progress widget.")
    ] = False,
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Subscribe to a task's NDJSON event stream."""

    async def _go() -> None:
        renderer = TaskProgressRenderer(console, plain=plain)
        async with Transport.from_config(_resolve(server, token)) as t:
            with renderer.live():
                async with t.stream_ndjson(
                    "GET",
                    f"/v1/tasks/{task_id}/events",
                    params={"from_seq": from_seq} if from_seq else None,
                ) as events:
                    final = await renderer.run(events)
        if final.status != "succeeded":
            console.print(f"[yellow]task {final.status}[/yellow]")
            if final.error:
                console.print(f"[dim]{final.error}[/dim]")
            raise typer.Exit(code=1)

    _run(_go())


@tasks_app.command("cancel")
def tasks_cancel_cmd(
    task_id: Annotated[str, typer.Argument(help="Task id (12-char hex).")],
    server: Annotated[str | None, _server_option()] = None,
    token: Annotated[str | None, _token_option()] = None,
) -> None:
    """Request cancellation of a running task."""

    async def _go() -> None:
        async with Transport.from_config(_resolve(server, token)) as t:
            payload = await t.post_json(f"/v1/tasks/{task_id}/cancel")
        if payload.get("already_terminal"):
            console.print(
                f"[dim]task {task_id} already terminal — no-op[/dim]"
            )
        else:
            console.print(
                f"[yellow]cancel requested[/yellow] for {task_id}"
            )

    _run(_go())


# ---- serve-and-run ----------------------------------------------------


@app.command(
    "serve-and-run",
    help=(
        "Start a local server, run an inner CLI command against it, "
        "and tear it down. Pass the inner command after ``--``."
    ),
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def serve_and_run_cmd(
    ctx: typer.Context,
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
            help="Interface to bind. 0.0.0.0 requires --token.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="TCP port to bind. ``0`` (default) picks a free one.",
        ),
    ] = 0,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help=(
                "Bearer token. Forwarded to both the server (--token) "
                "and the inner client (DIKW_SERVER_TOKEN). Required when "
                "--host is non-loopback."
            ),
        ),
    ] = None,
    ready_timeout: Annotated[
        float,
        typer.Option(
            "--ready-timeout",
            help="Seconds to wait for /v1/healthz before giving up.",
        ),
    ] = 30.0,
    keep_alive: Annotated[
        bool,
        typer.Option(
            "--keep-alive",
            help=(
                "After the inner command exits, leave the server "
                "running and print its connection details."
            ),
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="uvicorn log level for the spawned server.",
        ),
    ] = "warning",
) -> None:
    """One-shot server + inner-command lifecycle.

    Examples:

        dikw client serve-and-run -- status
        dikw client serve-and-run --wiki ./my-wiki -- ingest --no-embed
        dikw client serve-and-run --keep-alive -- query "..."
    """
    inner_cmd = list(ctx.args)
    opts = _sar.ServeAndRunOptions(
        wiki=wiki,
        host=host,
        port=port,
        token=token,
        ready_timeout=ready_timeout,
        keep_alive=keep_alive,
        log_level=log_level,
        inner_cmd=inner_cmd,
    )
    rc = _sar.run(opts)
    if rc != 0:
        raise typer.Exit(code=rc)


__all__ = ["app"]
