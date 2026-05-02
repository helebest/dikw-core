"""Remote CLI for dikw-core.

Thin Typer + httpx + rich client that talks to a ``dikw serve`` instance over
HTTP/NDJSON. ``client/*`` must not import any engine internals beyond
``schemas`` (used only for type alignment with server responses), so it can in
principle ship as a standalone wheel later.

Modules land in Phase 5 of the migration (see plan
`dikw-core-client-server-eventual-clarke`): ``cli_app``, ``transport``,
``progress``, ``upload``, ``config``, ``serve_and_run``.
"""
