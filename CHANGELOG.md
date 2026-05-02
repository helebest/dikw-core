# Changelog

All notable changes to `dikw-core` are tracked here. The project is
**pre-alpha** and follows [SemVer](https://semver.org) loosely — until
1.0, breaking changes can land in any minor version. The status notes
on each entry call out exactly what shape changes break.

## Unreleased

### **BREAKING**: client/server architecture replaces in-process CLI

Phases 0–6 of the `dikw-core-client-server-eventual-clarke` plan
collapse the in-process invocation model. The engine now runs as a
long-lived `dikw serve` process; the CLI is a thin httpx + NDJSON
client that talks to it over `/v1/`.

* **Removed**: `dikw mcp` subcommand and the entire `mcp_server.py`
  module. The MCP runtime dependency is gone from `pyproject.toml` and
  every `mcp.*` reference in code and docs is scrubbed (eval-dataset
  fixture text under `evals/datasets/` is left alone — that's corpus
  content, not engine docs).
* **Removed**: in-process implementations of `dikw status`,
  `dikw check`, `dikw ingest`, `dikw query`, `dikw synth`,
  `dikw lint`, `dikw distill`, `dikw review *`, `dikw eval`. These
  commands are now thin HTTP clients; running any of them requires a
  reachable `dikw serve` instance (or `dikw serve-and-run` for one-shot
  use).
* **Added**: `dikw serve --wiki <path>` — FastAPI + Uvicorn server.
  Defaults to `127.0.0.1:8765` with no auth on loopback; non-loopback
  hosts require `DIKW_SERVER_TOKEN`. Routes documented in
  [`docs/server.md`](./docs/server.md).
* **Added**: `dikw client *` subcommand group — full remote surface
  (status / check / init / ingest / query / synth / lint / distill /
  eval / review / tasks). Top-level aliases (`dikw status` etc.) keep
  the previous muscle memory working; they now route through the same
  HTTP client.
* **Added**: `dikw client serve-and-run -- <cmd> [args]` — spawns a
  local server, waits for `/v1/healthz`, runs the inner command
  against it, and tears it down. Use this for one-off ingest/query
  flows when you don't want to manage a long-lived server. Picks a
  free port automatically; `--keep-alive` leaves the server up.
* **Added**: NDJSON streaming for long ops (`ingest`, `synth`,
  `distill`, `eval`) and for `query`. Each event is a JSON line; the
  transport drops `heartbeat` events at the client layer. Task
  endpoints support `?from_seq=N` resume so a disconnected client can
  rejoin without missing events.
* **Added**: `POST /v1/upload/sources` — multipart tar.gz + manifest
  upload that the client packs from a local directory. Server
  validates sha256 per file before staging, and the ingest task
  references the staged tree by `upload_id`.
* **Added**: `dikw_core.progress.ProgressReporter` Protocol — the
  engine emits structured progress events (`progress`, `log`,
  `partial`) via this hook; the server bridges them onto the NDJSON
  task event stream, in-process callers (tests, the eval runner) can
  pass `NoopReporter()` and ignore them.
* **Changed**: dependencies. Added `fastapi`, `uvicorn[standard]`,
  `python-multipart`. Dropped `mcp`. The `dikw-core[postgres]` extra is
  unchanged.

### Migration

If you scripted against the old in-process CLI:

* **`dikw status`** etc. — still works, but the server must be
  running. Replace `dikw status` with either `dikw serve-and-run --
  status` or run `dikw serve` in a separate terminal first.
* **`dikw mcp --stdio`** — gone. There is no shim. Adapt to the HTTP
  surface (the wire shape is documented in
  [`docs/server.md`](./docs/server.md)) or pin to the last release
  containing MCP.
* **Configuration** — `dikw.yml` is unchanged. Client-side settings
  (server URL, token) live in `~/.config/dikw/client.toml` or
  `DIKW_SERVER_URL` / `DIKW_SERVER_TOKEN` env vars.
