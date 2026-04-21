# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`dikw-core` is a Python 3.12+ AI-native knowledge engine spanning the full
DIKW pyramid (**D**ata → **I**nformation → **K**nowledge → **W**isdom).
Status: **pre-alpha** — APIs and on-disk formats will change.

Two interfaces wrap the same core (`dikw_core.api`): a Typer-based CLI
(`dikw`) and an MCP server (`dikw mcp --stdio`).

Canonical docs (read these before designing changes):
- `docs/design.md` — approved design doc, source of truth for intent
- `docs/architecture.md` — module map, layer contracts, seams
- `docs/getting-started.md` — end-user walkthrough

## Dev workflow

Package manager is **`uv`** (not pip/poetry). Python **3.12+**.

```bash
uv sync --all-extras          # install (includes [postgres] + dev group)
uv run ruff check .           # lint
uv run mypy src               # strict type-check
uv run pytest -v              # tests (asyncio_mode=auto)
uv run pytest tests/test_storage_contract.py   # storage-contract tests (also run in CI against real Postgres)
uv run dikw <cmd>             # exercise the CLI against a scratch wiki
```

CI (`.github/workflows/ci.yml`) gates PRs on ruff + mypy + pytest across
Python 3.12 and 3.13, and runs the storage contract suite against a
`pgvector/pgvector:pg16` Postgres service. Release tags (`vX.Y.Z`) publish
to PyPI via trusted publishing (`.github/workflows/release.yml`).

Tooling config lives in `pyproject.toml`:
- ruff: line-length 100, rules `E,F,W,I,UP,B,SIM,C4,RUF` (E501 ignored)
- mypy: `strict = true`, `packages = ["dikw_core"]`, `mypy_path = "src"`
- pytest: `asyncio_mode = "auto"`, `testpaths = ["tests"]`

## Architecture at a glance

```
src/dikw_core/
├── api.py                 engine facade (ingest, query, synth, distill, review, lint, status)
├── cli.py                 Typer app → api
├── mcp_server.py          MCP tools grouped by layer
├── config.py              pydantic config + YAML loader (dikw.yml)
├── schemas.py             cross-layer DTOs (cross the Storage Protocol boundary — no SQL types)
├── data/                  D layer — sources + SourceBackend registry (markdown, html)
├── info/                  I layer — chunk, embed, RRF-fused hybrid search
├── knowledge/             K layer — wiki pages, [[wikilinks]], index.md, log.md, lint
├── wisdom/                W layer — distill, review state machine, apply-at-query
├── providers/             LLMProvider + EmbeddingProvider Protocols (anthropic, openai_compat)
├── storage/               Storage Protocol + adapters (sqlite, postgres, filesystem)
└── prompts/               versioned LLM prompts (importlib.resources)
```

### Named seams — extend here, not elsewhere

1. **`SourceBackend`** (`data/backends/base.py`) — new formats: one subclass + `register()`. Example: `data/backends/html.py` is stdlib-only.
2. **`Storage` Protocol** (`storage/base.py`) — three backends ship (sqlite, postgres, filesystem); engine code depends only on the Protocol. Hybrid-search fusion (RRF), chunking, link-graph parsing, and wisdom scoring live **outside** adapters — adapters expose primitives only.
3. **`LLMProvider` / `EmbeddingProvider`** (`providers/base.py`) — Anthropic uses `cache_control` on the system prompt; openai_compat works against any base URL.

### Core invariants

- **Karpathy's rule:** *scoping is deterministic, reasoning is probabilistic*. Navigation (source listing, chunk lookup, link traversal, wisdom lookup-by-title) is deterministic SQL/file I/O. LLMs enter only at synth, distill, and the final answer step of query.
- **W layer gate:** every wisdom item must cite **≥ 2 pieces of evidence** from K or D; state transitions go through `dikw review approve|reject`.
- **On-disk format is the product.** `wiki/` and `wisdom/` are plain markdown with YAML front-matter and `[[wikilinks]]` — an Obsidian vault the user owns. The engine writes, the user reads/edits with any editor.
- **Idempotent ingest.** Files whose content hash is unchanged are skipped.

## Conventions

- Types: code is fully typed; mypy runs strict. Don't widen types to silence errors — fix the root cause. Existing overrides (`sqlite_vec`, `frontmatter`, `markdown_it`, `pgvector`, `mcp_server`) are listed in `pyproject.toml`; extend deliberately.
- DTOs: anything crossing the Storage Protocol is a pydantic model in `schemas.py`. No SQL types, ORM handles, or cursors leak out of adapters.
- Tests: `tests/fakes.py` provides in-memory fakes; prefer them over mocks. Storage adapters are validated via `tests/test_storage_contract.py` — add new adapter behavior to the contract, not to ad-hoc tests.
- Prompts: versioned markdown files under `src/dikw_core/prompts/`, loaded via `importlib.resources`. Don't inline prompts in code.
- Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` are read from env; never hardcode or commit. `.env`/`.env.*` are gitignored (except `.env.example`).

## Things not to do

- Don't call SQL or touch adapter internals from engine code — go through the `Storage` Protocol.
- Don't implement search fusion inside a storage adapter — it belongs in `info/search.py`.
- Don't add a new source format without registering a `SourceBackend`.
- Don't change on-disk wiki/wisdom layout without updating `docs/design.md` first — users open these trees in Obsidian.
