# dikw-core

AI-native knowledge engine that turns your documents into **Data → Information → Knowledge → Wisdom**.

Inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), extended end-to-end across the full DIKW pyramid. Where Karpathy's pattern stops at a compounding markdown wiki (the K layer), `dikw-core` adds a first-class **Wisdom layer** — distilled principles, lessons, and patterns that apply beyond any single source.

> Status: pre-alpha. Under active construction; APIs and on-disk formats will change.

## What you get

- A local-first knowledge base where the wiki is a **plain markdown tree** your editor (Obsidian, VS Code, …) can open directly.
- Four explicit DIKW layers with their own operations:
  - **D**ata — raw sources you curate.
  - **I**nformation — parsed, chunked, embedded, indexed (FTS5 + vectors).
  - **K**nowledge — LLM-authored wiki pages with `[[wikilinks]]`, `index.md`, and an append-only `log.md`.
  - **W**isdom — evidence-backed principles / lessons / patterns, human-approved, surfaced at query time.
- Pluggable LLM providers (API-first): Anthropic + OpenAI-compatible (covers OpenAI, Azure, Ollama, DeepSeek, Gemini-compat).
- Pluggable storage: SQLite+sqlite-vec (default), Postgres+pgvector (enterprise), Filesystem/Vault (Obsidian-native, DB-less) — swap by config.
- Agent-first: MCP server exposes the engine to Claude Code, Claude Desktop, and any MCP client. A CLI (`dikw`) wraps the same core.

## Install & quick start

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/helebest/dikw-core
cd dikw-core
uv sync

uv run dikw init my-wiki --description "my research wiki"
cd my-wiki
# drop some markdown or HTML into sources/, then:
uv run dikw ingest --no-embed     # or: uv run dikw ingest (needs OPENAI_API_KEY)
uv run dikw status
uv run dikw synth                  # K layer (needs ANTHROPIC_API_KEY or OpenAI-compat)
uv run dikw distill                # W-layer candidates
uv run dikw review list
uv run dikw review approve W-abcdef123456
uv run dikw query "What does Karpathy mean by deterministic scoping?"
```

End-to-end walkthrough: [`docs/getting-started.md`](./docs/getting-started.md).
Architecture brief: [`docs/architecture.md`](./docs/architecture.md).
Approved design doc: [`docs/design.md`](./docs/design.md).

## Commands

| command                     | does                                                                          |
| --------------------------- | ----------------------------------------------------------------------------- |
| `dikw init <path>`          | scaffold a wiki directory (sources / wiki / wisdom / `.dikw/` + `dikw.yml`)   |
| `dikw ingest [--no-embed]`  | parse sources (MD, HTML), chunk, FTS-index, optionally embed                  |
| `dikw query "<q>"`          | hybrid search + LLM answer with citations and applicable wisdom               |
| `dikw synth [--all]`        | LLM turns source docs into K-layer wiki pages; maintains `index.md`+`log.md`  |
| `dikw lint`                 | report broken wikilinks, orphan pages, duplicate titles                       |
| `dikw distill`              | LLM proposes W-layer candidates (each needs ≥ 2 pieces of evidence)           |
| `dikw review {list,approve,reject}` | drive the candidate -> approved / archived state machine             |
| `dikw mcp [--stdio]`        | launch the MCP server (exposes the engine to Claude Code / Claude Desktop)    |
| `dikw status`               | counts across DIKW layers                                                     |

## Providers

Configured via `dikw.yml`:

```yaml
provider:
  llm: anthropic                # or: openai_compat
  llm_model: claude-sonnet-4-6
  llm_base_url: null            # set for openai_compat (Azure, Ollama, vLLM, DeepSeek, …)
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
```

- `anthropic` → uses the `anthropic` async SDK with `cache_control` on the
  system prompt, so repeated synth/query calls hit the prompt cache.
- `openai_compat` → uses the `openai` async SDK against any base URL that
  speaks the OpenAI HTTP surface.

## Source formats

Markdown and HTML ship out of the box. A new format is one `SourceBackend`
subclass + a `register()` call away — see
[`data/backends/html.py`](./src/dikw_core/data/backends/html.py) for a
stdlib-only reference.

## Storage

The default backend is SQLite + `sqlite-vec` + FTS5, stored at
`.dikw/index.sqlite` inside the wiki directory. Postgres+`pgvector`
(enterprise) and a DB-less filesystem adapter (Obsidian-native, ≤ ~200
pages) are planned Phase 5 additions as optional install extras; the
`Storage` Protocol is already in place.

## Releasing

Tagged pushes (`vX.Y.Z`) trigger
[`.github/workflows/release.yml`](./.github/workflows/release.yml), which
builds `sdist` + wheel, re-runs the full test gate, and publishes to PyPI
via **trusted publishing** (no token in repo secrets). One-time setup on
PyPI's side:

1. Create the `dikw-core` project on PyPI.
2. On the project's *Publishing* page, add a GitHub trusted publisher with:
   - owner: `helebest`
   - repository: `dikw-core`
   - workflow: `release.yml`
   - environment: `pypi`

After that, `git tag vX.Y.Z && git push --tags` is enough.

## License

MIT — see [LICENSE](./LICENSE).
