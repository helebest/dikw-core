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

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/helebest/dikw-core
cd dikw-core
uv sync

uv run dikw init my-wiki --description "my research wiki"
cd my-wiki
# drop some markdown or HTML into sources/, then:
uv run dikw ingest --no-embed     # or: uv run dikw ingest (needs DIKW_EMBEDDING_API_KEY)
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
| `dikw check`                | ping the configured LLM + embedding endpoints to verify `dikw.yml` + keys     |
| `dikw eval [--dataset]`     | run retrieval-quality evaluation against packaged or custom datasets          |

## Providers

Configured via `dikw.yml`:

```yaml
provider:
  llm: anthropic                # or: openai_compat
  llm_model: claude-sonnet-4-6
  llm_base_url: null            # set for any Anthropic-protocol-compatible endpoint
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
  embedding_dim: 1536           # required: must match what the endpoint returns
  embedding_revision: ""        # bump to force re-embed when vendor refreshes weights silently
  embedding_normalize: true
  embedding_distance: cosine
```

- `anthropic` → uses the `anthropic` async SDK with `cache_control` on the
  system prompt, so repeated synth/query calls hit the prompt cache. Set
  `llm_base_url` to retarget the SDK at any Anthropic-compatible endpoint
  (e.g., MiniMax); leave null for api.anthropic.com.
- `openai_compat` → uses the `openai` async SDK against any base URL that
  speaks the OpenAI HTTP surface (Azure, Ollama, vLLM, DeepSeek, MiniMax, …).

Full vendor cookbook (MiniMax, GLM, Gemini, DeepSeek, Gitee AI, Ollama, …)
and the production gotchas around batch size, embedding dimensions, and
retry/caching live in [`docs/providers.md`](./docs/providers.md).

### Using MiniMax LLM + Gitee AI embeddings

MiniMax has no embeddings endpoint — pair its Anthropic-compatible LLM surface
with an OpenAI-compatible embedding vendor. The example below uses
[Gitee AI](https://ai.gitee.com/v1) (`Qwen3-Embedding-8B`, matryoshka-truncatable).
Fill the URLs in by hand — dikw-core never auto-detects vendor endpoints:

```yaml
provider:
  llm: anthropic
  llm_model: <MiniMax Anthropic-compatible model name>
  llm_base_url: <MiniMax Anthropic endpoint, e.g. https://api.minimaxi.com/anthropic>
  embedding: openai_compat
  embedding_model: Qwen3-Embedding-8B
  embedding_base_url: https://ai.gitee.com/v1
  embedding_dim: 1024               # required: matryoshka truncation; locked at first ingest
  embedding_revision: ""            # bump to force re-embed when Qwen weights drift silently
  embedding_normalize: true
  embedding_distance: cosine
  embedding_batch_size: 16          # required: Gitee rejects batches >25
  embedding_provider_label: gitee-ai  # optional; shows up in `dikw check`
```

A working reference copy lives at
[`tests/fixtures/live-minimax-gitee.dikw.yml`](./tests/fixtures/live-minimax-gitee.dikw.yml)
— drop it into a fresh wiki and fill in your two keys.

Two keys for two vendors — the embedding leg reads `DIKW_EMBEDDING_API_KEY`
exclusively (no `OPENAI_API_KEY` fallback), so misconfigurations fail loudly
rather than cross-wiring credentials:

```bash
export ANTHROPIC_API_KEY=<your-MiniMax-key>
export DIKW_EMBEDDING_API_KEY=<your-Gitee-key>
```

Verify connectivity **before** running ingest/query. The two legs can be
probed separately, which is useful when you set up one vendor first:

```bash
uv run dikw check --llm-only     # just LLM — useful before Gitee is wired up
uv run dikw check --embed-only   # just embedding
uv run dikw check                # both
```

`dikw check` pings each provider with one tiny request and prints a status
table with endpoint, latency, and dim/tokens. Exit code is 0 on success,
1 on failure, 2 on flag misuse — scriptable in CI or a shell one-liner.

## Source formats

Markdown and HTML ship out of the box. A new format is one `SourceBackend`
subclass + a `register()` call away — see
[`data/backends/html.py`](./src/dikw_core/data/backends/html.py) for a
stdlib-only reference.

## Storage

Three backends ship, selected in `dikw.yml`:

```yaml
storage:
  backend: sqlite          # sqlite | postgres | filesystem

  # --- sqlite (default): single-user local ---
  path: .dikw/index.sqlite

  # --- postgres (enterprise): multi-user, pgvector + tsvector ---
  # backend: postgres
  # dsn: postgresql://user:pw@host:5432/dikw
  # schema: dikw
  # pool_size: 10

  # --- filesystem: DB-less, Obsidian-vault native (≤ ~300 pages, FTS-only) ---
  # backend: filesystem
  # root: .dikw/fs
```

- **SQLite + `sqlite-vec` + FTS5** — the default. No extras required.
- **Postgres + `pgvector`** — install via `uv pip install dikw-core[postgres]`.
  Requires the `pg_trgm` and `vector` extensions (standard on the
  `pgvector/pgvector:pg16` Docker image). The adapter uses `tsvector`+GIN
  for FTS and `vector(N)` for embeddings; the vector dimension is set at
  first insert.
- **Filesystem / vault** — zero extra deps. JSONL sidecars under
  `.dikw/fs/`, in-process inverted-index FTS, no dense retrieval (use
  the sqlite backend if you need vectors). Good for personal vaults;
  not a multi-writer story. Emits a hint to switch backends once the
  corpus outgrows the scale target.

Engine code talks only to the `Storage` Protocol
([`storage/base.py`](./src/dikw_core/storage/base.py)); each adapter
implements the same contract and is swappable by changing `dikw.yml`.

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
