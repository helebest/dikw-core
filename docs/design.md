# dikw-core — AI-Native Knowledge Engine (Plan)

## Context

`/Users/bytedance/Projects/dikw-core` is greenfield. The goal is an **AI-native knowledge engine** inspired by Karpathy's "LLM Wiki" pattern ([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)), but extended end-to-end across the **DIKW pyramid** — Data → Information → Knowledge → Wisdom.

Why this project exists:
- Karpathy's LLM-Wiki pattern captures a real gap in today's RAG stacks: **knowledge should be a compounding artifact, not a query-time search result.** His pattern stops at Knowledge (markdown wiki with index.md + log.md).
- Existing reference tools (`mineru-doc-explorer`, `qmd`) implement the pattern in TypeScript/Node, local-first with GGUF models and SQLite+sqlite-vec. They cover D→I→K well but do not treat Wisdom (principles, lessons, transferable judgment) as a first-class layer.
- The user wants a **Python-native** implementation that (a) makes all four DIKW layers first-class, (b) is pluggable across LLM providers via API, (c) targets personal and enterprise knowledge bases, (d) is packaged with `uv` and hosted on GitHub.

Design decisions already locked in (via clarifying Q&A):
- **Scope:** full D→I→K→W four layers, with Wisdom as the differentiator.
- **Providers:** API-first, pluggable. First-party: Anthropic + OpenAI-compatible (covers OpenAI, Azure, Ollama, DeepSeek, Gemini-compat, etc.). Local (llama-cpp-python) deferred.
- **MVP source format:** Markdown only. Other formats via a backend-registry extension point later.

## Vision & Principles

1. **DIKW as first-class layers** — each layer has its own storage, schemas, and operations. The pipeline between layers is explicit (not an implicit by-product of retrieval).
2. **Wiki-as-artifact** — Knowledge & Wisdom layers are plain markdown on disk, versioned with git by the user, editable by humans and LLMs. The engine is a tool; the wiki is the product.
3. **Scoping deterministic, reasoning probabilistic** (Karpathy) — navigation uses deterministic structure (index.md, link graph, FTS); LLM calls are reserved for synthesis, distillation, and answering.
4. **Agent-native first, CLI second** — primary interface is an MCP server exposing high-level operations; CLI wraps the same core for humans.
5. **Local-first data, pluggable compute** — the wiki lives on the user's filesystem; the default index is a local SQLite DB; only LLM calls leave the machine (and are provider-abstracted).
6. **Pluggable storage** — the engine talks to an abstract **Storage** interface, not to SQL directly. Three backends are planned: **SQLite+sqlite-vec** (MVP, single-user local), **Postgres+pgvector** (enterprise, multi-user), and **Filesystem/Vault** (DB-less, Obsidian-native — matches Karpathy's small-scale philosophy). Swapping backends is a config change.
7. **Obsidian-compatible on-disk format** — the K & W layers are written as a plain markdown tree that Obsidian (or any MD editor) opens as a vault: `[[wikilinks]]`, YAML front-matter with tags, folder-based organization, daily-note conventions. The engine is a collaborator, not a walled garden; the user owns the files.
8. **YAGNI + extension points** — ship a tight MVP, but put named seams (provider adapter, storage adapter, source-backend registry, prompt registry) where known growth vectors are.

## The Four Layers (concrete definitions)

| Layer | What it is | Storage | Who writes it |
|---|---|---|---|
| **D — Data** | Raw, immutable sources (markdown files the user curates) | filesystem + indexed `documents` table in SQLite (path, content hash, layer, active) | human |
| **I — Information** | Parsed, chunked, embedded, indexed — enables fast lookup | SQLite FTS5 + sqlite-vec (`.dikw/index.sqlite`) | engine (deterministic) |
| **K — Knowledge** | LLM-authored wiki pages: summaries, entities, concepts, cross-refs, `index.md`, `log.md` | markdown files in `wiki/` | LLM, human-editable |
| **W — Wisdom** | Distilled principles, heuristics, lessons, patterns — transferable beyond a single source | markdown files in `wisdom/` with explicit provenance & review status | LLM proposes, human confirms |

The W layer is the novel bit and is spelled out in "Wisdom Layer Design" below.

## Target Architecture

```
                 ┌──────────────────────────────────────────┐
 User & Agents → │  Interfaces:  MCP server   │   CLI       │
                 └────────────────┬─────────────────────────┘
                                  │
                 ┌────────────────▼─────────────────────────┐
                 │  Core API (dikw_core.api)                │
                 │  ingest · synthesize · distill · query · │
                 │  lint · status                           │
                 └────────────────┬─────────────────────────┘
          ┌───────────────────────┼────────────────────────┐
          ▼                       ▼                        ▼
 ┌────────────────┐     ┌───────────────────┐   ┌────────────────────┐
 │  Data (D)      │     │ Information (I)   │   │ Knowledge (K) /    │
 │  sources.py    │     │ chunk · embed ·   │   │ Wisdom (W)         │
 │  backends/md   │──▶  │ index · search    │◀─▶│ wiki/ · wisdom/    │
 │  (content-hash)│     │ (FTS5 + vec + RRF)│   │ links · log        │
 └────────┬───────┘     └─────────┬─────────┘   └──────────┬─────────┘
          │                       │                        │
          └───────────────────────▼────────────────────────┘
                ┌─────────────────▼───────────────────────────┐
                │  Storage adapter  (dikw_core.storage)       │
                │  base · sqlite (MVP) · postgres · filesystem│
                └─────────────────┬───────────────────────────┘
                                  │
        ┌─────────────────────────┼───────────────────────────────┐
        ▼                         ▼                               ▼
 SQLite+sqlite-vec+FTS5   Postgres+pgvector+tsvector     Filesystem/Vault
 (single-user, local)     (multi-user, enterprise)       (DB-less, Obsidian-native)
                                  │
                 ┌────────────────▼─────────────────────────┐
                 │ Providers (LLM + Embedding)              │
                 │ base · anthropic · openai_compat · local │
                 └──────────────────────────────────────────┘
```

Module boundaries are chosen so each subpackage fits in a single reading pass and has a named interface. Engine code depends only on the **Storage** Protocol — never on raw SQL or backend-specific tables — which keeps the SQLite/Postgres seam sharp.

## Tech Stack

- **Language**: Python 3.12+
- **Packaging**: `uv` → `pyproject.toml` (PEP 621), `uv.lock` committed; single source layout under `src/dikw_core/`
- **Storage (MVP, default)**: stdlib `sqlite3` + `sqlite-vec` (pip) for vectors; FTS5 built into SQLite. Behind a `Storage` Protocol.
- **Storage (planned, enterprise)**: Postgres 15+ with `pgvector` ≥0.6 and `tsvector`/`pg_trgm` for full-text, via `psycopg[binary,pool]`. Optional extra: `uv pip install dikw-core[postgres]`.
- **Storage (planned, vault-native)**: Filesystem backend — the vault IS the index. No DB. Chunks/links/wisdom-items live in `.dikw/` JSON sidecars; search falls back to LLM-driven navigation over `index.md` + link graph. Matches Karpathy's "scoping deterministic, reasoning probabilistic" claim at ≤~200 pages. No extra dep footprint.
- **Schemas**: Pydantic v2 for config, records, tool I/O
- **Markdown**: `markdown-it-py` + `python-frontmatter`; wiki-link parsing via a small in-repo module (not a heavy dep)
- **LLM SDKs**: `anthropic`, `openai` (the `openai` SDK covers all OpenAI-compatible endpoints), behind a thin provider interface
- **Embeddings**: default through an OpenAI-compatible `embeddings` endpoint (works for OpenAI, Ollama, TEI, etc.); Anthropic path uses OpenAI-compat for embeddings since Anthropic has no embeddings API
- **MCP**: `mcp` Python SDK (stdio transport first; HTTP optional)
- **CLI & output**: `typer` + `rich`
- **Quality**: `pytest`, `pytest-asyncio`, `ruff`, `mypy --strict` where practical
- **CI**: GitHub Actions — lint + type-check + tests on 3.12/3.13

Known patterns to reuse from references (concrete sources):
- **Hybrid search pipeline (BM25 + vector + RRF + rerank)** — `mineru-doc-explorer/src/hybrid-search.ts`, `mineru-doc-explorer/src/search.ts`. Port the RRF fusion + position-aware blending logic.
- **SQLite schema design + content-addressed storage** — `mineru-doc-explorer/src/db-schema.ts`, `mineru-doc-explorer/src/store.ts` (documents table with indexed content hash, links table, wiki_log).
- **Smart markdown chunking (~900 tokens, 15% overlap, heading-aware)** — `mineru-doc-explorer/src/store.ts` chunking section; `qmd/src/store.ts` lines ~257–310.
- **Wikilink parsing + forward/backward graph** — `mineru-doc-explorer/src/links.ts`, `mineru-doc-explorer/src/wiki/{log,lint,index-gen}.ts`. Port to a small `knowledge/links.py`.
- **MCP tool grouping** — `mineru-doc-explorer/src/mcp/tools/{core,document,wiki}.ts`. Mirror the grouping shape in Python.
- **YAML config + schema validation** — `mineru-doc-explorer/src/config-schema.ts` (Zod) → Pydantic v2 equivalent in `dikw_core/config.py`.
- **Strong-signal short-circuit** (skip expensive LLM expansion when FTS already gives a confident top hit) — `qmd/src/store.ts:4057–4076`.

## Package Layout

```
dikw-core/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── .python-version           # 3.12
├── .github/workflows/ci.yml
├── .gitignore
├── src/dikw_core/
│   ├── __init__.py
│   ├── api.py                # thin facade used by CLI + MCP
│   ├── config.py             # Pydantic models + YAML loader
│   ├── schemas.py            # cross-layer record types
│   │
│   ├── storage/              # Storage adapters — SQLite now; Postgres + Filesystem later
│   │   ├── __init__.py       # factory: resolves backend from config
│   │   ├── base.py           # Storage Protocol + typed DTOs
│   │   ├── sqlite.py         # SQLite + sqlite-vec + FTS5 implementation (MVP)
│   │   ├── postgres.py       # Phase 5 — tsvector + pgvector (optional extra)
│   │   ├── filesystem.py     # Phase 5 — DB-less, Obsidian-vault-native
│   │   ├── migrations/       # per-backend schema migrations
│   │   │   ├── sqlite/       #   001_init.sql … (MVP)
│   │   │   └── postgres/     #   placeholder, populated in Phase 5
│   │   └── _sql/             # backend-specific query fragments, kept out of engine code
│   │
│   ├── data/                 # D layer
│   │   ├── sources.py        # source registry, hashing, mtime tracking
│   │   └── backends/
│   │       ├── __init__.py   # backend registry (extension point)
│   │       └── markdown.py   # MD parser + front-matter + deep-read
│   │
│   ├── info/                 # I layer
│   │   ├── chunk.py          # heading-aware markdown chunking
│   │   ├── embed.py          # batched embedding via provider
│   │   ├── index.py          # FTS5 + sqlite-vec writes
│   │   └── search.py         # BM25 + vector + RRF + optional rerank
│   │
│   ├── knowledge/            # K layer
│   │   ├── wiki.py           # page read/write, front-matter conventions
│   │   ├── synthesize.py     # ingest → wiki pages (LLM-driven)
│   │   ├── links.py          # wikilink/markdown/URL link graph
│   │   ├── indexgen.py       # regenerate index.md from wiki/
│   │   └── log.py            # append-only wiki_log + log.md renderer
│   │
│   ├── wisdom/               # W layer (see dedicated section)
│   │   ├── distill.py        # propose principles/lessons/patterns
│   │   ├── review.py         # human-confirmation workflow
│   │   └── apply.py          # surface applicable wisdom at query time
│   │
│   ├── providers/            # LLM + embedding abstraction
│   │   ├── base.py           # LLMProvider, EmbeddingProvider protocols
│   │   ├── anthropic.py      # claude sonnet/haiku via anthropic SDK
│   │   └── openai_compat.py  # openai SDK pointed at any compat endpoint
│   │
│   ├── prompts/              # versioned prompt templates (Jinja2-lite strings)
│   │   ├── synthesize.md
│   │   ├── distill.md
│   │   ├── query.md
│   │   └── lint.md
│   │
│   ├── mcp_server.py         # MCP tools grouped by layer
│   └── cli.py                # typer app: init, ingest, query, synth, distill, lint, mcp
│
├── tests/
│   ├── fixtures/             # small MD corpora
│   ├── test_chunk.py
│   ├── test_search.py        # FTS + vector + RRF behavior on golden set
│   ├── test_wiki.py
│   ├── test_distill.py
│   ├── test_providers.py     # uses recorded responses
│   ├── test_storage_contract.py  # same contract test runs against every backend
│   └── test_mcp.py
└── examples/
    └── personal-wiki/        # runnable demo wiki
```

## On-Disk User Wiki Layout (convention, not code)

```
my-wiki/
├── dikw.yml                  # config: sources, provider, schema
├── sources/                  # user-curated raw markdown (D layer)
├── wiki/                     # K layer (LLM-authored, human-editable)
│   ├── index.md              # auto-generated catalog
│   ├── log.md                # append-only chronology
│   ├── entities/
│   ├── concepts/
│   └── notes/
├── wisdom/                   # W layer
│   ├── principles.md
│   ├── lessons.md
│   ├── patterns.md
│   └── _candidates/          # LLM proposals awaiting human review
└── .dikw/                    # engine-managed, gitignored by default
    ├── index.sqlite          # I layer when storage.backend=sqlite
    ├── fs/                   # I layer when storage.backend=filesystem (JSON sidecars)
    └── cache/                # model/artifact caches (backend-agnostic)
```

**Obsidian vault compatibility** — `my-wiki/` is itself a valid Obsidian vault. The engine follows these conventions so Obsidian (or any plain MD editor) can open it and edit alongside the engine without conflict:
- `[[Wikilinks]]` — the canonical link form in `wiki/` and `wisdom/`. `[[Page#Heading]]` and `[[Page|alias]]` supported.
- **YAML front-matter** — every engine-authored page has `---`-delimited front-matter with at least `id`, `kind` (for wisdom) or `type` (for wiki), `created`, `updated`, and `tags: [...]`. Obsidian reads `tags` natively.
- **Folder = category** — `wiki/entities/`, `wiki/concepts/`, `wiki/notes/`, `wisdom/_candidates/`. Matches Obsidian's default folder-sort behavior.
- **Daily-note style log** — `wiki/log.md` keeps Karpathy's chronological format; optionally daily files under `wiki/log/YYYY/MM/YYYY-MM-DD.md` for vaults that already use Obsidian's daily-notes plugin (opt-in via `schema.log_style: daily`).
- **Engine state stays out of the vault** — the `.dikw/` sidecar directory is gitignored and Obsidian-ignored (`.obsidian/app.json` `userIgnoreFilters` receives a `.dikw/` entry on `dikw init`).
- **No bespoke syntax in MD bodies** — only standard Markdown + wikilinks + front-matter, so a human editing in Obsidian never sees engine-only constructs that would get stripped on round-trip.

`dikw.yml` example:
```yaml
provider:
  llm: anthropic           # or: openai_compat
  llm_model: claude-sonnet-4-6
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
storage:
  backend: sqlite          # sqlite | postgres | filesystem
  # --- sqlite-specific (default) ---
  path: .dikw/index.sqlite
  # --- postgres-specific (Phase 5) ---
  # dsn: postgresql://user:pass@host:5432/dikw
  # schema: dikw            # isolates multi-tenant deployments
  # pool_size: 10
  # --- filesystem-specific (Phase 5, Obsidian-native) ---
  # root: .dikw/fs          # JSON sidecar directory inside the vault
  # embed: false            # skip embeddings; rely on LLM nav over index.md
  # max_pages_hint: 300     # warn + suggest switching to sqlite above this
schema:
  description: "Personal research wiki on AI safety"
  page_types: [entity, concept, note]
  wisdom_kinds: [principle, lesson, pattern]
sources:
  - path: ./sources
    pattern: "**/*.md"
    ignore: ["drafts/**"]
```

## Data Model

The logical model is backend-agnostic; the SQL below is the **SQLite reference schema** used by the MVP adapter. The Postgres adapter maps the same logical entities to equivalent structures — `tsvector` + GIN + `pg_trgm` for FTS, `pgvector` for embeddings, regular tables for the rest — behind the same `Storage` Protocol.

### SQLite reference schema (MVP)

```sql
-- D
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    path   TEXT UNIQUE NOT NULL,
    title  TEXT,
    hash   TEXT NOT NULL,                 -- sha256 of body; indexed for reverse lookup
    mtime  REAL,
    layer  TEXT CHECK (layer IN ('source','wiki','wisdom')) NOT NULL,
    active INTEGER DEFAULT 1
);
CREATE INDEX documents_hash_idx ON documents(hash);

-- I
CREATE VIRTUAL TABLE documents_fts USING fts5(path, title, body, content='');
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    doc_id   TEXT REFERENCES documents(doc_id),
    seq      INTEGER,
    start    INTEGER, end INTEGER,
    text     TEXT
);
CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[EMB_DIM]);
CREATE TABLE embed_meta (chunk_id INTEGER, model TEXT, PRIMARY KEY(chunk_id, model));

-- K (link graph spans K and W)
CREATE TABLE links (
    src_doc_id TEXT, dst_path TEXT,
    link_type  TEXT CHECK (link_type IN ('wikilink','markdown','url')),
    anchor     TEXT, line INTEGER,
    PRIMARY KEY (src_doc_id, dst_path, line)
);
CREATE TABLE wiki_log (
    ts INTEGER, action TEXT, src TEXT, dst TEXT, note TEXT
);

-- W
CREATE TABLE wisdom_items (
    item_id       TEXT PRIMARY KEY,
    kind          TEXT CHECK (kind IN ('principle','lesson','pattern')),
    status        TEXT CHECK (status IN ('candidate','approved','archived')) DEFAULT 'candidate',
    path          TEXT,          -- wisdom/<file>.md anchor
    title         TEXT,
    body          TEXT,
    confidence    REAL,
    created_ts    INTEGER, approved_ts INTEGER
);
CREATE TABLE wisdom_evidence (
    item_id  TEXT REFERENCES wisdom_items(item_id),
    doc_id   TEXT REFERENCES documents(doc_id),
    excerpt  TEXT, line INTEGER
);
```

## Core Operations

Each operation is implemented in `dikw_core.api` and surfaced identically in CLI and MCP.

| Op | Input | Output | Notes |
|---|---|---|---|
| `ingest(paths)` | file paths | updated `documents`/`chunks`/`documents_fts`/`chunks_vec` | D→I; deterministic; idempotent by content hash |
| `synthesize(scope)` | source doc_ids (or "new since log") | new/updated wiki pages + wiki_log entries | I→K; LLM call with prompts/synthesize.md |
| `distill(window)` | optional time/topic window | candidate wisdom items in `wisdom/_candidates/` + `wisdom_items(status='candidate')` | K→W; LLM call with prompts/distill.md; always produces candidates, never auto-approves |
| `review()` | — | interactive CLI workflow to approve/edit/reject candidates | W gate; writes final files to `wisdom/*.md` |
| `query(q)` | user question | answer + citations (from D, I, K, W) | uses hybrid search + page retrieval + applicable wisdom; prompts/query.md |
| `lint()` | — | report of broken links, stale claims, orphan pages, duplicated entities | K+W hygiene; prompts/lint.md |
| `status()` | — | counts per layer, last-ingest, last-synthesize, pending review | for CLI and MCP dashboards |

## Wisdom Layer Design (the novel bit)

**What "wisdom" means operationally** — a short, human-readable claim that (a) can be stated independent of any single source, (b) has at least N≥2 pieces of evidence drawn from K-layer pages or D-layer passages, (c) carries a kind and a status.

**Kinds** (opinionated, fixed small vocabulary; MVP):
- `principle` — normative: "Prefer deterministic scoping over probabilistic retrieval."
- `lesson` — retrospective: "Mocked DB tests hid the prod-migration failure last quarter."
- `pattern` — structural: "When ingesting PDFs, cache per-page text before chunking."

**Page format** — each wisdom file is markdown with front-matter:
```markdown
---
id: W-000042
kind: principle
status: approved
confidence: 0.82
created: 2026-04-21
approved: 2026-04-22
evidence:
  - doc: wiki/concepts/rag-vs-wiki.md
    excerpt: "Karpathy argues scoping should be deterministic..."
  - doc: sources/notes/2026-04-10-meeting.md
    excerpt: "We agreed to stop using live LLM calls for routing."
---

# Prefer deterministic scoping over probabilistic retrieval

Use the index and link graph to narrow scope; only invoke the LLM after
the candidate set is small and the question is concrete. This compresses
cost and improves reproducibility, especially for queries the agent will
re-ask many times.
```

**Distillation workflow** (`wisdom/distill.py`):
1. Scope: recent wiki activity (since last `distill` entry in wiki_log) or user-given topic.
2. Retrieve: top-K K-layer pages + their wikilinked neighbors via the link graph.
3. Prompt the LLM with `prompts/distill.md` — asks for candidate claims + evidence excerpts + kind + confidence.
4. Persist each candidate as `wisdom/_candidates/<slug>.md` AND as a row in `wisdom_items(status='candidate')`.
5. Never auto-promote. A candidate becomes approved only via `dikw review`.

**Application at query time** (`wisdom/apply.py`):
- After hybrid search, also match approved wisdom items by lexical overlap + a cheap semantic pass against the question.
- Include up to 3 applicable wisdom items in the answering prompt as "operating principles currently approved in this wiki" with citations.

**Why this design pulls its weight**:
- Evidence is required → reduces hallucinated "axioms."
- Explicit `status` and `_candidates/` queue → preserves human oversight without blocking throughput.
- Kind vocabulary is small and fixed → schema stability; wiki doesn't devolve into free-form prose.
- Surfacing at query time → the layer actually affects answers, not just archival.

## Storage Abstraction

Goal: one engine, many backends. The rest of `dikw_core` never touches SQL; it calls into a Protocol.

`storage/base.py` (sketch):
```python
class Storage(Protocol):
    # lifecycle
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def migrate(self) -> None: ...             # idempotent schema bring-up

    # D layer
    async def upsert_document(self, doc: DocumentRecord) -> None: ...
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    async def list_documents(self, *, layer: Layer, active: bool | None = True,
                             since_ts: float | None = None) -> Iterable[DocumentRecord]: ...
    async def deactivate_document(self, doc_id: str) -> None: ...

    # I layer
    async def replace_chunks(self, doc_id: str, chunks: Sequence[ChunkRecord]) -> None: ...
    async def upsert_embeddings(self, rows: Sequence[EmbeddingRow]) -> None: ...
    async def fts_search(self, q: str, *, limit: int, layer: Layer | None = None
                        ) -> list[FTSHit]: ...
    async def vec_search(self, embedding: list[float], *, limit: int,
                         layer: Layer | None = None) -> list[VecHit]: ...
    async def get_chunk(self, chunk_id: int) -> ChunkRecord | None: ...

    # K layer
    async def upsert_link(self, link: LinkRecord) -> None: ...
    async def links_from(self, src_doc_id: str) -> list[LinkRecord]: ...
    async def links_to(self, dst_path: str) -> list[LinkRecord]: ...
    async def append_wiki_log(self, entry: WikiLogEntry) -> None: ...

    # W layer
    async def put_wisdom(self, item: WisdomItem, evidence: Sequence[WisdomEvidence]) -> None: ...
    async def list_wisdom(self, *, status: WisdomStatus | None = None,
                          kind: WisdomKind | None = None) -> list[WisdomItem]: ...
    async def set_wisdom_status(self, item_id: str, status: WisdomStatus) -> None: ...

    # diagnostics
    async def counts(self) -> StorageCounts: ...
```

Design constraints:
- **No leaky query objects.** All inputs and outputs are plain Pydantic DTOs. No `cursor`, no `Session`, no backend-specific types crossing the boundary.
- **Hybrid search stays outside storage.** `info/search.py` owns RRF fusion / reranking; storage exposes only the two primitives (`fts_search`, `vec_search`) since the two backends express those very differently. This is the right abstraction seam — high enough to hide dialect, low enough to avoid re-implementing the fusion in each adapter.
- **Migrations are backend-owned.** `storage/migrations/sqlite/` ships SQL files; `storage/migrations/postgres/` will ship equivalents. A shared `Migrator` drives `await storage.migrate()`.
- **Contract tests.** `tests/test_storage_contract.py` defines a single pytest suite parameterized over `[sqlite, postgres]`; the Postgres variant skips unless `DIKW_TEST_POSTGRES_DSN` is set. This prevents the MVP from growing SQLite-only assumptions before Phase 5.
- **Transactional boundary.** One unit of work per engine operation (e.g., a single `ingest(path)` is one transaction on the adapter). Adapters are responsible for honoring that — SQLite via `BEGIN IMMEDIATE`, Postgres via `psycopg` transactions.

The Postgres adapter (Phase 5) is installed as an **optional extra** so SQLite users never pay for the `psycopg`/`asyncpg` dependency footprint:
```toml
[project.optional-dependencies]
postgres = ["psycopg[binary,pool] >=3.2", "pgvector >=0.3"]
```

### Filesystem / Vault backend (Phase 5, Obsidian-native)

Purpose: a DB-less mode where the Obsidian vault itself holds everything the engine needs, so the whole knowledge base is one portable, human-readable directory.

How it maps to the Storage Protocol:
- `upsert_document` — the files already exist on disk; this becomes a small JSON manifest in `.dikw/fs/documents.jsonl` keyed by content hash (path + title + mtime + layer).
- `fts_search` — simple in-process scan (lunr-style inverted index built at startup from the manifest) or just substring + front-matter tag filtering. Good enough at ≤200 pages.
- `vec_search` — **optional**. If `storage.embed: true`, embeddings are stored as per-chunk JSON sidecars under `.dikw/fs/vecs/<chunk_id>.json` and searched via in-process cosine (numpy). If `false`, `vec_search` raises `NotSupported` and `info/search.py` falls back to a **navigation mode**: pass `index.md` + relevant folders' file list to the LLM, let it pick pages to read, then answer.
- `upsert_link` / `links_from` / `links_to` — maintained in `.dikw/fs/links.jsonl`, regenerated from parsing wiki MD on `ingest` / `synth`.
- `put_wisdom` / `list_wisdom` / `set_wisdom_status` — wisdom items are already individual markdown files; their status is just the `status:` front-matter field. The adapter reads/writes that field rather than a separate table.

Constraints and honesty:
- Not a scale story — `max_pages_hint` defaults to 300; above that the engine emits a one-shot suggestion to `dikw migrate --to sqlite`.
- No cross-session locking — the filesystem backend assumes a single writer at a time. Enterprise deployments should use Postgres.
- The same storage contract test suite runs against filesystem with `embed=false` (with vec_search tests marked xfail) and `embed=true` (all tests).

## Provider Abstraction

`providers/base.py`:
```python
class LLMProvider(Protocol):
    async def complete(self, *, system: str, user: str, model: str,
                       max_tokens: int, temperature: float,
                       tools: list[ToolSpec] | None = None) -> LLMResponse: ...

class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...
```

`providers/anthropic.py` wraps the official `anthropic` SDK for LLM; raises for embedding (unsupported). `providers/openai_compat.py` wraps the official `openai` SDK and takes `base_url` + `api_key` from env/config, covering OpenAI proper, Azure OpenAI, Ollama, vLLM, TEI-style embedding endpoints, and any Claude Code-style OpenAI-compat. `providers/__init__.py` resolves instances from `dikw.yml`; swapping providers is a config-only change.

Prompt caching: when the provider is Anthropic, use the `cache_control` param on the system prompt and large wiki blocks in `synthesize`/`query`/`distill` — the wiki schema and the active-wisdom block are near-static per session and are the prime caching targets.

## Interfaces

**CLI** (`dikw ...`):
- `dikw init [path]` — scaffold `dikw.yml`, `sources/`, `wiki/`, `wisdom/`, `.dikw/`
- `dikw ingest [paths]` — D→I
- `dikw synth [--since TS|--all]` — K synthesis
- `dikw distill [--topic STR|--recent]` — produce W candidates
- `dikw review` — interactive approval of W candidates
- `dikw query "<q>"` — free-form Q&A with citations
- `dikw lint` — hygiene report
- `dikw status`
- `dikw mcp [--stdio|--http --port 8181]`

**MCP tools** (grouped to mirror the reference projects):
- `core.query`, `core.status`
- `doc.read`, `doc.search`, `doc.links`
- `wiki.list`, `wiki.get`, `wiki.synthesize`, `wiki.log`
- `wisdom.list`, `wisdom.apply`, `wisdom.distill`, `wisdom.review`
- `admin.ingest`, `admin.lint`

Each tool validates input with Pydantic and returns structured JSON plus a markdown-rendered companion string (so agents can paste answers directly).

## Phasing

- **Phase 0 — Scaffold (small):** repo layout, `uv` init, CI, ruff/mypy, typer CLI with `init`/`status`, config loader, **`Storage` Protocol + DTOs in `storage/base.py`**, SQLite bootstrap in `storage/sqlite.py`, `storage/__init__.py` factory, contract-test skeleton, minimal `providers/base.py` + Anthropic stub, a golden-path test that runs end-to-end on an empty wiki.
- **Phase 1 — D + I (foundation):** markdown backend, content-hash store, heading-aware chunker, embedding batch pipeline via OpenAI-compat, FTS5 index and sqlite-vec index implemented on the SQLite adapter, RRF hybrid `search` (fusion lives in `info/search.py`, calling `storage.fts_search` + `storage.vec_search`), `ingest` + `query` CLI + MCP tool. Acceptance: ingest a 50-file corpus, `query` returns citations in <2s warm.
- **Phase 2 — K (wiki):** `synthesize` prompt + worker, wiki page writer, link graph, `index.md` regenerator, `log.md` append, `lint`, `wiki.*` MCP tools. Acceptance: running `synth` on the Phase-1 corpus produces a non-empty `wiki/` with valid cross-links; `lint` reports 0 errors.
- **Phase 3 — W (wisdom, the differentiator):** `distill` prompt + worker, `wisdom_items` table, `_candidates/` flow, interactive `review`, `wisdom.apply` at query time, tests covering candidate→approved transitions and the "at least N=2 evidence" invariant.
- **Phase 4 — Polish:** OpenAI-compat provider completeness (Ollama and Azure verified), prompt-caching on Anthropic paths, packaging for PyPI (`pip install dikw-core`), docs site, GitHub Actions release automation, source-backend extension point exercised with one additional backend (likely `html` since it's trivial — keeps the seam real without committing to MinerU).
- **Phase 5 — Alternate storage adapters:**
  - **Postgres (enterprise):** `storage/postgres.py` using `psycopg[binary,pool]` + `pgvector`, `migrations/postgres/001_init.sql` with `tsvector`+GIN for FTS and `vector(N)` for embeddings. Contract test suite runs green against a `postgres:16`+`pgvector` container in CI. Packaged as `dikw-core[postgres]` optional extra.
  - **Filesystem / vault (Obsidian-native):** `storage/filesystem.py` — JSON sidecars under `.dikw/fs/`, in-process FTS, optional numpy-cosine vector search, and LLM-navigation fallback. No extra deps. `dikw init --vault` scaffolds a pure-vault layout. `dikw migrate --to sqlite` upgrades in place when the vault outgrows the filesystem backend.
  - Acceptance: the Phase 1–3 verification script runs end-to-end against each adapter with only `storage.backend` flipped in `dikw.yml`.

Each phase is a landable slice: CI green, tests added, docs updated.

## Critical Files to Create (first wave)

- `pyproject.toml` — declares package, pins runtime deps, `[project.optional-dependencies] postgres = [...]`, configures ruff/mypy/pytest
- `src/dikw_core/config.py` — Pydantic config + YAML loader (includes `storage:` block)
- `src/dikw_core/storage/base.py` — `Storage` Protocol + DTOs
- `src/dikw_core/storage/sqlite.py` — SQLite + sqlite-vec + FTS5 implementation
- `src/dikw_core/storage/migrations/sqlite/001_init.sql` — reference schema
- `src/dikw_core/storage/__init__.py` — factory resolving backend from config
- `src/dikw_core/data/backends/markdown.py` — MD parser + front-matter
- `src/dikw_core/info/chunk.py` — heading-aware chunker (port logic from qmd `store.ts:257–310`)
- `src/dikw_core/info/search.py` — RRF fusion on top of `storage.fts_search` + `storage.vec_search` (port from `mineru-doc-explorer/src/hybrid-search.ts`)
- `src/dikw_core/providers/{base,anthropic,openai_compat}.py`
- `src/dikw_core/cli.py`, `src/dikw_core/mcp_server.py`
- `tests/test_storage_contract.py` — parameterized over backends
- `.github/workflows/ci.yml`

## Verification (how we'll know it works end-to-end)

1. `uv sync` resolves cleanly; `uv run pytest` green; `uv run ruff check` + `uv run mypy src` clean.
2. `uv run dikw init examples/personal-wiki && cd examples/personal-wiki` scaffolds the expected tree.
3. Populate `sources/` with ~20 markdown notes (fixtures); `uv run dikw ingest`; confirm FTS and vec rows via a diagnostic `dikw status`.
4. `uv run dikw query "what is DIKW?"` returns an answer with at least one source citation.
5. `uv run dikw synth`; check `wiki/index.md` and `wiki/log.md` updated, at least one `entities/`/`concepts/` page created, all wikilinks resolve in `lint`.
6. `uv run dikw distill --recent` creates ≥1 candidate in `wisdom/_candidates/`; `uv run dikw review` accepts one; the corresponding `wisdom_items.status` flips to `approved` and a rendered page exists in `wisdom/principles.md` (or kind-appropriate file).
7. `uv run dikw query "what principles apply when choosing retrieval over a wiki?"` now cites the approved wisdom item.
8. `uv run dikw mcp --stdio` launches; a round-trip from an MCP client calls `core.query` and receives the same answer as step 7.
9. Swap provider in `dikw.yml` from Anthropic to OpenAI-compatible (pointed at Ollama locally or OpenAI) and repeat step 4 — works unchanged.
10. (After Phase 5, Postgres) `docker compose up postgres` (with `pgvector` image), set `storage.backend: postgres` in `dikw.yml`, rerun steps 3–8 against the Postgres adapter — every assertion holds, no engine code changes. The storage contract test suite runs green under `DIKW_TEST_POSTGRES_DSN=...` in CI.
11. (After Phase 5, Vault) `dikw init --vault examples/obsidian-vault`, open the folder in Obsidian, confirm wikilinks in `wiki/` render and files can be hand-edited; run `dikw synth` + `dikw distill` + `dikw query`; confirm that with `storage.embed: false` the engine falls back to LLM-navigation mode and still returns cited answers from `index.md`-driven retrieval; verify `.dikw/` is present in Obsidian's ignore list.

## Open execution-time decisions (not blockers for plan approval)

- Exact embedding model default (text-embedding-3-small vs bge-small) — pick in Phase 1 after a tiny retrieval eval on the fixtures.
- Whether `knowledge/links.py` parses MDX-style links (probably no — Karpathy's pattern is vanilla MD).
- Whether to ship a `dikw.yml` schema JSON for editor autocomplete — nice-to-have in Phase 4.
- License choice (MIT vs Apache-2.0) — ask user before publishing.

## Multimedia Assets — v1 (images only)

dikw-core's v1 brings images into the retrieval surface. The design honours
the four invariants from the rest of this doc — Obsidian-vault-native
on-disk format, deterministic scoping, the Storage Protocol as the only
seam, and reuse of named extension points.

**On disk.** Image binaries referenced from a markdown source (either the
standard `![alt](path)` form or Obsidian's `![[file|alias]]`) are copied
into an engine-managed directory under the project root:

```
<project_root>/assets/<sha256[:2]>/<sha256[:8]>-<sanitized-name>.<ext>
```

The 2-char hash prefix shards the directory (256-way) so the asset
folder stays Finder-, rsync-, and Dropbox-friendly even at six-figure
asset counts. The 8-hex prefix in the filename guarantees uniqueness
even when two different binaries share a sanitized stem; the trailing
sanitized name preserves human-readable semantics (Obsidian shows
`ab3f12ef-architecture-diagram.png`, not an opaque hash). Sanitization
NFC-normalizes Unicode and whitelists letters/numbers (any script,
including CJK / JP / KR / Cyrillic / Greek) plus `-` and `_`; everything
else collapses to `-`. Length is capped at 150 UTF-8 bytes on a
character boundary.

**Reference preservation across path rewrite.** The user's source
markdown is never modified — `![alt](./diagrams/foo.png)` survives
verbatim in `chunks.text` (Layer 1). Structural mapping lives in the
new `chunk_asset_refs` bridge table (Layer 2) — `(chunk_id, asset_id,
ord, alt, start_in_chunk, end_in_chunk)` — so the relationship
between chunk text positions and the materialized asset is recoverable
regardless of file moves. A pure `info/render.py::render_chunk` helper
produces a self-contained Markdown rendering with `stored_path`
substituted into the original positions on demand (Layer 3).

**Span-aware chunking.** The paragraph-aligned chunker takes an
`atomic_spans` parameter and hard-fails (rather than silently splitting)
if any image reference would cross a chunk boundary. For valid
single-line markdown image syntax this is a no-op; the explicit
post-condition catches pathological inputs and future chunker variants.

**Native multimodal embedding.** v1 introduces a
`MultimodalEmbeddingProvider` Protocol (text + image inputs → shared
vector space). The first concrete impl wraps Gitee AI's hosted
multimodal embedding endpoint over httpx; other providers (Voyage,
Cohere v4, native Jina) drop in as siblings with no engine changes.
Chunk text and asset binaries both flow through the same provider so
their vectors share one space — a text query naturally matches
against image semantics, no caption-to-text intermediary required.

**Embedding versioning.** Every embedding generation is identified by
a composite tuple `(provider, model, revision, dim, normalize, distance)`
in the new `embed_versions` table. The `revision` field is the
user-facing escape hatch when a vendor silently refreshes weights
behind a stable model name. Each version gets its own per-version
sqlite-vec virtual table (`vec_assets_v<id>`) so dim changes are
naturally isolated and prior data survives a model swap.

**Three-leg retrieval.** `info/search.HybridSearcher` now fuses three
ranked lists via RRF: BM25 over chunk text, vector search over chunk
vectors, and vector search over asset vectors with reverse-lookup
into parent chunks. Asset-vector hits promote chunks that *reference*
a matching image even when the chunk text contains no matching tokens.
The asset channel is opt-in — installations without multimodal config
get the original 2-leg behavior unchanged.

**Scope (v1) vs. roadmap.** v1 ships images only. Caption regeneration
via a `VisionProvider` (currently `assets.caption` stays NULL), audio /
video transcription with timestamp anchors, true mixed text+image
chunk encoding, and image-bytes-in-prompt LLM synthesis are all
designed to slot into the existing schema (`AssetKind` is an enum
with reserved values, `chunk_asset_refs` PK is `(chunk_id, ord)` not
tied to text-only assumptions, `LLMProvider` can grow an `images`
parameter behind a capability flag) but are deferred to v1.5/v2.

