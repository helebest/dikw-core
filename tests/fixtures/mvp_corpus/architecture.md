# Architecture

`dikw-core` is structured around two ideas, both fighting for air in most
RAG stacks today:

1. Knowledge should be a **compounding artifact**, not a query-time search
   result. The wiki you can read in Obsidian is the product; the engine is
   the scribe. (This is Karpathy's LLM-wiki framing.)
2. **All four DIKW layers deserve first-class treatment.** Data, Information,
   Knowledge, and Wisdom each have their own storage, schemas, and
   operations. The pipeline between them is explicit.

Everything else is plumbing.

## The four layers

| Layer                | What lives here                                  | Writer                         |
| -------------------- | ------------------------------------------------ | ------------------------------ |
| **D** — Data         | raw source files (markdown, HTML today)          | human                          |
| **I** — Information  | parsed, chunked, FTS-indexed, embedded           | engine (deterministic)         |
| **K** — Knowledge    | LLM-authored wiki pages, link graph, `index.md`  | LLM, human-editable            |
| **W** — Wisdom       | principles / lessons / patterns with evidence    | LLM proposes, human approves   |

The W layer is the differentiator. Every item must cite **≥ 2 pieces of
evidence** drawn from the K or D layers, and every state change passes
through the review flow (`dikw review approve|reject`). Approved items
surface automatically at query time as "operating principles".

## Module map

```text
src/dikw_core/
├── api.py                 thin facade (init_wiki, ingest, synth, distill,
│                          review, query, lint, status)
├── config.py              Pydantic config + YAML loader
├── schemas.py             cross-layer DTOs
├── data/
│   ├── sources.py         source-file scanner (glob + ignore)
│   └── backends/          registry-dispatched parsers
│       ├── base.py        SourceBackend Protocol + registry
│       ├── markdown.py    .md / .markdown
│       └── html.py        .html / .htm (stdlib-only)
├── info/
│   ├── chunk.py           heading-aware paragraph chunker
│   ├── embed.py           batched embedding worker
│   └── search.py          RRF-fused FTS + vector hybrid
├── knowledge/
│   ├── wiki.py            WikiPage I/O (Obsidian-compatible front-matter)
│   ├── synthesize.py      LLM -> <page> blocks -> WikiPage
│   ├── links.py           [[wikilinks]] + md + URL parser
│   ├── indexgen.py        regenerate wiki/index.md
│   ├── log.py             render wiki/log.md from wiki_log rows
│   └── lint.py            broken wikilinks, orphans, duplicate titles
├── wisdom/
│   ├── distill.py         LLM -> <wisdom> blocks; enforces N>=2 evidence
│   ├── io.py              candidate files + aggregate regenerators
│   ├── review.py          approve/reject state machine
│   └── apply.py           stem-aware token overlap at query time
├── providers/
│   ├── base.py            LLMProvider + EmbeddingProvider Protocols
│   ├── anthropic.py       anthropic SDK, system-prompt cache_control
│   └── openai_compat.py   openai SDK; any base_url
├── storage/
│   ├── base.py              Storage Protocol (engine depends only on this)
│   ├── sqlite.py            SQLite + sqlite-vec + FTS5 (default)
│   ├── postgres.py          Postgres + pgvector + tsvector (optional extra)
│   ├── filesystem.py        DB-less JSONL sidecars + in-proc FTS (zero deps)
│   └── migrations/
│       ├── sqlite/          schema SQL
│       └── postgres/        schema SQL (pg_trgm + vector extensions)
├── prompts/               versioned LLM prompts loaded via importlib.resources
├── mcp_server.py          MCP tools grouped by layer
└── cli.py                 typer app
```

## Seams on purpose

Three extension points are sharper than they look, because the rest of the
engine depends only on their Protocol / abstract interface:

1. **`SourceBackend`** — adding a format (PDF, Quarto, `.ipynb`) means
   writing one class and calling `register()`. No other change.
2. **`Storage`** — every I/O crosses typed Pydantic DTOs. Postgres
   (Phase 5) and the Obsidian-vault filesystem backend will slot in via
   `storage/__init__.py`'s factory without touching engine code.
3. **`LLMProvider` / `EmbeddingProvider`** — Anthropic and any
   OpenAI-compatible endpoint are wired today; llama-cpp-python for local
   inference is a drop-in.

## What stays out of the adapters

Hybrid search fusion (RRF), chunking, link-graph parsing, wisdom scoring,
and prompt templating all live **outside** the storage and provider
adapters. The Storage Protocol exposes only the primitives (`fts_search`,
`vec_search`, …); fusion happens in `info/search.py`. This is the right
abstraction height: high enough to hide SQL dialects, low enough that
each adapter doesn't re-implement RRF.

## Karpathy's rule, applied

> "Scoping should be deterministic, reasoning should be probabilistic."

We take that seriously. Every navigation step (source listing, chunk
lookup, link traversal, wisdom retrieval-by-title) is deterministic SQL +
file I/O. LLM calls only enter at synthesis, distillation, and the
natural-language answering step of `query`.
