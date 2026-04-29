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
│   ├── assets.py          image/asset materialization (sha-streamed)
│   ├── hashing.py         streaming + in-memory SHA-256 helpers
│   └── backends/          registry-dispatched parsers
│       ├── base.py        SourceBackend Protocol + registry
│       ├── markdown.py    .md / .markdown
│       └── html.py        .html / .htm (stdlib-only)
├── info/
│   ├── chunk.py           heading-aware paragraph chunker
│   ├── tokenize.py        CJK-aware preprocessing + token counting
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
│       └── postgres/        schema SQL (vector extension)
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

## Storage migrations

**Policy: pre-alpha "rebuild on incompatibility".** Each adapter ships
numbered SQL files under `storage/migrations/{sqlite,postgres}/` (`001_init.sql`,
`002_assets.sql`, `003_embed_cache.sql`). On every `migrate()` the
adapter applies any whose numeric prefix is greater than the value
recorded in `meta_kv['schema_version']`, then writes back the new high
water mark. Existing `IF NOT EXISTS` guards inside each script make
the body idempotent if the counter is missing — `meta_kv` is for
audit / diagnostics, not a full migration upgrade framework.

For schema changes that need actual data movement (column rename,
constraint flip, in-place rebuild), each adapter has a
`_verify_no_legacy_*` / `_migrate_legacy_*` helper family that runs
after the SQL files. They detect the old shape via PRAGMA / `to_regclass`
and either rebuild in place or fail loudly with rebuild instructions.
Because these touch row data, the helpers stay separate from the
numbered migration tracker.

**Deprecated tables / columns** (kept here so a future change doesn't
re-introduce a name collision; the legacy bail-out helpers reject any
SQLite/Postgres DB still carrying these):

| Name | Layer | Removed in | Replacement |
|---|---|---|---|
| `content` table | D | PR #19 | hash indexed on `documents.hash` |
| `chunks_vec` (singleton vec table) | I | PR #27 | per-version `vec_chunks_v<id>` |
| `embed_meta(chunk_id, model)` | I | PR #27 | `chunk_embed_meta(chunk_id, version_id)` |
| `chunks.start` / `chunks."end"` | I | PR #23 | `chunks.start_off` / `end_off` |
| `assets.width` / `height` | D | PR #25 | `media_meta` JSON discriminated union |
| `assets.caption` / `caption_model` | D | PR #25 | (placeholder, never used) |
| `assets.hash` | D | PR #37 | redundant with `asset_id` (= sha256 hex) |
| `documents.path UNIQUE` (column-level) | D | PR #39 | `documents.path_key UNIQUE` (NFC + casefold) |
| `documents_fts.{path,title,layer}` columns | I | this PR | body-only; doc/layer come from JOIN through `chunks` + `documents` |

### Cross-adapter shape: where the two SQL backends intentionally differ

The SQLite and Postgres tables match column-for-column with the
expected dialect aliases (`INTEGER`/`BIGINT`, `REAL`/`DOUBLE PRECISION`,
`BLOB`/`BYTEA`, `INTEGER 0/1`/`BOOLEAN`). One table is shaped
differently **on purpose** because the two engines implement FTS via
different mechanisms:

| Where text indexing lives | SQLite | Postgres |
|---|---|---|
| Search index | separate FTS5 virtual table `documents_fts` (body-only; `rowid` aligns with `chunks.chunk_id`) | generated `chunks.fts tsvector` column over `chunks.text` with a `GIN` index |

Both adapters expose the same `fts_search` method on the `Storage`
Protocol returning the same `FTSHit` DTOs — the engine never sees the
divergence. Schema-parity diff tools should treat `chunks.fts` (PG-only)
and `documents_fts` (SQLite-only) as the dual implementations of the
same logical capability. Their **column scope** is now identical
(both index only chunk body text); legacy SQLite DBs that carry the
old 4-column shape are auto-rebuilt by
`_migrate_legacy_documents_fts`.

**Tokenization** is also aligned: SQLite uses `unicode61
remove_diacritics 0` (the `0` is explicit because the unicode61
default is `1`, which still strips diacritics) so `café` and `cafe`
are different tokens — same byte-level behavior as PG's
`to_tsvector('simple', text)`. CJK input on the SQLite + filesystem
adapters flows through `info.tokenize.preprocess_for_fts` (jieba
when `cjk_tokenizer="jieba"`) on both ingest and query; PG does its
tokenization inside `to_tsvector` and is unaffected by the Python-
side preprocessor.

The PG `fts_search` consumes the `info/search.py:_sanitize_fts`
output via `to_tsquery` (with a small `_fts_to_tsquery_string`
adapter that translates SQLite's `'"foo" OR "bar"'` form into PG's
`'foo | bar'`). Earlier versions used `plainto_tsquery`, which
re-tokenized the sanitizer output and treated `OR` as a literal
search word — broken for any multi-word query.

Both adapters now apply the `documents.active = TRUE` filter inside
`fts_search` so soft-deleted documents never surface in BM25 hits.
Pre-PR the SQLite adapter skipped this filter (PG always applied
it); the post-PR JOIN on `documents` makes the alignment cheap and
removes a silent recall divergence on inactive docs.

`SQLiteStorage` also reports `notnull=0` on every `INTEGER`/`TEXT`
PRIMARY KEY column via `PRAGMA table_info` (a documented SQLite quirk
— PK columns implicitly enforce NOT NULL). Postgres `information_schema`
reports `notnull=1` for the same columns, which is the actual contract
both adapters honor at write time.

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
