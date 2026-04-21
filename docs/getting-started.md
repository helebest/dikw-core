# Getting started

This walkthrough takes a blank directory to a queryable knowledge base with a
curated Wisdom layer in about five minutes. It only needs Python 3.11+ and
`uv`; LLM keys are optional until you hit `dikw synth` or `dikw query`.

## 1. Install and scaffold

```bash
git clone https://github.com/helebest/dikw-core
cd dikw-core
uv sync

# Pick any directory — `my-wiki/` below — it will also be a valid Obsidian vault.
uv run dikw init ../my-wiki --description "my research wiki"
cd ../my-wiki
```

The init command creates this tree:

```text
my-wiki/
├── dikw.yml              # the config the engine reads on every command
├── sources/              # your raw documents go here (Data layer)
├── wiki/                 # LLM-authored knowledge pages, regenerated on synth
│   ├── index.md
│   ├── log.md
│   └── {entities,concepts,notes}/
├── wisdom/               # principles / lessons / patterns (LLM proposes, you approve)
│   ├── principles.md
│   ├── lessons.md
│   ├── patterns.md
│   └── _candidates/
└── .dikw/                # engine state (gitignored)
    └── index.sqlite
```

Open the folder in Obsidian and you'll see the wiki + wisdom pages render
natively thanks to the `[[wikilink]]` syntax and YAML front-matter the engine
emits.

## 2. Add source material and ingest

Drop markdown or HTML files anywhere under `sources/`, then:

```bash
# Offline mode — indexes FTS only, no API calls.
uv run dikw ingest --no-embed

# Or with embeddings (requires OPENAI_API_KEY or a compatible endpoint).
export OPENAI_API_KEY=sk-...
uv run dikw ingest
```

`dikw status` shows document, chunk, and embedding counts per DIKW layer.
Subsequent ingests are idempotent: files whose content hash hasn't changed
are skipped.

## 3. Ask questions (Information layer → LLM)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run dikw query "What does Karpathy mean by deterministic scoping?"
```

Output includes the answer and a table of citations. Every claim cites a
source excerpt by `[#N]`; approved wisdom items cite by `[W1]`.

## 4. Synthesise a Knowledge layer

```bash
uv run dikw synth
```

The LLM reads each source doc and produces a `wiki/<folder>/<slug>.md`
page, cross-linked via `[[wikilinks]]`. `wiki/index.md` and `wiki/log.md`
regenerate automatically. Re-running is a no-op until you add new sources
(or pass `--all` to resynthesise everything).

Run `dikw lint` to check for broken wikilinks, orphans, and duplicate titles.

## 5. Distil Wisdom (the W layer)

```bash
uv run dikw distill
uv run dikw review list
uv run dikw review approve W-abcdef123456
```

`distill` prompts the LLM for principles/lessons/patterns supported by **at
least two pieces of evidence** across the wiki. Approval deletes the
candidate file, promotes the item to approved status, and regenerates
`wisdom/<kind>s.md`. Rejected items are archived.

Approved wisdom automatically surfaces in future `dikw query` answers under
an "operating principles" block — cited as `[W1]`, `[W2]`, ….

## 6. Expose the engine as an MCP server

```bash
uv run dikw mcp --stdio
```

Point Claude Desktop, Claude Code, or any MCP client at that command and
the agent can call `core.query`, `wiki.synthesize`, `wisdom.distill`,
`admin.lint`, and friends directly.

## Pluggable providers

Edit `dikw.yml` to swap LLM or embedding providers without changing code:

```yaml
provider:
  llm: openai_compat           # anthropic | openai_compat
  llm_model: gpt-4.1-mini
  llm_base_url: http://localhost:11434/v1   # Ollama, vLLM, Azure, …
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://api.openai.com/v1
```

## Pluggable storage

The SQLite+sqlite-vec backend ships as the default. The Postgres+pgvector
and filesystem (DB-less, Obsidian-vault-native) adapters land in Phase 5;
the design doc documents the seam.
