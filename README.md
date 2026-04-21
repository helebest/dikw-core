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
git clone <repo>
cd dikw-core
uv sync
uv run dikw --help
uv run dikw init my-wiki
cd my-wiki
uv run dikw status
```

## Layout

The package lives under `src/dikw_core/`; the approved design doc is [`docs/design.md`](./docs/design.md).

## License

MIT — see [LICENSE](./LICENSE).
