---
title: Karpathy — LLM Wiki pattern
source: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
---

# LLM Wiki

A pattern for building personal knowledge bases using LLMs.

This is an idea file, designed to be copied to your own LLM Agent (e.g. OpenAI Codex, Claude Code, OpenCode / Pi, or etc.). Its goal is to communicate the high level idea, but your agent will build out the specifics in collaboration with you.

## The Core Idea

Most people's experience with LLMs and documents follows RAG: uploading files, retrieving relevant chunks at query time, and generating answers. This works, but the LLM rediscovers knowledge from scratch on every question. There's no accumulation.

The alternative: instead of retrieving from raw documents at query time, the LLM **incrementally builds and maintains a persistent wiki** — a structured, interlinked collection of markdown files sitting between you and raw sources. When adding a new source, the LLM reads it, extracts key information, and integrates it into the existing wiki — updating entity pages, revising summaries, noting contradictions, strengthening synthesis. Knowledge is compiled once and kept current, not re-derived on every query.

**The wiki is a persistent, compounding artifact.** Cross-references are already there. Contradictions are flagged. Synthesis reflects everything read. The wiki grows richer with every source and question.

You never write the wiki yourself — the LLM writes and maintains all of it. You curate sources, explore, and ask questions. The LLM does summarizing, cross-referencing, filing, and bookkeeping. In practice, one has the LLM agent open on one side and Obsidian on the other. The LLM makes edits based on conversation; you browse results in real time — following links, checking graphs, reading updated pages. Obsidian is the IDE; the LLM is the programmer; the wiki is the codebase.

This applies to various contexts:
- **Personal**: tracking goals, health, psychology, self-improvement
- **Research**: going deep on topics over weeks or months
- **Reading**: filing chapters, building companion wikis for characters, themes, plot threads
- **Business/team**: internal wikis maintained by LLMs, fed by Slack threads, transcripts, documents
- **Other**: competitive analysis, due diligence, trip planning, course notes, hobbies

## Architecture

Three layers:

**Raw sources** — curated collection of immutable source documents. Articles, papers, images, data files. The LLM reads from them but never modifies them.

**The wiki** — directory of LLM-generated markdown files. Summaries, entity pages, concept pages, comparisons, overviews, synthesis. The LLM owns this entirely, creating pages, updating them when new sources arrive, maintaining cross-references, keeping everything consistent.

**The schema** — document (e.g. CLAUDE.md or AGENTS.md) telling the LLM how the wiki is structured, conventions, and workflows for ingesting sources, answering questions, maintaining the wiki. This configuration makes the LLM a disciplined maintainer rather than a generic chatbot. Humans and LLM co-evolve this over time.

## Operations

**Ingest.** Drop a new source into the raw collection and tell the LLM to process it. The LLM reads the source, discusses takeaways with you, writes a summary page, updates the index, updates relevant pages across the wiki, appends a log entry. A single source might touch 10-15 pages. You can ingest one at a time, staying involved, or batch-ingest many sources with less supervision.

**Query.** Ask questions against the wiki. The LLM searches for relevant pages, reads them, synthesizes answers with citations. Forms vary — markdown pages, comparison tables, slide decks (Marp), charts (matplotlib), canvas. Good answers can be filed back into the wiki as new pages, so explorations compound like ingested sources.

**Lint.** Periodically ask the LLM to health-check the wiki. Look for contradictions, stale claims, orphan pages, important concepts lacking pages, missing cross-references, data gaps. The LLM suggests new questions and sources to investigate, keeping the wiki healthy.

### Scoping is deterministic; reasoning is probabilistic

Navigation — finding the right pages, enumerating cross-references, reading the index — should be deterministic: simple search or file-system lookups, not LLM-in-the-loop. The LLM is reserved for the probabilistic parts: synthesis, rewrites, summaries. Mixing them up — asking an LLM to "find" a page when `grep` would do — wastes tokens and adds latency for no quality gain.

## Indexing and Logging

Two special files help navigation as the wiki grows:

**index.md** is content-oriented. It's a catalog of everything in the wiki — each page with link, one-line summary, optionally metadata like date or source count. Organized by category. The LLM updates it on every ingest. When answering queries, the LLM reads the index first, then drills into pages. This works surprisingly well at moderate scale (~100 sources, ~hundreds of pages) without embedding-based RAG infrastructure.

**log.md** is chronological. It's an append-only record of what happened and when — ingests, queries, lint passes. If each entry starts with a consistent prefix (e.g. `## [2026-04-02] ingest | Article Title`), the log becomes parseable with simple tools — `grep "^## \[" log.md | tail -5` gives the last 5 entries. The log provides a timeline of wiki evolution and helps the LLM understand recent work.

## Optional: CLI Tools

At some point you may want small tools helping the LLM operate more efficiently. A search engine over wiki pages is most obvious — at small scale the index suffices, but as it grows you want proper search. `qmd` is good: local search engine for markdown with hybrid BM25/vector search and LLM re-ranking, all on-device. It has both a CLI (so LLM can shell out) and an MCP server (so LLM can use natively). You could build simpler tools yourself — the LLM can help vibe-code a naive search script as needed.

## Tips and Tricks

- **Obsidian Web Clipper** browser extension converts web articles to markdown. Useful for quickly getting sources into raw collection.
- **Download images locally.** In Obsidian Settings → Files and links, set "Attachment folder path" to fixed directory (e.g. `raw/assets/`). Then in Settings → Hotkeys, search for "Download" to find "Download attachments for current file" and bind to hotkey (e.g. Ctrl+Shift+D). LLMs can't read markdown with inline images in one pass natively; having local copies lets the LLM view them.
- **Obsidian's graph view** shows wiki shape — what's connected, which pages are hubs, which are orphans.
- **Marp** is a markdown-based slide deck format. Obsidian has a plugin for it.
- **Dataview** is an Obsidian plugin running queries over page frontmatter. If your LLM adds YAML frontmatter, Dataview generates dynamic tables and lists.
- The wiki is just a git repo of markdown files. You get version history, branching, and collaboration for free.

## Why This Works

The tedious part of maintaining knowledge bases is not reading or thinking — it's bookkeeping. Updating cross-references, keeping summaries current, noting contradictions, maintaining consistency across dozens of pages. Humans abandon wikis because maintenance burden grows faster than value. LLMs don't get bored, don't forget to update cross-references, and can touch 15 files in one pass. The wiki stays maintained because maintenance cost is near zero.

Human's job: curate sources, direct analysis, ask good questions, think about what it means. LLM's job: everything else.

The idea relates to Vannevar Bush's Memex (1945) — a personal, curated knowledge store with associative trails between documents. Bush's vision was closer to this than what the web became: private, actively curated, with connections between documents as valuable as documents themselves. The part he couldn't solve was who does maintenance. The LLM handles that.
