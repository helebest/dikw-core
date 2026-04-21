---
title: Karpathy's LLM Wiki pattern
tags: [pattern, llm]
---

# Karpathy's LLM Wiki pattern

Karpathy argues that scoping should be deterministic and reasoning should be
probabilistic. In the LLM Wiki pattern an index.md catalogues all pages and
log.md keeps a chronological activity log. The wiki is the compounding
artifact; the LLM is the maintainer that performs the tedious bookkeeping.

## Operations

The pattern identifies three operations — ingest, query, and lint — each of
which edits the wiki in place.

## Drafts

Drafts should live outside the wiki until they are cited twice.
