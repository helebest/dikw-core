---
title: Hybrid retrieval notes
tags: [search]
---

# Hybrid retrieval notes

A hybrid retriever combines BM25 lexical search with dense vector search and
fuses the rank lists via Reciprocal Rank Fusion. RRF uses a constant `k`
(commonly 60) and ignores raw scores — only the rank order matters.

Strong-signal short-circuits (skip expansion when BM25 is very confident)
are a useful optimisation but not required for correctness.
