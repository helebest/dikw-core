# Eval plan

Scope: how `dikw-core` measures whether answers are *good*. Captures the
tradeoff between the hand-rolled retrieval gate we just shipped and the
LLM-as-judge frameworks (RAGAS, TruLens, ARES) we might adopt next, so
the next iteration doesn't re-litigate the question.

Status: **current approach is Phase-A retrieval metrics only**. Decision
revisited when the triggers at the bottom fire.

## What we measure today

Eval is a first-class CLI subcommand — `dikw eval` — driven by the
runner at `src/dikw_core/eval/runner.py`. It ingests a dataset's corpus
into a temp wiki with deterministic `FakeEmbeddings`, runs the queries
through `HybridSearcher`, and compares aggregate `hit@3`, `hit@10`, and
`MRR` against the dataset's own thresholds.

The MVP dogfood dataset (project docs + Karpathy essays + 10 queries)
lives at `evals/datasets/mvp/` with its own `dataset.yaml` specifying
the thresholds. The full three-file contract ("how to add a dataset")
is in [`evals/README.md`](../evals/README.md). The pytest gate
`tests/test_retrieval_quality.py` is now a ~10-line wrapper over the
same runner, so the CLI and the gate can never drift.

- **What's covered.** Retrieval (I layer): chunking + RRF fusion + storage
  lookup. Catches: wrong chunk boundaries, broken vec/FTS wiring, RRF bugs,
  storage-adapter regressions.
- **What's not.** Generation (K-layer synth, query answering) is not
  measured. A retrieval hit doesn't mean the LLM will answer correctly.

## Options for generation-side eval

### Homegrown golden answers

Author ~20 Q/A pairs with **reference answers**, run `api.query` against
a live LLM, compare output to reference via string match or embedding
cosine.

- **Pros:** deterministic scoring, no extra deps, cheap to run.
- **Cons:** string/cosine match is brittle — paraphrased correct answers
  fail. Reference answers drift as the corpus evolves. Brittleness
  tempts you to either relax thresholds (useless) or constantly re-author
  references (expensive).

### RAGAS (LLM-as-judge)

RAGAS runs `faithfulness` (is the answer grounded in retrieved context?)
and `answer_relevancy` (does the answer address the question?) via a
separate LLM call per metric per query.

- **Pros:** robust to paraphrase; covers both retrieval and generation
  signals from one harness; decouples metric code from reference
  answers.
- **Cons:** ~2-4 extra LLM calls per query during eval — not free at 20+
  queries with MiniMax costs. LLM judges drift when the judge model is
  upgraded. Flaky at low sample counts.

### TruLens / ARES

Similar shape to RAGAS with different tradeoffs (TruLens is
heavier-weight infra; ARES uses synthetic-data training for the judge).
Neither clearly dominates RAGAS for our stage.

## Recommendation

Stay on **Phase-A retrieval metrics only** until one of the triggers
below fires. Rationale:

1. **Retrieval dominates answer quality at pre-alpha.** If the right
   chunks aren't in context, the judge can't rescue the answer. Fixing
   retrieval first is strictly higher ROI.
2. **Deterministic > noisy.** Phase A is hermetic; RAGAS is LLM-dependent
   and adds flakiness + spend. At 10 queries, LLM-judge variance would
   drown the signal.
3. **W-layer is the real differentiator.** Once the K/W pipeline is
   actually wiring approved wisdom into answers, the interesting
   generation-side metric is "does `[W#]` get cited correctly?" — which
   a bespoke check catches more cleanly than faithfulness does.

## Triggers for revisiting

Adopt an LLM-as-judge framework (default: RAGAS) when **any** of:

- Retrieval metrics saturate (hit@10 ≥ 0.95 on a 30+-query set) and
  user-perceived quality still disappoints. The issue is in generation,
  not retrieval.
- The corpus grows past ~50 docs or the query set past ~30 pairs, at
  which point authoring golden answers becomes a bottleneck but judging
  N questions still costs O(N) LLM calls, not O(N²) author-hours.
- We land the W-layer apply-at-query path and want to measure whether
  `[W#]` citations actually match the applicable wisdom for a question.

Until then: grow the Q/A set, keep Phase A green, don't spin up a
judge harness.

## 公开 benchmark 校准

Phase A also covers comparing dikw's retriever against published BEIR
/ CMTEB baselines via [`evals/tools/convert_{beir,cmteb}.py`](../evals/README.md#public-benchmarks)
+ `dikw eval --retrieval {bm25,vector,hybrid,all}`. The framing is
**calibration, not reproduction** — five things make exact-number
parity impossible (and not actually useful):

1. **Chunking at 900 tokens.** dikw runs every ingested doc through
   `info/chunk.py` before embedding / indexing. Most BEIR passages
   are 100–500 tokens (no fragmentation), but longer CMTEB passages
   split into multiple chunks; the doc-level hit@k still works
   correctly (chunks of the same doc share a stem) but the underlying
   index shape diverges from "passage retrieval as published".
2. **FTS5 ≠ Anserini BM25.** Our `bm25` mode goes through SQLite's
   FTS5 `bm25()` — same family of formulas, different IDF / length-norm
   constants and tokenizer. Treat ±0.10 nDCG@10 vs the published
   number as in-band; larger gaps suggest a real bug, not algorithm
   choice.
3. **RRF k=60, never tuned.** `info/search.py:RRF_K` is the value
   from the original RRF paper. Different k or different fusion (CombSUM,
   weighted) could move hybrid up or down by a few points.
4. **Embedding dim choice.** The benchmark stubs default to
   Qwen3-Embedding-8B at 1024-dim (matryoshka truncation) for cost.
   The 4096-dim native vectors would shift dense + hybrid numbers; pin
   the dim in `dataset.yaml`'s comments so re-runs reproduce.
5. **CMTEB sample sizing.** The Chinese benchmarks ship at 1M+
   passages; we sample down to ~5K. Fewer distractors → higher
   absolute metrics than the published full-corpus numbers. The
   sampling preserves all relevant docs (recall is honest), but
   precision-style metrics like hit@k will read higher than they
   would at full scale.

The useful signal across all five caveats is the **trend**:

* Does `bm25` land near published BM25 (within ±0.10)?
* Does `hybrid` beat both `bm25` and `vector` on the same chunking,
  on the same dataset?
* Does the same embedder do equally well on English (BEIR) and
  Chinese (CMTEB)?

If the answer to any of those is "no", the gap is informative — go
look at the FTS leg, the RRF weighting, or the embedder. If the
absolute number doesn't match the BEIR paper, that is *expected* and
not by itself a bug.
