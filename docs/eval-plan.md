# Eval plan

Scope: how `dikw-core` measures whether answers are *good*. Captures the
tradeoff between the hand-rolled retrieval gate we just shipped and the
LLM-as-judge frameworks (RAGAS, TruLens, ARES) we might adopt next, so
the next iteration doesn't re-litigate the question.

Status: **current approach is Phase-A retrieval metrics only**. Decision
revisited when the triggers at the bottom fire.

## What we measure today

`tests/test_retrieval_quality.py` ingests a curated dogfood corpus into
a temp wiki with deterministic `FakeEmbeddings`, runs ~10 hand-authored
queries through `HybridSearcher`, and asserts aggregate `hit@3`, `hit@10`,
and `MRR` thresholds. Ground truth lives in `tests/fixtures/mvp_queries.yaml`
as `q → expect_any: [doc_stem, …]`.

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
