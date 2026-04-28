# TODOS

Deferred items captured during reviews. Each has enough context to be picked up
in 3 months without needing to re-litigate the reasoning. Closed items live in
git history, not here.

---

## T3 — chunk-level nDCG@k / recall@k in eval runner

**What:** Phase 2.3 of the chunk-level retrieval plan computes only
`hit@k_chunk` and `mrr_chunk`. Extend `eval/runner.py::_compute_metrics`
with `ndcg_at_k_chunk` and `recall_at_k_chunk` for chunk-level datasets.

**Why:** Fusion-parameter sweeps (tuning `same_doc_penalty_alpha`, per-leg
weights) benefit from richer discrimination signals. Hit@k and MRR are
coarse; nDCG and recall give fractional graded feedback per query.

**Pros:**
- More informative parameter sweep surface when dogfood dataset grows
- Parity with BEIR/CMTEB doc-level metric set — apples-to-apples comparison
  across public benchmarks and private chunk-level datasets

**Cons:**
- Metrics bloat if not used — YAGNI until dogfood dataset has >100 queries
  and needs fractional scoring

**Context for pickup:**
- Phase 2.3 lands hit@k_chunk / mrr_chunk using chunk-level ranked lists
  (no stem duplicates, so metrics.py's existing ndcg_at_k / recall_at_k work
  without change — just call them on chunk_id lists)
- Trivial once chunk-level evaluation path exists; wait for a real need
  signal from dogfood

**Depends on / blocked by:** Phase 2 of chunk-level retrieval plan.

---

## T4 — chunk identity race: query vs. delete/re-ingest

**What:** Handle the race where a query's fused top-K references a
`chunk_id` that was deleted (doc removed) or replaced (re-ingest rotated
chunk_ids) between the fusion computation and the hit materialization step.

**Why:** Today `storage.get_chunk(chunk_id)` returns None on missing chunk
and `_build_excerpts` silently skips the hit. This violates design.md's
"no silent failure" principle and leaves the user with top-4 results when
they asked for top-5.

**Pros:**
- Principled failure mode (loud warning / partial result annotation)
- Protects against upcoming concurrent-user scenarios when HTTP API lands

**Cons:**
- Low priority: pre-alpha, single-user, race window is microseconds
- Fix touches `info/search.py` + `api.py::_build_excerpts`; not trivial

**Context for pickup:**
- Reproduce: start `dikw query "…"`, simultaneously `rm sources/some.md &&
  dikw ingest`, observe result count < requested limit
- Fix directions: (a) re-run fusion if missing-chunk ratio > threshold;
  (b) return partial results with a warning field in `QueryResult`;
  (c) lock at query time via storage-level snapshot

**Depends on / blocked by:** None, but don't address until a real user hits
it or the HTTP API lands. Observational.
