# TODOS

Deferred items captured during reviews. Each has enough context to be picked up
in 3 months without needing to re-litigate the reasoning.

---

## T1 — `doc.read(chunk_id=...)` MCP tool extension

**What:** Extend `mcp_server.py::doc.read` to accept an optional `chunk_id: int`
param; when present, return just that chunk's text (with optional surrounding
context) instead of the whole file.

**Why:** Phase 1 (chunk-level retrieval) makes `Hit.chunk_id` meaningful in
the MCP `doc.search` JSON payload. But MCP clients have no way to fetch a
specific chunk's context — they must pull the whole doc and re-chunk
client-side, which duplicates chunker logic outside the engine.

**Pros:**
- Closes the "producer says chunk_id matters; consumer has no chunk-level read" gap
- Keeps chunker logic single-sourced inside dikw-core
- Enables LLM clients to read just-enough context (token budget friendly)

**Cons:**
- New MCP surface — schema evolution contract
- Returning "chunk + N surrounding chunks" needs a policy decision (fixed N,
  token budget, same doc only?)

**Context for pickup:**
- Phase 1 of the chunk-level retrieval plan lands `chunk_id` as non-optional
  on `Hit`; this TODO closes the consumer side
- Existing `mcp_server.py::doc.read` at :281-289 takes only `path` and
  `wiki_path`; extending with `chunk_id` is additive
- `Storage.get_chunk(chunk_id)` already exists (`storage/base.py:89`)

**Depends on / blocked by:** Phase 1 of chunk-level retrieval plan.

---

## T2 — Move `Hit` DTO to `schemas.py`

**What:** Relocate `class Hit` from `src/dikw_core/info/search.py:82` to
`src/dikw_core/schemas.py`. Pure rename PR.

**Why:** CLAUDE.md mandates DTOs crossing the Storage Protocol live in
`schemas.py`. Hit doesn't cross Storage — it crosses `info → api → mcp_server`.
But it IS the public contract emitted to MCP clients via
`doc.search` / `core.query`. Peers `FTSHit` / `VecHit` / `AssetVecHit` already
live in schemas.py; Hit is the asymmetric outlier.

**Pros:**
- External consumers (MCP, future HTTP API) get a single schemas.py contract
  surface
- Consistent with FTSHit/VecHit/AssetVecHit siblings
- Makes the "stable external API" file easy to find

**Cons:**
- Creates an import-cycle risk if schemas.py ends up importing info/
  internals; need to verify AssetRecord etc. references are already clean
  (spot check: yes, schemas.py already has all needed types)

**Context for pickup:**
- Hit currently defined at `info/search.py:82-91`
- Used by `api.py` (return type of `HybridSearcher.search`), `mcp_server.py`
  (serialized via `model_dump()`), tests
- Pure move + adjust import paths; no behavior change

**Depends on / blocked by:** None. Can land any time.

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

---

## Closed

- **T5** — multimodal `q_vec` leaking into `chunks_vec.vec_search`. Fixed
  in PR #27 (PR-A): per-version `vec_chunks_v<id>` tables landed for the
  text channel; `HybridSearcher` now computes `q_vec_text` and `q_vec_mm`
  independently against their own `version_id`s. The dim-mismatch fallback
  at the storage boundary is gone.
- **T6** — `embed_versions` UNIQUE missing `modality`. Fixed in PR #27
  (PR-A): UNIQUE on both backends extends to
  `(provider, model, revision, dim, normalize, distance, modality)` and
  `upsert_embed_version`'s match key matches.

