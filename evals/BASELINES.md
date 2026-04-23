# Baselines

Archive of real-embedder benchmark runs. Each entry pins a reproducible
configuration and the observed numbers, so future runs can tell a
regression from a re-run variance.

Newest first. `dikw eval` thresholds in each dataset's `dataset.yaml`
are calibrated ~2-3 % below the most recent canonical-mode run.

## 2026-04-23 — SciFact (BEIR)

**Status:** first real-vector run. Established as the baseline.

### Configuration

| | |
|---|---|
| dikw commit | `62b596f` (branch `feat/evals-public-benchmarks`, pre-merge) |
| Dataset | BEIR/SciFact — 5,183 passages, 300 test queries, binary qrels |
| Converter | `evals/tools/convert_beir.py` — test split, baseline 0.665 |
| LLM provider | *(not exercised — retrieval-only eval)* |
| Embedding provider | openai_compat → Gitee AI (`https://ai.gitee.com/v1`) |
| Embedding model | `Qwen3-Embedding-8B`, **4096-dim native** (no matryoshka truncation) |
| Embedding batch size | 16 (Gitee AI caps at 25) |
| Chunking | default — heading-aware, 900 tokens, 15 % overlap |
| Fusion | RRF k = 60 (untuned default) |
| FTS5 sanitizer | OR-of-tokens (commit 62b596f) — reproduces Anserini behavior |
| Donor wiki | `tests/fixtures/live-minimax-gitee.dikw.yml` copied verbatim |
| Wall time | ~1 h 20 min (embedding ~28 min, query phase ~50 min) |
| Approximate cost | ~¥0.5 on Gitee AI (5183 passages × 4096-dim) |

### Results

```
dikw eval — scifact  (retrieval ablation: bm25 / vector / hybrid)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.707 │     0.817 │ 0.638 │      0.669 │         0.865 │
│ vector │    0.797 │     0.907 │ 0.738 │      0.773 │         0.947 │
│ hybrid │    0.757 │     0.883 │ 0.708 │      0.736 │         0.970 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

### Comparison vs published baselines

| Metric | Our bm25 | Published (BEIR paper, BM25 Anserini) | Δ |
|---|---|---|---|
| nDCG@10 | **0.669** | **0.665** | **+0.004** |

Within 0.004 of the published number — **FTS5 BM25 is functionally
equivalent to Anserini BM25 on SciFact** once the sanitizer tokenizes
correctly. This tightens `docs/eval-plan.md` caveat #2 ("FTS5 ≠
Anserini"); the algorithmic difference is below-noise on this dataset.

Dense retrieval in the original BEIR paper spans 0.48 – 0.67 nDCG@10
depending on the model (DPR, ANCE, GenQ, TAS-B, ColBERT). Our vector
leg at 0.773 is consistent with modern state-of-the-art embedders
(Qwen3, BGE-M3, E5-mistral family), which sit above the 2021 numbers.

### Known issues / observations

**1. Hybrid underperforms vector (regression risk).** On SciFact,
dense-only retrieval (`vector` mode) beats hybrid by +0.037 nDCG@10
and +0.024 hit@10. RRF with `k=60` gives equal weight to both legs;
because bm25 is ~0.10 nDCG@10 behind vector, averaging pulls the
ranking toward the weaker signal. Hybrid still wins on **recall@100
(0.970 vs 0.947)** — the bm25 leg covers docs vector misses, just at
ranks too low to help the position-weighted metrics.

This confirms `docs/eval-plan.md` caveat #3 ("RRF k=60 never tuned").
Not a blocker for this PR — the ablation surfacing the problem is
exactly what the new `--retrieval all` flag was built for — but the
**default behaviour of `dikw query` is currently suboptimal on
paraphrased / synonym-heavy queries where one leg dominates**.

**2. Real-embedder wall time dominates (~80 min for 5K corpus).** Most
of it is the 324 embedding batches at ~5 s each (Gitee AI latency
scales with output dim; 4096-dim is large). Using 1024-dim matryoshka
truncation would cut cost and time ~4×, at the quality cost of
roughly -0.02 nDCG@10 on Qwen3 (rule-of-thumb from matryoshka papers;
not measured here). Tradeoff to revisit if we ever want to run this
in CI — presently it stays manual.

**3. Chunking was not a factor.** SciFact passages median ~200 tokens;
no doc fragmented at the 900-token chunk boundary. Caveat #1 in
`docs/eval-plan.md` is effectively dormant for BEIR-scale corpora and
will only bite on CMTEB where passages routinely exceed 900 tokens.

### Follow-ups (not this PR)

Priority-ordered. Each is a self-contained piece of work; do not bundle.

- **Fix RRF weighting.** Two flavors worth trying, in order: (a)
  per-dataset tunable `RRF_K` / mode weights via `dikw.yml` — small,
  lets the user discover the right ratio per corpus; (b) score-
  normalized fusion (CombSUM / CombMNZ over min-max scaled scores) —
  more complex but solves the problem generally. Gate on whether (a)
  moves the SciFact hybrid number above vector.
- **Calibrate thresholds per-dataset after (a).** Thresholds currently
  track hybrid-as-canonical; if hybrid gets stronger, raise the floors.
- **Fix the dataset.yaml overwrite regression.** `convert_beir.py`
  always rewrites the file, stripping any curated description. Should
  merge into the existing stub (preserve user keys, only rewrite the
  block the converter owns).
- **Run CMTEB / T2Retrieval subset** against the same donor wiki —
  same Qwen3 embedder handles both languages; result shape should be
  similar but with 900-token chunking actually biting.
- **Sweep chunk size** (450 / 900 / 1800 tokens) on SciFact — the
  default is picked by feel, not measured. Might recover part of the
  hybrid gap if shorter chunks localise BM25 signal better.
