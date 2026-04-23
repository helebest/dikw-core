# Baselines

Archive of real-embedder benchmark runs. Each entry pins a reproducible
configuration and the observed numbers, so future runs can tell a
regression from a re-run variance.

Newest first. `dikw eval` thresholds in each dataset's `dataset.yaml`
are calibrated ~2-3 % below the most recent canonical-mode run.

## 2026-04-24 — CMTEB / T2Retrieval (sampled, 300 queries)

**Status:** First real-vector run on Chinese. **BM25 leg unusable on
CJK with the current FTS5 tokenizer** — see "Known issues" #1. The
fusion-weight generalisation question that motivated this run
(BASELINES v2 follow-up) is therefore **deferred** to a re-run after
the CJK BM25 fix lands. What we *can* report from this run: vector
retrieval works strongly on Chinese, and the v2 fusion defaults
`(rrf_k=60, bm25_weight=0.3, vector_weight=1.5)` "trivially generalise"
in the sense that they correctly down-weight a useless BM25 leg
(equal-weight fusion only loses 0.003 nDCG@10).

### Configuration

| | |
|---|---|
| dikw commit | branch `feat/evals-cmteb-calibration` |
| Dataset | C-MTEB / T2Retrieval — 118,605 passages, 22,812 queries (full); see sampling below |
| Sampling | query-first: 300 queries (seed=42) + 1,582 relevant pids + 3,418 random distractors → 5,000 corpus docs. See `evals/tools/prep_cmteb_t2.py` and `dataset.yaml`'s `_preprocessing` block. |
| LLM provider | *(not exercised — retrieval-only eval)* |
| Embedding provider | openai_compat → Gitee AI (`https://ai.gitee.com/v1`) |
| Embedding model | `Qwen3-Embedding-8B`, **4096-dim native** (matches SciFact v2) |
| Embedding batch size | 16 (Gitee AI caps at 25) |
| Chunking | default — heading-aware, 900 tokens, 15 % overlap |
| Fusion | RRF `(k=60, w_bm25=0.3, w_vec=1.5)` — shipped v2 defaults |
| FTS5 tokenizer | `unicode61` (default) — **the problem; see Known issues #1** |
| Donor wiki | `tests/fixtures/live-minimax-gitee.dikw.yml` copied verbatim |
| Wall time | ~4 h end-to-end (embedding ~28 min; query phase dominated by 300 × 3 modes × top-100) |
| Approximate cost | ~¥0.5 on Gitee AI (5,000 passages × 4096-dim) |

### Why the corpus is sampled query-first, not stratified

Unlike BEIR/SciFact, **every T2Retrieval corpus passage is a positive
qrel for at least one query** (118,605 unique pids referenced =
118,605 corpus total). The generic stratified sampler in
`convert_cmteb.py` therefore degenerates — it preserves the relevant
set, which equals the whole corpus, so any `--sample-size` < 118,605
returns all 118,605 passages. Realistic calibration on this dataset
requires sampling **queries** first, then collecting their positive
pids and padding with distractors drawn from docs not referenced by
the sampled queries. `evals/tools/prep_cmteb_t2.py` implements this
pre-pass.

### Results

```
dikw eval — cmteb-t2-subset  (retrieval ablation: bm25 / vector / hybrid)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.070 │     0.073 │ 0.070 │      0.031 │         0.025 │
│ vector │    0.967 │     0.983 │ 0.952 │      0.942 │         0.987 │
│ hybrid │    0.967 │     0.983 │ 0.952 │      0.942 │         0.987 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

`hybrid` matches `vector` to four decimal places on every position-
weighted metric and on recall@100. RRF with `(0.3, 1.5)` against a
near-zero BM25 ranking effectively returns the vector ranking
unmodified — see Known issues #2.

### Weight sweep (offline, same dump)

Same 48-combination grid as SciFact v2 (`rrf_k ∈ {40, 60, 100}` ×
`w_bm25 ∈ {0.3, 0.5, 0.7, 1.0}` × `w_vec ∈ {0.5, 1.0, 1.5, 2.0}`):

```
|   k | w_bm25 | w_vec | hit@3 | hit@10 |    mrr | nDCG@10 | recall@100 |
|----:|-------:|------:|------:|-------:|-------:|--------:|-----------:|
|  40 |   0.30 |  1.50 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  40 |   0.30 |  2.00 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  40 |   0.50 |  2.00 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  60 |   0.30 |  2.00 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |
|  60 |   0.30 |  1.50 | 0.967 |  0.983 |  0.952 |   0.942 |      0.987 |  ← shipped default
|  60 |   1.00 |  1.00 | 0.963 |  0.983 |  0.950 |   0.939 |      0.987 |  ← equal-weight (pre-tuning)
```

Every top-tier configuration produces identical numbers because the
BM25 leg contributes essentially no signal — see Known issues #1.
Equal-weight only drops 0.003 nDCG@10 vs the shipped default; that
0.003 is the entire "fusion-weight tuning" surface area on this
dataset, and it's within re-run noise. **The fusion-weight
generalisation question is unanswerable on this dataset until BM25
on CJK is fixed.**

### Comparison vs published baselines

CMTEB leaderboard numbers shift over time and depend on the model
under test, so this dataset's `dataset.yaml` deliberately omits a
``published_baselines`` block. For reference at the time of writing,
modern Chinese embedders on the MTEB-CN leaderboard cluster around
nDCG@10 ≈ 0.85–0.93 on T2Retrieval (full corpus, 22,812 queries);
our 0.942 on the **300-query / 5K-corpus** subset is naturally above
that range because the smaller corpus reduces distractor density.
**Treat this number as a calibration floor for re-runs, not a
leaderboard comparison.**

For BM25 on CJK, the published Anserini-with-jieba baselines on
T2Retrieval sit around nDCG@10 ≈ 0.50–0.65 (model- and
configuration-dependent). Our **0.031** is two orders of magnitude
below that — see Known issues #1.

### Known issues / observations

**1. FTS5 BM25 is unusable on CJK with the current `unicode61`
tokenizer (P0 follow-up).** 275 of 300 queries (91.7%) returned
**zero** BM25 results; the remaining 25 mostly returned 1–6 hits, and
only 4 reached the top-100 limit. Diagnosis: SQLite FTS5's default
`unicode61` tokenizer treats CJK characters as single-character
tokens, and the `_sanitize_fts` rewrite path (added in commit
`62b596f` for the SciFact ablation) feeds those characters as
OR-of-tokens. Per-character BM25 has no useful signal for Chinese
passage retrieval — high-frequency characters dominate IDF, and
discriminating multi-character terms never form. The fix is a
CJK-aware tokenizer; SQLite's built-in `trigram` (3-character
n-grams) is the lowest-friction option, jieba+full-text is the
heaviest. This needs its own focused branch.

**2. Hybrid degenerates to vector-only when one leg is dead, and the
v2 default's `bm25_weight=0.3` makes that degeneration safe.** With
a noise-quality BM25 ranking, RRF at any `(k, weights)` returns
something proportional to the vector ranking; the `(0.3, 1.5)`
weighting accelerates this collapse and produces an identical
top-100 to vector-only mode. This is a small but real win for the
v2 defaults: a keyword-tuned default (e.g. equal-weight) would have
let BM25's noise leak into the fused ranking and lose 0.003 nDCG@10.
Not enough to matter here, but the directional behaviour generalises
from SciFact (where BM25 was real but ~0.10 nDCG@10 weaker than
vector — same shape, different magnitude).

**3. Vector retrieval is strong on Chinese.** Qwen3-Embedding-8B at
4096-dim hits hit@3=0.967 / nDCG@10=0.942 / recall@100=0.987 on
this subset. Wall-time and cost track SciFact (5,183 passages →
~28 min embedding; this run with 5,000 → similar). Confirms
docs/providers.md's recommendation that Qwen3 handles both languages
without per-corpus retuning.

**4. Sampling caveat for re-runs.** The 5K subset is materially
denser-relevant than full T2Retrieval (1,582 / 5,000 = 31.6% of
docs are positive for some sampled query, vs 22,812 / 118,605 =
19.2% across the full benchmark). Higher hit@k vs the leaderboard
is structural, not a quality claim. The leaderboard comparison only
becomes meaningful after the BM25 fix lets us actually exercise both
legs — at which point we can also widen to the full corpus if we're
willing to spend ~¥10 / 10 hours embedding.

### Verdict on the original follow-up

> **Run CMTEB / T2Retrieval subset on the v2 defaults to confirm the
> weights generalise to Chinese + a different BM25/dense balance.**

Trivially yes (see above), but the test wasn't real — there was no
"different BM25/dense balance" to discover, only a broken BM25 leg
the v2 weights happened to silence cleanly. Real generalisation
verdict requires a re-run after the BM25 CJK fix.

### Follow-ups (priority-ordered)

1. **P0 — FTS5 CJK tokenizer.** Default `unicode61` doesn't segment
   CJK; switch to SQLite's built-in `trigram` (lowest friction, no
   new dep) or evaluate jieba-based segmentation. Schema change with
   a re-index migration path for existing wikis. Independent branch.
2. **After P0 — Re-run this baseline.** Same donor wiki, same
   sampling, same dump pipeline; should produce a non-trivial BM25
   leg whose nDCG@10 lands closer to the 0.50–0.65 published range,
   at which point the fusion-weight generalisation question becomes
   answerable.
3. **Score-normalised fusion (CombSUM / CombMNZ).** Same as in v2
   follow-ups; even after BM25 is fixed, RRF is rank-based and may
   not beat vector-only on datasets where the dense signal is much
   stronger than keyword. Independent branch.
4. **Optional — full-corpus run** for an apples-to-apples leaderboard
   comparison. ~10× the cost and time. Probably only worth it after
   #1 + #2 produce a credible single-corpus baseline.

## 2026-04-23 — SciFact (BEIR) v2 — RRF tuned

**Status:** supersedes v1 as the canonical SciFact baseline. Same
embeddings, new fusion defaults.

### What changed from v1

v1 exposed a structural regression: equal-weight RRF at `k=60` let the
~0.10-nDCG-weaker BM25 leg drag hybrid 0.037 points below the
vector-only mode on nDCG@10. The fix, from BASELINES v1 follow-up #1,
landed in three layers:

1. `reciprocal_rank_fusion` grew a `weights` parameter (it had always
   been equal-weight).
2. `RetrievalConfig` in `dikw.yml` exposes `rrf_k` + `bm25_weight` +
   `vector_weight` so users can tune per-corpus.
3. `dikw eval --retrieval all --dump-raw` + `evals/tools/sweep_rrf.py`
   re-fuse a single dump at arbitrary weights offline — no re-embedding.

### Weight sweep (offline, same dump as v1)

Full 48-combination grid: `rrf_k ∈ {40, 60, 100}` × `bm25_weight ∈ {0.3, 0.5, 0.7, 1.0}`
× `vector_weight ∈ {0.5, 1.0, 1.5, 2.0}`. Top 5 by nDCG@10 plus the
vanilla and shipped-default reference rows:

```
|   k | w_bm25 | w_vec | hit@3 | hit@10 |    mrr | nDCG@10 | recall@100 |
|----:|-------:|------:|------:|-------:|-------:|--------:|-----------:|
|  40 |   0.30 |  2.00 | 0.793 |  0.917 |  0.737 |   0.773 |      0.970 |
|  40 |   0.30 |  1.50 | 0.797 |  0.917 |  0.736 |   0.772 |      0.970 |
|  60 |   0.30 |  1.50 | 0.797 |  0.920 |  0.734 |   0.771 |      0.970 |  ← shipped default
|  40 |   0.50 |  2.00 | 0.793 |  0.920 |  0.734 |   0.770 |      0.970 |
|  60 |   0.30 |  2.00 | 0.790 |  0.917 |  0.733 |   0.770 |      0.970 |
|  60 |   1.00 |  1.00 | 0.757 |  0.883 |  0.708 |   0.736 |      0.970 |  ← equal-weight (pre-tuning)
```

Full grid + commentary in `/tmp/sweep-result.log` on the run host.

### Shipped defaults: `(rrf_k=60, bm25_weight=0.3, vector_weight=1.5)`

Why this point and not the grid top:

- **Keep `k=60` unchanged.** It's the original RRF-paper constant and
  what every downstream user's mental model pins on. `k=40` scores
  0.002 higher on SciFact — within noise, and the flatter `k=60` curve
  across the grid suggests it generalises better to other corpora.
- **Closest "k=60" row to the Pareto top.** Tiers at `w_bm25=0.3` all
  collapse to ~0.771 nDCG@10; picking a more moderate `w_bm25=0.5` or
  `0.7` loses 0.004–0.011 nDCG@10 for arguably less overfit but
  weakens the SciFact win we actually measured.
- **Recall@100 unchanged at 0.970.** Every top-tier config keeps this
  — the BM25 leg's recall contribution is invariant to its weight; only
  position-weighted metrics shift.

### Results (shipped defaults)

```
dikw eval — scifact  (retrieval ablation, v2 defaults)
┌────────┬──────────┬───────────┬───────┬────────────┬───────────────┐
│ mode   │ hit_at_3 │ hit_at_10 │   mrr │ ndcg_at_10 │ recall_at_100 │
├────────┼──────────┼───────────┼───────┼────────────┼───────────────┤
│ bm25   │    0.707 │     0.817 │ 0.638 │      0.669 │         0.865 │
│ vector │    0.797 │     0.907 │ 0.738 │      0.773 │         0.947 │
│ hybrid │    0.797 │     0.920 │ 0.734 │      0.771 │         0.970 │
└────────┴──────────┴───────────┴───────┴────────────┴───────────────┘
```

Hybrid now effectively **ties vector on every position-weighted metric
within re-run noise** (Δ ≤ 0.004), while keeping the BM25 leg's
recall@100 advantage (+0.023 over vector). The v1 regression ("hybrid
< vector on nDCG@10") is resolved.

### Comparison vs published baselines (unchanged from v1)

| Metric | Our bm25 | Published (BEIR paper, BM25 Anserini) | Δ |
|---|---|---|---|
| nDCG@10 | **0.669** | **0.665** | **+0.004** |

The BM25 leg is untouched by fusion tuning; the Anserini calibration
stays as before.

### Caveat: SciFact-tuned default is vector-heavy

The `(0.3, 1.5)` ratio over-favours dense semantics. On a
**keyword-heavy corpus** (code, identifiers, exact-term lookup) this
default will under-rank BM25 hits — raise `bm25_weight` in
`dikw.yml`'s `retrieval:` block. The `evals/tools/sweep_rrf.py` flow
is the intended tuning path; it finishes in milliseconds against a
`--dump-raw` JSONL, no second API call.

### Follow-ups (still not this PR)

- **Run CMTEB / T2Retrieval subset** on the v2 defaults to confirm the
  weights generalise to Chinese + a different BM25/dense balance.
- **Sweep chunk size** (450 / 900 / 1800 tokens) on SciFact. The
  default is picked by feel, not measured.
- **Score-normalized fusion** (CombSUM / CombMNZ) — the v2 tuning
  basically tied vector; going *above* it likely requires fusion that
  honours score magnitude, not just rank position. Independent branch,
  not blocked on anything above.

## 2026-04-23 — SciFact (BEIR)

**Status:** superseded by v2 above (same embeddings, equal-weight
fusion). Kept for historical reference — the pre-tuning numbers are
the fair comparison point for any future fusion work.

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
