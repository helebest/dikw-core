# Providers

How to pick, configure, and switch LLM / embedding vendors in `dikw-core`.
If you're adding a new vendor, start here before touching code — usually
you don't need to touch any.

## The design seam

`dikw-core` ships with two protocol-level providers, both pluggable via
`dikw.yml` alone:

- **`anthropic`** — uses the official `anthropic` async SDK. `base_url`
  can retarget it at any Anthropic-protocol-compatible endpoint (e.g.,
  MiniMax). Applies `cache_control: ephemeral` on the system prompt, so
  repeated synth / query / distill within the 5-minute TTL hit the
  prompt cache.
- **`openai_compat`** — uses the `openai` async SDK against any
  `base_url` that speaks the OpenAI HTTP surface. Covers OpenAI, Azure,
  Ollama, vLLM, TEI, DeepSeek, GLM, Gemini (OpenAI-compat mode),
  Gitee AI, and most others.

Every vendor falls under one of these two. **To add a new vendor you
don't write code; you edit `dikw.yml` and the right two env vars.**

## Vendor cookbook

Tested / known-compatible combinations. URLs may evolve — always
cross-check the vendor's own docs.

| Vendor | `llm` | `llm_base_url` | `embedding` | `embedding_base_url` | LLM key env | Embed key env |
|---|---|---|---|---|---|---|
| **OpenAI** (default) | `openai_compat` | `https://api.openai.com/v1` | `openai_compat` | same | `OPENAI_API_KEY` | `DIKW_EMBEDDING_API_KEY` |
| **Anthropic** | `anthropic` | leave `null` | *(no embed — pair elsewhere)* | — | `ANTHROPIC_API_KEY` | — |
| **MiniMax** | `anthropic` | `https://api.minimaxi.com/anthropic` | *(no embed — pair elsewhere)* | — | `ANTHROPIC_API_KEY` | — |
| **GLM / 智谱** | `openai_compat` | `https://open.bigmodel.cn/api/paas/v4` | `openai_compat` | same | `OPENAI_API_KEY` | `DIKW_EMBEDDING_API_KEY` |
| **Gemini** | `openai_compat` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `openai_compat` | same | `OPENAI_API_KEY` | `DIKW_EMBEDDING_API_KEY` |
| **DeepSeek** | `openai_compat` | `https://api.deepseek.com/v1` | *(no embed — pair elsewhere)* | — | `OPENAI_API_KEY` | — |
| **Gitee AI** | *(often paired as embed only)* | — | `openai_compat` | `https://ai.gitee.com/v1` | — | `DIKW_EMBEDDING_API_KEY` |
| **Ollama / vLLM / TEI** (local) | `openai_compat` | `http://localhost:<port>/v1` | `openai_compat` | same or localhost | `OPENAI_API_KEY` (any non-empty) | `DIKW_EMBEDDING_API_KEY` (any non-empty) |

**Reference configs** (committed in this repo):
- [`tests/fixtures/live-minimax-gitee.dikw.yml`](../tests/fixtures/live-minimax-gitee.dikw.yml)
  — MiniMax LLM + Gitee AI embeddings. Drop-in for a fresh wiki.

Add more fixtures over time as you verify combinations; PRs welcome.

## Switching procedure

1. **Edit `dikw.yml`** — replace the `provider:` block with the target
   vendor's values from the cookbook above. Don't forget
   `embedding_dim` and `embedding_batch_size` if the new embedder
   differs from the old (see gotchas).
2. **Update `.env`** with the new key values. Variable *names* don't
   change — only their *values*:
   - `ANTHROPIC_API_KEY` → LLM key (Anthropic or MiniMax).
   - `OPENAI_API_KEY` → LLM key (OpenAI, Azure, Ollama, GLM, Gemini, …).
   - `DIKW_EMBEDDING_API_KEY` → embedding key (same or different vendor).
3. **If the embedding model dim changed**, delete `.dikw/index.sqlite`
   (see gotcha #1). Reingestion is required.
4. **Verify**:
   ```bash
   uv run --env-file .env dikw check --path <wiki> --llm-only
   uv run --env-file .env dikw check --path <wiki> --embed-only
   ```
   Each variant pings one endpoint with one tiny request; failures
   print the error inline. Exit 0/1 is scriptable.
5. **`uv run dikw ingest`** to re-populate the I layer if you wiped the
   SQLite file.

## Production gotchas

In order of likely bite. Not blocking for experimentation, but worth
knowing before committing to a vendor on a real corpus.

### 1. Embedding dimensions are locked at first write

SQLite `vec0` locks vector dimension on the first `upsert_embeddings`
call. Changing `embedding_dim` or `embedding_model` to a
different dim afterwards produces a dimension-mismatch error on the
next insert.

**Only safe migration path today:** delete `.dikw/index.sqlite`, then
`dikw ingest` fresh. There is no incremental re-embed.

### 2. Batch size varies per vendor

Default `embedding_batch_size: 64` is safe for OpenAI (~2048 cap) but
violates other vendors:

| Vendor | Observed cap | Recommended config |
|---|---|---|
| OpenAI | ~2048 | default (64) |
| Gemini | 100 | default (64) |
| GLM | unverified | start with 32 |
| **Gitee AI** | **25** | `embedding_batch_size: 16` |

Exceeding the cap typically returns HTTP 400. Gitee AI emits
`"No schema matches, </input>"`.

### 3. No retry / backoff

LLM and embedding calls are one-shot; the underlying SDKs retry
transient network errors but not all 4xx/5xx. You'll see:

- MiniMax 529 overloaded (intermittent, no-op to re-run)
- Gemini 429 rate-limit (project-quota dependent)
- GLM 5xx occasionally

`dikw check` will print them as red cells; `dikw ingest` aborts on
first failure *but is idempotent via content hash*, so re-running
resumes without double-embedding unchanged docs. For production
automation, wrap the call in your own retry layer (e.g., `tenacity`).

### 4. Prompt caching only on the `anthropic` leg

The `AnthropicLLM` provider passes `cache_control: {"type": "ephemeral"}`
on the system prompt, cutting repeat-call input-token cost by ~90%
within a 5-minute TTL. Synth / query / distill all benefit.

**`openai_compat` does not expose prompt caching** — GLM / Gemini /
DeepSeek pay full price on every call even with a stable system
prompt. If you plan heavy synth work on an `openai_compat` vendor, the
cost model is different from the Anthropic leg.

### 5. `max_tokens` is hardcoded

Set in [`api.py`](../src/dikw_core/api.py):
`synth=2048, distill=2048, query=1024`. These are comfortable for all
tested vendors, but some cost-optimized models (a few GLM-Flash
variants, smaller Gemini Nano endpoints) cap responses below 2048 and
will return 400. If you hit this, the fix is a small patch making the
value configurable — not merged yet.

### 6. Two separate keys, on purpose

The embedding leg reads `DIKW_EMBEDDING_API_KEY` **exclusively** — no
silent fallback to `OPENAI_API_KEY`. When LLM and embedding point at
the same OpenAI-compat vendor, you still set two env vars to the same
value. This looks redundant but prevents cross-wiring when the two
legs diverge (the common case — MiniMax LLM + Gitee embeddings, or
Anthropic LLM + OpenAI embeddings).

### 7. CJK corpora need `cjk_tokenizer: jieba`

SQLite FTS5's default `unicode61` tokenizer splits CJK one character
at a time, which collapses BM25 on Chinese / Japanese into single-char
IDF. Measured on CMTEB T2Retrieval: **nDCG@10 = 0.031**, 91.7% of
queries zero-recall — vs ≈ 0.5–0.65 on the published Anserini+jieba
baselines. The default is now `jieba`, so a fresh wiki picks up the
right tokenizer with no config; pin it explicitly only if you need
the legacy whitespace path:

```yaml
retrieval:
  cjk_tokenizer: jieba       # default since 2026-04 (was "none")
```

…and install the optional extra:

```bash
uv sync --extra cjk          # pulls in jieba ≥ 0.42
```

The preprocessor runs `jieba.cut_for_search` over **CJK runs only** —
ASCII identifiers (``retrieval.rrf_k``, code snippets, …) are passed
verbatim, so mixed English/Chinese dev docs don't get their English
halves shredded.

**Locked at first ingest** — same shape as `embedding_dim`
(gotcha #1). The `documents_fts` rows store whatever segmentation was
in effect when they were written; flipping the config afterwards
produces a mismatch between indexed and queried tokens, silently
dropping CJK hits. To change: wipe `.dikw/index.sqlite` and `dikw
ingest` fresh.

## Public-benchmark calibration with Gitee AI

Reproducible workflow for running BEIR / CMTEB benchmarks against
dikw's hybrid retriever, using Gitee AI for embeddings (its free /
low-cost tier makes the 5K–60K passage runs financially trivial). The
benchmark datasets and the converter scripts live under `evals/`; the
runner is the same `dikw eval` you use on the dogfood mvp set.

Setup once:

```bash
uv run dikw init scratch-bench-wiki
cd scratch-bench-wiki
# Edit dikw.yml's provider block:
#   embedding: openai_compat
#   embedding_base_url: https://ai.gitee.com/v1
#   embedding_model: Qwen3-Embedding-8B
#   embedding_dim: 1024               # required: matryoshka truncation
#   embedding_revision: ""
#   embedding_normalize: true
#   embedding_distance: cosine
#   embedding_batch_size: 16          # gotcha #2 — Gitee caps at 25
# Then in .env: DIKW_EMBEDDING_API_KEY=<your gitee-ai key>
uv run --env-file .env dikw check --path . --embed-only
```

Materialise SciFact (BEIR English, 5K passages, ~1 minute on Gitee AI):

```bash
curl -L https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
    -o /tmp/scifact.zip
unzip /tmp/scifact.zip -d /tmp/scifact-src/
uv run python evals/tools/convert_beir.py \
    --source /tmp/scifact-src/scifact/ \
    --out evals/datasets/scifact/ \
    --baseline-bm25-ndcg10 0.665    # BEIR paper, Thakur et al. 2021
```

Run the full ablation and dump per-mode rankings for offline re-fusion:

```bash
uv run --env-file scratch-bench-wiki/.env \
    dikw eval --dataset scifact --embedder provider \
    --path ./scratch-bench-wiki --retrieval all \
    --dump-raw /tmp/scifact-raw.jsonl
```

Output is a 3-row × 5-metric table (bm25 / vector / hybrid × hit@3 /
hit@10 / mrr / nDCG@10 / recall@100). Compare the `bm25` row's
nDCG@10 to the published 0.665 baseline (treat ±0.10 as in-band —
FTS5 is not Anserini), and check whether `hybrid` actually beats both
single legs.

### Tuning RRF weights for your corpus

The shipped defaults are calibrated on BEIR/SciFact (vector-heavy:
`bm25_weight=0.3, vector_weight=1.5, rrf_k=60`). If your corpus has
different BM25 / dense balance — keyword-heavy code bases want more
BM25 influence, paraphrase-heavy prose wants more vector — tune via
the offline sweep, which re-fuses the same `--dump-raw` JSONL at
arbitrary weights in milliseconds:

```bash
uv run python evals/tools/sweep_rrf.py --raw-dump /tmp/scifact-raw.jsonl
```

The printed table shows the top-N (k, w_bm25, w_vec) combinations by
nDCG@10 plus two reference rows — vanilla `(1, 1, 60)` RRF and the
currently shipped defaults — so the absolute deltas are obvious. Pick
a winning row and pin it in the wiki's `dikw.yml`:

```yaml
retrieval:
  rrf_k: 60           # default; smaller = steeper rank decay
  bm25_weight: 0.5    # raise for keyword-heavy corpora
  vector_weight: 1.0  # raise for paraphrase / semantic match
```

No code change needed — `api.query` and the MCP `doc.search` tool pick
up the block on next call.

### Score-normalised fusion alternatives

`retrieval.fusion` selects the algorithm that combines BM25 + vector +
asset legs. Three options ship:

- `rrf` (default) — Reciprocal Rank Fusion. Rank-only, robust against
  heterogeneous score scales (BM25 unbounded vs cosine `[0, 2]`). The
  safe choice and what every existing baseline in `evals/BASELINES.md`
  was measured under.
- `combsum` — per-leg min-max normalises raw scores to `[0, 1]` then
  weighted-sums across legs. Preserves **magnitude**: a leg's clear
  leader keeps its margin where RRF would collapse it to `1/(k+1)`.
  Reach for it when one leg dominates (vector-strong corpora) or both
  legs are close at the head and rank-based fusion has nothing to
  discriminate (CMTEB-0.6B observation: hybrid `-0.003` vs vector
  nDCG@10 under RRF).
- `combmnz` — `CombSUM × (number of legs that retrieved each key)`.
  Boosts cross-leg consensus on top of CombSUM's magnitude
  preservation. Single-leg corpora collapse to plain CombSUM.

```yaml
retrieval:
  fusion: combsum     # default: rrf
  bm25_weight: 0.3    # weights still apply; per-leg cap on contribution
  vector_weight: 1.5
```

The same `bm25_weight` / `vector_weight` knobs cap the per-leg
contribution under all three modes (CombSUM/CombMNZ multiply the
normalised `[0, 1]` score by the weight; RRF multiplies the rank
reciprocal). `rrf_k` is RRF-only and is ignored under
`combsum` / `combmnz`. Today the offline sweep tool
(`evals/tools/sweep_rrf.py`) only re-fuses RRF — switching fusion mode
requires editing `dikw.yml` and replaying through
`evals/tools/run_phase15_from_snapshot.py` (no API spend, the
embeddings stay cached).

For a Chinese benchmark, repeat with `convert_cmteb.py` against a
HuggingFace download — same workflow, see
[`evals/README.md`](../evals/README.md#public-benchmarks) for the
full command. **Before running any CJK eval**, flip
`retrieval.cjk_tokenizer: jieba` in the scratch wiki's `dikw.yml`
(gotcha #7) — otherwise the BM25 row in the ablation table will
report 0.03 nDCG@10 regardless of fusion tuning, because FTS5's
default tokenizer doesn't segment Chinese.

## Pre-flight checklist for a new vendor

Before running `dikw ingest` against a real corpus with a new vendor
config:

- [ ] `embedding_dim` matches what the model actually returns.
      Run `dikw check --embed-only` and read `dim=…` from the output.
- [ ] `embedding_batch_size` is ≤ the vendor's observed cap.
- [ ] `dikw check --llm-only` and `dikw check --embed-only` each exit 0.
- [ ] If you're migrating from another vendor, `.dikw/index.sqlite` is
      deleted (see gotcha #1).
- [ ] Costs understood: if the LLM leg is `openai_compat`, you pay full
      input-token price on every synth / query — no prompt caching.

## See also

- [`README.md`](../README.md#providers) — quick config snippets.
- [`docs/getting-started.md`](./getting-started.md#pluggable-providers) —
  end-to-end walkthrough.
- [`docs/architecture.md`](./architecture.md) — where the provider
  seam sits in the module map.


## Multimodal embedding (v1: Gitee)

When `dikw.yml` declares an `assets.multimodal` block, the engine
routes both chunk text and image bytes through one
`MultimodalEmbeddingProvider` so they land in the same vector space.
Without the block, the legacy 2-leg text-embedding path is used
exactly as before — multimodal is opt-in.

### Wire format

Gitee's multimodal embeddings endpoint accepts **one shape across every
multimodal model it serves** (Qwen3-VL-Embedding-8B, jina-clip-v2, …):
a list of per-modality input dicts. The model name in
`assets.multimodal.model` discriminates which model runs server-side;
the wire payload does not change.

```http
POST /v1/embeddings
{
  "model": "Qwen3-VL-Embedding-8B",
  "input": [
    {"text": "a blue cat"},
    {"image": "data:image/png;base64,..."}
  ]
}
```

This is **distinct** from Gitee's text-only embeddings shape
(`input: "..."` or `input: ["...", "..."]`), which the legacy text
embedder uses on the `embedding_*` keys at the top of `dikw.yml`.

Each `MultimodalInput` becomes one wire input. Combined text+image on a
single `MultimodalInput` is rejected loudly (Gitee embeds per-modality,
no joint-encode mode), as is multiple images per input. The pipeline
never produces those shapes today (chunks are text-only, assets are
image-only), so the rejection just keeps a future config mistake from
silently dropping a modality.

### Cookbook: Qwen pair on Gitee (recommended)

Both legs on Gitee, same vendor and key. Empirical dims (probed
2026-04-25):

| Model | Native dim | `dimensions` knob | Default if unset |
|---|---|---|---|
| `Qwen3-Embedding-8B` (text-only) | 4096 | matryoshka 4096 / 1024 / 512 / 256 | **1024** |
| `Qwen3-VL-Embedding-8B` (multimodal) | **1024 fixed** | not accepted | 1024 |

```yaml
provider:
  llm: anthropic                       # or whichever LLM you've configured
  embedding: openai_compat             # text leg — single-string input shape
  embedding_base_url: https://ai.gitee.com/v1
  embedding_model: Qwen3-Embedding-8B
  embedding_dim: 1024                  # match Qwen3-VL dim so both legs of
                                       # hybrid retrieval live in equally-
                                       # priced spaces. Set to 4096 if you
                                       # want native quality on the text
                                       # leg and don't mind the dim split
                                       # (vec tables are independent).
                                       # WARNING: dim locks at first ingest
                                       # (gotcha #1), don't change later.
  embedding_revision: ""               # bump to force re-embed when Qwen
                                       # weights drift silently behind the
                                       # stable model name
  embedding_normalize: true
  embedding_distance: cosine
  embedding_batch_size: 16             # Gitee caps at 25 (gotcha #2)
  embedding_provider_label: gitee-ai

assets:
  dir: assets                          # relative to project root
  multimodal:
    provider: gitee_multimodal         # the only literal that ships today
    model: Qwen3-VL-Embedding-8B       # multimodal leg — per-modality dicts
    dim: 1024                          # fixed by the model; Gitee returns
                                       # 4096-byte float32 blobs = 1024 dim.
                                       # Setting any other value here makes
                                       # vec_assets_v<n> dim-mismatch on
                                       # first asset embed.
    revision: ""                       # bump (e.g., "2026-04") if Gitee
                                       # silently refreshes the weights
    normalize: true
    distance: cosine
    batch: 16                          # per-request input cap
    base_url: null                     # null = use the gitee_multimodal default
```

The text leg and the multimodal leg write **separate** vec tables
(`vec_chunks_v<n>` / `vec_assets_v<n>`, one per `embed_versions` row);
their dims do not have to match. Both legs read
`DIKW_EMBEDDING_API_KEY` — never `OPENAI_API_KEY`. If LLM and
embedding target different vendors, set them as distinct env vars.

**Note on chunk routing**: text and multimodal are strictly separate
channels. Chunk text always flows through the **text** embedder
(`provider.embedding_model`) into `vec_chunks_v<text_version_id>`,
even when `assets.multimodal` is configured. The multimodal embedder
embeds **assets only** (image bytes) into
`vec_assets_v<mm_version_id>`. Cross-modal retrieval works because
`info/search.HybridSearcher` runs both legs and asset-vec hits
promote the chunks that reference matching images via
`chunk_asset_refs` — the two vector spaces don't need to coincide.

### Verifying the config end-to-end

`dikw check --embed-only` automatically routes through the multimodal
embedder when `assets.multimodal` is present, sending one text + one
image input in **a single batched request** (no RTT stacking). Both
modalities probe the same endpoint Gitee will see at ingest time, so a
green check means real ingest will work:

```bash
$ uv run --env-file .env dikw check --path . --embed-only
Embedding | (provider default) | OK | 4234ms, dim=1024, modalities=text+image, provider=gitee-ai
```

(Latency dominated by the ~50ms TLS handshake forced per request to
work around Gitee's idle-keepalive drops — see fdd2cae for the
rationale. Plus a few seconds of Gitee server time on the multimodal
endpoint.)

If `assets.multimodal` is absent, the check falls back to the text-only
probe (one `"ping"` string) — same as before.

### What the multimodal pipeline buys you

- Image binaries referenced from your markdown (`![alt](path)` or
  `![[file]]`) are materialized into `assets/<h2>/<h8>-<name>.<ext>`
  inside your project root, visible in Obsidian.
- Each binary gets a vector via the multimodal model and lives in a
  per-version `vec_assets_v<id>` table (so dim/model changes don't
  collide with prior data).
- Hybrid search adds a third RRF leg over asset vectors that promotes
  chunks via the `chunk_asset_refs` reverse lookup — text queries
  retrieve chunks based on the images they reference even when the
  surrounding prose doesn't match the query.

### Switching the multimodal model

Change `model` (and `dim` if it differs) in `dikw.yml` and re-run
`dikw ingest`. The engine sees a new identity tuple, mints a new
version row, creates a fresh `vec_assets_v<new_id>` table, and writes
to it. The previous version's data stays in `vec_assets_v<old_id>`
until you run `dikw embed reindex` (v1 ships a stub; v1.5 implements
the real migration).

### Trying another multimodal provider

Drop a new file under `src/dikw_core/providers/` that implements the
`MultimodalEmbeddingProvider` Protocol (one method: `async embed(inputs:
list[MultimodalInput], *, model: str) -> list[list[float]]`), then
add a branch to `build_multimodal_embedder` in
`providers/__init__.py`. The Protocol signature accepts arbitrary
combinations of text and image inputs already — Voyage v3, Cohere v4,
Jina-direct, and self-hosted Nomic Embed Vision all slot in without
engine changes. (Their wire shapes diverge from Gitee's per-modality
dicts, which is why each vendor gets its own provider rather than
sharing one serializer.)
