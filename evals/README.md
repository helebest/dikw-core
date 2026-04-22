# evals

Retrieval-quality evaluation for `dikw-core`. Datasets live here as a
filesystem convention; `dikw eval` runs any of them.

## The three-file contract

Any subdirectory of `evals/datasets/` with these three things is a valid
dataset:

```
evals/datasets/<name>/
├── dataset.yaml       # name, description, thresholds
├── corpus/            # *.md or *.html — the docs to ingest
└── queries.yaml       # {q, expect_any: [doc_stem, …]} pairs
```

Missing any piece → `dikw eval` prints a specific error and exits 2.

### `dataset.yaml`

```yaml
name: mvp
description: short human-readable summary
thresholds:
  hit_at_3: 0.80        # optional; runner only asserts on metrics you list
  hit_at_10: 0.80
  mrr: 0.60
```

Supported metric keys: `hit_at_3`, `hit_at_10`, `mrr`. Unknown keys reject
the dataset. Omitting `thresholds:` entirely is fine — the runner still
computes metrics but `report.passed` is always True (useful for
exploratory datasets you haven't calibrated yet).

### `corpus/`

Any `*.md` or `*.html` under this directory is ingested; subdirectories
are fine. File stem (filename without extension) becomes the doc's
identity in `expect_any`.

### `queries.yaml`

```yaml
queries:
  - q: "What does Karpathy mean by deterministic scoping?"
    expect_any: [karpathy-gist, design]   # hit if any stem is in top-k
  - q: "How do I init a wiki?"
    expect_any: [getting-started, README]
```

Semantics: a query is a "hit at k" if *any* stem in `expect_any` appears
in the top-k retrieval result. Paraphrases often live in multiple docs,
and requiring *all* stems would be artificially punitive.

## Usage

```bash
uv run dikw eval                         # run every packaged dataset
uv run dikw eval --dataset mvp           # run by name (under evals/datasets/)
uv run dikw eval --dataset ./my-corpus   # run an arbitrary directory
```

Exit codes: `0` all passed, `1` any threshold failed, `2` bad spec /
dataset not found / no datasets to run.

### Real-vector mode

Default is hermetic — `FakeEmbeddings` (deterministic bag-of-words, 64
dim, <1s) with no network or API keys. For real-vector evaluation
against a configured provider, point at a wiki:

```bash
uv run --env-file .env dikw eval --dataset mvp --embedder provider --path ./my-wiki
```

The runner reads `provider.embedding_*` from the wiki's `dikw.yml` and
uses `build_embedder(cfg.provider)`. Gotchas around batch size,
dimensions, and dim-locking apply — see [`../docs/providers.md`](../docs/providers.md).

## Adding a dataset

Three steps:

1. `mkdir -p evals/datasets/<name>/{corpus,}`
2. Drop markdown files into `corpus/`; write `queries.yaml` with 5-20
   Q/A pairs (4 exact-fact + 3 paraphrase + 3 synthesis is a reasonable
   mix for retrieval).
3. First run: `uv run dikw eval --dataset <name>` — skip thresholds in
   `dataset.yaml` or leave them loose. Observe the numbers, then edit
   `dataset.yaml` to leave ~1-2 queries of slack below the observed
   values so corpus tweaks don't flake the gate while a real regression
   still fails it.

## Public benchmarks (future)

Not yet wired up; BEIR / MS-MARCO / TriviaQA come in their own formats
(JSONL + qrels). The plan is a one-shot converter (e.g.,
`tools/convert_beir.py`) that reads the public bundle and emits our
three-file shape into `evals/datasets/<name>/`. Once the directory
exists, `dikw eval` finds it automatically — no runtime format plugin.

## See also

- [`docs/eval-plan.md`](../docs/eval-plan.md) — why retrieval-only Phase
  A, when to revisit LLM-as-judge.
- [`docs/providers.md`](../docs/providers.md) — per-vendor config for
  `--embedder provider` mode.
- `src/dikw_core/eval/` — the runner, metrics, and dataset loader.
