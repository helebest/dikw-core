# Synth Quality Eval — Design Spec

Date: 2026-05-12
Status: Draft (brainstorming complete, awaiting user review)
Related: `docs/eval-plan.md`, `evals/BASELINES.md`, `src/dikw_core/eval/`

## Problem

`dikw-core` 的产品定位最近收窄到「知识库内核」——`query` 已经从 dikw-core 删除,由外层 agent 调用底层原语。内核的产出质量 = K 层 (`synth`) 和 W 层 (`distill`) 的输出质量。

但今天 synth/distill 的质量回路是缺失的:

- Retrieval (I 层) 有 Phase A 自动化:`dikw eval` + `tests/test_retrieval_quality.py` 把每个 PR 的 retrieval 退步挡在 CI。
- K 层 (synth) 只有 `elon-musk.md` 手工 baseline + 5+5 抽样规则,写在 `BASELINES.md`,作者每次改 `synthesize.md` prompt 都要手跑 + 手判。
- W 层 (distill) `BASELINES.md` 直接没条目。

结果是改一次 synth prompt 不知道是不是变好了,distill 几乎没人动也没法量化它好坏。"内核到极致"的前提是先把质量信号量化出来——本 spec 设计的就是这套基础设施。

## Goals

1. 把 K 层质量回路做到和 retrieval Phase A 同等水平:CI 必过的自动化 hard gate + 可手动跑的深度评测 (LLM judge soft score)。
2. dataset 契约、runner、CLI、report DTO 在设计上为 W 层预留;W 层激活时只加文件、不重做框架。
3. 沿用 `Karpathy's rule`:确定性指标承担 hard gate (PR 必过门),概率性 LLM judge 承担 soft score (作者手跑、写 BASELINES.md)。

## Non-goals

- 不在本 spec 内做 W 层激活——只做接口预留。
- 不替换 retrieval Phase A——和现有 `dikw eval <dataset>` 共存,共享 dataset 目录契约。
- 不引入新的 secret 或 LLM provider——judge 复用 `dikw.yml` 已配置的 LLM。

## High-level shape

复用 retrieval Phase A 的三文件契约 (`dataset.yaml` + `corpus/` + `queries.yaml` + `targets.yaml`),在同一 dataset 上**向后兼容地新增一个 mode**:

- `modes: [retrieval, synth]` — 声明这个 dataset 支持哪些评测模式;不写默认 `[retrieval]`,现有 retrieval-only dataset 一字不改
- 新增可选文件 `expected.yaml` — synth mode 的 reference (期望产出的 page 标题/关键词)
- `thresholds:` 沿用现有 `<view>/<metric>` namespace,新增 `synth/<metric>` 系列

CLI:

```bash
dikw eval mvp                          # 跑 dataset 声明的所有 mode
dikw eval mvp --mode synth             # 只跑 K 层
dikw eval mvp --mode synth --judge     # K 层 hard gate + LLM judge soft score
dikw eval mvp --mode synth --judge-sample N   # judge 抽样
dikw eval mvp --mode retrieval         # 当前 dikw eval mvp 的行为
```

## Code layout

不开新子目录,在 `src/dikw_core/eval/` 下扩展:

```
src/dikw_core/eval/
├── dataset.py        # 改: SUPPORTED_METRICS 扩 synth/*; DatasetSpec 加 expected 字段; SUPPORTED_VIEWS 加 "synth"
├── metrics.py        # 改: 增 K 层确定性指标函数 (6 个 hard + 1 个 informational)
├── runner.py         # 改: 增 run_synth_eval(); 现有 retrieval 入口不动
├── judge.py          # 新: LLM judge soft score (唯一新文件)
├── fake_embedder.py  # 不动
└── __init__.py
```

新 prompt:`src/dikw_core/prompts/eval_judge_synth.md`

测试:

```
tests/test_synth_quality.py    # 新: hard gate, hermetic, FakeLLM + FakeEmbeddings
tests/test_eval_metrics.py     # 新: unit, 每个 metric 函数 + edge case
tests/test_eval_judge.py       # 新: unit, judge 响应解析 + JudgeSummary 聚合
tests/test_retrieval_quality.py  # 不动
```

CLI 入口 (`src/dikw_core/cli.py` 或 `src/dikw_core/client/`,实现期决定) 增加 `--mode` / `--judge` / `--judge-sample` flag。

## Determinitic hard-gate metrics (K layer)

进 PR 必过门的 6 个指标 + 1 个 informational:

| Metric | 公式 | 抓什么 | 建议阈值 |
|---|---|---|---|
| `fact_grounding_ratio` | 每 page body 切 claim 句 (按 `.`/`。` + 去 wikilink/heading) → 对该 source 所有 chunk 嵌入,取最近邻 cosine ≥ τ_grounding 视为 grounded → 所有 page 的均值 | LLM 凭空编 (能抓"远离源"的胡编;曲解需要 judge) | ≥ 0.80 |
| `atomicity_score` | `1 − (non_atomic_pages / total_pages)`,直接复用现有 `non_atomic_page` lint | page 把多个主题塞一起 | ≥ 0.90 |
| `duplicate_ratio_max` | 所有 page pair 之间 page-embedding cosine ≥ τ_dup 的比例 | existing-pages awareness 失灵 / 同概念多次生成 | ≤ 0.05 |
| `wikilink_resolved_ratio` | `resolved_links / total_wikilinks`,从 `SynthReport.unresolved_wikilinks` 取 | fuzzy resolve 退化 / 大量断链 | ≥ 0.85 |
| `expected_coverage` | 每个 source 的 `expected.yaml.expected_titles` 中被生成 page 命中的比例;命中规则复用 `resolve_links` 的 NFKC + casefold + ASCII/CJK 标点剥离 + 复数 stem normalize | LLM 漏关键概念 | ≥ 0.80 |
| `language_fidelity` | 每 page body 前 200 字符语言检测 → 与 source 主语言比对 → 匹配比例 | source-language preservation 退化 | ≥ 0.95 |
| `page_density` (informational) | `pages / chunks` | LLM 发散/惜墨 — 不进 gate,只入 report | — |

参数(在 `dataset.yaml.synth:` 下):

- `grounding_threshold: 0.65` (τ_grounding, embedding cosine)
- `duplicate_threshold: 0.85` (τ_dup, embedding cosine)
- `page_types: [entity, concept, note]` (与 `SchemaConfig.page_types` 对齐)

反向阈值的命名约定:metric 名带 `_max` 后缀表示「上限,值越低越好」;不加后缀默认是下限。`_parse_thresholds` 不需要改,`check_thresholds` 根据后缀切方向。

`SUPPORTED_METRICS` 扩展:加 `fact_grounding_ratio`, `atomicity_score`, `duplicate_ratio_max`, `wikilink_resolved_ratio`, `expected_coverage`, `language_fidelity`, `page_density`。`SUPPORTED_VIEWS` 加 `synth`。

## LLM judge (soft score)

不进 gate,作者手动跑,数字贴进 `BASELINES.md`。每个 page 一次 LLM 调用,4 维度 0-5 整数评分:

| 维度 | 评分 prompt 大意 | 与 hard gate 互补点 |
|---|---|---|
| `judge_grounding` | "page 的 claim 是否都能从给出的 source chunk 里推出?" | 抓「embedding 命中但其实曲解了」 |
| `judge_atomicity` | "page 是否聚焦单一主题/实体/概念?" | 抓 lint 没抓到的细微跑题 |
| `judge_completeness` | "对 source 关于该主题的核心信息,page 是否覆盖完整?" | hard gate 完全没这维度 |
| `judge_clarity` | "新读者(没读 source)看 page 能否理解?" | 写作质量,hard gate 完全抓不到 |

聚合形态:

```python
class JudgeScore(BaseModel):
    grounding: int       # 0-5
    atomicity: int
    completeness: int
    clarity: int
    rationale: str       # 一句话原因, 调试用

class JudgeSummary(BaseModel):
    n_judged: int
    n_errors: int
    mean_grounding: float
    mean_atomicity: float
    mean_completeness: float
    mean_clarity: float
    per_page: list[tuple[str, JudgeScore]]  # (page_path, score)

async def judge_synthesis(
    pages: Sequence[WikiPage],
    *,
    sources: Mapping[str, str],   # path -> raw source text
    llm: LLMProvider,
    model: str,
    sample: int | None = None,
) -> JudgeSummary: ...
```

策略:

- **跑哪些 page**:默认全跑;`--judge-sample N` 随机抽 N 个
- **用哪个 LLM**:默认复用 `dikw.yml` 的 LLM provider;`dataset.yaml.judge.{provider, model}` 可 pin 覆盖
- **单 page 失败**:计入 `n_errors`,继续跑其他 page,不 fail-fast
- **prompt**:`src/dikw_core/prompts/eval_judge_synth.md`,strict JSON 输出 + schema validation

## Dataset contract extension

`dataset.yaml` 完整 schema(向后兼容,所有新增字段都可选):

```yaml
name: mvp
description: ...

modes: [retrieval, synth]               # 新; 缺省 [retrieval]

thresholds:
  # retrieval (现状, 不动)
  hit_at_3: 0.80
  mrr: 0.60
  # synth (新)
  synth/fact_grounding_ratio: 0.80
  synth/atomicity_score: 0.90
  synth/duplicate_ratio_max: 0.05
  synth/wikilink_resolved_ratio: 0.85
  synth/expected_coverage: 0.80
  synth/language_fidelity: 0.95
  # synth/page_density 不设阈值 → informational

synth:                                  # 新; mode=synth 参数
  grounding_threshold: 0.65
  duplicate_threshold: 0.85
  page_types: [entity, concept, note]

judge:                                  # 新; --judge 用
  provider: null                        # null → 沿用 dikw.yml
  model: null
```

`expected.yaml` (synth mode 可选;缺失则 `expected_coverage` 跳过):

```yaml
sources:
  - path: karpathy-software-2-0.md
    expected_titles:                    # 进 expected_coverage 计算
      - "Software 2.0"
      - "Neural Networks"
      - "Gradient Descent"
    expected_keywords:                  # informational, 任意 page body 命中即可
      - "differentiable"
```

## Data flow

```
load_dataset(mvp)
  → DatasetSpec(modes=[retrieval, synth], expected=KWExpected(...) or None)

run_synth_eval(spec, llm, embedder, progress):
  for source in spec.corpus_dir:
      progress.emit("source_start", path=source.path)
      ingest → chunks → synth → pages    # 走真实 api.ingest 路径
      progress.emit("source_done", path=source.path, n_pages=...)
  metrics = compute_synth_metrics(all_pages, all_sources, expected, cfg)
  threshold_results = check_thresholds(metrics, spec.thresholds)
  return SynthEvalReport(metrics, threshold_results, ...)

# 可选 soft layer (--judge)
if judge_enabled:
    summary = await judge_synthesis(pages, sources, llm, model, sample)
    report.judge_summary = summary
```

`run_synth_eval` 是 `async def`,CLI 走 `asyncio.run`,与 `dikw.api` 一致。

## Report DTO

```python
class SynthEvalReport(BaseModel):
    dataset: str
    mode: Literal["synth"]
    n_sources: int
    n_pages: int
    metrics: dict[str, float]                   # {"fact_grounding_ratio": 0.84, ...}
    threshold_results: list[ThresholdResult]    # name / observed / threshold / direction / passed
    pages_per_source: dict[str, int]
    informational: dict[str, float]             # page_density 等
    judge_summary: JudgeSummary | None = None
    warnings: list[str]
```

输出沿用现状:默认 NDJSON (agent-friendly, 每行 `mode:` 字段做区分),`--pretty` 走 rich table。

`RetrievalReport` 和 `SynthEvalReport` 独立两份 DTO,各自字段不一样,多 mode 跑时输出多条 NDJSON 行。

## Error handling

| 情况 | 行为 |
|---|---|
| `expected.yaml` 缺失 | `expected_coverage` 不出现在 `metrics` 里;runner 在 `report.warnings` 加一条 |
| `expected.yaml` 引用不存在的 source path | `load_dataset` **loud fail**,仿现有 `_validate_query_targets` 的 typo-catching 精神 |
| `pages = 0` (LLM 罢工 / 全部 parse 错) | runner raise `SynthEvalError` — 0 pages 不是 0 重复 |
| `pages = 1` | `duplicate_ratio_max = 0.0` (无 pair),OK |
| FakeLLM 下指标数字不真实 | 阈值在 hermetic 测试模式下**不生效** — 测试只验「指标算得出来 + 数值在合法 range」;真实阈值留给作者用真实 LLM 跑 baseline 校准 |
| 单 page judge 返回的 JSON schema 错 | 计入 `JudgeSummary.n_errors`,其余 page 继续 |
| `--mode synth` 在 dataset 未声明 synth | CLI raise with hint:"dataset.yaml 没有 `synth` in modes,要支持需要加 `expected.yaml`+ `synth/*` thresholds" |

## Testing strategy

### Hermetic gate test

```python
# tests/test_synth_quality.py
async def test_synth_quality_mvp() -> None:
    spec = load_dataset("mvp")
    llm = FakeLLM(responses=load_fixture("synth_responses/mvp_*.txt"))
    embedder = FakeEmbeddings(dim=64)
    report = await run_synth_eval(spec, llm=llm, embedder=embedder, ...)
    # 不卡阈值, 只验存在性 + range + 类型
    for m in ("fact_grounding_ratio", "atomicity_score", "duplicate_ratio_max",
              "wikilink_resolved_ratio", "expected_coverage", "language_fidelity"):
        assert m in report.metrics
        assert 0.0 <= report.metrics[m] <= 1.0
    assert report.n_pages > 0
```

Fixture 形态:`tests/fixtures/synth_responses/mvp_<source_stem>.txt`,每条是从真实 LLM run 里 capture 的 `<page>` 块输出,提交进 repo,record-replay。

### Unit tests

`tests/test_eval_metrics.py`:每个 metric 函数 (`fact_grounding_ratio`, `atomicity_score`, `duplicate_ratio_max`, `wikilink_resolved_ratio`, `expected_coverage`, `language_fidelity`, `page_density`) 独立 unit test。用合成 `WikiPage` + 合成 source text。覆盖:

- 正常情况
- `pages = 0` / `pages = 1` 边界
- `expected.yaml` 缺失 → `expected_coverage` 不算
- reverse 阈值 (`_max` 后缀) 方向正确
- embedding 阈值 τ_grounding / τ_dup 在边界附近

`tests/test_eval_judge.py`:

- `parse_judge_response`:malformed JSON、缺字段、越界整数 (>5 / <0) 各种情况
- FakeLLM 模拟 judge LLM 输出,验 `JudgeSummary` 聚合正确(均值、`n_errors` 计数、`per_page` 顺序)
- `--judge-sample` 抽样行为(N > n_pages、N == 0)

### Real-LLM baseline

不在 CI 里跑。作者改了 `synthesize.md` / `domains/knowledge/` 后,本地手跑:

```bash
dikw eval mvp --mode synth --judge --pretty
```

把数字贴进 `evals/BASELINES.md` 的 synth 章节(新增)。CLAUDE.md 里 `Acceptance gates for K-layer and Retrieval changes` 那段同步更新:K 层改动现在多一个量化数据点要求。

### Observability

`run_synth_eval` 通过 `ProgressReporter` Protocol emit 事件:`source_start`, `source_done`, `synth_calling`, `synth_returned`, `judge_calling`, `judge_returned`, `judge_error`。沿用 memory `feedback_synth_observability.md` 的纪律——慢内循环必须每次迭代发事件,否则 UI 看不出死锁 vs 进行中。

## W 层预留(本 spec 不实装,只占座)

W 层激活时,框架不变,只加文件 / metric:

- `modes:` 列表加 `distill`
- 新文件 `evals/datasets/<name>/expected_wisdom.yaml`(schema 类似 `expected.yaml`,但条目是期望的 wisdom 标题 + kind)
- `metrics.py` 增 W 系列确定性指标:`evidence_in_page_ratio` (distill 给出的 evidence excerpt 是否真在 page 里)、`wisdom_dup_ratio` (与已有 wisdom 的嵌入相似度上限)、`kind_distribution`,等等
- `runner.py` 增 `run_distill_eval`,可独立调用,也可被 `run_synth_eval` 在 dataset 声明了 distill mode 时链式调用
- `judge.py` 增 `judge_wisdom`,4 维度评分:`grounding` / `clarity` / `transferability` / `novelty`
- `dataset.yaml` 增可选 `distill:` section,与 `synth:` 平行
- `SUPPORTED_METRICS` 扩 W 系列;`SUPPORTED_VIEWS` 加 `distill`

## Open questions

(brainstorming 阶段已收敛,留作实现期复核)

- `dataset.yaml.synth.page_types` 与 `dikw.yml.schema.page_types` 是否要强制一致?当前提议是 dataset 内独立(eval 跑出来的 page 不进真实 wiki,允许评测 dataset 用更严格的子集)。
- FakeLLM fixture 怎么生成?手动 capture 一次,提交进 repo,后续 LLM prompt 改了再 regenerate。需要在 `tests/fixtures/synth_responses/README.md` 写明 regenerate 流程。
- `language_fidelity` 用什么语言检测?最轻:看 chunk 里 ASCII 占比阈值切英文/CJK;重一点用 `langdetect` 但要加依赖。倾向轻方案,精度够用。

## Implementation notes for writing-plans

实现拆分建议(实际 plan skill 接管):

1. **dataset.py 扩展** — `SUPPORTED_METRICS` / `SUPPORTED_VIEWS` / `DatasetSpec.expected` / `expected.yaml` loader
2. **metrics.py 扩展** — 6 个 K 层确定性指标 + page_density,各自 unit test
3. **runner.py 扩展** — `run_synth_eval` + `SynthEvalReport` + `check_thresholds` 方向感知
4. **judge.py 新文件** + prompt + `JudgeSummary` + `judge_synthesis` async + unit test
5. **CLI 集成** — `--mode` / `--judge` / `--judge-sample` flag
6. **测试 gate** — `test_synth_quality.py` + fixture capture
7. **dogfood** — 升级 `evals/datasets/mvp/dataset.yaml` 加 synth thresholds + 新建 `expected.yaml`;真 LLM 跑一遍校准阈值;BASELINES.md 加 synth 章节;CLAUDE.md 同步 K 层 acceptance gate 段
