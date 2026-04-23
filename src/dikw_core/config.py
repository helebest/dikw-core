"""Configuration loader for `dikw.yml`.

The config mirrors the top-level sections in the design doc: `provider`, `storage`,
`schema`, `sources`. Storage-specific fields live under a single `storage` block
and are validated per backend via a discriminated union.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ProviderConfig(BaseModel):
    llm: Literal["anthropic", "openai_compat"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    embedding: Literal["openai_compat"] = "openai_compat"
    embedding_model: str = "text-embedding-3-small"
    # The OpenAI-compat base URL is used for BOTH `openai_compat` LLM calls and
    # for embeddings when the LLM provider is Anthropic (which has no embeddings API).
    embedding_base_url: str = "https://api.openai.com/v1"
    # Optional embedding output dimension (matryoshka truncation). ``None`` keeps
    # the provider's native default (e.g., 1536 for text-embedding-3-small);
    # set it explicitly for models that support trimming (e.g.,
    # Qwen3-Embedding-8B on Gitee AI).
    embedding_dimensions: int | None = None
    # Max texts per ``/v1/embeddings`` request. OpenAI accepts ~2048;
    # Gitee AI caps at ~25. Keep the default safe for OpenAI and drop it
    # via config when hitting a stricter backend.
    embedding_batch_size: int = 64
    # Optional free-form display label surfaced in ``dikw check`` output
    # (e.g., "gitee-ai", "openai", "azure-east"). Describes which vendor
    # the embedding endpoint points at; purely for human diagnostics.
    embedding_provider_label: str | None = None
    # Used by both "anthropic" and "openai_compat" LLM providers. For anthropic,
    # points the Anthropic SDK at an Anthropic-protocol-compatible endpoint
    # (e.g., MiniMax's Anthropic surface). Leave null to use the provider default.
    llm_base_url: str | None = None
    # Per-operation response budget handed to ``LLMProvider.complete`` via
    # ``max_tokens``. Defaults match the values previously hardcoded in
    # ``api.py``; shrink for cost-optimised models (some GLM-Flash / Gemini
    # Nano variants cap below 2048), grow if synth/distill responses get
    # truncated.
    llm_max_tokens_query: int = 1024
    llm_max_tokens_synth: int = 2048
    llm_max_tokens_distill: int = 2048
    # Per-leg SDK retry budget. Anthropic and OpenAI SDKs retry 408/409/429/5xx
    # (incl. MiniMax 529) with exponential backoff + jitter; their default is
    # 2. We bump to 5 by default to absorb intermittent overload without
    # pulling in a third-party retry layer. Split per-leg because LLM and
    # embedding frequently target different vendors with different failure
    # profiles (e.g., MiniMax LLM + Gitee AI embeddings).
    llm_max_retries: int = 5
    embedding_max_retries: int = 5


class RetrievalConfig(BaseModel):
    """Fusion knobs for ``HybridSearcher``.

    Defaults match the pre-2026-04 behaviour (equal-weight RRF, k=60) so
    a wiki whose ``dikw.yml`` omits the ``retrieval:`` block sees zero
    change. Tune these per-corpus when one leg is systematically stronger
    than the other — SciFact/BEIR benefits from a vector-heavier setting
    because dense retrieval beats the BM25 leg by ~0.10 nDCG@10 there.
    See ``docs/eval-plan.md`` + ``evals/BASELINES.md`` for how to measure.
    """

    # Reciprocal Rank Fusion's rank-offset constant. Smaller = steeper
    # decay (rank-1 wins by more). 60 is the value used in the original
    # RRF paper and by both reference projects.
    rrf_k: int = 60
    # Per-leg contribution factor. 1.0 each = vanilla RRF. Lowering one
    # leg's weight keeps its recall contribution (docs it alone found
    # still enter the fused pool) while shrinking its rank influence.
    bm25_weight: float = 1.0
    vector_weight: float = 1.0


class SQLiteStorageConfig(BaseModel):
    backend: Literal["sqlite"] = "sqlite"
    path: str = ".dikw/index.sqlite"


class PostgresStorageConfig(BaseModel):
    backend: Literal["postgres"] = "postgres"
    dsn: str
    schema_: str = Field(default="dikw", alias="schema")
    pool_size: int = 10

    model_config = {"populate_by_name": True}


class FilesystemStorageConfig(BaseModel):
    backend: Literal["filesystem"] = "filesystem"
    root: str = ".dikw/fs"
    embed: bool = False
    max_pages_hint: int = 300


StorageConfig = Annotated[
    SQLiteStorageConfig | PostgresStorageConfig | FilesystemStorageConfig,
    Field(discriminator="backend"),
]


class SchemaConfig(BaseModel):
    description: str = ""
    page_types: list[str] = Field(default_factory=lambda: ["entity", "concept", "note"])
    wisdom_kinds: list[str] = Field(
        default_factory=lambda: ["principle", "lesson", "pattern"]
    )
    log_style: Literal["append", "daily"] = "append"


class SourceConfig(BaseModel):
    path: str
    pattern: str = "**/*.md"
    ignore: list[str] = Field(default_factory=list)


class DikwConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    storage: StorageConfig = Field(default_factory=SQLiteStorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    sources: list[SourceConfig] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @field_validator("sources")
    @classmethod
    def _require_at_least_one_source_path(cls, v: list[SourceConfig]) -> list[SourceConfig]:
        # allow an empty list at init time (newly scaffolded wiki); engine-level
        # operations that need sources can validate at call time.
        return v


CONFIG_FILENAME = "dikw.yml"


def load_config(path: str | Path) -> DikwConfig:
    """Load and validate a `dikw.yml` file."""
    p = Path(path)
    if p.is_dir():
        p = p / CONFIG_FILENAME
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top-level YAML must be a mapping, got {type(raw).__name__}")
    return DikwConfig.model_validate(raw)


def find_config(start: str | Path) -> Path | None:
    """Walk up from `start` looking for `dikw.yml`. Returns None if not found."""
    p = Path(start).resolve()
    for candidate in (p, *p.parents):
        cfg = candidate / CONFIG_FILENAME
        if cfg.is_file():
            return cfg
    return None


def default_config(description: str = "A dikw-core knowledge base") -> DikwConfig:
    """Return a DikwConfig populated with sensible defaults for `dikw init`.

    Two source entries ship out of the box — one per built-in backend — so a
    fresh wiki picks up both markdown and HTML sources without extra config.
    """
    return DikwConfig(
        provider=ProviderConfig(),
        storage=SQLiteStorageConfig(),
        schema=SchemaConfig(description=description),
        sources=[
            SourceConfig(path="./sources", pattern="**/*.md"),
            SourceConfig(path="./sources", pattern="**/*.html"),
        ],
    )


def dump_config_yaml(cfg: DikwConfig) -> str:
    """Render a DikwConfig as a YAML string suitable for `dikw.yml`."""
    data = cfg.model_dump(mode="json", by_alias=True, exclude_defaults=False)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
