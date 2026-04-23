from __future__ import annotations

from pathlib import Path

import pytest

from dikw_core.config import (
    CONFIG_FILENAME,
    DikwConfig,
    FilesystemStorageConfig,
    PostgresStorageConfig,
    ProviderConfig,
    RetrievalConfig,
    SQLiteStorageConfig,
    default_config,
    dump_config_yaml,
    find_config,
    load_config,
)


def test_default_config_roundtrip(tmp_path: Path) -> None:
    cfg = default_config(description="unit-test wiki")
    yaml_text = dump_config_yaml(cfg)
    path = tmp_path / CONFIG_FILENAME
    path.write_text(yaml_text, encoding="utf-8")

    loaded = load_config(path)
    assert isinstance(loaded, DikwConfig)
    assert loaded.schema_.description == "unit-test wiki"
    assert isinstance(loaded.storage, SQLiteStorageConfig)
    assert loaded.storage.backend == "sqlite"


def test_load_config_discriminated_storage(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  llm: anthropic
  llm_model: claude-sonnet-4-6
  embedding: openai_compat
  embedding_model: text-embedding-3-small
  embedding_base_url: https://example.invalid/v1
storage:
  backend: postgres
  dsn: postgresql://u:p@h:5432/db
  schema: dikw
  pool_size: 4
schema:
  description: pg wiki
sources:
  - path: ./sources
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert isinstance(cfg.storage, PostgresStorageConfig)
    assert cfg.storage.dsn.startswith("postgresql://")
    assert cfg.storage.schema_ == "dikw"


def test_load_config_filesystem_storage(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
storage:
  backend: filesystem
  root: .dikw/fs
  embed: false
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert isinstance(cfg.storage, FilesystemStorageConfig)
    assert cfg.storage.embed is False


def test_find_config_walks_up(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (root / CONFIG_FILENAME).write_text(dump_config_yaml(default_config()), encoding="utf-8")

    found = find_config(nested)
    assert found is not None
    assert found.parent == root


def test_find_config_returns_none_when_missing(tmp_path: Path) -> None:
    assert find_config(tmp_path) is None


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / CONFIG_FILENAME
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_provider_config_llm_max_tokens_defaults() -> None:
    """Per-op max_tokens fields default to the values currently hardcoded in api.py."""
    cfg = ProviderConfig()
    assert cfg.llm_max_tokens_query == 1024
    assert cfg.llm_max_tokens_synth == 2048
    assert cfg.llm_max_tokens_distill == 2048


def test_provider_config_llm_max_tokens_override_via_yaml(tmp_path: Path) -> None:
    """Users can shrink (or grow) per-op budgets via dikw.yml to fit their vendor."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  llm_max_tokens_query: 512
  llm_max_tokens_synth: 4096
  llm_max_tokens_distill: 1536
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.provider.llm_max_tokens_query == 512
    assert cfg.provider.llm_max_tokens_synth == 4096
    assert cfg.provider.llm_max_tokens_distill == 1536


def test_provider_config_max_retries_defaults() -> None:
    """Both legs default to 5 retries — above the SDK default of 2 to give
    MiniMax 529 / Gemini 429 class errors a bit more breathing room.
    """
    cfg = ProviderConfig()
    assert cfg.llm_max_retries == 5
    assert cfg.embedding_max_retries == 5


def test_provider_config_max_retries_round_trip(tmp_path: Path) -> None:
    """Retry budgets are independently tunable per leg via dikw.yml."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  llm_max_retries: 3
  embedding_max_retries: 7
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.provider.llm_max_retries == 3
    assert cfg.provider.embedding_max_retries == 7


def test_retrieval_config_defaults_match_legacy_behavior() -> None:
    """Default RetrievalConfig = pre-weighting behaviour (k=60, weights=1.0).

    Regression guard: a silent default change would shift every hybrid
    ranking across every user wiki.
    """
    cfg = RetrievalConfig()
    assert cfg.rrf_k == 60
    assert cfg.bm25_weight == 1.0
    assert cfg.vector_weight == 1.0


def test_dikw_config_retrieval_block_omitted_fills_defaults(tmp_path: Path) -> None:
    """A wiki whose dikw.yml predates this feature loads cleanly."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
provider:
  llm: anthropic
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.retrieval.rrf_k == 60
    assert cfg.retrieval.bm25_weight == 1.0
    assert cfg.retrieval.vector_weight == 1.0


def test_dikw_config_retrieval_block_round_trip(tmp_path: Path) -> None:
    """Fusion knobs parse from YAML + survive dump → load."""
    path = tmp_path / CONFIG_FILENAME
    path.write_text(
        """
retrieval:
  rrf_k: 40
  bm25_weight: 0.5
  vector_weight: 1.5
sources: []
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.retrieval.rrf_k == 40
    assert cfg.retrieval.bm25_weight == 0.5
    assert cfg.retrieval.vector_weight == 1.5

    # round-trip: dump → re-load yields identical values
    yaml_text = dump_config_yaml(cfg)
    path.write_text(yaml_text, encoding="utf-8")
    cfg2 = load_config(path)
    assert cfg2.retrieval.rrf_k == 40
    assert cfg2.retrieval.bm25_weight == 0.5
    assert cfg2.retrieval.vector_weight == 1.5
