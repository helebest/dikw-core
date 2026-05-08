"""``init_logging()`` contract: env-driven, idempotent, isolates foreign handlers."""

from __future__ import annotations

import logging

import pytest

from dikw_core.logging import init_logging


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dikw_core.logging._initialized", False, raising=True)
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    yield
    root.handlers[:] = saved
    root.setLevel(saved_level)
    monkeypatch.setattr("dikw_core.logging._initialized", False, raising=True)


def test_init_logging_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_LOG_LEVEL", "DEBUG")
    init_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_init_logging_default_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIKW_LOG_LEVEL", raising=False)
    init_logging()
    assert logging.getLogger().level == logging.INFO


def test_init_logging_invalid_level_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A junk env value must NOT crash CLI startup."""
    monkeypatch.setenv("DIKW_LOG_LEVEL", "BOGUS")
    init_logging()
    assert logging.getLogger().level == logging.INFO


def test_init_logging_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_LOG_LEVEL", "INFO")
    init_logging()
    handler_count_after_first = len(logging.getLogger().handlers)
    init_logging()
    assert len(logging.getLogger().handlers) == handler_count_after_first


def test_init_logging_quiets_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIKW_LOG_LEVEL", "INFO")
    init_logging()
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING
