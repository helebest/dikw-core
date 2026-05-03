"""Resolution order tests for ``dikw_core.client.config``.

The hierarchy is: explicit kwarg > env var > toml file > default. Each
test pins one layer at a time so a regression that breaks the priority
ordering surfaces with the precise layer that misbehaved.
"""

from __future__ import annotations

from pathlib import Path

from dikw_core.client.config import (
    DEFAULT_SERVER_URL,
    ENV_SERVER_TOKEN,
    ENV_SERVER_URL,
    resolve,
)


def _write_toml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_default_when_nothing_set(tmp_path: Path) -> None:
    cfg = resolve(env={}, config_path=tmp_path / "missing.toml")
    assert cfg.server_url == DEFAULT_SERVER_URL
    assert cfg.token is None


def test_explicit_kwargs_win_over_env_and_file(tmp_path: Path) -> None:
    toml = _write_toml(
        tmp_path / "client.toml",
        '[default]\nserver_url = "http://from-toml"\ntoken = "toml-tok"\n',
    )
    cfg = resolve(
        server_url="http://from-arg",
        token="arg-tok",
        env={ENV_SERVER_URL: "http://from-env", ENV_SERVER_TOKEN: "env-tok"},
        config_path=toml,
    )
    assert cfg.server_url == "http://from-arg"
    assert cfg.token == "arg-tok"


def test_env_wins_over_file(tmp_path: Path) -> None:
    toml = _write_toml(
        tmp_path / "client.toml",
        '[default]\nserver_url = "http://from-toml"\ntoken = "toml-tok"\n',
    )
    cfg = resolve(
        env={ENV_SERVER_URL: "http://from-env", ENV_SERVER_TOKEN: "env-tok"},
        config_path=toml,
    )
    assert cfg.server_url == "http://from-env"
    assert cfg.token == "env-tok"


def test_file_wins_when_env_missing(tmp_path: Path) -> None:
    toml = _write_toml(
        tmp_path / "client.toml",
        '[default]\nserver_url = "http://from-toml"\ntoken = "toml-tok"\n',
    )
    cfg = resolve(env={}, config_path=toml)
    assert cfg.server_url == "http://from-toml"
    assert cfg.token == "toml-tok"


def test_trailing_slash_normalised(tmp_path: Path) -> None:
    cfg = resolve(
        server_url="http://example.com/",
        env={},
        config_path=tmp_path / "missing.toml",
    )
    assert cfg.server_url == "http://example.com"  # no trailing slash


def test_empty_token_string_is_treated_as_none(tmp_path: Path) -> None:
    """Empty string in the toml is "unset" — falls through to env/default.

    This matters when a user comments out a real token by replacing it
    with an empty string: we must NOT pass ``""`` as a bearer header.
    """
    toml = _write_toml(
        tmp_path / "client.toml",
        '[default]\ntoken = ""\n',
    )
    cfg = resolve(env={ENV_SERVER_TOKEN: "from-env"}, config_path=toml)
    assert cfg.token == "from-env"
