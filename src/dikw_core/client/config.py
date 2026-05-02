"""Client config resolution.

The client picks the server URL + bearer token from four sources, in
descending precedence:

1. Explicit args (``--server`` / ``--token``) — highest, defeats anything
   else for a single command.
2. Environment variables (``DIKW_SERVER_URL``, ``DIKW_SERVER_TOKEN``) —
   for shells/CI; the env always wins over file-based config.
3. ``~/.config/dikw/client.toml`` — per-user persistent default.
4. Built-in defaults (``http://127.0.0.1:8765``, no token).

Anything missing falls through to the next layer rather than erroring,
so the typical "I started ``dikw serve`` on this machine and want to
talk to it" flow needs no configuration at all.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SERVER_URL = "http://127.0.0.1:8765"
ENV_SERVER_URL = "DIKW_SERVER_URL"
ENV_SERVER_TOKEN = "DIKW_SERVER_TOKEN"


@dataclass(frozen=True)
class ClientConfig:
    """Resolved server URL + token the transport will use."""

    server_url: str
    token: str | None


def default_config_path() -> Path:
    """``~/.config/dikw/client.toml`` on Linux/macOS;
    ``%APPDATA%\\dikw\\client.toml`` on Windows.

    Falls back to the XDG default on platforms without ``%APPDATA%``."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "dikw" / "client.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "dikw" / "client.toml"


def _load_file(path: Path) -> dict[str, object]:
    """Parse ``client.toml``; missing file → empty dict (not an error).

    The file shape is intentionally minimal — a flat ``[default]`` table
    with ``server_url`` and ``token`` keys. We don't grow named profiles
    here until there's a real need; Phase-5 plan keeps it single-tenant.
    """
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    default = data.get("default")
    if not isinstance(default, dict):
        return {}
    return {str(k): v for k, v in default.items()}


def resolve(
    *,
    server_url: str | None = None,
    token: str | None = None,
    config_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> ClientConfig:
    """Compute the effective ``ClientConfig`` from CLI flags + env + file.

    Tests pass ``env`` and ``config_path`` to keep the lookup hermetic;
    the production CLI calls this with no kwargs, accepting the defaults.
    """
    src_env = os.environ if env is None else env
    file_cfg = _load_file(config_path or default_config_path())

    resolved_url = (
        server_url
        or src_env.get(ENV_SERVER_URL)
        or _strip_or_none(file_cfg.get("server_url"))
        or DEFAULT_SERVER_URL
    )
    resolved_token = (
        token
        or src_env.get(ENV_SERVER_TOKEN)
        or _strip_or_none(file_cfg.get("token"))
    )
    return ClientConfig(
        server_url=resolved_url.rstrip("/"),
        token=resolved_token or None,
    )


def _strip_or_none(value: object) -> str | None:
    """Coerce a TOML value to ``str | None``; empty strings become None.

    Empty-string overrides on disk are treated as "unset" rather than
    "explicitly empty" — otherwise a user who comments-out ``token = ""``
    can't fall through to env on the next run.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s or None


__all__ = [
    "DEFAULT_SERVER_URL",
    "ENV_SERVER_TOKEN",
    "ENV_SERVER_URL",
    "ClientConfig",
    "default_config_path",
    "resolve",
]
