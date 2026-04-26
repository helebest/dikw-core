"""Streaming SHA-256 of on-disk files."""

from __future__ import annotations

import hashlib
from pathlib import Path

_DEFAULT_CHUNK_SIZE = 1 << 20


def hash_file(path: Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """SHA-256 hex digest of ``path``, with O(chunk_size) peak memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()
