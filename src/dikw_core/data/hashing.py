"""Streaming SHA-256 of on-disk files.

Asset binaries (images, eventually video / audio per the v2 roadmap) can
exceed the practical memory budget of a one-shot ``read_bytes`` →
``sha256`` pipeline. ``hash_file`` reads the file in fixed-size blocks
and folds each block into the hash via ``.update()`` so peak memory
stays at ``chunk_size`` regardless of file size.

Kept separate from ``data/backends/markdown.content_hash`` deliberately:
that one hashes an in-memory string (a parsed document body) and has no
streaming story — the body is realised before it is hashed. Conflating
"hash bytes from disk" with "hash bytes from memory" would muddy the
intent at the call sites.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB — comfortable trade-off; empirically
# saturates SHA-256 throughput on commodity disks without bloating RSS.


def hash_file(path: Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """SHA-256 hex digest of ``path``'s contents, computed incrementally.

    Peak memory is O(chunk_size) regardless of file size.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()
