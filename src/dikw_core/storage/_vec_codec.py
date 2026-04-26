"""Float32 vector packing shared between adapters.

The migration comment on ``embed_cache`` declares the on-disk vector
encoding byte-exact across SQLite and Postgres so a snapshot taken
under one backend is interchangeable with the other. Both adapters
must call through these helpers — never inline ``struct.pack``.
"""

from __future__ import annotations

import struct


def serialize_vec(values: list[float]) -> bytes:
    """Pack a float32 vector (LE)."""
    return struct.pack(f"{len(values)}f", *values)


def deserialize_vec(blob: bytes, dim: int) -> list[float]:
    """Unpack a float32 vector blob (inverse of ``serialize_vec``)."""
    return list(struct.unpack(f"{dim}f", blob))


__all__ = ["deserialize_vec", "serialize_vec"]
