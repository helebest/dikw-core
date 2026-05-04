"""Streaming SHA-256 helper — ``src/dikw_core/domains/data/hashing.py``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dikw_core.domains.data.hashing import hash_file

# SHA-256 of zero bytes — well-known constant.
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_file_matches_oneshot_sha256(tmp_path: Path) -> None:
    """Reference: streaming digest must equal the one-shot digest."""
    p = tmp_path / "sample.bin"
    payload = b"hello world\n" * 17
    p.write_bytes(payload)
    assert hash_file(p) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1 << 20])
def test_hash_file_chunk_size_invariant(tmp_path: Path, chunk_size: int) -> None:
    """Digest must be identical regardless of chunk_size — guards against
    off-by-one in the iter()/sentinel loop.
    """
    p = tmp_path / "sample.bin"
    payload = bytes(range(256)) * 91  # 23_296 bytes; not aligned to any size
    p.write_bytes(payload)
    assert hash_file(p, chunk_size=chunk_size) == hashlib.sha256(payload).hexdigest()


def test_hash_file_empty_file(tmp_path: Path) -> None:
    """Empty file hashes to the well-known SHA-256 of empty bytes."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert hash_file(p) == _EMPTY_SHA256


def test_hash_file_large_streamed_correctness(tmp_path: Path) -> None:
    """Hash a 4 MiB file with a tiny chunk_size so the iterator runs many
    laps. Memory bound is hard to assert in pytest reliably; what we *can*
    pin is that the streamed digest still matches the one-shot reference.
    """
    p = tmp_path / "big.bin"
    payload = (b"abcdefghij" * 1024) * 410  # ~4 MiB
    p.write_bytes(payload)
    assert hash_file(p, chunk_size=1 << 16) == hashlib.sha256(payload).hexdigest()
