"""Local sources/+assets/ → tar.gz + manifest packing tests.

Verifies that ``build_upload`` produces a tarball whose contents,
manifest, and sha256 hashes match what the server's
``POST /v1/upload/sources`` validator expects. Round-tripping through
the actual server validator lives in ``tests/server/test_upload.py``;
this file focuses on the client side alone (no server, no httpx) so a
broken manifest builder shows up before any I/O.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from dikw_core.client.upload import UploadError, build_upload


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_build_from_layout_with_sources_subtree(tmp_path: Path) -> None:
    src = tmp_path / "inbox"
    (src / "sources").mkdir(parents=True)
    (src / "sources" / "alpha.md").write_text("# A\n", encoding="utf-8")
    (src / "assets").mkdir()
    asset_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    (src / "assets" / "diagram.png").write_bytes(asset_bytes)

    bundle = build_upload(src)
    try:
        manifest = json.loads(bundle.manifest_json)
        paths = sorted(e["path"] for e in manifest["files"])
        assert paths == ["assets/diagram.png", "sources/alpha.md"]
        # sha256 in manifest matches the on-disk content.
        for entry in manifest["files"]:
            on_disk = (src / entry["path"]).read_bytes()
            assert entry["sha256"] == _sha256(on_disk)
            assert entry["size"] == len(on_disk)

        # The actual tarball matches the manifest entry for entry.
        with tarfile.open(fileobj=bundle.payload, mode="r:gz") as tf:
            tar_paths = sorted(m.name for m in tf.getmembers() if m.isfile())
            assert tar_paths == paths
    finally:
        bundle.close()


def test_build_promotes_top_level_md_into_sources(tmp_path: Path) -> None:
    """Bare ``foo.md`` at the top is auto-rooted under ``sources/``."""
    src = tmp_path / "inbox-flat"
    src.mkdir()
    (src / "foo.md").write_text("# foo\n", encoding="utf-8")
    (src / "bar.md").write_text("# bar\n", encoding="utf-8")

    bundle = build_upload(src)
    try:
        manifest = json.loads(bundle.manifest_json)
        paths = sorted(e["path"] for e in manifest["files"])
        assert paths == ["sources/bar.md", "sources/foo.md"]
    finally:
        bundle.close()


def test_build_rejects_mixed_top_and_subtree(tmp_path: Path) -> None:
    src = tmp_path / "inbox-mixed"
    (src / "sources").mkdir(parents=True)
    (src / "sources" / "alpha.md").write_text("# A\n", encoding="utf-8")
    (src / "stray.md").write_text("# stray\n", encoding="utf-8")

    with pytest.raises(UploadError, match=r"ambiguous|disambiguate"):
        build_upload(src)


def test_build_rejects_unsupported_extension(tmp_path: Path) -> None:
    src = tmp_path / "inbox-bad"
    (src / "sources").mkdir(parents=True)
    (src / "sources" / "code.py").write_text("print(1)\n", encoding="utf-8")

    with pytest.raises(UploadError, match="unsupported"):
        build_upload(src)


def test_build_rejects_empty_input(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(UploadError, match="no files"):
        build_upload(src)


def test_extra_extensions_admitted(tmp_path: Path) -> None:
    """Custom extensions extend, not replace, the default whitelist."""
    src = tmp_path / "inbox-extra"
    (src / "sources").mkdir(parents=True)
    (src / "sources" / "snippet.org").write_text("* heading\n", encoding="utf-8")
    (src / "sources" / "alpha.md").write_text("# A\n", encoding="utf-8")

    bundle = build_upload(src, extra_extensions=[".org"])
    try:
        manifest = json.loads(bundle.manifest_json)
        paths = sorted(e["path"] for e in manifest["files"])
        assert paths == ["sources/alpha.md", "sources/snippet.org"]
    finally:
        bundle.close()
