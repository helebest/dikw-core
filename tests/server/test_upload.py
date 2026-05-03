"""HTTP-level tests for ``POST /v1/upload/sources``.

Covers the wire-format validation surface end-to-end via the in-memory
ASGI transport — every failure path is asserted by ``error.code`` so a
client implementation can branch on the error machine-readably.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, body in files.items():
            ti = tarfile.TarInfo(path)
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
    return buf.getvalue()


def _manifest_for(files: dict[str, bytes]) -> dict[str, Any]:
    entries = [
        {
            "path": p,
            "size": len(b),
            "sha256": hashlib.sha256(b).hexdigest(),
        }
        for p, b in files.items()
    ]
    return {
        "files": entries,
        "total_bytes": sum(len(b) for b in files.values()),
    }


def _post_upload(
    client: httpx.AsyncClient,
    files: dict[str, bytes],
    *,
    manifest: dict[str, Any] | None = None,
) -> Any:
    """Build the multipart parts and POST. Returns the awaitable."""
    payload = _tar_bytes(files)
    body = manifest if manifest is not None else _manifest_for(files)
    return client.post(
        "/v1/upload/sources",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": json.dumps(body)},
    )


# ---- happy path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_lands_in_staging(
    server_client: httpx.AsyncClient,
    wiki_root: Path,
) -> None:
    files = {
        "sources/notes/x.md": b"# x\nhello\n",
        "assets/img/y.png": b"\x89PNG\r\n\x1a\nfake",
    }
    resp = await _post_upload(server_client, files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["files_count"] == 2
    assert body["bytes"] > 0
    upload_id = body["upload_id"]
    assert len(upload_id) >= 8

    staging = wiki_root / ".dikw" / "upload-staging" / upload_id
    assert (staging / "sources" / "notes" / "x.md").read_bytes() == files[
        "sources/notes/x.md"
    ]
    assert (staging / "assets" / "img" / "y.png").read_bytes() == files[
        "assets/img/y.png"
    ]


# ---- manifest validation -----------------------------------------------


@pytest.mark.asyncio
async def test_manifest_sha256_mismatch_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    files = {"sources/x.md": b"hello"}
    bad_manifest = _manifest_for(files)
    bad_manifest["files"][0]["sha256"] = "0" * 64
    resp = await _post_upload(server_client, files, manifest=bad_manifest)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_sha256_mismatch"


@pytest.mark.asyncio
async def test_manifest_missing_file_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    files = {"sources/x.md": b"hello"}
    manifest = _manifest_for(files)
    manifest["files"].append(
        {"path": "sources/ghost.md", "size": 5, "sha256": "0" * 64}
    )
    resp = await _post_upload(server_client, files, manifest=manifest)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_missing_files"


@pytest.mark.asyncio
async def test_manifest_extra_file_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    files = {"sources/x.md": b"hello", "sources/y.md": b"world"}
    manifest = _manifest_for({"sources/x.md": files["sources/x.md"]})
    resp = await _post_upload(server_client, files, manifest=manifest)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_extra_files"


@pytest.mark.asyncio
async def test_manifest_invalid_json_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    payload = _tar_bytes({"sources/x.md": b"hello"})
    resp = await server_client.post(
        "/v1/upload/sources",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={"manifest": "{not json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "manifest_malformed"


# ---- tarball shape validation -------------------------------------------


@pytest.mark.asyncio
async def test_tar_path_traversal_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    # Build the tar by hand because tarfile's high-level API normalises
    # leading separators — we want the literal escape attempt.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("../escape.md")
        body = b"nope"
        ti.size = len(body)
        tf.addfile(ti, io.BytesIO(body))
    payload = buf.getvalue()
    resp = await server_client.post(
        "/v1/upload/sources",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={
            "manifest": json.dumps(
                {
                    "files": [
                        {
                            "path": "../escape.md",
                            "size": 4,
                            "sha256": hashlib.sha256(b"nope").hexdigest(),
                        }
                    ],
                    "total_bytes": 4,
                }
            )
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tar_path_traversal"


@pytest.mark.asyncio
async def test_tar_outside_allowed_top_dirs_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    files = {"wiki/index.md": b"# index"}
    resp = await _post_upload(server_client, files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tar_unexpected_path"


@pytest.mark.asyncio
async def test_tar_with_symlink_rejected(
    server_client: httpx.AsyncClient,
) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("sources/link.md")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    payload = buf.getvalue()
    resp = await server_client.post(
        "/v1/upload/sources",
        files={"payload": ("u.tar.gz", payload, "application/gzip")},
        data={
            "manifest": json.dumps(
                {"files": [], "total_bytes": 0}
            )
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "tar_link_forbidden"


@pytest.mark.asyncio
async def test_upload_too_large_rejected(
    server_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cap at a tiny size so the test stays cheap.
    monkeypatch.setenv("DIKW_SERVER_MAX_UPLOAD_BYTES", "256")
    big = b"a" * 4096
    files = {"sources/big.md": big}
    resp = await _post_upload(server_client, files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "upload_too_large"
