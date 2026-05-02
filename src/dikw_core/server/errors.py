"""HTTP error mapping for the FastAPI app.

Engine-internal exceptions translate to a small uniform JSON envelope
``{ "error": { "code": ..., "message": ..., "detail": ... } }`` with a
matching HTTP status. NDJSON streams use the ``error`` event type for
non-final, in-stream errors (handled in ``ndjson.py``).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Server-thrown error with explicit HTTP semantics.

    ``code`` is a stable string clients can branch on (``not_found``,
    ``cancelled``, ``unauthorized`` …) — derived from the exception class
    name by default but overridable when raising.
    """

    status_code: int = 500
    default_code: str = "server_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        detail: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class BadRequest(ApiError):
    status_code = 400
    default_code = "bad_request"


class Unauthorized(ApiError):
    status_code = 401
    default_code = "unauthorized"


class NotFoundError(ApiError):
    status_code = 404
    default_code = "not_found"


class Conflict(ApiError):
    status_code = 409
    default_code = "conflict"


def _envelope(err: ApiError) -> dict[str, Any]:
    body: dict[str, Any] = {"code": err.code, "message": err.message}
    if err.detail is not None:
        body["detail"] = err.detail
    return {"error": body}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _on_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_envelope(exc))


__all__ = [
    "ApiError",
    "BadRequest",
    "Conflict",
    "NotFoundError",
    "Unauthorized",
    "install_handlers",
]
