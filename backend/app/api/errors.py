"""Uniform JSON error responses."""

from __future__ import annotations


class ApiError(Exception):
    """An error with an HTTP status, a stable machine code and a message."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}

    def body(self, request_id: str | None = None) -> dict[str, object]:
        err: dict[str, object] = {"code": self.code, "message": self.message}
        if request_id:
            err["request_id"] = request_id
        return {"error": err}
