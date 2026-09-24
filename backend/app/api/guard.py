"""Pre-body request guard for upload endpoints.

FastAPI reads (and spools to disk) the whole multipart body *before* it runs
endpoint dependencies. Authentication, rate limiting and the upload size cap
therefore have to happen at the ASGI layer, or an anonymous client could make
the server buffer arbitrarily large bodies before being rejected.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from .errors import ApiError
from .security import ApiKeyAuth, RateLimiter, client_ip
from .settings import Settings

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

UPLOAD_PATHS = frozenset({"/v1/analyze", "/v1/render", "/v1/jobs"})
# Multipart framing + the JSON options field on top of the file itself.
_MULTIPART_OVERHEAD = 1024 * 1024


class BodyTooLarge(StarletteHTTPException):
    """Raised from ``receive`` once the streamed body exceeds the cap."""


class UploadGuard:
    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: Settings,
        auth: ApiKeyAuth,
        limiter: RateLimiter,
    ) -> None:
        self.app = app
        self.settings = settings
        self.auth = auth
        self.limiter = limiter
        self.max_upload_bytes = settings.max_upload_mb * 1024 * 1024
        self.max_body = self.max_upload_bytes + _MULTIPART_OVERHEAD

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in UPLOAD_PATHS
        ):
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        try:
            owner = self.auth.identify(request)
            self.limiter.check(owner or "ip:" + client_ip(request, self.settings))
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > self.max_body:
                raise self._too_large()
        except ApiError as err:
            rid = _request_id(request)
            headers = {**err.headers, "X-Request-ID": rid}
            await JSONResponse(err.body(rid), status_code=err.status, headers=headers)(
                scope, receive, send
            )
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body:
                    raise BodyTooLarge(413, detail=self._too_large().message)
            return message

        await self.app(scope, limited_receive, send)

    def _too_large(self) -> ApiError:
        return ApiError(
            413,
            "payload_too_large",
            f"File exceeds the {self.max_upload_bytes // 1048576} MB upload limit",
        )


def _request_id(request: Request) -> str:
    incoming = request.headers.get("x-request-id", "")
    if 0 < len(incoming) <= 64 and incoming.isascii():
        return incoming
    return uuid.uuid4().hex
