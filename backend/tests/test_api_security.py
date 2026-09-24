"""Unit tests for API key auth, rate limiting and client IP resolution."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from starlette.requests import Request  # noqa: E402

from backend.app.api.errors import ApiError  # noqa: E402
from backend.app.api.security import ApiKeyAuth, RateLimiter, client_ip  # noqa: E402
from backend.app.api.settings import Settings  # noqa: E402


def _request(headers: dict[str, str], host: str = "10.0.0.1") -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (host, 1234),
    })


def test_forwarded_for_uses_rightmost_proxy_appended_entry():
    settings = Settings(trust_proxy_headers=True)
    req = _request({"X-Forwarded-For": "6.6.6.6, 203.0.113.9"})
    assert client_ip(req, settings) == "203.0.113.9"


def test_forwarded_for_ignored_unless_trusted():
    req = _request({"X-Forwarded-For": "6.6.6.6"})
    assert client_ip(req, Settings()) == "10.0.0.1"


def test_rate_limiter_bounds_client_table():
    limiter = RateLimiter(per_minute=60, max_clients=100)
    for i in range(1000):
        limiter.check(f"ip:{i}")
    assert len(limiter._buckets) <= 101


def test_rate_limiter_blocks_after_burst():
    limiter = RateLimiter(per_minute=3)
    for _ in range(3):
        limiter.check("a")
    with pytest.raises(ApiError) as exc:
        limiter.check("a")
    assert exc.value.status == 429
    limiter.check("b")  # other identities unaffected


def test_api_key_identity_is_stable_and_not_the_key():
    auth = ApiKeyAuth(("super-secret",))
    ident = auth.identify(_request({"X-API-Key": "super-secret"}))
    assert ident == auth.identify(_request({"Authorization": "Bearer super-secret"}))
    assert "super-secret" not in ident
