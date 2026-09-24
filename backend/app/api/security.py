"""API-key authentication and per-client rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time

from fastapi import Request

from .errors import ApiError
from .settings import Settings


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


class ApiKeyAuth:
    """Validates ``X-API-Key`` / ``Authorization: Bearer`` against configured keys.

    Keys are compared as SHA-256 digests with :func:`hmac.compare_digest` so
    neither the key length nor its prefix leaks through timing.
    """

    def __init__(self, keys: tuple[str, ...]) -> None:
        self._digests = [_digest(k) for k in keys]

    @property
    def enabled(self) -> bool:
        return bool(self._digests)

    @staticmethod
    def extract(request: Request) -> str | None:
        key = request.headers.get("x-api-key")
        if key:
            return key.strip()
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def identify(self, request: Request) -> str | None:
        """Return a stable, non-secret id for the caller's key, or raise 401."""
        if not self.enabled:
            return None
        key = self.extract(request)
        if not key:
            raise ApiError(401, "unauthorized", "Missing API key (X-API-Key header)")
        candidate = _digest(key)
        matched = False
        for digest in self._digests:
            matched |= hmac.compare_digest(candidate, digest)
        if not matched:
            raise ApiError(401, "unauthorized", "Invalid API key")
        return "key:" + candidate.hex()[:12]


class RateLimiter:
    """Token bucket per client identity (API key id or client IP)."""

    def __init__(self, per_minute: int, burst: int | None = None) -> None:
        self.rate = per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(1, per_minute))
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def check(self, identity: str) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(identity, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                retry = max(1, int((1.0 - tokens) / self.rate) + 1)
                self._buckets[identity] = (tokens, now)
                raise ApiError(
                    429, "rate_limited", "Too many requests", headers={"Retry-After": str(retry)}
                )
            self._buckets[identity] = (tokens - 1.0, now)
            if len(self._buckets) > 10_000:  # bound memory under IP churn
                stale = [k for k, (_, ts) in self._buckets.items() if now - ts > 600]
                for k in stale:
                    del self._buckets[k]


def client_ip(request: Request, settings: Settings) -> str:
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
