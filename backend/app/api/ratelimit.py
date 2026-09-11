"""Per-owner token-bucket rate limiting for mutating requests.

Configured with ``AUTOBANNER_RATE_LIMIT`` such as ``60/minute`` or ``600/hour``;
unset means no limit. Buckets live in memory (one process); a multi-node
deployment needs a shared store, which is not provided.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass

_UNITS = {
    "second": 1.0,
    "sec": 1.0,
    "s": 1.0,
    "minute": 60.0,
    "min": 60.0,
    "m": 60.0,
    "hour": 3600.0,
    "h": 3600.0,
}


@dataclass(frozen=True)
class Rate:
    count: int
    per_seconds: float

    @property
    def per_second(self) -> float:
        return self.count / self.per_seconds


def parse_rate(raw: str | None) -> Rate | None:
    """``"60/minute"`` → 60 requests per 60 s; empty or unset → ``None`` (no limit)."""
    if raw is None or not raw.strip():
        return None
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d*)\s*([a-zA-Z]+)\s*", raw)
    if not m:
        raise ValueError(f"invalid rate limit '{raw}' (expected e.g. 60/minute)")
    count, multiplier, unit = int(m.group(1)), m.group(2), m.group(3).lower()
    if unit not in _UNITS or count <= 0:
        raise ValueError(f"invalid rate limit '{raw}'")
    seconds = _UNITS[unit] * (int(multiplier) if multiplier else 1)
    return Rate(count, seconds)


class RateLimiter:
    """Token bucket per key (owner). ``allow`` returns (allowed, retry_after_seconds)."""

    def __init__(self, rate: Rate | None, clock=time.monotonic) -> None:
        self.rate = rate
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens: dict[str, float] = {}
        self._updated: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.rate is not None

    def allow(self, key: str) -> tuple[bool, int]:
        if self.rate is None:
            return True, 0
        now = self._clock()
        with self._lock:
            tokens = self._tokens.get(key, float(self.rate.count))
            last = self._updated.get(key, now)
            tokens = min(float(self.rate.count), tokens + (now - last) * self.rate.per_second)
            if tokens >= 1.0:
                self._tokens[key] = tokens - 1.0
                self._updated[key] = now
                return True, 0
            self._tokens[key] = tokens
            self._updated[key] = now
            retry = math.ceil((1.0 - tokens) / self.rate.per_second)
            return False, max(1, retry)

    def describe(self) -> str | None:
        if self.rate is None:
            return None
        return f"{self.rate.count}/{int(self.rate.per_seconds)}s"
