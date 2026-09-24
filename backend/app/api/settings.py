"""Runtime settings for the HTTP server, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Server configuration. Every field maps to an ``AUTOBANNER_*`` env var."""

    api_keys: tuple[str, ...] = ()
    use_ai: bool = False
    workers: int = 2
    max_queue: int = 32
    job_ttl_seconds: int = 3600
    max_jobs_retained: int = 500
    rate_limit_per_minute: int = 60
    max_upload_mb: int = 150
    cors_origins: tuple[str, ...] = ()
    enable_studio: bool = True
    enable_docs: bool = True
    trust_proxy_headers: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def auth_required(self) -> bool:
        return bool(self.api_keys)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_keys=_env_list("AUTOBANNER_API_KEYS"),
            use_ai=_env_bool("AUTOBANNER_USE_AI", False),
            workers=max(1, _env_int("AUTOBANNER_WORKERS", 2)),
            max_queue=max(1, _env_int("AUTOBANNER_MAX_QUEUE", 32)),
            job_ttl_seconds=max(60, _env_int("AUTOBANNER_JOB_TTL_SECONDS", 3600)),
            max_jobs_retained=max(10, _env_int("AUTOBANNER_MAX_JOBS_RETAINED", 500)),
            rate_limit_per_minute=max(0, _env_int("AUTOBANNER_RATE_LIMIT_PER_MINUTE", 60)),
            max_upload_mb=max(1, _env_int("AUTOBANNER_MAX_UPLOAD_MB", 150)),
            cors_origins=_env_list("AUTOBANNER_CORS_ORIGINS"),
            enable_studio=_env_bool("AUTOBANNER_ENABLE_STUDIO", True),
            enable_docs=_env_bool("AUTOBANNER_ENABLE_DOCS", True),
            trust_proxy_headers=_env_bool("AUTOBANNER_TRUST_PROXY_HEADERS", False),
        )
