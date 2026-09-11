"""Local users, sessions and personal API tokens.

AutoBanner runs self-hosted, so identity is local: a JSON-backed user store with
PBKDF2-hashed passwords, opaque server-side sessions (delivered as an HttpOnly cookie)
and personal API tokens for automation. No external identity provider is contacted.

Workspaces
    A *workspace* is the owner id every project, job, usage counter and event is
    scoped to (the same id ``AUTOBANNER_API_KEYS`` calls an owner). Each user belongs
    to exactly one workspace and holds one role there (``viewer``, ``editor``,
    ``approver``, ``admin``). Only the operator-held setup token can create a
    workspace with its first administrator; that administrator manages members.

Security notes
    Only SHA-256 digests of sessions and tokens are stored, so a copied store file
    grants nothing. Password checks are constant-time and failed logins are throttled
    per username. Everything is written atomically under a process lock; the store is
    per process (one server), like the rest of the file-backed state.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..design.atomic import atomic_write_json

logger = logging.getLogger("autobanner.api.auth")

ROLES = ("viewer", "editor", "approver", "admin")
USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{1,63}$")
WORKSPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MIN_PASSWORD = 8
SESSION_IDLE = timedelta(hours=12)
SESSION_MAX = timedelta(days=7)
LOCKOUT_FAILURES = 8
LOCKOUT_SECONDS = 60
TOKEN_PREFIX = "abt_"


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Principal:
    """Who is acting: the workspace (owner) and role, plus the user when known."""

    owner: str
    role: str
    username: str | None = None
    via: str = "open"  # open | api_key | token | session


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def hash_password(password: str, *, iterations: int, salt: bytes | None = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {"alg": "pbkdf2_sha256", "iterations": iterations, "salt": salt.hex(), "hash": dk.hex()}


def verify_password(password: str, record: dict) -> bool:
    try:
        salt = bytes.fromhex(record["salt"])
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(record["iterations"])
        )
        return hmac.compare_digest(dk.hex(), str(record["hash"]))
    except (KeyError, ValueError, TypeError):
        return False


class UserStore:
    """JSON-backed users, sessions and tokens for one deployment."""

    def __init__(self, path: str | Path, *, iterations: int | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.iterations = iterations or int(
            os.environ.get("AUTOBANNER_PBKDF2_ITERATIONS", "600000")
        )
        self._lock = threading.RLock()
        self._failures: dict[str, tuple[int, float]] = {}
        self._data = {"users": {}, "sessions": {}, "tokens": {}}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise AuthError(f"cannot read user store {self.path}: {exc}", 500) from exc
            for key in ("users", "sessions", "tokens"):
                self._data.setdefault(key, {})

    # ---- persistence -------------------------------------------------------------------
    def _save(self) -> None:
        atomic_write_json(self.path, self._data, ensure_ascii=True)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)

    # ---- users -------------------------------------------------------------------------
    def count(self) -> int:
        return len(self._data["users"])

    def setup_required(self) -> bool:
        return self.count() == 0

    def workspaces(self) -> list[str]:
        return sorted({u["owner"] for u in self._data["users"].values()})

    def public(self, user: dict) -> dict:
        return {
            "username": user["username"],
            "owner": user["owner"],
            "role": user["role"],
            "created_at": user.get("created_at"),
            "created_by": user.get("created_by"),
        }

    def get(self, username: str) -> dict | None:
        return self._data["users"].get(username.lower())

    def create_user(
        self, username: str, password: str, owner: str, role: str, *, created_by: str | None
    ) -> dict:
        username = username.strip()
        if not USERNAME.match(username):
            raise AuthError("username: 2-64 letters, digits, . _ @ -", 400)
        if not WORKSPACE.match(owner):
            raise AuthError("workspace: letters, digits, . _ -", 400)
        if role not in ROLES:
            raise AuthError(f"role must be one of {', '.join(ROLES)}", 400)
        if len(password) < MIN_PASSWORD:
            raise AuthError(f"password must be at least {MIN_PASSWORD} characters", 400)
        key = username.lower()
        with self._lock:
            if key in self._data["users"]:
                raise AuthError("username already exists", 409)
            user = {
                "username": username,
                "owner": owner,
                "role": role,
                "password": hash_password(password, iterations=self.iterations),
                "created_at": _iso(_now()),
                "created_by": created_by,
            }
            self._data["users"][key] = user
            self._save()
        logger.info("user created: %s (%s@%s) by %s", username, role, owner, created_by)
        return user

    def list_users(self, owner: str) -> list[dict]:
        return [self.public(u) for u in self._data["users"].values() if u["owner"] == owner]

    def delete_user(self, username: str, owner: str) -> None:
        key = username.lower()
        with self._lock:
            user = self._data["users"].get(key)
            if user is None or user["owner"] != owner:
                raise AuthError("user not found", 404)
            admins = [
                u for u in self._data["users"].values()
                if u["owner"] == owner and u["role"] == "admin"
            ]
            if user["role"] == "admin" and len(admins) <= 1:
                raise AuthError("cannot remove the last administrator of a workspace", 409)
            del self._data["users"][key]
            self._data["sessions"] = {
                k: s for k, s in self._data["sessions"].items() if s["username"] != key
            }
            self._data["tokens"] = {
                k: t for k, t in self._data["tokens"].items() if t["username"] != key
            }
            self._save()

    def set_role(self, username: str, owner: str, role: str) -> dict:
        if role not in ROLES:
            raise AuthError(f"role must be one of {', '.join(ROLES)}", 400)
        key = username.lower()
        with self._lock:
            user = self._data["users"].get(key)
            if user is None or user["owner"] != owner:
                raise AuthError("user not found", 404)
            if user["role"] == "admin" and role != "admin":
                admins = [
                    u for u in self._data["users"].values()
                    if u["owner"] == owner and u["role"] == "admin"
                ]
                if len(admins) <= 1:
                    raise AuthError("a workspace needs at least one administrator", 409)
            user["role"] = role
            self._save()
        return user

    def set_password(self, username: str, new_password: str) -> None:
        if len(new_password) < MIN_PASSWORD:
            raise AuthError(f"password must be at least {MIN_PASSWORD} characters", 400)
        key = username.lower()
        with self._lock:
            user = self._data["users"].get(key)
            if user is None:
                raise AuthError("user not found", 404)
            user["password"] = hash_password(new_password, iterations=self.iterations)
            # Other sessions of this user are ended; tokens stay (explicitly revoked).
            self._data["sessions"] = {
                k: s for k, s in self._data["sessions"].items() if s["username"] != key
            }
            self._save()

    # ---- login -------------------------------------------------------------------------
    def authenticate(self, username: str, password: str) -> dict:
        """Return the user for valid credentials; raise 401 or 429 otherwise."""
        key = username.strip().lower()
        now = time.monotonic()
        failures, until = self._failures.get(key, (0, 0.0))
        if until > now:
            raise AuthError("too many failed logins; try again shortly", 429)
        user = self._data["users"].get(key)
        # Always run one hash so a missing user takes as long as a wrong password.
        record = user["password"] if user else hash_password("", iterations=self.iterations)
        ok = user is not None and verify_password(password, record)
        if not ok:
            failures += 1
            lock_until = now + LOCKOUT_SECONDS if failures >= LOCKOUT_FAILURES else 0.0
            self._failures[key] = (failures if lock_until == 0.0 else 0, lock_until)
            logger.warning("failed login for %s (%d)", key, failures)
            raise AuthError("invalid username or password", 401)
        self._failures.pop(key, None)
        return user

    # ---- sessions ----------------------------------------------------------------------
    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock:
            self._prune_sessions(now)
            self._data["sessions"][_digest(token)] = {
                "username": username.lower(),
                "created_at": _iso(now),
                "last_seen": _iso(now),
            }
            self._save()
        return token

    def resolve_session(self, token: str | None) -> dict | None:
        if not token:
            return None
        key = _digest(token)
        now = _now()
        with self._lock:
            sess = self._data["sessions"].get(key)
            if sess is None:
                return None
            try:
                created = datetime.fromisoformat(sess["created_at"])
                seen = datetime.fromisoformat(sess["last_seen"])
            except (KeyError, ValueError):
                self._data["sessions"].pop(key, None)
                return None
            if now - created > SESSION_MAX or now - seen > SESSION_IDLE:
                self._data["sessions"].pop(key, None)
                self._save()
                return None
            if now - seen > timedelta(minutes=5):
                sess["last_seen"] = _iso(now)
                self._save()
            return self._data["users"].get(sess["username"])

    def revoke_session(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            if self._data["sessions"].pop(_digest(token), None) is not None:
                self._save()

    def _prune_sessions(self, now: datetime) -> None:
        keep = {}
        for k, s in self._data["sessions"].items():
            try:
                created = datetime.fromisoformat(s["created_at"])
                seen = datetime.fromisoformat(s["last_seen"])
            except (KeyError, ValueError):
                continue
            if now - created <= SESSION_MAX and now - seen <= SESSION_IDLE:
                keep[k] = s
        self._data["sessions"] = keep

    # ---- personal API tokens -----------------------------------------------------------
    def create_token(self, username: str, name: str) -> tuple[str, dict]:
        raw = TOKEN_PREFIX + secrets.token_urlsafe(24)
        token_id = secrets.token_hex(4)
        record = {
            "id": token_id,
            "username": username.lower(),
            "name": (name or "token").strip()[:64],
            "created_at": _iso(_now()),
            "last_used": None,
        }
        with self._lock:
            self._data["tokens"][_digest(raw)] = record
            self._save()
        return raw, dict(record)

    def resolve_token(self, raw: str | None) -> dict | None:
        if not raw or not raw.startswith(TOKEN_PREFIX):
            return None
        with self._lock:
            rec = self._data["tokens"].get(_digest(raw))
            if rec is None:
                return None
            rec["last_used"] = _iso(_now())
            return self._data["users"].get(rec["username"])

    def list_tokens(self, username: str) -> list[dict]:
        key = username.lower()
        return [dict(t) for t in self._data["tokens"].values() if t["username"] == key]

    def revoke_token(self, username: str, token_id: str) -> None:
        key = username.lower()
        with self._lock:
            for k, t in list(self._data["tokens"].items()):
                if t["username"] == key and t["id"] == token_id:
                    del self._data["tokens"][k]
                    self._save()
                    return
        raise AuthError("token not found", 404)
