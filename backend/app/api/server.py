"""FastAPI application exposing the project journey and serving the web UI.

Run locally::

    AUTOBANNER_DATA_DIR=./data uvicorn backend.app.api.server:app --port 8000

Authentication and ownership (``AUTOBANNER_AUTH`` = ``local`` | ``api_key`` | ``open``):

- ``local`` (the documented deployment mode): users, sessions and personal API
  tokens live in ``$AUTOBANNER_DATA_DIR/auth/users.json`` (see ``auth.py``). The
  first workspace administrator is created through ``POST /api/auth/setup`` with
  the operator's setup token (``AUTOBANNER_SETUP_TOKEN`` or the one logged at
  startup); the browser UI then uses an HttpOnly session cookie, automation uses
  ``X-API-Key`` with a personal token. Chosen automatically once a user exists.
- ``api_key``: ``AUTOBANNER_API_KEYS="key1:owner-a[:role],key2:owner-b[:role]"``
  maps static keys to owners (workspaces) and roles; ``AUTOBANNER_API_KEY=key`` is
  the single-owner form (owner ``default``). Static keys also work in ``local`` mode.
- ``open`` (default when neither users nor keys exist): everything belongs to
  owner ``local`` with the admin role. Development only; the health endpoint and
  the UI say so.

Every workspace only sees its own projects and jobs (others read as 404).
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..logging_config import setup_logging
from .auth import AuthError, Principal, UserStore
from .offline import install_offline_guard
from .presets import preset_catalog
from .ratelimit import RateLimiter, parse_rate
from .service import LOCAL_OWNER, MAX_UPLOAD_BYTES, ProjectService, ServiceError, env_flag

logger = logging.getLogger("autobanner.api.server")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_UPLOAD = File(...)


def create_app(data_dir: str | Path | None = None, *, max_workers: int | None = None) -> FastAPI:
    setup_logging(os.environ.get("AUTOBANNER_LOG_LEVEL", "INFO"))
    offline = env_flag("AUTOBANNER_OFFLINE", False)
    if offline:
        install_offline_guard()
    data_root = Path(data_dir or os.environ.get("AUTOBANNER_DATA_DIR", "data"))
    workers = int(max_workers or os.environ.get("AUTOBANNER_JOB_WORKERS", "2"))
    service = ProjectService(
        data_root, max_workers=workers, use_ai=env_flag("AUTOBANNER_USE_AI", False)
    )
    key_principals = _key_owner_map()
    users = UserStore(data_root / "auth" / "users.json")
    auth_mode = _auth_mode(users, key_principals)
    setup_token = os.environ.get("AUTOBANNER_SETUP_TOKEN", "").strip()
    if auth_mode == "local" and not setup_token:
        setup_token = secrets.token_urlsafe(12)
        if users.setup_required():
            logger.warning(
                "AUTOBANNER_SETUP_TOKEN is not set; use this one-time setup token to create the "
                "first administrator: %s",
                setup_token,
            )
    cookie_secure = env_flag("AUTOBANNER_COOKIE_SECURE", False)
    limiter = RateLimiter(parse_rate(os.environ.get("AUTOBANNER_RATE_LIMIT")))
    retention_days = _int_env("AUTOBANNER_RETENTION_DAYS")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if retention_days:
            service.start_retention(retention_days)
        yield
        service.shutdown()

    app = FastAPI(
        title="AutoBanner API",
        version="0.2.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.users = users
    app.state.auth_mode = auth_mode
    app.state.setup_token = setup_token

    def resolve_principal(request: Request) -> Principal | None:
        """Identify the caller: static key, personal token, session cookie, or open mode.

        Returns ``None`` for an anonymous request in a mode that needs credentials and
        raises 401 for credentials that are presented but wrong.
        """
        cached = getattr(request.state, "principal", None)
        if cached is not None:
            return cached
        principal: Principal | None = None
        key = request.headers.get("x-api-key") or ""
        if key:
            if key in key_principals:
                owner, role = key_principals[key]
                principal = Principal(owner, role, via="api_key")
            else:
                user = users.resolve_token(key)
                if user is None:
                    raise HTTPException(status_code=401, detail="invalid or missing API key")
                principal = Principal(user["owner"], user["role"], user["username"], "token")
        if principal is None:
            user = users.resolve_session(request.cookies.get(SESSION_COOKIE))
            if user is not None:
                principal = Principal(user["owner"], user["role"], user["username"], "session")
        if principal is None and auth_mode == "open":
            principal = Principal(LOCAL_OWNER, "admin")
        request.state.principal = principal
        return principal

    async def current_principal(request: Request) -> Principal:
        principal = resolve_principal(request)
        if principal is None:
            raise HTTPException(status_code=401, detail="sign in required")
        if principal.via == "session" and request.method in MUTATING and not request.headers.get(
            "x-requested-with"
        ):
            # Cookie-authenticated writes must come from our own scripts (CSRF).
            raise HTTPException(
                status_code=403, detail="missing X-Requested-With header (CSRF protection)"
            )
        return principal

    Who = Depends(current_principal)  # noqa: N806

    async def current_owner(principal: Principal = Who) -> str:
        return principal.owner

    async def current_role(principal: Principal = Who) -> str:
        return principal.role

    def require(*roles: str):
        async def _guard(role: str = Depends(current_role)) -> None:
            if role != "admin" and role not in roles:
                needs = "/".join(roles)
                raise HTTPException(
                    status_code=403, detail=f"role '{role}' may not do this (needs {needs})"
                )

        return Depends(_guard)

    dep = [Depends(current_owner)]
    edit = [Depends(current_owner), require("editor")]
    approve = [Depends(current_owner), require("approver")]
    admin = [Depends(current_owner), require("admin")]
    Owner = Depends(current_owner)  # noqa: N806
    Role = Depends(current_role)  # noqa: N806

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if limiter.enabled and request.method in MUTATING:
            try:
                principal = resolve_principal(request)
            except HTTPException:
                principal = None
            owner = principal.owner if principal is not None else "anonymous"
            allowed, retry = limiter.allow(owner)
            if not allowed:
                service.events.record(
                    owner, "rate_limited", path=request.url.path, retry_after=retry
                )
                return JSONResponse(
                    {"detail": "rate limit exceeded", "retry_after": retry},
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
        return await call_next(request)

    async def read_limited(upload: UploadFile, limit: int) -> bytes:
        """Read an upload without buffering more than ``limit`` bytes."""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise HTTPException(status_code=413, detail="file too large")
            chunks.append(chunk)
        return b"".join(chunks)

    @app.exception_handler(ServiceError)
    async def _service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @app.exception_handler(AuthError)
    async def _auth_error(_request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    # ---- auth --------------------------------------------------------------------------
    def set_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            SESSION_COOKIE, token, httponly=True, samesite="lax", secure=cookie_secure,
            max_age=7 * 24 * 3600, path="/",
        )

    def auth_status(request: Request) -> dict:
        try:
            principal = resolve_principal(request)
        except HTTPException:
            principal = None
        return {
            "mode": auth_mode,
            "setup_required": auth_mode == "local" and users.setup_required(),
            "authenticated": principal is not None,
            "user": principal.username if principal else None,
            "role": principal.role if principal else None,
            "workspace": principal.owner if principal else None,
            "via": principal.via if principal else None,
            "roles": list(ROLES),
        }

    @app.get("/api/auth/status")
    def get_auth_status(request: Request) -> dict:
        return auth_status(request)

    @app.post("/api/auth/setup", status_code=201)
    async def setup(request: Request, response: Response) -> dict:
        """Create a workspace with its first administrator (operator setup token)."""
        if auth_mode != "local":
            raise HTTPException(status_code=409, detail=f"auth mode is '{auth_mode}'")
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        if not hmac.compare_digest(str(body.get("setup_token", "")), setup_token):
            raise HTTPException(status_code=403, detail="invalid setup token")
        workspace = str(body.get("workspace") or "default").strip()
        if workspace in users.workspaces():
            raise HTTPException(status_code=409, detail="workspace already set up")
        user = users.create_user(
            str(body.get("username", "")), str(body.get("password", "")), workspace, "admin",
            created_by="setup",
        )
        set_session_cookie(response, users.create_session(user["username"]))
        service.events.record(workspace, "workspace_setup", user=user["username"])
        return {"user": users.public(user)}

    @app.post("/api/auth/login")
    async def login(request: Request, response: Response) -> dict:
        if auth_mode != "local":
            raise HTTPException(status_code=409, detail=f"auth mode is '{auth_mode}'")
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        user = users.authenticate(str(body.get("username", "")), str(body.get("password", "")))
        set_session_cookie(response, users.create_session(user["username"]))
        return {"user": users.public(user)}

    @app.post("/api/auth/logout")
    async def logout(request: Request, response: Response) -> dict:
        users.revoke_session(request.cookies.get(SESSION_COOKIE))
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/auth/me", dependencies=dep)
    def me(request: Request, who: Principal = Who) -> dict:
        data = auth_status(request)
        if who.username:
            data["tokens"] = users.list_tokens(who.username)
        return data

    @app.post("/api/auth/password", dependencies=dep)
    async def change_password(request: Request, response: Response, who: Principal = Who) -> dict:
        if not who.username:
            raise HTTPException(status_code=400, detail="no user account for this credential")
        body = await request.json()
        users.authenticate(who.username, str(body.get("current", "")))
        users.set_password(who.username, str(body.get("new", "")))
        set_session_cookie(response, users.create_session(who.username))
        return {"ok": True}

    @app.get("/api/auth/users", dependencies=admin)
    def list_users(owner: str = Owner) -> dict:
        return {"users": users.list_users(owner)}

    @app.post("/api/auth/users", dependencies=admin, status_code=201)
    async def create_user(request: Request, who: Principal = Who) -> dict:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        service.check_members(who.owner, len(users.list_users(who.owner)))
        user = users.create_user(
            str(body.get("username", "")), str(body.get("password", "")), who.owner,
            str(body.get("role", "editor")), created_by=who.username or who.via,
        )
        return {"user": users.public(user)}

    @app.patch("/api/auth/users/{username}", dependencies=admin)
    async def update_user(username: str, request: Request, who: Principal = Who) -> dict:
        body = await request.json()
        if "role" in body:
            users.set_role(username, who.owner, str(body["role"]))
        if body.get("password"):
            target = users.get(username)
            if target is None or target["owner"] != who.owner:
                raise HTTPException(status_code=404, detail="user not found")
            users.set_password(username, str(body["password"]))
        target = users.get(username)
        if target is None:
            raise HTTPException(status_code=404, detail="user not found")
        return {"user": users.public(target)}

    @app.delete("/api/auth/users/{username}", dependencies=admin, status_code=204)
    def delete_user(username: str, who: Principal = Who) -> Response:
        if who.username and username.lower() == who.username.lower():
            raise HTTPException(status_code=409, detail="cannot delete yourself")
        users.delete_user(username, who.owner)
        return Response(status_code=204)

    @app.post("/api/auth/tokens", dependencies=dep, status_code=201)
    async def create_token(request: Request, who: Principal = Who) -> dict:
        if not who.username:
            raise HTTPException(status_code=400, detail="no user account for this credential")
        body = await request.json()
        raw, record = users.create_token(who.username, str(body.get("name", "token")))
        return {"token": raw, "record": record}

    @app.delete("/api/auth/tokens/{token_id}", dependencies=dep, status_code=204)
    def revoke_token(token_id: str, who: Principal = Who) -> Response:
        if not who.username:
            raise HTTPException(status_code=400, detail="no user account for this credential")
        users.revoke_token(who.username, token_id)
        return Response(status_code=204)

    # ---- meta --------------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        from ..quality.evaluate import environment_fingerprint

        return {
            "status": "ok",
            "version": app.version,
            "environment": environment_fingerprint(),
            "auth": auth_mode, "offline": offline,
            "setup_required": auth_mode == "local" and users.setup_required(),
            "roles": sorted({role for _o, role in key_principals.values()}) or ["admin"],
            "rate_limit": limiter.describe(),
            "retention_days": retention_days,
            "limits": {
                "max_target_side": Config.MAX_IMAGE_SIZE,
                "max_upload_bytes": MAX_UPLOAD_BYTES,
            },
        }

    @app.get("/api/usage", dependencies=dep)
    def usage(owner: str = Owner) -> dict:
        members = len(users.list_users(owner)) if auth_mode == "local" else None
        return service.usage_summary(owner, members=members)

    @app.get("/api/plans", dependencies=dep)
    def plans(owner: str = Owner) -> dict:
        ent = service.entitlements
        return {"plans": ent.plans, "default": ent.default, "workspace": owner,
                "current": ent.plan_name(owner)}

    @app.put("/api/plans/{workspace}")
    async def set_plan(workspace: str, request: Request) -> dict:
        """Operator action: assign a plan to a workspace (needs the setup token)."""
        if auth_mode != "local":
            raise HTTPException(status_code=409, detail="plans are assigned in local auth mode "
                                "(or with AUTOBANNER_WORKSPACE_PLANS)")
        given = request.headers.get("X-Setup-Token", "")
        if not given or not hmac.compare_digest(given, setup_token):
            raise HTTPException(status_code=403, detail="setup token required")
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("plan"), str):
            raise HTTPException(status_code=400, detail="body needs a plan name")
        name = service.entitlements.set_plan(workspace, body["plan"])
        service.events.record(workspace, "plan_set", plan=name)
        return {"workspace": workspace, "plan": name,
                "limits": service.entitlements.limits(workspace)}

    @app.get("/api/pilot/summary", dependencies=dep)
    def pilot_summary(owner: str = Owner) -> dict:
        """Local pilot instruments: first-pass acceptance, time to decision, corrections."""
        return service.pilot_summary(owner)

    @app.get("/api/presets", dependencies=dep)
    def presets() -> dict:
        return preset_catalog()

    @app.get("/api/fonts", dependencies=dep)
    def fonts() -> dict:
        return {"families": service.registry.families}

    # ---- projects ------------------------------------------------------------------------
    @app.get("/api/projects", dependencies=dep)
    def list_projects(owner: str = Owner) -> dict:
        return {"projects": service.list_projects(owner)}

    @app.post("/api/projects", dependencies=edit, status_code=201)
    async def create_project(
        file: UploadFile = _UPLOAD,
        name: str | None = Form(default=None),
        brand: str | None = Form(default=None),
        owner: str = Owner,
    ) -> dict:
        data = await read_limited(file, MAX_UPLOAD_BYTES)
        project = service.create_project_from_upload(
            file.filename or "upload.png", data, name=name, brand=brand, owner=owner
        )
        return service.project_payload(project)

    @app.post("/api/projects/blank", dependencies=edit, status_code=201)
    async def create_blank(request: Request, owner: str = Owner) -> dict:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        project = service.create_blank_project(
            name=str(body.get("name") or "Untitled"),
            width=int(body.get("width", 1200)),
            height=int(body.get("height", 628)),
            background=str(body.get("background", "#ffffff")),
            brand=body.get("brand"),
            owner=owner,
        )
        return service.project_payload(project)

    @app.post("/api/projects/{project_id}/elements", dependencies=edit, status_code=201)
    async def add_element(
        project_id: str,
        kind: str = Form(...),
        name: str = Form(default=""),
        role: str = Form(default="unknown"),
        x: float = Form(default=0),
        y: float = Form(default=0),
        width: float = Form(default=100),
        height: float = Form(default=50),
        text: str | None = Form(default=None),
        style: str | None = Form(default=None),
        file: UploadFile | None = None,
        owner: str = Owner,
    ) -> dict:
        import json as _json

        style_dict = None
        if style:
            try:
                style_dict = _json.loads(style)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="style must be JSON") from exc
        data = await read_limited(file, MAX_UPLOAD_BYTES) if file is not None else None
        project = service.add_element(
            project_id,
            owner=owner,
            kind=kind,
            name=name,
            role=role,
            geometry={"x": x, "y": y, "width": width, "height": height},
            text=text,
            style=style_dict,
            image_bytes=data,
            image_name=(file.filename if file is not None else "") or "",
        )
        return service.project_payload(project)

    @app.post("/api/projects/import", dependencies=edit, status_code=201)
    async def import_project(
        file: UploadFile = _UPLOAD, owner: str = Owner, role: str = Role
    ) -> dict:
        data = await read_limited(file, MAX_UPLOAD_BYTES * 4)
        project = service.import_project(data, owner=owner, role=role)
        return service.project_payload(project)

    @app.get("/api/projects/{project_id}", dependencies=dep)
    def get_project(project_id: str, owner: str = Owner) -> dict:
        return service.project_payload(service.get_project(project_id, owner))

    @app.get("/api/projects/{project_id}/learned", dependencies=dep)
    def learned_rules(project_id: str, owner: str = Owner) -> dict:
        return service.learned_rules(project_id, owner)

    @app.get("/api/brands/{brand}/rules", dependencies=dep)
    def brand_rules(brand: str, owner: str = Owner) -> dict:
        return {"brand": brand, "rules": service.brand_rules(owner, brand)}

    @app.get("/api/brands", dependencies=dep)
    def list_brands(owner: str = Owner) -> dict:
        return {"brands": service.list_brand_profiles(owner)}

    @app.get("/api/brands/{brand}", dependencies=dep)
    def get_brand(brand: str, owner: str = Owner) -> dict:
        return {
            "brand": brand,
            "profile": service.get_brand_profile(owner, brand),
            "rules": service.brand_rules(owner, brand),
        }

    @app.put("/api/brands/{brand}", dependencies=edit)
    async def put_brand(brand: str, request: Request, principal: Principal = Who) -> dict:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        profile = service.save_brand_profile(
            principal.owner, brand, body, by=principal.username or principal.owner
        )
        return {"brand": brand, "profile": profile}

    @app.delete("/api/brands/{brand}", dependencies=edit, status_code=204)
    def delete_brand(brand: str, owner: str = Owner) -> Response:
        service.delete_brand_profile(owner, brand)
        return Response(status_code=204)

    @app.post("/api/projects/{project_id}/learned/corrections/{index}/apply", dependencies=edit)
    def apply_correction(project_id: str, index: int, owner: str = Owner) -> dict:
        return service.apply_correction(project_id, index, owner)

    @app.delete("/api/projects/{project_id}", dependencies=edit, status_code=204)
    def delete_project(project_id: str, owner: str = Owner) -> Response:
        service.delete_project(project_id, owner)
        return Response(status_code=204)

    # ---- document ------------------------------------------------------------------------
    @app.patch("/api/projects/{project_id}/document", dependencies=edit)
    async def patch_document(project_id: str, request: Request, owner: str = Owner) -> dict:
        body = await request.json()
        ops = body.get("ops") if isinstance(body, dict) else None
        label = str(body.get("label", "edit")) if isinstance(body, dict) else "edit"
        project = service.apply_operations(project_id, ops or [], label=label, owner=owner)
        return service.project_payload(project)

    @app.post("/api/projects/{project_id}/undo", dependencies=edit)
    def undo(project_id: str, owner: str = Owner) -> dict:
        changed = service.undo(project_id, owner)
        payload = service.project_payload(service.get_project(project_id, owner))
        payload["undone"] = changed
        return payload

    @app.post("/api/projects/{project_id}/restore/{version}", dependencies=edit)
    def restore(project_id: str, version: int, owner: str = Owner) -> dict:
        service.restore(project_id, version, owner)
        return service.project_payload(service.get_project(project_id, owner))

    @app.get("/api/projects/{project_id}/preview.png", dependencies=dep)
    def preview(project_id: str, max_side: int = 1600, owner: str = Owner) -> Response:
        img = service.master_preview(
            project_id, max_side=max(200, min(4096, max_side)), owner=owner
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/projects/{project_id}/assets/{asset_id}", dependencies=dep)
    def asset(project_id: str, asset_id: str, owner: str = Owner) -> FileResponse:
        return FileResponse(
            service.asset_path(project_id, asset_id, owner), media_type="image/png"
        )

    # ---- variants ------------------------------------------------------------------------
    @app.post("/api/projects/{project_id}/variants", dependencies=edit, status_code=202)
    async def request_variants(
        project_id: str,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        owner: str = Owner,
    ) -> dict:
        spec = await request.json()
        if not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        job = service.request_variants(
            project_id, spec, idempotency_key=idempotency_key, owner=owner
        )
        payload = service.project_payload(service.get_project(project_id, owner))
        return {"job": job.to_dict(), **payload}

    @app.post("/api/projects/{project_id}/campaign/rows", dependencies=edit)
    async def parse_campaign_rows(project_id: str, request: Request, owner: str = Owner) -> dict:
        """Read a pasted CSV/TSV table into campaign rows (no side effects)."""
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("csv"), str):
            raise HTTPException(status_code=400, detail="body needs a csv string")
        if len(body["csv"]) > 2_000_000:
            raise HTTPException(status_code=413, detail="table too large")
        return service.parse_campaign_rows(project_id, body["csv"], owner=owner)

    @app.get("/api/projects/{project_id}/variants", dependencies=dep)
    def list_variants(project_id: str, owner: str = Owner) -> dict:
        project = service.get_project(project_id, owner)
        return {"variants": [v.to_dict() for v in project.variants.values()]}

    @app.get("/api/projects/{project_id}/variants/{variant_id}", dependencies=dep)
    def variant_detail(project_id: str, variant_id: str, owner: str = Owner) -> dict:
        return service.variant_detail(project_id, variant_id, owner)

    @app.get("/api/projects/{project_id}/variants/{variant_id}/image.png", dependencies=dep)
    def variant_image(project_id: str, variant_id: str, owner: str = Owner) -> FileResponse:
        return FileResponse(service.variant_image_path(project_id, variant_id, owner),
                            media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/projects/{project_id}/variants/{variant_id}/approval", dependencies=approve)
    async def approval(
        project_id: str, variant_id: str, request: Request, owner: str = Owner
    ) -> dict:
        body = await request.json()
        rec = service.set_approval(
            project_id,
            variant_id,
            str(body.get("approval", "none")),
            str(body.get("reason", "")),
            owner=owner,
        )
        return {"variant": rec.to_dict()}

    @app.post("/api/projects/{project_id}/variants/refresh", dependencies=edit,
              status_code=202)
    async def refresh_variants(project_id: str, request: Request, owner: str = Owner) -> dict:
        """Re-render only the variants a document change touched (incremental refresh)."""
        body: Any = {}
        if int(request.headers.get("content-length", "0") or 0) > 0:
            body = await request.json()
        keep = bool(body.get("keep_layout", True)) if isinstance(body, dict) else True
        return service.refresh_variants(project_id, keep_layout=keep, owner=owner)

    @app.post("/api/projects/{project_id}/variants/{variant_id}/regenerate", dependencies=edit,
              status_code=202)
    async def regenerate(
        project_id: str, variant_id: str, request: Request, owner: str = Owner
    ) -> dict:
        body: Any = {}
        if int(request.headers.get("content-length", "0") or 0) > 0:
            body = await request.json()
        spec = body if isinstance(body, dict) else None
        job = service.regenerate_variant(project_id, variant_id, spec, owner=owner)
        return {"job": job.to_dict()}

    @app.delete("/api/projects/{project_id}/variants/{variant_id}", dependencies=edit,
                status_code=204)
    def delete_variant(project_id: str, variant_id: str, owner: str = Owner) -> Response:
        service.delete_variant(project_id, variant_id, owner)
        return Response(status_code=204)

    # ---- jobs ------------------------------------------------------------------------------
    @app.get("/api/jobs/{job_id}", dependencies=dep)
    def job_status(job_id: str, owner: str = Owner) -> dict:
        return {"job": service.get_job(job_id, owner).to_dict()}

    @app.post("/api/jobs/{job_id}/cancel", dependencies=edit)
    def cancel_job(job_id: str, owner: str = Owner) -> dict:
        job = service.get_job(job_id, owner)
        ok = service.jobs.cancel(job.id)
        return {"cancelled": ok, "job": job.to_dict()}

    # ---- export --------------------------------------------------------------------------
    @app.get("/api/projects/{project_id}/export", dependencies=dep)
    def export(
        project_id: str,
        fmt: str = Query(default="png", alias="format"),
        only: str = "all",
        quality: int = 90,
        owner: str = Owner,
    ) -> Response:
        data = service.export_deliverables(
            project_id, fmt=fmt, only=only, quality=max(30, min(100, quality)), owner=owner
        )
        disposition = f'attachment; filename="{project_id}_{only}.zip"'
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": disposition},
        )

    @app.get("/api/projects/{project_id}/export/project", dependencies=dep)
    def export_project(project_id: str, owner: str = Owner) -> Response:
        data = service.export_project(project_id, owner)
        disposition = f'attachment; filename="{project_id}.autobanner.zip"'
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": disposition},
        )

    # ---- web UI --------------------------------------------------------------------------
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


ROLES = ("viewer", "editor", "approver", "admin")
SESSION_COOKIE = "ab_session"
MUTATING = ("POST", "PATCH", "PUT", "DELETE")


def _auth_mode(users: UserStore, key_principals: dict) -> str:
    """``AUTOBANNER_AUTH`` wins; otherwise local once a user exists, keys, else open."""
    raw = os.environ.get("AUTOBANNER_AUTH", "").strip().lower()
    if raw in ("local", "api_key", "open"):
        return raw
    if raw:
        raise ValueError("AUTOBANNER_AUTH must be local, api_key or open")
    if users.count():
        return "local"
    return "api_key" if key_principals else "open"


def _key_owner_map() -> dict[str, tuple[str, str]]:
    """Parse AUTOBANNER_API_KEYS ("key:owner[:role],...") and AUTOBANNER_API_KEY.

    Roles: ``viewer`` (read only), ``editor`` (edit, generate, export; no approvals),
    ``approver`` (approve/reject only), ``admin`` (everything, the default).
    """
    mapping: dict[str, tuple[str, str]] = {}
    multi = os.environ.get("AUTOBANNER_API_KEYS", "").strip()
    if multi:
        for entry in multi.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = [x.strip() for x in entry.split(":")]
            key = parts[0]
            owner = (parts[1] if len(parts) > 1 and parts[1] else "default")[:64]
            role = parts[2].lower() if len(parts) > 2 and parts[2] else "admin"
            if role not in ROLES:
                raise ValueError(f"unknown role '{role}' in AUTOBANNER_API_KEYS (use {ROLES})")
            if key:
                mapping[key] = (owner, role)
    single = os.environ.get("AUTOBANNER_API_KEY", "").strip()
    if single:
        mapping.setdefault(single, ("default", "admin"))
    return mapping


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


app = create_app()
