"""FastAPI application exposing the project journey and serving the web UI.

Run locally::

    AUTOBANNER_DATA_DIR=./data uvicorn backend.app.api.server:app --port 8000

Authentication and ownership:

- ``AUTOBANNER_API_KEYS="key1:owner-a,key2:owner-b"`` maps API keys to owner
  ids; every ``/api`` request must send a known key as ``X-API-Key`` and only
  sees projects and jobs of its owner (others read as 404).
- ``AUTOBANNER_API_KEY=key`` is the single-owner form (owner ``default``).
- Without either, the server is open and everything belongs to owner ``local``
  (development only).
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..logging_config import setup_logging
from .presets import preset_catalog
from .ratelimit import RateLimiter, parse_rate
from .service import LOCAL_OWNER, MAX_UPLOAD_BYTES, ProjectService, ServiceError, env_flag

logger = logging.getLogger("autobanner.api.server")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_UPLOAD = File(...)


def create_app(data_dir: str | Path | None = None, *, max_workers: int | None = None) -> FastAPI:
    setup_logging(os.environ.get("AUTOBANNER_LOG_LEVEL", "INFO"))
    data_root = Path(data_dir or os.environ.get("AUTOBANNER_DATA_DIR", "data"))
    workers = int(max_workers or os.environ.get("AUTOBANNER_JOB_WORKERS", "2"))
    service = ProjectService(
        data_root, max_workers=workers, use_ai=env_flag("AUTOBANNER_USE_AI", False)
    )
    key_principals = _key_owner_map()
    key_owners = {k: owner for k, (owner, _role) in key_principals.items()}
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

    async def current_owner(x_api_key: str | None = Header(default=None)) -> str:
        if not key_owners:
            return LOCAL_OWNER
        owner = key_owners.get(x_api_key or "")
        if owner is None:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return owner

    async def current_role(x_api_key: str | None = Header(default=None)) -> str:
        if not key_principals:
            return "admin"
        principal = key_principals.get(x_api_key or "")
        if principal is None:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return principal[1]

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
    Owner = Depends(current_owner)  # noqa: N806

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if limiter.enabled and request.method in ("POST", "PATCH", "PUT", "DELETE"):
            key = request.headers.get("x-api-key") or ""
            owner = key_owners.get(key, LOCAL_OWNER if not key_owners else "anonymous")
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

    # ---- meta --------------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        from ..quality.evaluate import environment_fingerprint

        return {
            "status": "ok",
            "version": app.version,
            "environment": environment_fingerprint(),
            "auth": "api_key" if key_owners else "open",
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
        return service.usage_summary(owner)

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
    async def import_project(file: UploadFile = _UPLOAD, owner: str = Owner) -> dict:
        data = await read_limited(file, MAX_UPLOAD_BYTES * 4)
        project = service.import_project(data, owner=owner)
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
