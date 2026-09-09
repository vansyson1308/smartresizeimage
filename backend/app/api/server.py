"""FastAPI application exposing the project journey and serving the web UI.

Run locally::

    AUTOBANNER_DATA_DIR=./data uvicorn backend.app.api.server:app --port 8000

Authentication: when ``AUTOBANNER_API_KEY`` is set, every ``/api`` request must
send it as ``X-API-Key``. Without it the server is open (local development).
Multi-tenant isolation is not implemented yet; one data directory is one
tenant.
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
from .service import ProjectService, ServiceError, env_flag

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
    api_key = os.environ.get("AUTOBANNER_API_KEY", "").strip()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
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

    async def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    dep = [Depends(require_key)]

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
            "auth": "api_key" if api_key else "open",
            "limits": {"max_target_side": Config.MAX_IMAGE_SIZE},
        }

    @app.get("/api/presets", dependencies=dep)
    def presets() -> dict:
        return preset_catalog()

    @app.get("/api/fonts", dependencies=dep)
    def fonts() -> dict:
        return {"families": service.registry.families}

    # ---- projects ------------------------------------------------------------------------
    @app.get("/api/projects", dependencies=dep)
    def list_projects() -> dict:
        return {"projects": service.list_projects()}

    @app.post("/api/projects", dependencies=dep, status_code=201)
    async def create_project(
        file: UploadFile = _UPLOAD,
        name: str | None = Form(default=None),
        brand: str | None = Form(default=None),
    ) -> dict:
        data = await file.read()
        project = service.create_project_from_upload(
            file.filename or "upload.png", data, name=name, brand=brand
        )
        return service.project_payload(project)

    @app.post("/api/projects/blank", dependencies=dep, status_code=201)
    async def create_blank(request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        project = service.create_blank_project(
            name=str(body.get("name") or "Untitled"),
            width=int(body.get("width", 1200)),
            height=int(body.get("height", 628)),
            background=str(body.get("background", "#ffffff")),
            brand=body.get("brand"),
        )
        return service.project_payload(project)

    @app.post("/api/projects/{project_id}/elements", dependencies=dep, status_code=201)
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
    ) -> dict:
        import json as _json

        style_dict = None
        if style:
            try:
                style_dict = _json.loads(style)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="style must be JSON") from exc
        data = await file.read() if file is not None else None
        project = service.add_element(
            project_id,
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

    @app.post("/api/projects/import", dependencies=dep, status_code=201)
    async def import_project(file: UploadFile = _UPLOAD) -> dict:
        data = await file.read()
        project = service.import_project(data)
        return service.project_payload(project)

    @app.get("/api/projects/{project_id}", dependencies=dep)
    def get_project(project_id: str) -> dict:
        return service.project_payload(service.get_project(project_id))

    @app.delete("/api/projects/{project_id}", dependencies=dep, status_code=204)
    def delete_project(project_id: str) -> Response:
        service.delete_project(project_id)
        return Response(status_code=204)

    # ---- document ------------------------------------------------------------------------
    @app.patch("/api/projects/{project_id}/document", dependencies=dep)
    async def patch_document(project_id: str, request: Request) -> dict:
        body = await request.json()
        ops = body.get("ops") if isinstance(body, dict) else None
        label = str(body.get("label", "edit")) if isinstance(body, dict) else "edit"
        project = service.apply_operations(project_id, ops or [], label=label)
        return service.project_payload(project)

    @app.post("/api/projects/{project_id}/undo", dependencies=dep)
    def undo(project_id: str) -> dict:
        changed = service.undo(project_id)
        payload = service.project_payload(service.get_project(project_id))
        payload["undone"] = changed
        return payload

    @app.post("/api/projects/{project_id}/restore/{version}", dependencies=dep)
    def restore(project_id: str, version: int) -> dict:
        service.restore(project_id, version)
        return service.project_payload(service.get_project(project_id))

    @app.get("/api/projects/{project_id}/preview.png", dependencies=dep)
    def preview(project_id: str, max_side: int = 1600) -> Response:
        img = service.master_preview(project_id, max_side=max(200, min(4096, max_side)))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/projects/{project_id}/assets/{asset_id}", dependencies=dep)
    def asset(project_id: str, asset_id: str) -> FileResponse:
        return FileResponse(service.asset_path(project_id, asset_id), media_type="image/png")

    # ---- variants ------------------------------------------------------------------------
    @app.post("/api/projects/{project_id}/variants", dependencies=dep, status_code=202)
    async def request_variants(
        project_id: str,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict:
        spec = await request.json()
        if not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        job = service.request_variants(project_id, spec, idempotency_key=idempotency_key)
        return {"job": job.to_dict(), **service.project_payload(service.get_project(project_id))}

    @app.get("/api/projects/{project_id}/variants", dependencies=dep)
    def list_variants(project_id: str) -> dict:
        project = service.get_project(project_id)
        return {"variants": [v.to_dict() for v in project.variants.values()]}

    @app.get("/api/projects/{project_id}/variants/{variant_id}", dependencies=dep)
    def variant_detail(project_id: str, variant_id: str) -> dict:
        return service.variant_detail(project_id, variant_id)

    @app.get("/api/projects/{project_id}/variants/{variant_id}/image.png", dependencies=dep)
    def variant_image(project_id: str, variant_id: str) -> FileResponse:
        return FileResponse(service.variant_image_path(project_id, variant_id),
                            media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/projects/{project_id}/variants/{variant_id}/approval", dependencies=dep)
    async def approval(project_id: str, variant_id: str, request: Request) -> dict:
        body = await request.json()
        rec = service.set_approval(
            project_id, variant_id, str(body.get("approval", "none")), str(body.get("reason", ""))
        )
        return {"variant": rec.to_dict()}

    @app.post("/api/projects/{project_id}/variants/{variant_id}/regenerate", dependencies=dep,
              status_code=202)
    async def regenerate(project_id: str, variant_id: str, request: Request) -> dict:
        body: Any = {}
        if int(request.headers.get("content-length", "0") or 0) > 0:
            body = await request.json()
        spec = body if isinstance(body, dict) else None
        job = service.regenerate_variant(project_id, variant_id, spec)
        return {"job": job.to_dict()}

    @app.delete("/api/projects/{project_id}/variants/{variant_id}", dependencies=dep,
                status_code=204)
    def delete_variant(project_id: str, variant_id: str) -> Response:
        service.delete_variant(project_id, variant_id)
        return Response(status_code=204)

    # ---- jobs ------------------------------------------------------------------------------
    @app.get("/api/jobs/{job_id}", dependencies=dep)
    def job_status(job_id: str) -> dict:
        job = service.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job": job.to_dict()}

    @app.post("/api/jobs/{job_id}/cancel", dependencies=dep)
    def cancel_job(job_id: str) -> dict:
        ok = service.jobs.cancel(job_id)
        job = service.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"cancelled": ok, "job": job.to_dict()}

    # ---- export --------------------------------------------------------------------------
    @app.get("/api/projects/{project_id}/export", dependencies=dep)
    def export(
        project_id: str,
        fmt: str = Query(default="png", alias="format"),
        only: str = "all",
        quality: int = 90,
    ) -> Response:
        data = service.export_deliverables(
            project_id, fmt=fmt, only=only, quality=max(30, min(100, quality))
        )
        disposition = f'attachment; filename="{project_id}_{only}.zip"'
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": disposition},
        )

    @app.get("/api/projects/{project_id}/export/project", dependencies=dep)
    def export_project(project_id: str) -> Response:
        data = service.export_project(project_id)
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


app = create_app()
