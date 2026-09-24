"""FastAPI application: REST API + the bundled web studio.

Note: no ``from __future__ import annotations`` here - FastAPI resolves the
``Annotated[..., Depends(...)]`` hints of the nested endpoint functions at
runtime and cannot see closure variables through string annotations.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..config import Config
from ..exceptions import AutoBannerError
from ..export import ExportOptions
from ..presets import PACK_DESCRIPTIONS, PACKS, list_presets, resolve_targets
from ..service import ANCHOR_PRESETS, MODES, RenderRequest, RenderService
from ..validators import safe_filename, validate_upload
from .errors import ApiError
from .guard import UploadGuard
from .jobs import SUCCEEDED, Job, JobManager
from .metrics import Metrics
from .security import ApiKeyAuth, RateLimiter
from .settings import Settings

logger = logging.getLogger("autobanner.api")

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
_CHUNK = 1 << 20


class RenderOptions(BaseModel):
    """Options accepted by the render endpoints (sent as the ``options`` JSON form field)."""

    model_config = {"extra": "forbid"}

    presets: list[str] = Field(default_factory=list, description="Preset ids, see /v1/presets")
    packs: list[str] = Field(default_factory=list, description="Pack ids, e.g. google-display")
    sizes: list[str] = Field(default_factory=list, description='Custom sizes, e.g. "1200x628"')
    mode: Literal["phase21", "phase3"] = "phase21"
    format: Literal["png", "jpeg", "webp"] = "png"
    quality: int = Field(90, ge=1, le=100)
    max_kb: int | None = Field(None, gt=0, description="Override per-file size budget (KB)")
    respect_platform_limits: bool = True
    enforce_safe_zones: bool = True
    anchor_preset: Literal["none", "flat_banner_3anchors"] = "none"
    anchors: list[dict[str, Any]] | None = None
    role_overrides: dict[str, str] = Field(default_factory=dict)
    auto_layers: bool = Field(
        False, description="Flat images: detect headline/CTA/logo/hero as movable layers"
    )

    def to_request(self) -> RenderRequest:
        targets = resolve_targets(presets=self.presets, packs=self.packs, sizes=self.sizes)
        return RenderRequest(
            targets=targets,
            mode=self.mode,
            export=ExportOptions(format=self.format, quality=self.quality, max_kb=self.max_kb),
            respect_platform_limits=self.respect_platform_limits,
            manual_anchors=self.anchors,
            anchor_preset=self.anchor_preset,
            role_overrides=dict(self.role_overrides),
            enforce_safe_zones=self.enforce_safe_zones,
            auto_layers=self.auto_layers,
        )


def _parse_options(raw: str | None) -> RenderOptions:
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise ApiError(422, "invalid_options", f"options is not valid JSON: {e.msg}") from e
    if not isinstance(data, dict):
        raise ApiError(422, "invalid_options", "options must be a JSON object")
    try:
        return RenderOptions.model_validate(data)
    except PydanticValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "options"
        raise ApiError(422, "invalid_options", f"{loc}: {first.get('msg')}") from e


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application."""
    settings = settings or Settings.from_env()
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024

    metrics = Metrics()
    service = RenderService(use_ai=settings.use_ai, max_upload_bytes=max_upload_bytes)
    jobs = JobManager(
        service,
        metrics,
        workers=settings.workers,
        max_queue=settings.max_queue,
        ttl_seconds=settings.job_ttl_seconds,
        max_retained=settings.max_jobs_retained,
    )
    auth = ApiKeyAuth(settings.api_keys)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    data_dir = Path(settings.data_dir) if settings.data_dir else Path(tempfile.gettempdir())
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="autobanner-", dir=data_dir))

    if not auth.enabled:
        logger.warning(
            "AUTOBANNER_API_KEYS is not set: the API is open to anyone who can reach it."
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        yield
        jobs.shutdown()
        shutil.rmtree(tmp_root, ignore_errors=True)

    app = FastAPI(
        title="AutoBanner API",
        version=__version__,
        description=(
            "Turn one master banner (PSD/PNG/JPG/WEBP) into every ad and social size, "
            "with brand elements preserved, platform safe zones respected and file-size "
            "budgets met."
        ),
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.metrics = metrics

    # Outermost: reject unauthenticated, rate-limited or oversized uploads
    # before a single body byte is buffered.
    app.add_middleware(UploadGuard, settings=settings, auth=auth, limiter=limiter)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )

    # -- middleware / error handling --------------------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if 0 < len(incoming) <= 64 and incoming.isascii() else ""
        request.state.request_id = request_id or uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled errors bypass exception handlers at this layer; build the
            # 500 here so it still carries the request id, headers and metrics.
            logger.exception("Unhandled error rid=%s", request.state.request_id)
            err = ApiError(500, "internal_error", "Internal server error")
            response = JSONResponse(err.body(request.state.request_id), status_code=500)
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.inc(
            "autobanner_http_requests_total", "HTTP requests",
            method=request.method, route=path, status=str(response.status_code),
        )
        metrics.observe("autobanner_http_request_duration_seconds", elapsed, "HTTP latency")
        logger.info(
            "%s %s %s %.0fms rid=%s",
            request.method, request.url.path, response.status_code, elapsed * 1000,
            request.state.request_id,
        )
        return response

    def _rid(request: Request) -> str | None:
        return getattr(request.state, "request_id", None)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body(_rid(request)), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(AutoBannerError)
    async def domain_error_handler(request: Request, exc: AutoBannerError) -> JSONResponse:
        err = ApiError(422, "invalid_input", str(exc))
        return JSONResponse(err.body(_rid(request)), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}.get(
            exc.status_code, "http_error"
        )
        err = ApiError(exc.status_code, code, str(exc.detail))
        return JSONResponse(
            err.body(_rid(request)), status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()))
        err = ApiError(422, "invalid_request", f"{loc}: {first.get('msg', 'invalid')}")
        return JSONResponse(err.body(_rid(request)), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error rid=%s", _rid(request))
        err = ApiError(500, "internal_error", "Internal server error")
        return JSONResponse(err.body(_rid(request)), status_code=500)

    # -- dependencies ---------------------------------------------------------------------
    def caller(request: Request) -> str | None:
        """Authenticate; returns the caller's key id (None when auth is disabled)."""
        return auth.identify(request)

    # Rate limiting of upload endpoints happens in UploadGuard (before the body
    # is read); the dependency only resolves the caller's identity.
    limited_caller = caller

    async def save_upload(upload: UploadFile) -> tuple[Path, Path, str]:
        """Stream an upload to a private temp dir enforcing the size limit."""
        original = upload.filename or "upload"
        suffix = Path(original).suffix.lower()
        if suffix not in (".psd", ".png", ".jpg", ".jpeg", ".webp"):
            raise ApiError(
                415, "unsupported_format", "Upload a PSD, PNG, JPG/JPEG or WEBP file"
            )
        workdir = Path(tempfile.mkdtemp(dir=tmp_root))
        dest = workdir / f"source{suffix}"
        limit = max_upload_bytes
        written = 0
        try:
            with dest.open("wb") as fh:
                while chunk := await upload.read(_CHUNK):
                    written += len(chunk)
                    if written > limit:
                        raise ApiError(
                            413, "payload_too_large",
                            f"File exceeds the {limit // 1048576} MB upload limit",
                        )
                    fh.write(chunk)
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        display = safe_filename(Path(original).stem, "design") + suffix
        return dest, workdir, display

    # -- public endpoints -------------------------------------------------------------------
    @app.get("/healthz", tags=["ops"], summary="Liveness probe")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"], summary="Readiness probe")
    def readyz() -> JSONResponse:
        ready = jobs.accepting
        return JSONResponse(
            {"status": "ready" if ready else "busy", "pending_jobs": jobs.pending},
            status_code=200 if ready else 503,
        )

    @app.get("/metrics", tags=["ops"], summary="Prometheus metrics",
             response_class=PlainTextResponse)
    def metrics_endpoint(_: Annotated[str | None, Depends(caller)]) -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/config", tags=["meta"], summary="Public server capabilities")
    def public_config() -> dict[str, Any]:
        return {
            "version": __version__,
            "auth_required": auth.enabled,
            "modes": list(MODES),
            "formats": ["png", "jpeg", "webp"],
            "anchor_presets": list(ANCHOR_PRESETS),
            "limits": {
                "max_upload_mb": settings.max_upload_mb,
                "max_targets_per_job": Config.MAX_TARGETS_PER_JOB,
                "max_target_dimension": Config.MAX_IMAGE_SIZE,
                "max_source_dimension": Config.MAX_SOURCE_DIMENSION,
            },
        }

    @app.get("/v1/presets", tags=["catalog"], summary="List output sizes and packs")
    def presets(platform: str | None = None) -> dict[str, Any]:
        return {
            "presets": [p.to_dict() for p in list_presets(platform)],
            "packs": {
                pid: {"description": PACK_DESCRIPTIONS.get(pid, ""), "presets": list(ids)}
                for pid, ids in PACKS.items()
            },
        }

    # -- rendering ------------------------------------------------------------------------
    @app.post("/v1/analyze", tags=["render"], summary="Detect layers and their roles")
    async def analyze(
        owner: Annotated[str | None, Depends(limited_caller)],
        file: Annotated[UploadFile, File(description="PSD, PNG, JPG or WEBP")],
        auto_layers: bool = False,
    ) -> dict[str, Any]:
        path, workdir, display = await save_upload(file)
        try:
            future = jobs.submit_sync(
                lambda: service.analyze(path, display_name=display, auto_layers=auto_layers)
            )
            return await asyncio.wrap_future(future)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.post(
        "/v1/render",
        tags=["render"],
        summary="Render synchronously and return a ZIP",
        response_class=Response,
        responses={200: {"content": {"application/zip": {}}}},
    )
    async def render_sync(
        owner: Annotated[str | None, Depends(limited_caller)],
        file: Annotated[UploadFile, File(description="PSD, PNG, JPG or WEBP")],
        options: Annotated[str | None, Form(description="RenderOptions as JSON")] = None,
    ) -> Response:
        opts = _parse_options(options)
        request = opts.to_request()
        request.validate()
        path, workdir, display = await save_upload(file)
        try:
            future = jobs.submit_sync(
                lambda: service.render(path, request, display_name=display)
            )
            report = await asyncio.wrap_future(future)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        stem = safe_filename(Path(display).stem, "design")
        summary = report.manifest()["summary"]
        return Response(
            await run_in_threadpool(report.to_zip),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}_autobanner.zip"',
                "X-AutoBanner-Succeeded": str(summary["succeeded"]),
                "X-AutoBanner-Failed": str(summary["failed"]),
            },
        )

    @app.post("/v1/jobs", tags=["jobs"], status_code=202, summary="Queue an async render job")
    async def create_job(
        owner: Annotated[str | None, Depends(limited_caller)],
        file: Annotated[UploadFile, File(description="PSD, PNG, JPG or WEBP")],
        options: Annotated[str | None, Form(description="RenderOptions as JSON")] = None,
    ) -> dict[str, Any]:
        opts = _parse_options(options)
        request = opts.to_request()
        request.validate()
        path, workdir, display = await save_upload(file)
        try:
            # Validate before queueing so bad files fail fast with a 422.
            await run_in_threadpool(validate_upload, str(path), max_upload_bytes)
            job = jobs.submit(
                owner=owner, source_path=path, source_name=display,
                workdir=workdir, request=request,
            )
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        return job.describe()

    @app.get("/v1/jobs/{job_id}", tags=["jobs"], summary="Job status and manifest")
    def get_job(job_id: str, owner: Annotated[str | None, Depends(caller)]) -> dict[str, Any]:
        return jobs.get(job_id, owner).describe()

    @app.delete("/v1/jobs/{job_id}", tags=["jobs"], status_code=204, summary="Delete a job")
    def delete_job(job_id: str, owner: Annotated[str | None, Depends(caller)]) -> Response:
        jobs.delete(job_id, owner)
        return Response(status_code=204)

    def _finished(job_id: str, owner: str | None) -> Job:
        job = jobs.get(job_id, owner)
        if job.status != SUCCEEDED or job.zip_path is None:
            raise ApiError(409, "not_ready", f"Job is {job.status}")
        return job

    @app.get(
        "/v1/jobs/{job_id}/download",
        tags=["jobs"],
        summary="Download all outputs as ZIP",
        response_class=FileResponse,
        responses={200: {"content": {"application/zip": {}}}},
    )
    def download_job(job_id: str, owner: Annotated[str | None, Depends(caller)]) -> Response:
        job = _finished(job_id, owner)
        stem = safe_filename(Path(job.source_name).stem, "design")
        return FileResponse(
            job.zip_path,
            media_type="application/zip",
            filename=f"{stem}_autobanner.zip",
        )

    @app.get(
        "/v1/jobs/{job_id}/assets/{filename}",
        tags=["jobs"],
        summary="Download one output image",
        response_class=FileResponse,
    )
    def job_asset(
        job_id: str, filename: str, owner: Annotated[str | None, Depends(caller)]
    ) -> Response:
        job = _finished(job_id, owner)
        stored = job.asset_files.get(filename)
        if stored is None:
            raise ApiError(404, "not_found", "Asset not found")
        path, mime = stored
        return FileResponse(
            path,
            media_type=mime,
            content_disposition_type="inline",
            filename=filename,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    # -- studio -----------------------------------------------------------------------------
    if settings.enable_studio and STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def studio() -> FileResponse:
            return FileResponse(
                STATIC_DIR / "index.html",
                headers={
                    "Cache-Control": "no-cache",
                    "Content-Security-Policy": (
                        "default-src 'self'; img-src 'self' blob: data:; "
                        "style-src 'self' 'unsafe-inline'; script-src 'self'; "
                        "connect-src 'self'; frame-ancestors 'none'"
                    ),
                },
            )

    return app

