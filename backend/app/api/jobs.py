"""Bounded background job execution with TTL-based retention."""

from __future__ import annotations

import logging
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from ..exceptions import AutoBannerError
from ..service import RenderReport, RenderRequest, RenderService
from .errors import ApiError
from .metrics import Metrics

logger = logging.getLogger("autobanner.api.jobs")

T = TypeVar("T")

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"


@dataclass
class Job:
    id: str
    owner: str | None
    source_name: str
    workdir: Path
    request: RenderRequest
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress_done: int = 0
    progress_total: int = 0
    current: str = ""
    error: dict[str, str] | None = None
    # Results live on disk (workdir), not in memory: only the manifest is kept.
    manifest: dict[str, Any] | None = None
    zip_path: Path | None = None
    asset_files: dict[str, tuple[Path, str]] = field(default_factory=dict)

    def persist(self, report: RenderReport) -> None:
        """Write outputs + ZIP into the job's workdir and keep only metadata."""
        assets_dir = self.workdir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for asset in report.succeeded:
            assert asset.encoded is not None
            path = assets_dir / asset.filename
            path.write_bytes(asset.encoded.data)
            self.asset_files[asset.filename] = (path, asset.encoded.mime_type)
        zip_path = self.workdir / "result.zip"
        report.write_zip(zip_path)
        self.manifest = report.manifest()
        self.zip_path = zip_path

    def describe(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "source": self.source_name,
            "mode": self.request.mode,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {
                "done": self.progress_done,
                "total": self.progress_total,
                "current": self.current,
            },
            "error": self.error,
        }
        if self.manifest is not None:
            data["manifest"] = self.manifest
            data["links"] = {
                "download": f"/v1/jobs/{self.id}/download",
                "assets": {
                    name: f"/v1/jobs/{self.id}/assets/{name}" for name in self.asset_files
                },
            }
        return data


class JobManager:
    """Runs render jobs on a bounded thread pool.

    * At most ``workers`` renders execute concurrently (CPU bound).
    * At most ``max_queue`` jobs may be waiting; beyond that callers get 503
      so a traffic spike degrades gracefully instead of exhausting memory.
    * Finished jobs (and their temp files) are purged after ``ttl`` seconds.
    """

    def __init__(
        self,
        service: RenderService,
        metrics: Metrics,
        workers: int = 2,
        max_queue: int = 32,
        ttl_seconds: int = 3600,
        max_retained: int = 500,
    ) -> None:
        self.service = service
        self.metrics = metrics
        self.workers = workers
        self.max_queue = max_queue
        self.ttl_seconds = ttl_seconds
        self.max_retained = max_retained
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pending = 0
        self._closed = False

    # -- lifecycle ------------------------------------------------------------------------
    def shutdown(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            shutil.rmtree(job.workdir, ignore_errors=True)

    @property
    def accepting(self) -> bool:
        return not self._closed and self._pending < self.max_queue + self.workers

    @property
    def pending(self) -> int:
        return self._pending

    # -- submission -----------------------------------------------------------------------
    def _reserve(self) -> None:
        with self._lock:
            if self._closed:
                raise ApiError(503, "shutting_down", "Server is shutting down")
            if self._pending >= self.max_queue + self.workers:
                raise ApiError(
                    503, "overloaded", "Render queue is full, retry shortly",
                    headers={"Retry-After": "10"},
                )
            self._pending += 1
            self.metrics.set_gauge("autobanner_jobs_pending", self._pending, "Queued+running jobs")

    def _release(self) -> None:
        with self._lock:
            self._pending -= 1
            self.metrics.set_gauge("autobanner_jobs_pending", self._pending, "Queued+running jobs")

    def submit_sync(self, fn: Callable[[], T]) -> Future[T]:
        """Run ``fn`` on the shared pool (bounded concurrency) and return its future."""
        self._reserve()
        try:
            return self._executor.submit(self._wrapped, fn)
        except BaseException:
            self._release()
            raise

    def _wrapped(self, fn: Callable[[], T]) -> T:
        try:
            return fn()
        finally:
            self._release()

    def submit(
        self,
        *,
        owner: str | None,
        source_path: Path,
        source_name: str,
        workdir: Path,
        request: RenderRequest,
    ) -> Job:
        self._purge()
        self._reserve()
        job = Job(
            id=secrets.token_urlsafe(12),
            owner=owner,
            source_name=source_name,
            workdir=workdir,
            request=request,
            progress_total=len(request.targets),
        )
        with self._lock:
            self._jobs[job.id] = job
        self.metrics.inc("autobanner_jobs_submitted_total", "Jobs submitted")
        self._executor.submit(self._run, job, source_path)
        return job

    def _run(self, job: Job, source_path: Path) -> None:
        job.status = RUNNING
        job.started_at = time.time()

        def progress(done: int, total: int, current: str) -> None:
            job.progress_done = done
            job.progress_total = total
            job.current = current

        try:
            report = self.service.render(
                source_path, job.request, display_name=job.source_name, progress=progress
            )
            job.persist(report)
            job.status = SUCCEEDED
        except AutoBannerError as e:
            job.status = FAILED
            job.error = {"code": "invalid_input", "message": str(e)}
        except Exception:
            logger.exception("Job %s crashed", job.id)
            job.status = FAILED
            job.error = {"code": "internal_error", "message": "Rendering failed unexpectedly"}
        finally:
            job.finished_at = time.time()
            self._release()
            # The source is no longer needed once rendering finished.
            source_path.unlink(missing_ok=True)
            elapsed = job.finished_at - (job.started_at or job.finished_at)
            self.metrics.observe(
                "autobanner_job_duration_seconds", elapsed, "Job wall-clock duration"
            )
            self.metrics.inc(
                "autobanner_jobs_finished_total", "Jobs finished", status=job.status
            )

    # -- queries --------------------------------------------------------------------------
    def get(self, job_id: str, owner: str | None) -> Job:
        self._purge()
        with self._lock:
            job = self._jobs.get(job_id)
        # Jobs are only visible to the key that created them; 404 (not 403)
        # so job ids cannot be probed across tenants.
        if job is None or job.owner != owner:
            raise ApiError(404, "not_found", "Job not found or expired")
        return job

    def delete(self, job_id: str, owner: str | None) -> None:
        job = self.get(job_id, owner)
        if job.status in (QUEUED, RUNNING):
            raise ApiError(409, "job_active", "Job is still running")
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(job.workdir, ignore_errors=True)

    def _purge(self) -> None:
        now = time.time()
        expired: list[Job] = []
        with self._lock:
            finished = [
                j for j in self._jobs.values() if j.status in (SUCCEEDED, FAILED)
            ]
            for job in finished:
                if job.finished_at and now - job.finished_at > self.ttl_seconds:
                    expired.append(job)
            overflow = len(self._jobs) - len(expired) - self.max_retained
            if overflow > 0:
                remaining = sorted(
                    (j for j in finished if j not in expired),
                    key=lambda j: j.finished_at or 0,
                )
                expired.extend(remaining[:overflow])
            for job in expired:
                self._jobs.pop(job.id, None)
        for job in expired:
            shutil.rmtree(job.workdir, ignore_errors=True)
