"""In-process job runner with progress, cancellation and partial completion.

Jobs run on a small thread pool. Each job owns a cancellation event; a
cancelled or failed variant never erases other variants' outputs. Job state is
also mirrored into the project's variant records so it survives restarts as
"interrupted" rather than silently pending.
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..design.document import new_id, utc_now

logger = logging.getLogger("autobanner.api.jobs")


@dataclass
class JobItem:
    item_id: str
    label: str
    status: str = "pending"  # pending | running | done | failed | cancelled
    progress: float = 0.0
    stage: str = ""
    error: str | None = None
    result: dict = field(default_factory=dict)


@dataclass
class Job:
    id: str
    kind: str
    project_id: str
    items: list[JobItem]
    status: str = "pending"  # pending | running | done | failed | cancelled | partial
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    idempotency_key: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "project_id": self.project_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "idempotency_key": self.idempotency_key,
            "error": self.error,
            "progress": (
                sum(i.progress for i in self.items) / len(self.items) if self.items else 0.0
            ),
            "items": [
                {
                    "item_id": i.item_id,
                    "label": i.label,
                    "status": i.status,
                    "progress": round(i.progress, 3),
                    "stage": i.stage,
                    "error": i.error,
                    "result": i.result,
                }
                for i in self.items
            ],
        }


ItemRunner = Callable[[Job, JobItem, Callable[[str, float], None]], dict]


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="autobanner-job"
        )
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def find_by_key(self, idempotency_key: str | None) -> Job | None:
        if not idempotency_key:
            return None
        with self._lock:
            job_id = self._by_key.get(idempotency_key)
        return self._jobs.get(job_id) if job_id else None

    def list(self, project_id: str | None = None) -> list[Job]:
        jobs = list(self._jobs.values())
        if project_id:
            jobs = [j for j in jobs if j.project_id == project_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def submit(
        self,
        kind: str,
        project_id: str,
        items: list[tuple[str, str]],
        runner: ItemRunner,
        *,
        idempotency_key: str | None = None,
        on_finish: Callable[[Job], None] | None = None,
    ) -> Job:
        with self._lock:
            if idempotency_key and idempotency_key in self._by_key:
                existing = self._jobs[self._by_key[idempotency_key]]
                return existing
            job = Job(
                id=new_id("job"),
                kind=kind,
                project_id=project_id,
                items=[JobItem(item_id=i, label=label) for i, label in items],
                idempotency_key=idempotency_key,
            )
            self._jobs[job.id] = job
            if idempotency_key:
                self._by_key[idempotency_key] = job.id
            self._futures[job.id] = self._pool.submit(self._run, job, runner, on_finish)
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("done", "failed", "cancelled", "partial"):
            return False
        job.cancel.set()
        return True

    def _run(self, job: Job, runner: ItemRunner, on_finish: Callable[[Job], None] | None) -> None:
        job.status = "running"
        job.started_at = utc_now()
        any_done = any_failed = any_cancelled = False
        for item in job.items:
            if job.cancel.is_set():
                item.status = "cancelled"
                any_cancelled = True
                continue
            item.status = "running"

            def progress(stage: str, frac: float, _item: JobItem = item) -> None:
                _item.stage = stage
                _item.progress = max(_item.progress, min(1.0, float(frac)))

            try:
                item.result = runner(job, item, progress)
                item.status = "done"
                item.progress = 1.0
                any_done = True
            except Exception as exc:  # noqa: BLE001
                if job.cancel.is_set() or type(exc).__name__ == "CancelledError":
                    item.status = "cancelled"
                    any_cancelled = True
                else:
                    item.status = "failed"
                    item.error = f"{type(exc).__name__}: {exc}"
                    logger.error("job %s item %s failed: %s", job.id, item.item_id, exc)
                    logger.debug(traceback.format_exc())
                    any_failed = True
        if any_cancelled and not any_done:
            job.status = "cancelled"
        elif any_failed and not any_done:
            job.status = "failed"
        elif any_failed or any_cancelled:
            job.status = "partial"
        else:
            job.status = "done"
        job.finished_at = utc_now()
        if on_finish:
            try:
                on_finish(job)
            except Exception as exc:  # noqa: BLE001
                logger.error("job %s on_finish failed: %s", job.id, exc)

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        fut = self._futures.get(job_id)
        if fut is not None:
            fut.result(timeout=timeout)
        return self._jobs.get(job_id)

    def shutdown(self) -> None:
        for job in self._jobs.values():
            job.cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)


def as_dict(job: Job | None) -> dict[str, Any] | None:
    return job.to_dict() if job else None
