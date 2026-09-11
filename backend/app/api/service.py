"""Project service: the operations behind the HTTP API (also usable in-process).

Keeps HTTP concerns out of the domain logic so tests and CLI tools can drive
the same code path users hit through the web UI.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ..classifier import SemanticClassifier
from ..config import Config
from ..constants import SUPPORTED_EXTENSIONS
from ..design.adapter import document_from_elements, elements_from_document
from ..design.assets import AssetStore
from ..design.brand import (
    collect_brand_rules,
    normalize_profile,
    propose_brand_rules,
    propose_profile_rules,
)
from ..design.corrections import RejectionSnapshot, derive_corrections
from ..design.decompose import decompose_flat_image
from ..design.document import (
    AllowedTransforms,
    Constraint,
    DesignDocument,
    Element,
    Geometry,
    Provenance,
    new_id,
)
from ..design.examples import example_from_plan, infer_families
from ..design.fonts import FontRegistry
from ..design.grammar import direction_key, parse_direction
from ..design.planner import Family, Region, aspect_class, choose_families
from ..design.project import Project, VariantRecord, is_safe_id, safe_relative_path
from ..design.render import render_master
from ..design.serialize import document_from_dict, document_to_dict
from ..design.variant import VariantBrief, generate_variant
from ..parser import get_parser
from ..quality.contract import CheckResult, CheckStatus, Severity, derive_verdict, summarize
from ..quality.family import VariantSnapshot, family_consistency_checks
from .events import EventLog
from .jobs import Job, JobItem, JobManager
from .presets import find_preset

logger = logging.getLogger("autobanner.api.service")

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 5000
MAX_PIXELS = 40_000_000
MAX_VARIANTS_PER_JOB = 144  # e.g. 24 campaign rows x 6 formats
MAX_CAMPAIGN_ROWS = 200
MAX_TARGET_SIDE = Config.MAX_IMAGE_SIZE


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


LOCAL_OWNER = "local"


@dataclass
class Quota:
    """Per-owner limits. ``None`` means unlimited."""

    variants_per_day: int | None = None
    projects: int | None = None
    storage_bytes: int | None = None

    @staticmethod
    def from_env() -> Quota:
        def _int(name: str) -> int | None:
            raw = os.environ.get(name, "").strip()
            return int(raw) if raw.isdigit() else None

        return Quota(
            variants_per_day=_int("AUTOBANNER_QUOTA_VARIANTS_PER_DAY"),
            projects=_int("AUTOBANNER_QUOTA_PROJECTS"),
            storage_bytes=_int("AUTOBANNER_QUOTA_STORAGE_BYTES"),
        )


PLAN_LIMIT_KEYS = ("variants_per_day", "projects", "members", "campaign_rows", "storage_bytes")
DEFAULT_PLANS: dict[str, dict[str, int | None]] = {
    "free": {"variants_per_day": 50, "projects": 3, "members": 2, "campaign_rows": 3,
             "storage_bytes": 512 * 1024 * 1024},
    "team": {"variants_per_day": 1000, "projects": 100, "members": 10, "campaign_rows": 50,
             "storage_bytes": 20 * 1024 ** 3},
    "business": {"variants_per_day": 10000, "projects": 1000, "members": 100,
                 "campaign_rows": 200, "storage_bytes": 200 * 1024 ** 3},
    "unlimited": dict.fromkeys(PLAN_LIMIT_KEYS, None),
}


class Entitlements:
    """Which plan a workspace is on and what that plan allows.

    Plan definitions come from ``DEFAULT_PLANS`` merged with ``AUTOBANNER_PLANS`` (JSON);
    assignments from ``data/plans.json`` (set by the operator through the API with the
    setup token) overridden by ``AUTOBANNER_WORKSPACE_PLANS`` (``acme:team,globex:free``).
    Unassigned workspaces get ``AUTOBANNER_DEFAULT_PLAN`` (default ``unlimited`` so an
    existing deployment gains no new limits by surprise). Global ``AUTOBANNER_QUOTA_*``
    caps still apply on top: the stricter limit wins.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.plans: dict[str, dict[str, int | None]] = {
            k: dict(v) for k, v in DEFAULT_PLANS.items()
        }
        raw = os.environ.get("AUTOBANNER_PLANS", "").strip()
        if raw:
            try:
                for name, limits in json.loads(raw).items():
                    plan = dict(self.plans.get(name, dict.fromkeys(PLAN_LIMIT_KEYS, None)))
                    for key in PLAN_LIMIT_KEYS:
                        if key in limits:
                            plan[key] = None if limits[key] is None else int(limits[key])
                    self.plans[str(name)] = plan
            except (ValueError, TypeError, AttributeError) as exc:
                logger.warning("AUTOBANNER_PLANS ignored: %s", exc)
        self.default = os.environ.get("AUTOBANNER_DEFAULT_PLAN", "unlimited").strip() or "unlimited"
        if self.default not in self.plans:
            logger.warning("AUTOBANNER_DEFAULT_PLAN %r unknown; using unlimited", self.default)
            self.default = "unlimited"
        self._assigned: dict[str, str] = {}
        if self.path.exists():
            try:
                self._assigned = {
                    str(k): str(v)
                    for k, v in (json.loads(self.path.read_text(encoding="utf-8"))
                                 .get("workspaces") or {}).items()
                }
            except ValueError as exc:
                logger.warning("unreadable %s: %s", self.path, exc)
        self._env_assigned: dict[str, str] = {}
        for pair in os.environ.get("AUTOBANNER_WORKSPACE_PLANS", "").split(","):
            if ":" in pair:
                ws, plan = pair.split(":", 1)
                if plan.strip() in self.plans:
                    self._env_assigned[ws.strip()] = plan.strip()
                else:
                    logger.warning("AUTOBANNER_WORKSPACE_PLANS: unknown plan %r", plan)
        self._lock = threading.Lock()

    def plan_name(self, owner: str | None) -> str:
        owner = owner or LOCAL_OWNER
        return self._env_assigned.get(owner) or self._assigned.get(owner) or self.default

    def limits(self, owner: str | None) -> dict[str, int | None]:
        return dict(self.plans[self.plan_name(owner)])

    def set_plan(self, owner: str, name: str) -> str:
        if name not in self.plans:
            raise ServiceError(f"unknown plan {name!r}; known: {', '.join(self.plans)}", 400)
        with self._lock:
            self._assigned[owner] = name
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"workspaces": self._assigned}, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        logger.info("workspace %s set to plan %s", owner, name)
        return name


def _stricter(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


class UsageMeter:
    """Per-owner usage counters persisted as JSON (one file per owner)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, owner: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in owner)[:64]
        return self.root / f"{safe or 'owner'}.json"

    def read(self, owner: str) -> dict:
        p = self._path(owner)
        if not p.exists():
            return {"owner": owner, "days": {}, "totals": {}}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {"owner": owner, "days": {}, "totals": {}}

    def add(self, owner: str, metric: str, amount: int = 1) -> dict:
        with self._lock:
            data = self.read(owner)
            day = _now()[:10]
            data.setdefault("days", {}).setdefault(day, {})
            data["days"][day][metric] = int(data["days"][day].get(metric, 0)) + amount
            data.setdefault("totals", {})
            data["totals"][metric] = int(data["totals"].get(metric, 0)) + amount
            data["owner"] = owner
            data["updated_at"] = _now()
            tmp = self._path(owner).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self._path(owner))
            return data

    def today(self, owner: str, metric: str) -> int:
        data = self.read(owner)
        return int(data.get("days", {}).get(_now()[:10], {}).get(metric, 0))


class ProjectService:
    def __init__(
        self,
        data_dir: str | Path,
        *,
        max_workers: int = 2,
        use_ai: bool = False,
        quota: Quota | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir = self.data_dir / "fonts"
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.brands_dir = self.data_dir / "brands"
        self.brands_dir.mkdir(parents=True, exist_ok=True)
        self.registry = FontRegistry(extra_dirs=[self.fonts_dir])
        self.jobs = JobManager(max_workers=max_workers, persist_dir=self.data_dir / "jobs")
        self.usage = UsageMeter(self.data_dir / "usage")
        self.entitlements = Entitlements(self.data_dir / "plans.json")
        self.events = EventLog(self.data_dir / "events")
        self.quota = quota or Quota.from_env()
        self.use_ai = use_ai
        self._locks: dict[str, threading.RLock] = {}
        self._cache: dict[str, Project] = {}
        self._recover_interrupted()

    # ---- ownership -----------------------------------------------------------------------
    @staticmethod
    def _owned_by(project: Project, owner: str | None) -> bool:
        """Projects without an owner belong to the local owner; others must match."""
        if owner is None:
            return True
        current = project.meta.get("owner") or LOCAL_OWNER
        return current == owner

    def storage_bytes(self, owner: str) -> int:
        total = 0
        for root in self.projects_dir.iterdir() if self.projects_dir.exists() else []:
            if not (root / "project.json").exists():
                continue
            try:
                project = self.get_project(root.name, owner=None)
            except ServiceError:
                continue
            if not self._owned_by(project, owner):
                continue
            total += sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        return total

    def usage_summary(self, owner: str, *, members: int | None = None) -> dict:
        data = self.usage.read(owner)
        plan = self.entitlements.limits(owner)
        used = {
            "variants_per_day": self.usage.today(owner, "variants"),
            "projects": len(self.list_projects(owner)),
            "storage_bytes": self.storage_bytes(owner),
        }
        if members is not None:
            used["members"] = members
        remaining = {
            key: (None if plan.get(key) is None else max(0, int(plan[key]) - used[key]))
            for key in used
        }
        return {
            "owner": owner,
            "today": data.get("days", {}).get(_now()[:10], {}),
            "totals": data.get("totals", {}),
            "storage_bytes": used["storage_bytes"],
            "quota": {
                "variants_per_day": self.quota.variants_per_day,
                "projects": self.quota.projects,
                "storage_bytes": self.quota.storage_bytes,
            },
            "plan": {
                "name": self.entitlements.plan_name(owner),
                "limits": plan,
                "used": used,
                "remaining": remaining,
            },
        }

    def plan_limit(self, owner: str, key: str) -> int | None:
        """The plan's limit for ``key`` (None when unlimited)."""
        return self.entitlements.limits(owner).get(key)

    def _plan_error(self, owner: str, what: str) -> ServiceError:
        name = self.entitlements.plan_name(owner)
        return ServiceError(f"plan '{name}' {what}", 402)

    def _check_quota_variants(self, owner: str, requested: int) -> None:
        used = self.usage.today(owner, "variants")
        plan_limit = self.plan_limit(owner, "variants_per_day")
        if plan_limit is not None and used + requested > plan_limit:
            raise self._plan_error(
                owner, f"allows {plan_limit} variants per day ({used} used, {requested} requested)"
            )
        limit = self.quota.variants_per_day
        if limit is None:
            return
        if used + requested > limit:
            raise ServiceError(
                f"daily variant quota exceeded ({used}/{limit} used, {requested} requested)", 429
            )

    def _check_quota_projects(self, owner: str) -> None:
        count = len(self.list_projects(owner))
        plan_limit = self.plan_limit(owner, "projects")
        if plan_limit is not None and count >= plan_limit:
            raise self._plan_error(owner, f"allows {plan_limit} projects ({count} in use)")
        plan_storage = self.plan_limit(owner, "storage_bytes")
        if plan_storage is not None and self.storage_bytes(owner) >= plan_storage:
            raise self._plan_error(owner, f"allows {plan_storage} bytes of storage")
        limit = self.quota.projects
        if limit is not None and count >= limit:
            raise ServiceError(f"project quota reached ({count}/{limit})", 429)
        cap = self.quota.storage_bytes
        if cap is not None and self.storage_bytes(owner) >= cap:
            raise ServiceError("storage quota reached", 429)

    def check_members(self, owner: str, current: int) -> None:
        """Refuse a new member when the workspace's plan is full."""
        plan_limit = self.plan_limit(owner, "members")
        if plan_limit is not None and current >= plan_limit:
            raise self._plan_error(owner, f"allows {plan_limit} members ({current} in use)")

    def check_campaign_rows(self, owner: str, rows: int) -> None:
        plan_limit = self.plan_limit(owner, "campaign_rows")
        if plan_limit is not None and rows > plan_limit:
            raise self._plan_error(
                owner, f"allows {plan_limit} campaign rows per job ({rows} requested)"
            )

    # ---- helpers ----------------------------------------------------------------------
    def _lock(self, project_id: str) -> threading.RLock:
        return self._locks.setdefault(project_id, threading.RLock())

    def _root(self, project_id: str) -> Path:
        if not project_id or "/" in project_id or ".." in project_id:
            raise ServiceError("invalid project id", 400)
        return self.projects_dir / project_id

    def _recover_interrupted(self) -> None:
        """Continue the variant jobs a restart cut short (durable queue).

        Every variant record carries its brief and every job record what the run
        needs (the layout families chosen jointly for the size set), so unfinished
        items are queued again in a new job that points back at the interrupted one.
        With ``AUTOBANNER_RESUME_JOBS=0`` the previous behaviour applies: unfinished
        variants are marked failed with "interrupted by restart".
        """
        resume = os.environ.get("AUTOBANNER_RESUME_JOBS", "1").strip().lower() not in (
            "0", "false", "no", "off"
        )
        for root in self.projects_dir.iterdir() if self.projects_dir.exists() else []:
            if not (root / "project.json").exists():
                continue
            try:
                project = Project.load(root)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cannot load project %s: %s", root, exc)
                continue
            unfinished = [
                r for r in project.variants.values() if r.status in ("pending", "running")
            ]
            if not unfinished:
                continue
            by_job: dict[str | None, list[VariantRecord]] = {}
            for rec in unfinished:
                by_job.setdefault(rec.job_id, []).append(rec)
            project_id = project.meta.get("id") or root.name
            self._cache[project_id] = project
            for job_id, recs in by_job.items():
                old = self.jobs.get(job_id) if job_id else None
                resumable = old is not None and old.status == "interrupted" and not old.resumed_by
                if resume and resumable:
                    try:
                        new = self._resume_job(project_id, project, old, recs)
                        logger.warning(
                            "job %s interrupted by restart: %d variant(s) continue in job %s",
                            old.id, len(recs), new.id,
                        )
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.error("cannot resume job %s: %s", old.id, exc)
                for rec in recs:
                    rec.status = "failed"
                    rec.error = "interrupted by restart"
            project.save()

    def _resume_job(self, project_id: str, project: Project, old: Job,
                    recs: list[VariantRecord]) -> Job:
        labels = {i.item_id: i.label for i in old.items}
        briefs = {rec.id: _brief_from_record(rec) for rec in recs}
        families = _families_from_meta(old.meta)
        references: dict[str, dict | None] = {}
        if old.kind == "regenerate" and old.meta.get("keep_layout"):
            for rec in recs:
                references[rec.id] = self._reference_for(project, rec.id)
        for rec in recs:
            rec.status = "pending"
            rec.error = None
        items = [(rec.id, labels.get(rec.id, rec.name)) for rec in recs]
        job = self._submit_variant_job(
            project_id, project, old.kind, items, briefs, families, references,
            owner=old.owner, meta=old.meta, resumed_from=old.id,
        )
        self.jobs.mark_resumed(old.id, job.id)
        for rec in recs:
            rec.job_id = job.id
        self.events.record(
            old.owner or LOCAL_OWNER, "job_resumed", project_id=project_id,
            job_id=old.id, resumed_by=job.id, count=len(recs),
        )
        return job

    def _submit_variant_job(
        self,
        project_id: str,
        project: Project,
        kind: str,
        items: list[tuple[str, str]],
        briefs: dict[str, VariantBrief],
        families: dict[str, Family],
        references: dict[str, dict | None] | None = None,
        *,
        owner: str | None,
        idempotency_key: str | None = None,
        meta: dict | None = None,
        resumed_from: str | None = None,
    ) -> Job:
        """Queue variant renders; shared by fresh requests, regeneration and resume."""
        references = references or {}

        def runner(job: Job, item: JobItem, progress) -> dict:
            brief = briefs[item.item_id]
            fams = families.get(direction_key(brief.direction)) or {}
            fam = fams.get(aspect_class(brief.width / max(1, brief.height))) or None
            return self._run_variant(
                project_id, item.item_id, brief, job, progress, family=fam,
                reference=references.get(item.item_id),
            )

        def on_finish(job: Job) -> None:
            with self._lock(project_id):
                for item in job.items:
                    rec = project.variants.get(item.item_id)
                    if rec is None:
                        continue
                    if item.status == "cancelled":
                        rec.status = "cancelled"
                    elif item.status == "failed":
                        rec.status = "failed"
                        rec.error = item.error
                    rec.job_id = job.id
                project.save()
            # Cross-variant consistency is judged within a row: rows carry different copy.
            groups: dict[str | None, list[str]] = {}
            for item in job.items:
                row_ref = briefs[item.item_id].row or {}
                groups.setdefault(row_ref.get("id"), []).append(item.item_id)
            for ids in groups.values():
                self._apply_family_checks(project_id, ids)

        return self.jobs.submit(
            kind, project_id, items, runner, idempotency_key=idempotency_key,
            on_finish=on_finish, owner=owner, meta=meta, resumed_from=resumed_from,
        )

    def _reference_for(self, project: Project, variant_id: str) -> dict | None:
        """The previous plan of a finished variant, for a layout-keeping regeneration."""
        detail = project.variant_detail(variant_id) or {}
        plan = detail.get("plan") or {}
        if plan.get("placements") and plan.get("planner") == "constraints":
            return {
                "placements": plan["placements"],
                "typography": plan.get("typography") or {},
                "text_plate_rects": plan.get("text_plate_rects") or [],
                "variant_id": variant_id,
            }
        return None

    def get_project(self, project_id: str, owner: str | None = LOCAL_OWNER) -> Project:
        """Load a project; a project that belongs to another owner reads as not found."""
        root = self._root(project_id)
        with self._lock(project_id):
            project = self._cache.get(project_id)
            if project is None:
                if not (root / "project.json").exists():
                    raise ServiceError("project not found", 404)
                project = Project.load(root)
                self._cache[project_id] = project
        if not self._owned_by(project, owner):
            raise ServiceError("project not found", 404)
        return project

    # ---- projects ----------------------------------------------------------------------
    def list_projects(self, owner: str | None = LOCAL_OWNER) -> list[dict]:
        out = []
        for root in sorted(self.projects_dir.iterdir()):
            if (root / "project.json").exists():
                try:
                    project = self.get_project(root.name, owner=None)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("skip project %s: %s", root, exc)
                    continue
                if self._owned_by(project, owner):
                    out.append(project.summary())
        return sorted(out, key=lambda p: p.get("updated_at") or "", reverse=True)

    def create_project_from_upload(
        self,
        filename: str,
        data: bytes,
        *,
        name: str | None = None,
        brand: str | None = None,
        owner: str | None = None,
        decompose: bool = True,
    ) -> Project:
        owner = owner or LOCAL_OWNER
        self._check_quota_projects(owner)
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ServiceError(f"unsupported file type '{ext}'", 415)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ServiceError("file too large", 413)
        if len(data) == 0:
            raise ServiceError("empty upload", 400)

        with tempfile.TemporaryDirectory() as tmp:
            safe_name = "upload" + ext
            path = Path(tmp) / safe_name
            path.write_bytes(data)
            if ext != ".psd":
                try:
                    with Image.open(path) as probe:
                        w, h = probe.size
                except Exception as exc:  # noqa: BLE001
                    raise ServiceError(f"cannot read image: {exc}", 400) from exc
                if w * h > MAX_PIXELS:
                    raise ServiceError("image exceeds pixel limit", 413)
            parser = get_parser(str(path))
            try:
                elements, source_size = parser.parse(str(path))
            except Exception as exc:  # noqa: BLE001
                raise ServiceError(f"parse failed: {exc}", 400) from exc
            if source_size[0] * source_size[1] > MAX_PIXELS:
                raise ServiceError("design exceeds pixel limit", 413)
            classifier = SemanticClassifier(use_ai=self.use_ai)
            elements = classifier.classify_all(elements, source_size)
            decomposition = None
            if ext != ".psd" and decompose:
                try:
                    decomposition = decompose_flat_image(elements[0].image)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("decomposition failed, keeping flat import: %s", exc)

            project_id = new_id("proj")
            root = self._root(project_id)
            root.mkdir(parents=True, exist_ok=False)
            (root / "source").mkdir()
            digest = hashlib.sha256(data).hexdigest()
            original_path = root / "source" / f"original{ext}"
            shutil.copyfile(path, original_path)
            store = AssetStore(root / "assets")
            doc = document_from_elements(
                elements,
                source_size,
                store,
                name=name or Path(filename).stem,
                source_ref=f"sha256:{digest[:16]}",
                origin="psd_layer" if ext == ".psd" else "flat_image",
                registry=self.registry,
            )
            if decomposition is not None:
                _apply_decomposition(doc, decomposition, store, source_size)
            doc.metadata.update(
                {
                    "source_filename": Path(filename).name,
                    "source_sha256": digest,
                    "source_path": f"source/original{ext}",
                    "import_notes": _import_notes(doc, ext, decomposition),
                }
            )
            project = Project.create(root, doc, name=name or doc.name, owner=owner, brand=brand)
            project.meta["id"] = project_id
            project.save()
            with self._lock(project_id):
                self._cache[project_id] = project
            try:
                self._apply_brand_rules(project, owner)
            except Exception as exc:  # noqa: BLE001
                logger.warning("brand rule proposal failed: %s", exc)
            self.usage.add(owner, "projects")
            self.usage.add(owner, "upload_bytes", len(data))
            self.events.record(
                owner, "project_created", project_id=project_id, source=ext,
                elements=len(doc.elements),
            )
            return project

    def create_blank_project(
        self,
        *,
        name: str,
        width: int,
        height: int,
        background: str = "#ffffff",
        brand: str | None = None,
        owner: str | None = None,
    ) -> Project:
        """Start a master from scratch (compose from separated assets and copy)."""
        owner = owner or LOCAL_OWNER
        self._check_quota_projects(owner)
        width, height = int(width), int(height)
        if width < 16 or height < 16 or width * height > MAX_PIXELS:
            raise ServiceError("canvas size out of range", 400)
        project_id = new_id("proj")
        root = self._root(project_id)
        root.mkdir(parents=True, exist_ok=False)
        doc = DesignDocument(id=new_id("doc"), name=name or "Untitled", canvas_width=width,
                             canvas_height=height)
        from ..design.document import Element

        doc.elements.append(
            Element(
                id="background",
                kind="shape",
                name="Background",
                role="background",
                geometry=Geometry(0, 0, width, height),
                z_index=0,
                shape={"type": "rect", "fill": _normalize_hex(background)},
                priority=9,
                allowed=AllowedTransforms(move=False, scale_free=True, crop=True),
                provenance=Provenance(origin="user"),
            )
        )
        doc.metadata["import_notes"] = ["Started from a blank canvas; add assets and copy."]
        project = Project.create(root, doc, name=name, owner=owner, brand=brand)
        project.meta["id"] = project_id
        project.save()
        with self._lock(project_id):
            self._cache[project_id] = project
        self.usage.add(owner, "projects")
        return project

    def add_element(
        self,
        project_id: str,
        *,
        owner: str | None = LOCAL_OWNER,
        kind: str,
        name: str,
        role: str,
        geometry: dict,
        text: str | None = None,
        style: dict | None = None,
        image_bytes: bytes | None = None,
        image_name: str = "",
    ) -> Project:
        """Add a text or image element (original asset stored by content hash)."""
        project = self.get_project(project_id, owner)
        from ..design.document import Element, TextContent, TextRun, TextStyle

        if kind not in ("text", "image"):
            raise ServiceError("kind must be text or image", 400)
        try:
            g = Geometry(float(geometry.get("x", 0)), float(geometry.get("y", 0)),
                         float(geometry.get("width", 100)), float(geometry.get("height", 50)))
        except (TypeError, ValueError) as exc:
            raise ServiceError("invalid geometry", 400) from exc
        if g.width <= 0 or g.height <= 0:
            raise ServiceError("geometry needs positive size", 400)
        with self._lock(project_id):
            doc = project.document
            element_id = new_id("el")
            z = max((e.z_index for e in doc.elements), default=-1) + 1
            if kind == "text":
                if not text or not text.strip():
                    raise ServiceError("text is required", 400)
                st = TextStyle()
                for key in ("font_family", "weight", "color", "align"):
                    if style and key in style:
                        setattr(st, key, str(style[key]))
                if style and "font_size" in style:
                    st.font_size = float(style["font_size"])
                else:
                    st.font_size = max(8.0, g.height * 0.7)
                resolved = self.registry.resolve(st.font_family, st.weight, st.italic)
                from ..design.document import FontRef

                if not any(f.family == st.font_family for f in doc.fonts):
                    doc.fonts.append(FontRef(
                        family=st.font_family, weight=st.weight, italic=st.italic,
                        path=resolved.path, status=resolved.status,
                        substitute=resolved.family if resolved.substituted else None,
                    ))
                element = Element(
                    id=element_id, kind="text", name=name or role, role=role, geometry=g,
                    z_index=z, text=TextContent(runs=[TextRun(text=text, style=st)]),
                    allowed=AllowedTransforms(reflow=True),
                    provenance=Provenance(origin="user"), role_confidence=1.0,
                    priority=1 if role in ("headline", "logo") else 2,
                )
            else:
                if not image_bytes:
                    raise ServiceError("image file is required", 400)
                if len(image_bytes) > MAX_UPLOAD_BYTES:
                    raise ServiceError("file too large", 413)
                try:
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        if img.width * img.height > MAX_PIXELS:
                            raise ServiceError("image exceeds pixel limit", 413)
                        rgba = img.convert("RGBA").copy()
                except ServiceError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise ServiceError(f"cannot read image: {exc}", 400) from exc
                ref = project.assets.put(rgba, image_name or name)
                element = Element(
                    id=element_id, kind="image", name=name or role, role=role, geometry=g,
                    z_index=z, asset=ref,
                    allowed=AllowedTransforms(scale_free=False, crop=False),
                    provenance=Provenance(origin="user", source_ref=image_name),
                    role_confidence=1.0,
                    priority=1 if role == "logo" else 2,
                )
            doc.elements.append(element)
            doc.normalize_z()
            problems = doc.validate()
            if problems:
                doc.elements.pop()
                raise ServiceError("; ".join(problems), 400)
            project.save(snapshot=True, label=f"add {element.name}")
        return project

    def delete_project(self, project_id: str, owner: str | None = LOCAL_OWNER) -> None:
        project = self.get_project(project_id, owner)
        with self._lock(project_id):
            project.delete()
            self._cache.pop(project_id, None)

    def project_payload(self, project: Project) -> dict:
        return {
            "project": project.summary(),
            "document": document_to_dict(project.document),
            "variants": [v.to_dict() for v in project.variants.values()],
            "history": project.history()[-20:],
        }

    # ---- document edits ---------------------------------------------------------------
    def apply_operations(
        self,
        project_id: str,
        ops: list[dict],
        label: str = "edit",
        owner: str | None = LOCAL_OWNER,
    ) -> Project:
        project = self.get_project(project_id, owner)
        if not isinstance(ops, list) or not ops:
            raise ServiceError("ops must be a non-empty list", 400)
        with self._lock(project_id):
            doc = document_from_dict(document_to_dict(project.document))  # work on a copy
            for op in ops:
                _apply_op(doc, op)
            problems = doc.validate()
            if problems:
                raise ServiceError("invalid document after edit: " + "; ".join(problems), 400)
            project.document = doc
            project.save(snapshot=True, label=label)
        self.events.record(
            owner or LOCAL_OWNER, "document_ops", project_id=project_id, count=len(ops),
            kinds=sorted({str(o.get("op")) for o in ops}),
        )
        return project

    def undo(self, project_id: str, owner: str | None = LOCAL_OWNER) -> bool:
        project = self.get_project(project_id, owner)
        with self._lock(project_id):
            return project.undo()

    def restore(self, project_id: str, version: int, owner: str | None = LOCAL_OWNER) -> None:
        project = self.get_project(project_id, owner)
        with self._lock(project_id):
            try:
                project.restore_version(version)
            except KeyError as exc:
                raise ServiceError("version not found", 404) from exc

    # ---- rendering ---------------------------------------------------------------------
    def master_preview(
        self, project_id: str, max_side: int = 1600, owner: str | None = LOCAL_OWNER
    ) -> Image.Image:
        project = self.get_project(project_id, owner)
        doc = project.document
        scale = min(1.0, max_side / max(doc.canvas_width, doc.canvas_height))
        with self._lock(project_id):
            return render_master(doc, project.assets, self.registry, scale=scale)

    def asset_path(self, project_id: str, asset_id: str, owner: str | None = LOCAL_OWNER) -> Path:
        project = self.get_project(project_id, owner)
        if not asset_id.isalnum():
            raise ServiceError("invalid asset id", 400)
        p = project.assets.path(asset_id)
        if not p.exists():
            raise ServiceError("asset not found", 404)
        return p

    # ---- variants ----------------------------------------------------------------------
    def request_variants(
        self,
        project_id: str,
        spec: dict,
        idempotency_key: str | None = None,
        owner: str | None = LOCAL_OWNER,
    ) -> Job:
        project = self.get_project(project_id, owner)
        existing = self.jobs.find_by_key(idempotency_key)
        if existing is not None:
            if existing.project_id != project_id:
                raise ServiceError("idempotency key already used by another project", 409)
            return existing
        targets = _normalize_targets(spec)
        if not targets:
            raise ServiceError("no targets requested", 400)
        rows = _normalize_rows(spec, project.document)
        total = len(targets) * max(1, len(rows))
        if total > MAX_VARIANTS_PER_JOB:
            raise ServiceError(
                f"too many variants in one job ({total}; max {MAX_VARIANTS_PER_JOB})", 400
            )
        meter_owner = owner or project.meta.get("owner") or LOCAL_OWNER
        self.check_campaign_rows(meter_owner, len(rows))
        self._check_quota_variants(meter_owner, total)
        text_overrides = dict(spec.get("text_overrides") or {})
        for eid in text_overrides:
            if not project.document.has_element(eid):
                raise ServiceError(f"unknown element '{eid}' in text_overrides", 400)
        hidden = list(spec.get("hidden_elements") or [])
        locale = spec.get("locale")
        direction = _normalize_direction(spec.get("direction"), "direction")

        # Brand rules (H5 follow-up): rules confirmed in the owner's other projects of the
        # same brand are proposed into this document before planning (reviewable, soft).
        try:
            self._apply_brand_rules(project, meter_owner)
        except Exception as exc:  # noqa: BLE001
            logger.warning("brand rule proposal failed: %s", exc)

        # Joint planning (H2): one layout family per orientation for the whole job.
        # Approved variants of this project act as examples (H1): their inferred
        # families compete with the hand-written ones for every orientation.
        # One joint family choice per distinct creative direction (rows may differ).
        directions: dict[str, dict | None] = {direction_key(direction): direction}
        for row in rows:
            row_dir = row["direction"] if row.get("direction") is not None else direction
            directions.setdefault(direction_key(row_dir), row_dir)
        families: dict[str, dict[str, Family]] = {}
        learned: dict[str, Family] = {}
        if Config.DESIGN_PLANNER == "constraints":
            try:
                learned = {
                    cls: inf.family
                    for cls, inf in self._learned_from_approved(project).items()
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("learning from approved variants failed: %s", exc)
        if Config.DESIGN_PLANNER == "constraints" and (len(targets) > 1 or learned):
            try:
                engine_elements = elements_from_document(
                    project.document, project.assets, registry=self.registry,
                    text_overrides=text_overrides, locale=locale,
                )
                engine_elements = [
                    e for e in engine_elements if e.id not in set(hidden)
                ]
                for dkey, dirn in directions.items():
                    families[dkey] = choose_families(
                        project.document, engine_elements,
                        [(t["width"], t["height"]) for t in targets], registry=self.registry,
                        learned=learned or None, direction=dirn,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("joint family choice failed; planning per variant: %s", exc)
                families = {}

        with self._lock(project_id):
            items: list[tuple[str, str]] = []
            briefs: dict[str, VariantBrief] = {}
            # A campaign table generates every row in every size; the job-level copy,
            # locale and hidden elements are the defaults a row can override.
            for row in rows or [None]:
                row_overrides = dict(text_overrides)
                row_locale, row_hidden = locale, list(hidden)
                row_ref = None
                row_direction = direction
                if row is not None:
                    row_overrides.update(row["text_overrides"])
                    row_locale = row["locale"] or locale
                    row_hidden = sorted(set(hidden) | set(row["hidden_elements"]))
                    row_ref = {"id": row["id"], "label": row["label"]}
                    if row.get("direction") is not None:
                        row_direction = row["direction"]
                for t in targets:
                    brief = VariantBrief(
                        width=t["width"],
                        height=t["height"],
                        name=t["name"],
                        text_overrides=row_overrides,
                        locale=row_locale,
                        hidden_elements=row_hidden,
                        channel_preset=t.get("preset_id"),
                        row=row_ref,
                        direction=row_direction,
                    )
                    rec = project.new_variant(t["name"], t["width"], t["height"], brief.to_dict())
                    label = f"{row['label']} · {t['name']}" if row is not None else t["name"]
                    items.append((rec.id, label))
                    briefs[rec.id] = brief
            project.save()

        job = self._submit_variant_job(
            project_id, project, "variants", items, briefs, families,
            owner=meter_owner, idempotency_key=idempotency_key,
            meta={"families": {cls: asdict(f) for cls, f in families.items()}},
        )
        self.usage.add(meter_owner, "variants", len(items))
        self.usage.add(meter_owner, "jobs")
        with self._lock(project_id):
            for rec_id, _ in items:
                project.variants[rec_id].job_id = job.id
            project.save()
        return job

    def _run_variant(self, project_id: str, variant_id: str, brief: VariantBrief, job: Job,
                     progress, family: Family | None = None,
                     reference: dict | None = None) -> dict:
        project = self.get_project(project_id, owner=None)
        with self._lock(project_id):
            project.mark_variant(variant_id, "running")
            doc_snapshot = document_from_dict(document_to_dict(project.document))
        # The brand profile travels with the snapshot so the verifier can check palette and
        # fonts; the stored document is not changed.
        profile = self.get_brand_profile(
            job.owner or project.meta.get("owner"), project.meta.get("brand")
        )
        if profile:
            doc_snapshot.metadata["brand_profile"] = profile
        result = generate_variant(
            doc_snapshot,
            project.assets,
            brief,
            registry=self.registry,
            progress=progress,
            cancel=job.cancel,
            family=family,
            reference=reference,
        )
        with self._lock(project_id):
            rec = project.store_variant_output(
                variant_id,
                result.image,
                result.report.to_dict(),
                {**result.plan, "warnings": result.warnings},
                verdict=result.verdict,
            )
            project.save()
        self.events.record(
            job.owner or LOCAL_OWNER, "variant_done", project_id=project_id,
            variant_id=variant_id, verdict=rec.verdict, size=f"{rec.width}x{rec.height}",
            repair_steps=len(result.repair_steps),
        )
        return {"variant_id": rec.id, "verdict": rec.verdict}

    # ---- learning from approved variants (H1) -----------------------------------------
    def _approved_examples(self, project: Project) -> tuple[list[str], list]:
        """Approved, finished variants of a project as example documents."""
        ids: list[str] = []
        examples = []
        for rec in project.variants.values():
            if rec.status != "done" or rec.approval != "approved":
                continue
            detail = project.variant_detail(rec.id)
            plan = (detail or {}).get("plan") or {}
            placements = plan.get("placements") or []
            if not placements:
                continue
            examples.append(
                example_from_plan(
                    project.document,
                    (rec.width, rec.height),
                    placements,
                    plan.get("typography") or {},
                    example_id=f"{project.id}_{rec.id}",
                )
            )
            ids.append(rec.id)
        return ids, examples

    def _learned_from_approved(self, project: Project) -> dict:
        ids, examples = self._approved_examples(project)
        if not examples:
            return {}
        return infer_families(project.document, examples)

    def learned_rules(self, project_id: str, owner: str | None = LOCAL_OWNER) -> dict:
        """Reviewable summary of what the planner learned from approved variants."""
        project = self.get_project(project_id, owner)
        ids, examples = self._approved_examples(project)
        inferred = infer_families(project.document, examples) if examples else {}
        proposals, unresolved = self._corrections(project)
        return {
            "examples": ids,
            "families": [inf.to_dict() for inf in inferred.values()],
            "corrections": [p.to_dict() for p in proposals],
            "unresolved_corrections": [u.to_dict() for u in unresolved],
            "note": (
                "Approved variants are examples: their composition per orientation "
                "competes with the built-in layout families on the next generation. "
                "Nothing is trained; rules are inferred per project and can be reviewed here."
            ),
        }

    def _apply_family_checks(self, project_id: str, variant_ids: list[str]) -> None:
        """Run cross-variant checks for the variants of one job and update their reports."""
        project = self.get_project(project_id, owner=None)
        snapshots: list[VariantSnapshot] = []
        details: dict[str, dict] = {}
        with self._lock(project_id):
            for vid in variant_ids:
                rec = project.variants.get(vid)
                if rec is None or rec.status != "done":
                    continue
                detail = project.variant_detail(vid)
                if not detail:
                    continue
                details[vid] = detail
                plan = detail.get("plan") or {}
                hideable = {
                    e.id for e in project.document.elements if e.allowed.hide
                }
                snapshots.append(
                    VariantSnapshot(
                        variant_id=vid,
                        target=(rec.width, rec.height),
                        family=(plan.get("planner_meta") or {}).get("family"),
                        placements=plan.get("placements", []),
                        typography=plan.get("typography", {}),
                        roles={e.id: e.role for e in project.document.elements},
                        hideable=hideable,
                        master_px={
                            e.id: float(e.text.primary_style.font_size)
                            for e in project.document.elements
                            if e.kind == "text" and e.text is not None
                        },
                    )
                )
            if len(snapshots) < 2:
                return
            results = family_consistency_checks(snapshots)
            for vid, checks in results.items():
                detail = details[vid]
                quality = detail.get("quality") or {}
                existing = [
                    c
                    for c in quality.get("checks", [])
                    if not str(c.get("check_id", "")).startswith("family_")
                ]
                all_checks = existing + [c.to_dict() for c in checks]
                rebuilt = [
                    CheckResult(
                        check_id=c["check_id"],
                        status=CheckStatus(c["status"]),
                        severity=Severity(c["severity"]),
                        message=c["message"],
                        subject_id=c.get("subject_id"),
                        details=dict(c.get("details") or {}),
                    )
                    for c in all_checks
                ]
                verdict = derive_verdict(rebuilt)
                quality["checks"] = all_checks
                quality["verdict"] = verdict.value
                quality["summary"] = summarize(rebuilt)
                detail["quality"] = quality
                rec = project.variants[vid]
                rec.verdict = verdict.value
                _atomic_write_json(project.root / rec.report_path, detail)  # type: ignore[arg-type]
            project.save()

    def regenerate_variant(
        self,
        project_id: str,
        variant_id: str,
        spec: dict | None = None,
        owner: str | None = LOCAL_OWNER,
    ) -> Job:
        project = self.get_project(project_id, owner)
        meter_owner = owner or project.meta.get("owner") or LOCAL_OWNER
        self._check_quota_variants(meter_owner, 1)
        rec = project.variants.get(variant_id)
        if rec is None:
            raise ServiceError("variant not found", 404)
        brief_dict = dict(rec.brief)
        if spec:
            for key in ("text_overrides", "hidden_elements", "locale"):
                if key in spec:
                    brief_dict[key] = spec[key]
            if "direction" in spec:
                brief_dict["direction"] = _normalize_direction(spec["direction"], "direction")
        # Local edits (H3): keep the previous layout of this variant unless asked to re-plan.
        keep_layout = bool((spec or {}).get("keep_layout", True))
        reference: dict | None = None
        if keep_layout and rec.status == "done":
            reference = self._reference_for(project, variant_id)
        brief = VariantBrief(
            width=rec.width,
            height=rec.height,
            name=rec.name,
            text_overrides=dict(brief_dict.get("text_overrides") or {}),
            locale=brief_dict.get("locale"),
            hidden_elements=list(brief_dict.get("hidden_elements") or []),
            channel_preset=brief_dict.get("channel_preset"),
            row=brief_dict.get("row") or None,
            direction=brief_dict.get("direction") or None,
        )
        with self._lock(project_id):
            rec.brief = brief.to_dict()
            rec.status = "pending"
            rec.approval = "none"
            rec.approval_reason = ""
            rec.error = None
            rec.document_version = project.document_version
            project.save()

        job = self._submit_variant_job(
            project_id, project, "regenerate", [(variant_id, rec.name)], {variant_id: brief},
            {}, {variant_id: reference}, owner=meter_owner,
            meta={"keep_layout": bool(reference)},
        )
        self.usage.add(meter_owner, "variants")
        self.usage.add(meter_owner, "jobs")
        return job

    def set_approval(
        self,
        project_id: str,
        variant_id: str,
        approval: str,
        reason: str,
        owner: str | None = LOCAL_OWNER,
    ) -> VariantRecord:
        project = self.get_project(project_id, owner)
        if variant_id not in project.variants:
            raise ServiceError("variant not found", 404)
        if approval == "rejected" and not reason.strip():
            raise ServiceError("a rejection needs a reason", 400)
        with self._lock(project_id):
            rec = project.set_approval(variant_id, approval, reason)
            if approval == "rejected":
                # Keep the rejected plan as evidence for correction rules (H5).
                detail = project.variant_detail(variant_id) or {}
                plan = detail.get("plan") or {}
                if plan.get("placements"):
                    project.record_rejection(variant_id, reason, plan)
            project.save()
        self.events.record(
            owner or LOCAL_OWNER, "approval", project_id=project_id, variant_id=variant_id,
            approval=approval, reason=reason[:200], verdict=rec.verdict,
        )
        return rec

    # ---- brand-level rules (H5 follow-up) ----------------------------------------------
    @staticmethod
    def _brand_key(brand: str | None) -> str:
        return (brand or "").strip().lower()

    def brand_rules(
        self, owner: str | None, brand: str, *, exclude_project: str | None = None
    ) -> list[dict]:
        """Union of carried rules across the owner's projects with this brand."""
        docs = self._brand_documents(owner, brand, exclude_project=exclude_project)
        return [r.to_dict() for r in collect_brand_rules(docs)]

    def _apply_brand_rules(self, project: Project, owner: str | None) -> list[str]:
        """Propose the brand's rules into a project: profile rules first, then carried ones."""
        brand = project.meta.get("brand")
        if not self._brand_key(brand):
            return []
        profile = self.get_brand_profile(owner, brand)
        rules = collect_brand_rules(
            [
                (name, doc)
                for name, doc in self._brand_documents(owner, brand, exclude_project=project.id)
            ]
        )
        if not rules and not profile:
            return []
        with self._lock(project.id):
            added = []
            if profile:
                added.extend(propose_profile_rules(project.document, profile, brand=str(brand)))
            if rules:
                added.extend(propose_brand_rules(project.document, rules, brand=str(brand)))
            if added:
                project.save(snapshot=True, label=f"brand rules proposed ({len(added)})")
        if added:
            self.events.record(
                owner or LOCAL_OWNER, "brand_rules_proposed", project_id=project.id,
                brand=str(brand), count=len(added),
            )
        return [c.id for c in added]

    # ---- brand profiles ---------------------------------------------------------------
    def _brand_profile_path(self, owner: str | None, brand: str) -> Path:
        safe_owner = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in (owner or LOCAL_OWNER)
        )[:64] or "owner"
        safe_brand = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in self._brand_key(brand)
        )[:64]
        return self.brands_dir / safe_owner / f"{safe_brand}.json"

    def get_brand_profile(self, owner: str | None, brand: str | None) -> dict | None:
        if not self._brand_key(brand):
            return None
        path = self._brand_profile_path(owner, str(brand))
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("unreadable brand profile %s", path)
            return None

    def list_brand_profiles(self, owner: str | None) -> list[dict]:
        folder = self._brand_profile_path(owner, "x").parent
        out = []
        for path in sorted(folder.glob("*.json")) if folder.exists() else []:
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return out

    def save_brand_profile(self, owner: str | None, brand: str, raw: dict,
                           *, by: str | None = None) -> dict:
        brand = str(brand).strip()
        if not self._brand_key(brand) or len(brand) > 80:
            raise ServiceError("brand: 1-80 characters", 400)
        try:
            profile = normalize_profile(raw, brand=brand)
        except ValueError as exc:
            raise ServiceError(str(exc), 400) from exc
        profile["updated_at"] = _now()
        profile["updated_by"] = by
        path = self._brand_profile_path(owner, brand)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        self.events.record(owner or LOCAL_OWNER, "brand_profile_saved", brand=brand, by=by)
        return profile

    def delete_brand_profile(self, owner: str | None, brand: str) -> None:
        path = self._brand_profile_path(owner, brand)
        if not path.exists():
            raise ServiceError("brand profile not found", 404)
        path.unlink()

    def _brand_documents(
        self, owner: str | None, brand: str | None, *, exclude_project: str | None = None
    ) -> list[tuple[str, DesignDocument]]:
        key = self._brand_key(brand)
        out: list[tuple[str, DesignDocument]] = []
        if not key or not self.projects_dir.exists():
            return out
        for root in sorted(self.projects_dir.iterdir()):
            if not (root / "project.json").exists() or root.name == exclude_project:
                continue
            try:
                other = self.get_project(root.name, owner=None)
            except Exception:  # noqa: BLE001
                continue
            if self._owned_by(other, owner) and self._brand_key(other.meta.get("brand")) == key:
                out.append((other.name, other.document))
        return out

    # ---- rules from correction history (H5) --------------------------------------------
    def _corrections(self, project: Project):
        rejections = [RejectionSnapshot.from_dict(d) for d in project.rejections()]
        if not rejections:
            return [], []
        approved = []
        for rec in project.variants.values():
            if rec.status != "done" or rec.approval != "approved":
                continue
            plan = (project.variant_detail(rec.id) or {}).get("plan") or {}
            if plan.get("placements"):
                approved.append((rec.to_dict(), plan))
        return derive_corrections(project.document, rejections, approved)

    def apply_correction(
        self, project_id: str, index: int, owner: str | None = LOCAL_OWNER
    ) -> dict:
        """Add a derived correction rule to the document as a reviewable soft constraint."""
        project = self.get_project(project_id, owner)
        proposals, _ = self._corrections(project)
        if index < 0 or index >= len(proposals):
            raise ServiceError("correction not found", 404)
        proposal = proposals[index]
        constraint = proposal.to_constraint()
        with self._lock(project_id):
            project.document.add_constraint(constraint)
            project.save(snapshot=True, label=f"rule from correction: {proposal.reason[:40]}")
        self.events.record(
            owner or LOCAL_OWNER, "document_ops", project_id=project_id, ops=1,
            kinds=["add_constraint"], source="correction",
        )
        return self.project_payload(project)

    def delete_variant(
        self, project_id: str, variant_id: str, owner: str | None = LOCAL_OWNER
    ) -> None:
        project = self.get_project(project_id, owner)
        with self._lock(project_id):
            if not project.delete_variant(variant_id):
                raise ServiceError("variant not found", 404)
            project.save()

    def variant_detail(
        self, project_id: str, variant_id: str, owner: str | None = LOCAL_OWNER
    ) -> dict:
        project = self.get_project(project_id, owner)
        rec = project.variants.get(variant_id)
        if rec is None:
            raise ServiceError("variant not found", 404)
        detail = project.variant_detail(variant_id) or {}
        return {
            "variant": rec.to_dict(),
            "plan": detail.get("plan"),
            "quality": detail.get("quality"),
        }

    def variant_image_path(
        self, project_id: str, variant_id: str, owner: str | None = LOCAL_OWNER
    ) -> Path:
        project = self.get_project(project_id, owner)
        path = project.variant_file(variant_id, "image")
        if path is None:
            raise ServiceError("variant image not available", 404)
        return path

    # ---- export / import ------------------------------------------------------------------
    def parse_campaign_rows(self, project_id: str, text: str,
                            owner: str | None = LOCAL_OWNER) -> dict:
        """Map a pasted CSV/TSV table onto this project's text elements (no side effects)."""
        project = self.get_project(project_id, owner)
        return parse_campaign_table(project.document, text)

    def export_deliverables(self, project_id: str, *, fmt: str = "png", only: str = "all",
                            quality: int = 90, owner: str | None = LOCAL_OWNER) -> bytes:
        project = self.get_project(project_id, owner)
        fmt = fmt.lower()
        if fmt not in ("png", "jpeg", "jpg", "webp", "pdf"):
            raise ServiceError("format must be png, jpeg, webp or pdf", 400)
        pil_fmt = {"png": "PNG", "jpeg": "JPEG", "jpg": "JPEG", "webp": "WEBP", "pdf": "PDF"}[fmt]
        ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "PDF": "pdf"}[pil_fmt]
        pdf_pages: list[Image.Image] = []
        manifest: dict[str, Any] = {
            "project": project.summary(),
            "format": pil_fmt,
            "filter": only,
            "font_disclosure": [f.__dict__ for f in project.document.fonts],
            "variants": [],
            "skipped": [],
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf, self._lock(project_id):
            for rec in project.variants.values():
                if rec.status != "done" or not rec.image_path:
                    manifest["skipped"].append({"id": rec.id, "reason": rec.status})
                    continue
                if only == "approved" and rec.approval != "approved":
                    manifest["skipped"].append({"id": rec.id, "reason": "not approved"})
                    continue
                if only == "accepted" and rec.verdict != "accepted" and rec.approval != "approved":
                    manifest["skipped"].append({"id": rec.id, "reason": "not accepted"})
                    continue
                img = project.variant_image(rec.id)
                if img is None:
                    continue
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rec.name)
                row_ref = (rec.brief or {}).get("row") or None
                if row_ref:
                    folder = "".join(
                        ch if ch.isalnum() or ch in "-_" else "_" for ch in str(row_ref["label"])
                    ) or str(row_ref["id"])
                    safe = f"{folder}/{safe}"
                out = io.BytesIO()
                if pil_fmt == "PDF":
                    # one page per variant at pixel size; the PDF is written once at the end
                    page = img.convert("RGB")
                    pdf_pages.append(page)
                    page.save(out, format="PNG", optimize=True)  # hashed content of the page
                    fname = "deliverables.pdf"
                elif pil_fmt == "JPEG":
                    img.convert("RGB").save(out, format="JPEG", quality=quality, optimize=True)
                    fname = f"{safe}_{rec.width}x{rec.height}.{ext}"
                    zf.writestr(fname, out.getvalue())
                elif pil_fmt == "WEBP":
                    img.save(out, format="WEBP", quality=quality)
                    fname = f"{safe}_{rec.width}x{rec.height}.{ext}"
                    zf.writestr(fname, out.getvalue())
                else:
                    img.save(out, format="PNG", optimize=True)
                    fname = f"{safe}_{rec.width}x{rec.height}.{ext}"
                    zf.writestr(fname, out.getvalue())
                detail = project.variant_detail(rec.id) or {}
                zf.writestr(
                    f"reports/{safe}_{rec.width}x{rec.height}.json", json.dumps(detail, indent=2)
                )
                entry = {
                    "id": rec.id,
                    "name": rec.name,
                    "row": row_ref,
                    "file": fname,
                    "width": rec.width,
                    "height": rec.height,
                    "verdict": rec.verdict,
                    "approval": rec.approval,
                    "sha256": hashlib.sha256(out.getvalue()).hexdigest(),
                }
                if pil_fmt == "PDF":
                    entry["page"] = len(pdf_pages)
                manifest["variants"].append(entry)
            if pil_fmt == "PDF" and pdf_pages:
                pdf = io.BytesIO()
                pdf_pages[0].save(
                    pdf, format="PDF", save_all=True, append_images=pdf_pages[1:], resolution=150
                )
                zf.writestr("deliverables.pdf", pdf.getvalue())
                manifest["pdf"] = {
                    "file": "deliverables.pdf",
                    "pages": len(pdf_pages),
                    "resolution_dpi": 150,
                    "sha256": hashlib.sha256(pdf.getvalue()).hexdigest(),
                }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        self.events.record(
            owner or LOCAL_OWNER, "export", project_id=project_id, fmt=pil_fmt, only=only,
            count=len(manifest["variants"]),
        )
        return buf.getvalue()

    def export_project(self, project_id: str, owner: str | None = LOCAL_OWNER) -> bytes:
        project = self.get_project(project_id, owner)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf, self._lock(project_id):
            project.save()
            for path in sorted(project.root.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(project.root)))
        return buf.getvalue()

    def import_project(
        self, data: bytes, owner: str | None = LOCAL_OWNER, *, role: str = "admin"
    ) -> Project:
        """Reopen an editable project archive as a new project of ``owner``.

        Everything inside the archive is untrusted: member names, the variant index,
        asset references and the trust-bearing metadata (owner, id, approvals). The
        archive can neither place files outside its new project directory nor carry
        approvals the importer could not have granted (``role`` below approver).
        """
        owner = owner or LOCAL_OWNER
        self._check_quota_projects(owner)
        if len(data) > MAX_UPLOAD_BYTES * 4:
            raise ServiceError("archive too large", 413)
        project_id = new_id("proj")
        root = self._root(project_id)
        root.mkdir(parents=True, exist_ok=False)
        total = 0
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                infos = zf.infolist()
                if len(infos) > MAX_ARCHIVE_MEMBERS:
                    raise ServiceError("archive has too many members", 400)
                if "project.json" not in {i.filename for i in infos}:
                    raise ServiceError("archive is not an AutoBanner project", 400)
                for info in infos:
                    name = info.filename
                    if _is_symlink_member(info):
                        raise ServiceError("unsafe path in archive", 400)
                    if safe_relative_path(root, name.rstrip("/")) is None:
                        raise ServiceError("unsafe path in archive", 400)
                    total += info.file_size
                    if total > MAX_UPLOAD_BYTES * 8:
                        raise ServiceError("archive expands too large", 413)
                    if info.is_dir():
                        continue
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(info))
            try:
                project = Project.load(root, strict=True)
            except ValueError as exc:
                if "unsafe" in str(exc):
                    raise ServiceError(f"unsafe archive: {exc}", 400) from exc
                raise
            project.meta["id"] = project_id
            project.meta["owner"] = owner
            project.meta["imported_at"] = _now()
            project.meta["imported_by_role"] = role
            if role not in ("approver", "admin"):
                # Reviewer decisions are not editor-supplied data: a claimed approval
                # travels as history only, never as authority.
                for rec in project.variants.values():
                    if rec.approval != "none":
                        rec.approval_reason = (
                            f"reset on import by {role} (archive claimed '{rec.approval}')"
                        )
                        rec.approval = "none"
            project.save()
            self.usage.add(owner, "projects")
        except ServiceError:
            shutil.rmtree(root, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(root, ignore_errors=True)
            raise ServiceError(f"import failed: {exc}", 400) from exc
        with self._lock(project_id):
            self._cache[project_id] = project
        return project

    def pilot_summary(self, owner: str) -> dict:
        return self.events.summary(owner)

    def get_job(self, job_id: str, owner: str | None = LOCAL_OWNER) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise ServiceError("job not found", 404)
        if owner is not None and (job.owner or LOCAL_OWNER) != owner:
            raise ServiceError("job not found", 404)
        return job

    # ---- retention -----------------------------------------------------------------------
    def purge_stale_projects(self, days: int, *, now: str | None = None) -> list[str]:
        """Delete projects (with variants, history, corrections) untouched for ``days``.

        "Untouched" = neither the project nor any of its variants was updated after the
        cutoff and no job of the project is pending/running. Returns the deleted ids.
        Operators are expected to export project zips they want to keep; there is no
        undo for a purge.
        """
        from datetime import UTC, datetime, timedelta

        if days <= 0:
            return []
        cutoff = (
            datetime.fromisoformat(now) if now else datetime.now(UTC)
        ) - timedelta(days=days)
        deleted: list[str] = []
        if not self.projects_dir.exists():
            return deleted
        for root in sorted(self.projects_dir.iterdir()):
            if not (root / "project.json").exists():
                continue
            try:
                project = self.get_project(root.name, owner=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("retention: skip unreadable project %s: %s", root, exc)
                continue
            stamps = [str(project.meta.get("updated_at") or project.meta.get("created_at") or "")]
            stamps += [str(v.updated_at) for v in project.variants.values()]
            latest = max((t for t in stamps if t), default="")
            if not latest:
                continue
            try:
                last_touch = datetime.fromisoformat(latest)
            except ValueError:
                continue
            if last_touch.tzinfo is None:
                last_touch = last_touch.replace(tzinfo=UTC)
            if last_touch >= cutoff:
                continue
            active = any(
                j.project_id == project.id and j.status in ("pending", "running")
                for j in self.jobs.list()
            )
            if active:
                continue
            owner = project.meta.get("owner") or LOCAL_OWNER
            with self._lock(project.id):
                project.delete()
                self._cache.pop(project.id, None)
            self.events.record(
                owner, "project_purged", project_id=project.id, last_touch=latest, days=days
            )
            deleted.append(project.id)
        if deleted:
            logger.info("retention: purged %d project(s) older than %d days", len(deleted), days)
        for owner_name in self.events.owners():
            trimmed = self.events.trim(owner_name, days, now=now)
            if trimmed:
                logger.info("retention: trimmed %d event(s) for %s", trimmed, owner_name)
        return deleted

    def start_retention(self, days: int, interval_s: float = 24 * 3600) -> None:
        """Purge now, then once per ``interval_s`` on a daemon thread until shutdown."""
        self._retention_stop = threading.Event()

        def _loop() -> None:
            while not self._retention_stop.is_set():
                try:
                    self.purge_stale_projects(days)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("retention sweep failed: %s", exc)
                self._retention_stop.wait(interval_s)

        self._retention_thread = threading.Thread(target=_loop, name="retention", daemon=True)
        self._retention_thread.start()

    def shutdown(self) -> None:
        stop = getattr(self, "_retention_stop", None)
        if stop is not None:
            stop.set()
        self.jobs.shutdown()


# ---- helpers -----------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _normalize_hex(color: str) -> str:
    c = (color or "").strip().lstrip("#")
    if len(c) in (6, 8) and all(ch in "0123456789abcdefABCDEF" for ch in c):
        return "#" + c.lower()
    return "#ffffff"


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    """Unix zip members record their mode in the high 16 bits of external_attr."""
    return stat.S_ISLNK(info.external_attr >> 16)


def _now() -> str:
    from ..design.document import utc_now

    return utc_now()


def _apply_decomposition(doc: DesignDocument, result, store: AssetStore, size) -> None:
    """Replace the single flat element with recovered elements (provenance: recovered)."""
    from ..design.document import Element

    w, h = size
    bg_ref = store.put(result.background, "background_inpainted")
    doc.elements = [
        Element(
            id="background",
            kind="image",
            name="Background (inferred)",
            role="background",
            geometry=Geometry(0, 0, w, h),
            z_index=0,
            asset=bg_ref,
            priority=9,
            allowed=AllowedTransforms(move=False, scale_free=True, crop=True),
            provenance=Provenance(
                origin="recovered",
                confidence=0.6,
                notes=f"background fill: {result.method.get('background_fill')}",
            ),
            role_confidence=0.9,
        )
    ]
    for i, rec in enumerate(result.elements, start=1):
        ref = store.put(rec.image, f"{rec.role_guess}_{i}")
        x, y, bw, bh = rec.bbox
        effects = {}
        if rec.text:
            effects["recovered_text"] = rec.text
            effects["recovered_text_confidence"] = round(rec.confidence, 3)
        if rec.text_color:
            effects["recovered_text_color"] = rec.text_color
        priority = 1 if rec.role_guess in ("headline", "logo") else 2
        doc.elements.append(
            Element(
                id=new_id("el"),
                kind="image",
                name=(
                    rec.text.splitlines()[0][:32] if rec.text else rec.role_guess.replace("_", " ")
                ),
                role=rec.role_guess,
                geometry=Geometry(x, y, bw, bh),
                z_index=i,
                asset=ref,
                priority=priority,
                allowed=AllowedTransforms(scale_free=False, crop=False),
                provenance=Provenance(
                    origin="recovered",
                    confidence=rec.confidence,
                    notes="; ".join(rec.notes)[:300],
                ),
                role_confidence=rec.confidence,
                effects=effects,
            )
        )
    doc.normalize_z()
    from ..design.adapter import propose_constraints

    doc.constraints = propose_constraints(doc)


_STYLE_STR = ("font_family", "weight", "color", "align")
_STYLE_NUM = ("font_size", "line_height", "letter_spacing")
_STYLE_BOOL = ("italic", "uppercase")


def _apply_style_to_runs(text, style: dict) -> None:
    """Apply a style edit to every run, keeping what makes the runs different.

    Only fields that actually change against the primary run are touched, so a
    colour or size edit made in the single-style editor does not flatten a bold or
    differently coloured second run: sizes scale by the same ratio and every run
    keeps its own value for fields the edit leaves alone.
    """
    if not style or not text.runs:
        return
    primary = text.primary_style
    changes: dict[str, object] = {}
    for key in _STYLE_STR:
        if key in style and str(style[key]) != getattr(primary, key):
            changes[key] = str(style[key])
    for key in _STYLE_NUM:
        if key in style and float(style[key]) != float(getattr(primary, key)):
            changes[key] = float(style[key])
    for key in _STYLE_BOOL:
        if key in style and bool(style[key]) != bool(getattr(primary, key)):
            changes[key] = bool(style[key])
    if not changes:
        return
    ratio = None
    if "font_size" in changes:
        base = float(primary.font_size) or 1.0
        ratio = float(changes["font_size"]) / base
    for run in text.runs:
        st = run.style
        for key, value in changes.items():
            if key == "font_size":
                st.font_size = round(float(st.font_size) * ratio, 2) if ratio else float(value)
            elif key == "letter_spacing" and run is not text.runs[0] and st.letter_spacing:
                # keep a run's own tracking unless the edit clears it
                st.letter_spacing = float(value) if not value else st.letter_spacing
            else:
                setattr(st, key, value)


def _runs_from_payload(raw, primary) -> list:
    """Validate an explicit ``runs`` edit: [{"text": ..., "style": {...}}, ...]."""
    from ..design.document import TextRun, TextStyle

    if not isinstance(raw, list) or not raw:
        raise ServiceError("runs must be a non-empty list of {text, style}", 400)
    runs = []
    for item in raw[:64]:
        if not isinstance(item, dict) or "text" not in item:
            raise ServiceError("each run needs a text", 400)
        st = TextStyle(**{k: getattr(primary, k) for k in TextStyle.__dataclass_fields__})
        style = item.get("style") or {}
        if not isinstance(style, dict):
            raise ServiceError("run style must be an object", 400)
        for key in _STYLE_STR:
            if key in style:
                setattr(st, key, str(style[key]))
        for key in _STYLE_NUM:
            if key in style:
                setattr(st, key, float(style[key]))
        for key in _STYLE_BOOL:
            if key in style:
                setattr(st, key, bool(style[key]))
        text = str(item["text"])
        if text:
            runs.append(TextRun(text=text, style=st))
    if not runs:
        raise ServiceError("runs must contain text", 400)
    return runs


def _import_notes(doc: DesignDocument, ext: str, decomposition=None) -> list[str]:
    notes: list[str] = []
    if ext != ".psd" and decomposition is None:
        notes.append(
            "Flat image imported as a single picture. Text, logo and product were not separated; "
            "variants will need review."
        )
    elif ext != ".psd":
        recovered = [
            e for e in doc.elements if e.provenance.origin == "recovered" and not e.is_background
        ]
        texts = [e for e in recovered if e.effects.get("recovered_text")]
        notes.append(
            f"Flat image decomposed: {len(recovered)} element(s) recovered "
            f"({len(texts)} text block(s) read by OCR, unverified). Confirm roles, then convert "
            "text blocks to editable text."
        )
        notes.extend(decomposition.notes)
    low = [e for e in doc.elements if e.role_confidence < 0.6 and not e.is_background]
    if low:
        notes.append(f"{len(low)} element(s) have low-confidence roles; confirm them.")
    missing = [f for f in doc.fonts if f.status != "available"]
    if missing:
        names = ", ".join(sorted({f.family for f in missing}))
        notes.append(f"Fonts not available and substituted: {names}.")
    unsupported = sorted({
        k for e in doc.elements for k in e.effects
        if k not in ("drop_shadow", "unsupported_text_attributes")
    })
    if unsupported:
        notes.append("Unsupported layer effects are ignored: " + ", ".join(unsupported[:6]))
    attrs = sorted({
        a for e in doc.elements for a in (e.effects.get("unsupported_text_attributes") or [])
    })
    if attrs:
        notes.append(
            "Text attributes not reproduced by the renderer (kept as plain runs): "
            + ", ".join(attrs) + "."
        )
    mixed = [e for e in doc.elements if e.text is not None and len(e.text.runs) > 1]
    if mixed:
        notes.append(
            f"{len(mixed)} text element(s) keep several styled runs (mixed weight, size or "
            "colour); they are fitted and rendered per run."
        )
    return notes


def _brief_from_record(rec: VariantRecord) -> VariantBrief:
    d = dict(rec.brief or {})
    return VariantBrief(
        width=int(d.get("width") or rec.width),
        height=int(d.get("height") or rec.height),
        name=str(d.get("name") or rec.name),
        text_overrides=dict(d.get("text_overrides") or {}),
        locale=d.get("locale"),
        hidden_elements=list(d.get("hidden_elements") or []),
        channel_preset=d.get("channel_preset"),
        row=d.get("row") or None,
        direction=d.get("direction") or None,
    )


def _family_from_dict(d: dict) -> Family:
    return Family(
        name=str(d["name"]),
        text=Region(**d["text"]),
        subject=Region(**d["subject"]),
        logo=Region(**d["logo"]),
        text_align=str(d.get("text_align", "left")),
        subject_first=bool(d.get("subject_first", False)),
        traits=dict(d.get("traits") or {}),
    )


def _families_from_meta(meta: dict) -> dict[str, dict[str, Family]]:
    """Stored families keyed by direction key, then aspect class (legacy: flat by class)."""
    raw = meta.get("families") or {}
    nested = raw if all(
        isinstance(v, dict) and "name" not in v for v in raw.values()
    ) else {"": raw}
    out: dict[str, dict[str, Family]] = {}
    for dkey, fams in nested.items():
        for cls, d in (fams or {}).items():
            try:
                out.setdefault(dkey, {})[cls] = _family_from_dict(d)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("ignoring stored family for %s: %s", cls, exc)
    return out


def _normalize_direction(raw: object, where: str) -> dict | None:
    try:
        return parse_direction(raw)
    except ValueError as exc:
        raise ServiceError(f"{where}: {exc}", 400) from exc


def _normalize_rows(spec: dict, doc: DesignDocument) -> list[dict]:
    """Validate the campaign rows of a variants request (``[]`` when none were given)."""
    raw = spec.get("rows")
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ServiceError("rows must be a list of {id, label, text_overrides, locale}", 400)
    if len(raw) > MAX_CAMPAIGN_ROWS:
        raise ServiceError(f"too many rows (max {MAX_CAMPAIGN_ROWS})", 400)
    rows: list[dict] = []
    seen: set[str] = set()
    for idx, r in enumerate(raw, 1):
        if not isinstance(r, dict):
            raise ServiceError("each row must be an object", 400)
        rid = str(r.get("id") or f"row{idx}")[:64]
        if not is_safe_id(rid):
            raise ServiceError(f"row id {rid!r}: use letters, digits, _ and - only", 400)
        if rid in seen:
            raise ServiceError(f"duplicate row id {rid!r}", 400)
        seen.add(rid)
        label = str(r.get("label") or rid).strip()[:120] or rid
        overrides = r.get("text_overrides") or {}
        if not isinstance(overrides, dict):
            raise ServiceError(f"row {rid!r}: text_overrides must be an object", 400)
        clean: dict[str, str] = {}
        for eid, text in overrides.items():
            if not doc.has_element(str(eid)):
                raise ServiceError(f"row {rid!r}: unknown element '{eid}' in text_overrides", 400)
            if text is None or str(text) == "":
                continue
            clean[str(eid)] = str(text)
        hidden = [str(h) for h in (r.get("hidden_elements") or [])]
        for h in hidden:
            if not doc.has_element(h):
                raise ServiceError(f"row {rid!r}: unknown element '{h}' in hidden_elements", 400)
        locale = str(r["locale"]).strip()[:16] if r.get("locale") else None
        rows.append({
            "id": rid, "label": label, "text_overrides": clean,
            "hidden_elements": hidden, "locale": locale,
            "direction": _normalize_direction(r.get("direction"), f"row {rid!r} direction"),
        })
    return rows


_ROW_ID_COLUMNS = ("row", "id", "row_id")
_LABEL_COLUMNS = ("label", "name", "title", "message")
_LOCALE_COLUMNS = ("locale", "language", "lang")
_DIRECTION_COLUMNS = ("direction", "creative", "style")


def parse_campaign_table(doc: DesignDocument, text: str) -> dict:
    """Read a CSV/TSV table into campaign rows.

    The header names the text elements of the design (by element name, id or, when
    unique, role) plus optional ``row``/``id``, ``label`` and ``locale`` columns. Cells
    left empty keep the master copy. Columns that match nothing are reported and
    ignored; verbatim (protected) copy is never overridden and is reported too.
    """
    text = (text or "").lstrip("\ufeff")
    if not text.strip():
        raise ServiceError("the table is empty", 400)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    raw_rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if len(raw_rows) < 2:
        raise ServiceError("the table needs a header row and at least one data row", 400)
    header = [h.strip() for h in raw_rows[0]]
    texts = [e for e in doc.elements if e.kind == "text" and e.text is not None]
    by_key: dict[str, Element] = {}
    for e in texts:
        by_key.setdefault(e.id.lower(), e)
        by_key.setdefault(e.name.strip().lower(), e)
    roles: dict[str, list[Element]] = {}
    for e in texts:
        roles.setdefault(e.role.lower(), []).append(e)
    for role, elems in roles.items():
        if len(elems) == 1:
            by_key.setdefault(role, elems[0])
    columns: list[tuple[str, str | None]] = []
    ignored: list[str] = []
    protected_cols: list[str] = []
    for h in header:
        key = h.lower()
        if key in _ROW_ID_COLUMNS:
            columns.append(("id", None))
        elif key in _LABEL_COLUMNS:
            columns.append(("label", None))
        elif key in _LOCALE_COLUMNS:
            columns.append(("locale", None))
        elif key in _DIRECTION_COLUMNS:
            columns.append(("direction", None))
        elif key in by_key:
            e = by_key[key]
            if e.text is not None and e.text.protected:
                columns.append(("protected", e.id))
                protected_cols.append(e.name)
            else:
                columns.append(("text", e.id))
        else:
            columns.append(("ignore", None))
            if h:
                ignored.append(h)
    if len(raw_rows) - 1 > MAX_CAMPAIGN_ROWS:
        raise ServiceError(f"too many rows (max {MAX_CAMPAIGN_ROWS})", 400)
    rows: list[dict] = []
    seen: set[str] = set()
    bad_directions: list[str] = []
    for idx, cells in enumerate(raw_rows[1:], 1):
        row = {"id": f"row{idx}", "label": "", "text_overrides": {}, "locale": None,
               "direction": None}
        for (kind, eid), cell in zip(columns, cells, strict=False):
            cell = cell.strip()
            if not cell:
                continue
            if kind == "id":
                row["id"] = re.sub(r"[^A-Za-z0-9_-]", "_", cell)[:64] or row["id"]
            elif kind == "label":
                row["label"] = cell[:120]
            elif kind == "locale":
                row["locale"] = cell[:16]
            elif kind == "direction":
                try:
                    row["direction"] = parse_direction(cell)
                except ValueError as exc:
                    bad_directions.append(f"row {idx}: {exc}")
            elif kind == "text":
                row["text_overrides"][eid] = cell
        if row["id"] in seen:
            base = row["id"][:56]
            n = 2
            while f"{base}_{n}" in seen:
                n += 1
            row["id"] = f"{base}_{n}"
        seen.add(row["id"])
        row["label"] = row["label"] or row["id"]
        rows.append(row)
    notes: list[str] = []
    if ignored:
        notes.append("Ignored columns (no text element with that name, id or role): "
                     + ", ".join(ignored) + ".")
    if protected_cols:
        notes.append("Verbatim copy is never overridden by a table; column(s) ignored: "
                     + ", ".join(sorted(set(protected_cols))) + ".")
    used = sorted({eid for _, eid in columns if eid} - {eid for k, eid in columns
                                                        if k == "protected"})
    if not used:
        notes.append("No column matched a text element; every row would repeat the master copy.")
    if bad_directions:
        notes.append("Directions not understood (left empty): " + "; ".join(bad_directions[:5])
                     + ". Use tokens such as copy, subject, text-left, text-right, text-top, "
                       "subject-top, center, stack, side.")
    return {
        "rows": rows,
        "columns": [{"header": h, "kind": k, "element_id": eid}
                    for h, (k, eid) in zip(header, columns, strict=False)],
        "ignored_columns": ignored,
        "notes": notes,
        "elements": [{"id": e.id, "name": e.name, "role": e.role,
                      "protected": bool(e.text is not None and e.text.protected)} for e in texts],
    }


def _normalize_targets(spec: dict) -> list[dict]:
    targets: list[dict] = []
    for pid in spec.get("preset_ids") or []:
        p = find_preset(str(pid))
        if p is None:
            raise ServiceError(f"unknown preset '{pid}'", 400)
        targets.append({"width": p.width, "height": p.height, "name": p.name, "preset_id": p.id})
    for t in spec.get("targets") or []:
        try:
            w, h = int(t["width"]), int(t["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceError("targets need integer width and height", 400) from exc
        if w < Config.MIN_ELEMENT_SIZE or h < Config.MIN_ELEMENT_SIZE:
            raise ServiceError("target too small", 400)
        if w > MAX_TARGET_SIDE or h > MAX_TARGET_SIDE:
            raise ServiceError(f"target exceeds {MAX_TARGET_SIDE}px", 400)
        targets.append({"width": w, "height": h, "name": str(t.get("name") or f"{w}x{h}")})
    return targets


def _apply_op(doc: DesignDocument, op: dict) -> None:
    kind = op.get("op")
    if kind == "set_geometry":
        e = doc.element(op["element_id"])
        if e.locked:
            raise ServiceError(f"element {e.id} is locked", 409)
        g = op["geometry"]
        e.geometry = Geometry(
            float(g.get("x", e.geometry.x)),
            float(g.get("y", e.geometry.y)),
            float(g.get("width", e.geometry.width)),
            float(g.get("height", e.geometry.height)),
            float(g.get("rotation", e.geometry.rotation)),
        )
        e.provenance = Provenance(origin="user", confidence=1.0, notes="geometry edited")
    elif kind == "set_text":
        e = doc.element(op["element_id"])
        if e.kind != "text" or e.text is None:
            raise ServiceError("element is not text", 400)
        if e.locked:
            raise ServiceError(f"element {e.id} is locked", 409)
        if "runs" in op:
            e.text.runs = _runs_from_payload(op["runs"], e.text.primary_style)
        elif "text" in op and op["text"] is not None:
            # keeps the styled runs around the edited characters (finding 3)
            e.text.edit_text(str(op["text"]))
        style = op.get("style") or {}
        _apply_style_to_runs(e.text, style)
        if "protected" in op:
            e.text.protected = bool(op["protected"])
        if "max_lines" in op:
            e.text.max_lines = int(op["max_lines"]) if op["max_lines"] else None
        if "translations" in op:
            raw = op.get("translations") or {}
            if not isinstance(raw, dict):
                raise ServiceError("translations must be an object of locale -> text", 400)
            e.text.translations = {
                str(k).strip(): str(v) for k, v in raw.items() if str(k).strip() and str(v).strip()
            }
        if "locale" in op and op["locale"]:
            e.text.locale = str(op["locale"])[:16]
        e.provenance = Provenance(origin="user", confidence=1.0, notes="text edited")
    elif kind == "convert_to_text":
        e = doc.element(op["element_id"])
        if e.kind == "text":
            raise ServiceError("element is already text", 400)
        from ..design.document import TextContent, TextRun, TextStyle

        text = str(op.get("text") or e.effects.get("recovered_text") or "").strip()
        if not text:
            raise ServiceError("text is required to convert this element", 400)
        style_in = op.get("style") or {}
        lines = max(1, text.count("\n") + 1)
        st = TextStyle(
            font_family=str(style_in.get("font_family", "DejaVu Sans")),
            font_size=float(style_in.get("font_size", max(8.0, e.geometry.height / lines * 0.75))),
            weight=str(style_in.get("weight", "regular")),
            color=str(style_in.get("color", e.effects.get("recovered_text_color", "#000000"))),
            align=str(style_in.get("align", "left")),
        )
        e.kind = "text"
        e.text = TextContent(runs=[TextRun(text=text, style=st)])
        e.asset = None  # the raster crop is no longer the source of truth
        e.allowed = AllowedTransforms(reflow=True)
        e.role_confidence = 1.0
        e.provenance = Provenance(
            origin="user", confidence=1.0, notes="converted from recovered raster"
        )
        for key in ("recovered_text", "recovered_text_confidence", "recovered_text_color"):
            e.effects.pop(key, None)
    elif kind == "set_role":
        e = doc.element(op["element_id"])
        e.role = str(op["role"])
        e.role_confidence = 1.0
        e.provenance = Provenance(origin="user", confidence=1.0, notes="role confirmed")
    elif kind == "set_flags":
        e = doc.element(op["element_id"])
        for key in ("locked", "visible"):
            if key in op:
                setattr(e, key, bool(op[key]))
        if "priority" in op:
            e.priority = max(1, min(9, int(op["priority"])))
        if "name" in op:
            e.name = str(op["name"])[:120]
    elif kind == "set_allowed":
        e = doc.element(op["element_id"])
        a = op.get("allowed") or {}
        e.allowed = AllowedTransforms(
            move=bool(a.get("move", e.allowed.move)),
            scale_uniform=bool(a.get("scale_uniform", e.allowed.scale_uniform)),
            scale_free=bool(a.get("scale_free", e.allowed.scale_free)),
            crop=bool(a.get("crop", e.allowed.crop)),
            reflow=bool(a.get("reflow", e.allowed.reflow)),
            hide=bool(a.get("hide", e.allowed.hide)),
        )
    elif kind == "set_z":
        doc.reorder(op["element_id"], int(op["z_index"]))
    elif kind == "delete_element":
        if not doc.remove_element(op["element_id"]):
            raise ServiceError("element not found", 404)
    elif kind == "add_constraint":
        c = op.get("constraint") or {}
        constraint = Constraint(
            id=str(c.get("id") or new_id("c")),
            type=str(c["type"]),
            elements=[str(x) for x in c.get("elements", [])],
            params=dict(c.get("params") or {}),
            hard=bool(c.get("hard", True)),
            provenance=Provenance(origin="user", confidence=1.0),
        )
        doc.add_constraint(constraint)
    elif kind == "remove_constraint":
        if not doc.remove_constraint(op["constraint_id"]):
            raise ServiceError("constraint not found", 404)
    elif kind == "set_constraint":
        for c in doc.constraints:
            if c.id == op["constraint_id"]:
                if "enabled" in op:
                    c.enabled = bool(op["enabled"])
                if "hard" in op:
                    c.hard = bool(op["hard"])
                if "params" in op:
                    c.params.update(dict(op["params"]))
                if c.provenance.origin == "generated":
                    c.provenance = Provenance(origin="user", confidence=1.0, notes="reviewed")
                break
        else:
            raise ServiceError("constraint not found", 404)
    elif kind == "set_metadata":
        doc.metadata.update(dict(op.get("metadata") or {}))
    elif kind == "rename":
        doc.name = str(op["name"])[:120]
    else:
        raise ServiceError(f"unknown op '{kind}'", 400)


def env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")
