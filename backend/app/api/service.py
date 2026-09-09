"""Project service: the operations behind the HTTP API (also usable in-process).

Keeps HTTP concerns out of the domain logic so tests and CLI tools can drive
the same code path users hit through the web UI.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from ..classifier import SemanticClassifier
from ..config import Config
from ..constants import SUPPORTED_EXTENSIONS
from ..design.adapter import document_from_elements
from ..design.assets import AssetStore
from ..design.decompose import decompose_flat_image
from ..design.document import (
    AllowedTransforms,
    Constraint,
    DesignDocument,
    Geometry,
    Provenance,
    new_id,
)
from ..design.fonts import FontRegistry
from ..design.project import Project, VariantRecord
from ..design.render import render_master
from ..design.serialize import document_from_dict, document_to_dict
from ..design.variant import VariantBrief, generate_variant
from ..parser import get_parser
from .jobs import Job, JobItem, JobManager
from .presets import find_preset

logger = logging.getLogger("autobanner.api.service")

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_VARIANTS_PER_JOB = 48
MAX_TARGET_SIDE = Config.MAX_IMAGE_SIZE


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class ProjectService:
    def __init__(self, data_dir: str | Path, *, max_workers: int = 2, use_ai: bool = False):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir = self.data_dir / "fonts"
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.registry = FontRegistry(extra_dirs=[self.fonts_dir])
        self.jobs = JobManager(max_workers=max_workers)
        self.use_ai = use_ai
        self._locks: dict[str, threading.RLock] = {}
        self._cache: dict[str, Project] = {}
        self._mark_interrupted_variants()

    # ---- helpers ----------------------------------------------------------------------
    def _lock(self, project_id: str) -> threading.RLock:
        return self._locks.setdefault(project_id, threading.RLock())

    def _root(self, project_id: str) -> Path:
        if not project_id or "/" in project_id or ".." in project_id:
            raise ServiceError("invalid project id", 400)
        return self.projects_dir / project_id

    def _mark_interrupted_variants(self) -> None:
        for root in self.projects_dir.iterdir() if self.projects_dir.exists() else []:
            if not (root / "project.json").exists():
                continue
            try:
                project = Project.load(root)
            except Exception as exc:  # noqa: BLE001
                logger.warning("cannot load project %s: %s", root, exc)
                continue
            changed = False
            for rec in project.variants.values():
                if rec.status in ("pending", "running"):
                    rec.status = "failed"
                    rec.error = "interrupted by restart"
                    changed = True
            if changed:
                project.save()

    def get_project(self, project_id: str) -> Project:
        root = self._root(project_id)
        with self._lock(project_id):
            if project_id in self._cache:
                return self._cache[project_id]
            if not (root / "project.json").exists():
                raise ServiceError("project not found", 404)
            project = Project.load(root)
            self._cache[project_id] = project
            return project

    # ---- projects ----------------------------------------------------------------------
    def list_projects(self) -> list[dict]:
        out = []
        for root in sorted(self.projects_dir.iterdir()):
            if (root / "project.json").exists():
                try:
                    out.append(self.get_project(root.name).summary())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("skip project %s: %s", root, exc)
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
        return project

    def add_element(
        self,
        project_id: str,
        *,
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
        project = self.get_project(project_id)
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

    def delete_project(self, project_id: str) -> None:
        project = self.get_project(project_id)
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
    def apply_operations(self, project_id: str, ops: list[dict], label: str = "edit") -> Project:
        project = self.get_project(project_id)
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
        return project

    def undo(self, project_id: str) -> bool:
        project = self.get_project(project_id)
        with self._lock(project_id):
            return project.undo()

    def restore(self, project_id: str, version: int) -> None:
        project = self.get_project(project_id)
        with self._lock(project_id):
            try:
                project.restore_version(version)
            except KeyError as exc:
                raise ServiceError("version not found", 404) from exc

    # ---- rendering ---------------------------------------------------------------------
    def master_preview(self, project_id: str, max_side: int = 1600) -> Image.Image:
        project = self.get_project(project_id)
        doc = project.document
        scale = min(1.0, max_side / max(doc.canvas_width, doc.canvas_height))
        with self._lock(project_id):
            return render_master(doc, project.assets, self.registry, scale=scale)

    def asset_path(self, project_id: str, asset_id: str) -> Path:
        project = self.get_project(project_id)
        if not asset_id.isalnum():
            raise ServiceError("invalid asset id", 400)
        p = project.assets.path(asset_id)
        if not p.exists():
            raise ServiceError("asset not found", 404)
        return p

    # ---- variants ----------------------------------------------------------------------
    def request_variants(
        self, project_id: str, spec: dict, idempotency_key: str | None = None
    ) -> Job:
        project = self.get_project(project_id)
        existing = self.jobs.find_by_key(idempotency_key)
        if existing is not None:
            if existing.project_id != project_id:
                raise ServiceError("idempotency key already used by another project", 409)
            return existing
        targets = _normalize_targets(spec)
        if not targets:
            raise ServiceError("no targets requested", 400)
        if len(targets) > MAX_VARIANTS_PER_JOB:
            raise ServiceError(f"too many targets (max {MAX_VARIANTS_PER_JOB})", 400)
        text_overrides = dict(spec.get("text_overrides") or {})
        for eid in text_overrides:
            if not project.document.has_element(eid):
                raise ServiceError(f"unknown element '{eid}' in text_overrides", 400)
        hidden = list(spec.get("hidden_elements") or [])
        locale = spec.get("locale")

        with self._lock(project_id):
            items: list[tuple[str, str]] = []
            briefs: dict[str, VariantBrief] = {}
            for t in targets:
                brief = VariantBrief(
                    width=t["width"],
                    height=t["height"],
                    name=t["name"],
                    text_overrides=text_overrides,
                    locale=locale,
                    hidden_elements=hidden,
                    channel_preset=t.get("preset_id"),
                )
                rec = project.new_variant(t["name"], t["width"], t["height"], brief.to_dict())
                items.append((rec.id, t["name"]))
                briefs[rec.id] = brief
            project.save()

        def runner(job: Job, item: JobItem, progress) -> dict:
            return self._run_variant(project_id, item.item_id, briefs[item.item_id], job, progress)

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

        job = self.jobs.submit(
            "variants", project_id, items, runner, idempotency_key=idempotency_key,
            on_finish=on_finish,
        )
        with self._lock(project_id):
            for rec_id, _ in items:
                project.variants[rec_id].job_id = job.id
            project.save()
        return job

    def _run_variant(self, project_id: str, variant_id: str, brief: VariantBrief, job: Job,
                     progress) -> dict:
        project = self.get_project(project_id)
        with self._lock(project_id):
            project.mark_variant(variant_id, "running")
            doc_snapshot = document_from_dict(document_to_dict(project.document))
        result = generate_variant(
            doc_snapshot,
            project.assets,
            brief,
            registry=self.registry,
            progress=progress,
            cancel=job.cancel,
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
        return {"variant_id": rec.id, "verdict": rec.verdict}

    def regenerate_variant(self, project_id: str, variant_id: str, spec: dict | None = None) -> Job:
        project = self.get_project(project_id)
        rec = project.variants.get(variant_id)
        if rec is None:
            raise ServiceError("variant not found", 404)
        brief_dict = dict(rec.brief)
        if spec:
            for key in ("text_overrides", "hidden_elements", "locale"):
                if key in spec:
                    brief_dict[key] = spec[key]
        brief = VariantBrief(
            width=rec.width,
            height=rec.height,
            name=rec.name,
            text_overrides=dict(brief_dict.get("text_overrides") or {}),
            locale=brief_dict.get("locale"),
            hidden_elements=list(brief_dict.get("hidden_elements") or []),
            channel_preset=brief_dict.get("channel_preset"),
        )
        with self._lock(project_id):
            rec.brief = brief.to_dict()
            rec.status = "pending"
            rec.approval = "none"
            rec.approval_reason = ""
            rec.error = None
            rec.document_version = project.document_version
            project.save()

        def runner(job: Job, item: JobItem, progress) -> dict:
            return self._run_variant(project_id, item.item_id, brief, job, progress)

        return self.jobs.submit("regenerate", project_id, [(variant_id, rec.name)], runner)

    def set_approval(
        self, project_id: str, variant_id: str, approval: str, reason: str
    ) -> VariantRecord:
        project = self.get_project(project_id)
        if variant_id not in project.variants:
            raise ServiceError("variant not found", 404)
        if approval == "rejected" and not reason.strip():
            raise ServiceError("a rejection needs a reason", 400)
        with self._lock(project_id):
            rec = project.set_approval(variant_id, approval, reason)
            project.save()
        return rec

    def delete_variant(self, project_id: str, variant_id: str) -> None:
        project = self.get_project(project_id)
        with self._lock(project_id):
            if not project.delete_variant(variant_id):
                raise ServiceError("variant not found", 404)
            project.save()

    def variant_detail(self, project_id: str, variant_id: str) -> dict:
        project = self.get_project(project_id)
        rec = project.variants.get(variant_id)
        if rec is None:
            raise ServiceError("variant not found", 404)
        detail = project.variant_detail(variant_id) or {}
        return {
            "variant": rec.to_dict(),
            "plan": detail.get("plan"),
            "quality": detail.get("quality"),
        }

    def variant_image_path(self, project_id: str, variant_id: str) -> Path:
        project = self.get_project(project_id)
        rec = project.variants.get(variant_id)
        if rec is None or not rec.image_path:
            raise ServiceError("variant image not available", 404)
        return project.root / rec.image_path

    # ---- export / import ------------------------------------------------------------------
    def export_deliverables(self, project_id: str, *, fmt: str = "png", only: str = "all",
                            quality: int = 90) -> bytes:
        project = self.get_project(project_id)
        fmt = fmt.lower()
        if fmt not in ("png", "jpeg", "jpg", "webp"):
            raise ServiceError("format must be png, jpeg or webp", 400)
        pil_fmt = {"png": "PNG", "jpeg": "JPEG", "jpg": "JPEG", "webp": "WEBP"}[fmt]
        ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[pil_fmt]
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
                out = io.BytesIO()
                if pil_fmt == "JPEG":
                    img.convert("RGB").save(out, format="JPEG", quality=quality, optimize=True)
                elif pil_fmt == "WEBP":
                    img.save(out, format="WEBP", quality=quality)
                else:
                    img.save(out, format="PNG", optimize=True)
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rec.name)
                fname = f"{safe}_{rec.width}x{rec.height}.{ext}"
                zf.writestr(fname, out.getvalue())
                detail = project.variant_detail(rec.id) or {}
                zf.writestr(
                    f"reports/{safe}_{rec.width}x{rec.height}.json", json.dumps(detail, indent=2)
                )
                manifest["variants"].append(
                    {
                        "id": rec.id,
                        "file": fname,
                        "width": rec.width,
                        "height": rec.height,
                        "verdict": rec.verdict,
                        "approval": rec.approval,
                        "sha256": hashlib.sha256(out.getvalue()).hexdigest(),
                    }
                )
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        return buf.getvalue()

    def export_project(self, project_id: str) -> bytes:
        project = self.get_project(project_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf, self._lock(project_id):
            project.save()
            for path in sorted(project.root.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(project.root)))
        return buf.getvalue()

    def import_project(self, data: bytes) -> Project:
        if len(data) > MAX_UPLOAD_BYTES * 4:
            raise ServiceError("archive too large", 413)
        project_id = new_id("proj")
        root = self._root(project_id)
        root.mkdir(parents=True, exist_ok=False)
        total = 0
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if "project.json" not in names:
                    raise ServiceError("archive is not an AutoBanner project", 400)
                for info in zf.infolist():
                    name = info.filename
                    if name.startswith("/") or ".." in Path(name).parts:
                        raise ServiceError("unsafe path in archive", 400)
                    total += info.file_size
                    if total > MAX_UPLOAD_BYTES * 8:
                        raise ServiceError("archive expands too large", 413)
                    if info.is_dir():
                        continue
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(info))
            project = Project.load(root)
            project.meta["id"] = project_id
            project.meta["imported_at"] = _now()
            project.save()
        except ServiceError:
            shutil.rmtree(root, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(root, ignore_errors=True)
            raise ServiceError(f"import failed: {exc}", 400) from exc
        with self._lock(project_id):
            self._cache[project_id] = project
        return project

    def shutdown(self) -> None:
        self.jobs.shutdown()


# ---- helpers -----------------------------------------------------------------------------


def _normalize_hex(color: str) -> str:
    c = (color or "").strip().lstrip("#")
    if len(c) in (6, 8) and all(ch in "0123456789abcdefABCDEF" for ch in c):
        return "#" + c.lower()
    return "#ffffff"


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
    unsupported = sorted({k for e in doc.elements for k in e.effects if k != "drop_shadow"})
    if unsupported:
        notes.append("Unsupported layer effects are ignored: " + ", ".join(unsupported[:6]))
    return notes


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
        if "text" in op and op["text"] is not None:
            e.text.replace_text(str(op["text"]))
        style = op.get("style") or {}
        st = e.text.primary_style
        for key in ("font_family", "weight", "color", "align"):
            if key in style:
                setattr(st, key, str(style[key]))
        for key in ("font_size", "line_height", "letter_spacing"):
            if key in style:
                setattr(st, key, float(style[key]))
        if "italic" in style:
            st.italic = bool(style["italic"])
        if "uppercase" in style:
            st.uppercase = bool(style["uppercase"])
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
