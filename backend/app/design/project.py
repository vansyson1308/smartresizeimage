"""File-backed project: document, assets, variants, history and undo.

Layout of a project directory::

    project.json          # ProjectMeta + current DesignDocument
    assets/<hash>.png     # content-addressed originals
    variants/<id>.png     # rendered outputs
    variants/<id>.json    # plan + quality report + approval state
    history/<n>.json      # document snapshots (for undo / version history)

Everything is plain files so a project survives process restarts and can be
zipped, moved and reopened.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .assets import AssetStore
from .document import DesignDocument, new_id, utc_now
from .serialize import document_from_dict, document_to_dict

logger = logging.getLogger("autobanner.design.project")

PROJECT_SCHEMA_VERSION = "1.0"
MAX_HISTORY = 200
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def safe_relative_path(root: str | Path, rel: str | None) -> Path | None:
    """Resolve ``rel`` inside ``root`` or return ``None`` when it would escape.

    Paths recorded in a project (variant images, reports, assets) come from files a
    user can hand-edit or import, so they are untrusted: absolute paths, traversal,
    backslashes, NUL bytes and symlinks that point outside the project root are all
    refused. The returned path is ``root / rel`` (unresolved) so callers keep working
    with the project directory they know.
    """
    if not rel or not isinstance(rel, str):
        return None
    if "\x00" in rel or "\\" in rel or rel.startswith("/") or rel.startswith("~"):
        return None
    parts = Path(rel).parts
    if not parts or any(part in ("..", ".") for part in parts) or Path(rel).is_absolute():
        return None
    root = Path(root)
    candidate = root / rel
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved == resolved_root or resolved_root not in resolved.parents:
        return None
    return candidate


def is_safe_id(value: object) -> bool:
    return isinstance(value, str) and bool(_SAFE_ID.match(value))


@dataclass
class VariantRecord:
    """Index entry for one rendered variant."""

    id: str
    name: str
    width: int
    height: int
    status: str = "pending"  # pending | running | done | failed | cancelled
    verdict: str = "not_evaluated"  # accepted | needs_review | failed | not_evaluated
    approval: str = "none"  # none | approved | rejected
    approval_reason: str = ""
    image_path: str | None = None
    report_path: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    brief: dict = field(default_factory=dict)
    document_version: int = 0
    job_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> VariantRecord:
        fields = VariantRecord.__dataclass_fields__
        return VariantRecord(**{k: v for k, v in d.items() if k in fields})


class Project:
    """A campaign/brand project rooted at a directory."""

    def __init__(self, root: str | Path, document: DesignDocument, meta: dict | None = None):
        self.root = Path(root)
        self.document = document
        self.meta: dict[str, Any] = meta or {}
        self.assets = AssetStore(self.root / "assets")
        self.variants: dict[str, VariantRecord] = {}
        self._history_dir = self.root / "history"
        self._variants_dir = self.root / "variants"
        self._corrections_dir = self.root / "corrections"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._variants_dir.mkdir(parents=True, exist_ok=True)
        self._corrections_dir.mkdir(parents=True, exist_ok=True)

    # ---- lifecycle -----------------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, document: DesignDocument, *, name: str | None = None,
               owner: str | None = None, brand: str | None = None) -> Project:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "project_schema_version": PROJECT_SCHEMA_VERSION,
            "id": new_id("proj"),
            "name": name or document.name,
            "owner": owner,
            "brand": brand,
            "created_at": utc_now(),
            "document_version": 0,
        }
        project = cls(root, document, meta)
        project.save(snapshot=True, label="created")
        return project

    @classmethod
    def load(cls, root: str | Path, *, strict: bool = False) -> Project:
        """Load a project directory.

        The variant index is untrusted input (it travels inside editable archives):
        records whose id or file paths would escape the project are neutralised, or
        rejected with ``ValueError`` when ``strict`` is set (used by imports).
        """
        root = Path(root)
        payload = json.loads((root / "project.json").read_text(encoding="utf-8"))
        version = str(payload.get("project_schema_version", PROJECT_SCHEMA_VERSION))
        if version != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema {version}")
        document = document_from_dict(payload["document"])
        meta = {k: v for k, v in payload.items() if k != "document"}
        project = cls(root, document, meta)
        for element in document.elements:
            if element.asset is not None:
                try:
                    project.assets.path(element.asset)
                except ValueError as exc:
                    if strict:
                        raise ValueError(f"unsafe asset path for '{element.name}'") from exc
                    logger.warning("project %s: %s", root, exc)
        vindex = root / "variants" / "index.json"
        if vindex.exists():
            entries = json.loads(vindex.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                raise ValueError("variant index must be a list")
            for d in entries:
                if not isinstance(d, dict):
                    raise ValueError("variant index entries must be objects")
                rec = VariantRecord.from_dict(d)
                if not is_safe_id(rec.id):
                    if strict:
                        raise ValueError("unsafe variant id in variant index")
                    logger.warning("project %s: dropping variant with unsafe id", root)
                    continue
                for attr in ("image_path", "report_path"):
                    rel = getattr(rec, attr)
                    if rel is None:
                        continue
                    if safe_relative_path(root, rel) is None:
                        if strict:
                            raise ValueError("unsafe path in variant index")
                        logger.warning(
                            "project %s: variant %s has an unsafe %s; neutralised",
                            root, rec.id, attr,
                        )
                        setattr(rec, attr, None)
                        rec.status = "failed"
                        rec.error = "unsafe path in variant index"
                project.variants[rec.id] = rec
        return project

    def variant_file(self, variant_id: str, kind: str) -> Path | None:
        """Safe on-disk path of a variant's ``image`` or ``report``, else ``None``."""
        rec = self.variants.get(variant_id)
        if rec is None:
            return None
        rel = rec.image_path if kind == "image" else rec.report_path
        path = safe_relative_path(self.root, rel)
        if path is None or not path.is_file():
            return None
        return path

    @property
    def id(self) -> str:
        return str(self.meta.get("id"))

    @property
    def name(self) -> str:
        return str(self.meta.get("name", self.document.name))

    @property
    def document_version(self) -> int:
        return int(self.meta.get("document_version", 0))

    def save(self, *, snapshot: bool = False, label: str = "") -> None:
        """Persist document + meta atomically; optionally snapshot for undo."""
        if snapshot:
            self.meta["document_version"] = self.document_version + 1
            self._write_snapshot(label)
            if not label.startswith("undo to v"):
                # A real edit resets the undo cursor to the newest version.
                self.meta["undo_cursor"] = self.document_version
        self.meta["updated_at"] = utc_now()
        self.document.touch()
        payload = {**self.meta, "document": document_to_dict(self.document)}
        _atomic_write_json(self.root / "project.json", payload)
        self._save_variant_index()

    def _save_variant_index(self) -> None:
        _atomic_write_json(
            self._variants_dir / "index.json",
            [v.to_dict() for v in self.variants.values()],
        )

    # ---- history / undo -------------------------------------------------------------
    def _write_snapshot(self, label: str) -> None:
        version = self.document_version
        payload = {
            "version": version,
            "label": label,
            "saved_at": utc_now(),
            "document": document_to_dict(self.document),
        }
        _atomic_write_json(self._history_dir / f"{version:05d}.json", payload)
        snapshots = sorted(self._history_dir.glob("*.json"))
        for old in snapshots[:-MAX_HISTORY]:
            old.unlink()

    def history(self) -> list[dict]:
        out = []
        for p in sorted(self._history_dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append(
                {"version": d["version"], "label": d.get("label", ""), "saved_at": d["saved_at"]}
            )
        return out

    def undo(self) -> bool:
        """Step back one edit. Returns False when there is nothing left to undo.

        Every undo is recorded as a new version (so history stays linear and a
        later restore can "redo"), while a cursor remembers how far back the
        chain of undos has walked so repeated undos keep going backwards.
        """
        cursor = int(self.meta.get("undo_cursor", self.document_version))
        candidates = []
        for p in sorted(self._history_dir.glob("*.json")):
            version = int(p.stem)
            if version >= cursor:
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if str(d.get("label", "")).startswith("undo to v"):
                continue
            candidates.append(version)
        if not candidates:
            return False
        target = candidates[-1]
        d = json.loads((self._history_dir / f"{target:05d}.json").read_text(encoding="utf-8"))
        self.document = document_from_dict(d["document"])
        self.save(snapshot=True, label=f"undo to v{target}")
        self.meta["undo_cursor"] = target
        self.save()
        return True

    def restore_version(self, version: int) -> None:
        p = self._history_dir / f"{int(version):05d}.json"
        if not p.exists():
            raise KeyError(version)
        d = json.loads(p.read_text(encoding="utf-8"))
        self.document = document_from_dict(d["document"])
        self.save(snapshot=True, label=f"restore v{version}")

    # ---- variants -------------------------------------------------------------------
    def new_variant(self, name: str, width: int, height: int, brief: dict | None = None,
                    job_id: str | None = None) -> VariantRecord:
        rec = VariantRecord(
            id=new_id("var"),
            name=name,
            width=int(width),
            height=int(height),
            brief=dict(brief or {}),
            document_version=self.document_version,
            job_id=job_id,
        )
        self.variants[rec.id] = rec
        self._save_variant_index()
        return rec

    def store_variant_output(
        self,
        variant_id: str,
        image: Image.Image,
        report: dict,
        plan: dict,
        *,
        verdict: str,
    ) -> VariantRecord:
        rec = self.variants[variant_id]
        img_path = self._variants_dir / f"{variant_id}.png"
        image.convert("RGB").save(img_path, format="PNG", optimize=True)
        payload = {"variant": rec.to_dict(), "plan": plan, "quality": report}
        _atomic_write_json(self._variants_dir / f"{variant_id}.json", payload)
        rec.image_path = f"variants/{variant_id}.png"
        rec.report_path = f"variants/{variant_id}.json"
        rec.status = "done"
        rec.verdict = verdict
        rec.updated_at = utc_now()
        self._save_variant_index()
        return rec

    def mark_variant(self, variant_id: str, status: str, error: str | None = None) -> VariantRecord:
        rec = self.variants[variant_id]
        rec.status = status
        rec.error = error
        rec.updated_at = utc_now()
        self._save_variant_index()
        return rec

    def set_approval(self, variant_id: str, approval: str, reason: str = "") -> VariantRecord:
        if approval not in ("none", "approved", "rejected"):
            raise ValueError(approval)
        rec = self.variants[variant_id]
        rec.approval = approval
        rec.approval_reason = reason
        rec.updated_at = utc_now()
        self._save_variant_index()
        return rec

    # ---- correction history (H5) ------------------------------------------------------
    def record_rejection(self, variant_id: str, reason: str, plan: dict) -> Path:
        """Snapshot a rejected variant's plan and reason (the plan is overwritten on
        regeneration, so the evidence must be kept here)."""
        rec = self.variants[variant_id]
        snap = {
            "variant_id": variant_id,
            "name": rec.name,
            "width": rec.width,
            "height": rec.height,
            "reason": reason,
            "ts": utc_now(),
            "placements": list(plan.get("placements") or []),
            "typography": dict(plan.get("typography") or {}),
        }
        path = self._corrections_dir / f"{snap['ts'].replace(':', '')}_{variant_id}.json"
        _atomic_write_json(path, snap)
        return path

    def rejections(self) -> list[dict]:
        out = []
        for path in sorted(self._corrections_dir.glob("*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return out

    def variant_image(self, variant_id: str) -> Image.Image | None:
        path = self.variant_file(variant_id, "image")
        if path is None:
            return None
        with Image.open(path) as img:
            return img.convert("RGB").copy()

    def variant_detail(self, variant_id: str) -> dict | None:
        path = self.variant_file(variant_id, "report")
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_variant(self, variant_id: str) -> bool:
        if variant_id not in self.variants:
            return False
        for kind in ("image", "report"):
            path = self.variant_file(variant_id, kind)
            if path is not None and not path.is_symlink():
                path.unlink()
        self.variants.pop(variant_id, None)
        self._save_variant_index()
        return True

    # ---- housekeeping ------------------------------------------------------------------
    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "brand": self.meta.get("brand"),
            "owner": self.meta.get("owner"),
            "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"),
            "document_version": self.document_version,
            "canvas": {"width": self.document.canvas_width, "height": self.document.canvas_height},
            "element_count": len(self.document.elements),
            "constraint_count": len(self.document.constraints),
            "variant_count": len(self.variants),
            "fonts": [asdict(f) for f in self.document.fonts],
        }


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
