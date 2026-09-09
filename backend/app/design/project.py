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
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._variants_dir.mkdir(parents=True, exist_ok=True)

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
    def load(cls, root: str | Path) -> Project:
        root = Path(root)
        payload = json.loads((root / "project.json").read_text(encoding="utf-8"))
        version = str(payload.get("project_schema_version", PROJECT_SCHEMA_VERSION))
        if version != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema {version}")
        document = document_from_dict(payload["document"])
        meta = {k: v for k, v in payload.items() if k != "document"}
        project = cls(root, document, meta)
        vindex = root / "variants" / "index.json"
        if vindex.exists():
            for d in json.loads(vindex.read_text(encoding="utf-8")):
                rec = VariantRecord.from_dict(d)
                project.variants[rec.id] = rec
        return project

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

    def variant_image(self, variant_id: str) -> Image.Image | None:
        rec = self.variants.get(variant_id)
        if rec is None or not rec.image_path:
            return None
        with Image.open(self.root / rec.image_path) as img:
            return img.convert("RGB").copy()

    def variant_detail(self, variant_id: str) -> dict | None:
        rec = self.variants.get(variant_id)
        if rec is None or not rec.report_path:
            return None
        return json.loads((self.root / rec.report_path).read_text(encoding="utf-8"))

    def delete_variant(self, variant_id: str) -> bool:
        rec = self.variants.pop(variant_id, None)
        if rec is None:
            return False
        for rel in (rec.image_path, rec.report_path):
            if rel:
                p = self.root / rel
                if p.exists():
                    p.unlink()
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
