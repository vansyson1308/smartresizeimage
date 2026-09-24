"""Stateless, thread-safe rendering service.

This is the single entry point the REST API, the CLI and the web studio use.
Every call builds its own :class:`ReLayoutEngine`, so concurrent jobs never
share mutable layout state, and per-size failures are reported instead of being
silently dropped.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from . import __version__
from .classifier import SemanticClassifier
from .config import Config
from .enums import ElementRole
from .exceptions import AutoBannerError, ValidationError
from .export import EncodedImage, ExportOptions, encode_image
from .models import CompositionResult, DesignElement
from .presets import SizePreset
from .relayout import ReLayoutEngine
from .validators import safe_filename, validate_manual_anchors, validate_upload

logger = logging.getLogger("autobanner.service")

MODES = ("phase21", "phase3")
ANCHOR_PRESETS = ("none", "flat_banner_3anchors")

# Roles whose legibility / visibility a media buyer cares about.
_CRITICAL_ROLES = frozenset(
    {
        ElementRole.HEADLINE,
        ElementRole.SUBHEADLINE,
        ElementRole.BODY_TEXT,
        ElementRole.CTA,
        ElementRole.LOGO,
        ElementRole.BADGE,
        ElementRole.LABEL,
    }
)
_TEXT_ROLES = frozenset(
    {ElementRole.HEADLINE, ElementRole.SUBHEADLINE, ElementRole.BODY_TEXT, ElementRole.CTA}
)
_MIN_TEXT_HEIGHT_PX = 9

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class RenderRequest:
    """What to render for one source design."""

    targets: list[SizePreset]
    mode: str = "phase21"
    export: ExportOptions = field(default_factory=ExportOptions)
    # When True and ``export.max_kb`` is unset, each preset's network cap applies.
    respect_platform_limits: bool = True
    manual_anchors: list[dict[str, Any]] | None = None
    anchor_preset: str = "none"
    role_overrides: dict[str, str] = field(default_factory=dict)
    # Move key elements out of platform UI overlay zones (Stories, Reels, ...).
    enforce_safe_zones: bool = True

    def validate(self) -> None:
        if not self.targets:
            raise ValidationError("Select at least one target size")
        if len(self.targets) > Config.MAX_TARGETS_PER_JOB:
            raise ValidationError(
                f"At most {Config.MAX_TARGETS_PER_JOB} target sizes per job "
                f"(got {len(self.targets)})"
            )
        if self.mode not in MODES:
            raise ValidationError(f"Unknown mode '{self.mode}'. Use one of: {', '.join(MODES)}")
        if self.anchor_preset not in ANCHOR_PRESETS:
            raise ValidationError(
                f"Unknown anchor preset '{self.anchor_preset}'. "
                f"Use one of: {', '.join(ANCHOR_PRESETS)}"
            )
        valid_roles = {r.value for r in ElementRole}
        for elem_id, role in self.role_overrides.items():
            if role not in valid_roles:
                raise ValidationError(f"Unknown role '{role}' for element '{elem_id}'")


@dataclass
class RenderedAsset:
    """Outcome of rendering one target size."""

    preset: SizePreset
    filename: str
    status: str  # "ok" | "failed"
    encoded: EncodedImage | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    qa: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    used_fallback: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def describe(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.preset.id,
            "name": self.preset.name,
            "platform": self.preset.platform,
            "width": self.preset.width,
            "height": self.preset.height,
            "file": self.filename if self.ok else None,
            "status": self.status,
            "error": self.error,
            "warnings": list(self.warnings),
            "qa": self.qa,
            "duration_ms": self.duration_ms,
            "used_fallback": self.used_fallback,
        }
        if self.encoded is not None:
            data["export"] = self.encoded.describe()
        return data


@dataclass
class RenderReport:
    """Everything produced by one render call."""

    source: dict[str, Any]
    mode: str
    assets: list[RenderedAsset]
    created_at: str
    duration_ms: int

    @property
    def succeeded(self) -> list[RenderedAsset]:
        return [a for a in self.assets if a.ok]

    @property
    def failed(self) -> list[RenderedAsset]:
        return [a for a in self.assets if not a.ok]

    def manifest(self) -> dict[str, Any]:
        return {
            "generator": f"autobanner/{__version__}",
            "created_at": self.created_at,
            "mode": self.mode,
            "duration_ms": self.duration_ms,
            "source": self.source,
            "summary": {
                "total": len(self.assets),
                "succeeded": len(self.succeeded),
                "failed": len(self.failed),
                "with_warnings": sum(1 for a in self.assets if a.warnings),
            },
            "assets": [a.describe() for a in self.assets],
        }

    def to_zip(self) -> bytes:
        """Return a ZIP with every successful asset and ``manifest.json``."""
        buf = io.BytesIO()
        self.write_zip(buf)
        return buf.getvalue()

    def write_zip(self, target: str | Path | IO[bytes]) -> None:
        """Write the ZIP to a path or binary file object."""
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for asset in self.succeeded:
                assert asset.encoded is not None
                # Encoded images are already compressed; don't waste CPU.
                zf.writestr(
                    zipfile.ZipInfo(asset.filename, date_time=(2020, 1, 1, 0, 0, 0)),
                    asset.encoded.data,
                    compress_type=zipfile.ZIP_STORED,
                )
            zf.writestr("manifest.json", json.dumps(self.manifest(), indent=2))

    def write_to(self, out_dir: str | Path) -> list[Path]:
        """Write assets and ``manifest.json`` into ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for asset in self.succeeded:
            assert asset.encoded is not None
            path = out / asset.filename
            path.write_bytes(asset.encoded.data)
            written.append(path)
        manifest_path = out / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest(), indent=2))
        written.append(manifest_path)
        return written


def flat_banner_anchor_preset(source_size: tuple[int, int]) -> list[dict[str, int | str]]:
    """Default Mascot / MainText / CTA boxes for a typical flat landscape banner."""
    sw, sh = source_size
    return [
        {
            "id": "Mascot",
            "role": "hero_image",
            "x": int(0.56 * sw),
            "y": int(0.22 * sh),
            "width": int(0.32 * sw),
            "height": int(0.66 * sh),
        },
        {
            "id": "MainText",
            "role": "headline",
            "x": int(0.06 * sw),
            "y": int(0.10 * sh),
            "width": int(0.50 * sw),
            "height": int(0.34 * sh),
        },
        {
            "id": "CTA",
            "role": "cta",
            "x": int(0.08 * sw),
            "y": int(0.64 * sh),
            "width": int(0.28 * sw),
            "height": int(0.18 * sh),
        },
    ]


def describe_elements(engine: ReLayoutEngine) -> list[dict[str, Any]]:
    """JSON-friendly summary of the elements detected in the loaded source."""
    out = []
    for elem in engine.elements:
        text = elem.text_content
        if text and len(text) > 80:
            text = text[:80] + "..."
        out.append(
            {
                "id": elem.id,
                "name": elem.name,
                "type": elem.layer_type,
                "role": elem.role.value,
                "priority": elem.priority,
                "bbox": {
                    "x": elem.bbox.x,
                    "y": elem.bbox.y,
                    "width": elem.bbox.width,
                    "height": elem.bbox.height,
                },
                "text": text,
            }
        )
    return out


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_inside(box: tuple[int, int, int, int], safe: tuple[int, int, int, int]) -> bool:
    return box[0] >= safe[0] and box[1] >= safe[1] and box[2] <= safe[2] and box[3] <= safe[3]


def evaluate_qa(
    preset: SizePreset,
    result: CompositionResult,
    elements: list[DesignElement],
    mode: str,
) -> tuple[dict[str, Any], list[str]]:
    """Platform QA: safe-zone compliance and minimum text size.

    Returns a QA dict for the manifest and a list of human readable warnings.
    """
    boxes: list[tuple[str, ElementRole, tuple[int, int, int, int], float]] = []
    by_id = {e.id: e for e in elements}

    if mode == "phase3":
        for anchor in result.metadata.get("redesign", {}).get("anchors", []):
            try:
                role = ElementRole(anchor.get("role", "unknown"))
            except ValueError:
                role = ElementRole.UNKNOWN
            b = anchor["bbox"]
            boxes.append(
                (anchor["element_id"], role,
                 (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]),
                 float(anchor.get("scale", 1.0)))
            )
    else:
        for lr in result.layout_results:
            elem = by_id.get(lr.element_id)
            if elem is None or not lr.visible:
                continue
            boxes.append((elem.id, elem.role, lr.new_bbox.to_tuple(), float(lr.scale_factor)))

    critical = [b for b in boxes if b[1] in _CRITICAL_ROLES]
    warnings: list[str] = []
    qa: dict[str, Any] = {
        "evaluated": bool(critical),
        "critical_elements": len(critical),
    }
    if not critical:
        qa["note"] = "No text/logo/CTA layers detected; supply a layered PSD or anchors for QA."

    if not preset.safe_zone.is_empty:
        safe = preset.safe_zone.to_pixels(preset.width, preset.height)
        violations = [eid for eid, _, box, _ in critical if not _box_inside(box, safe)]
        qa["safe_zone"] = {
            "rect": {"x1": safe[0], "y1": safe[1], "x2": safe[2], "y2": safe[3]},
            "violations": violations,
        }
        if violations:
            warnings.append(
                f"{len(violations)} key element(s) sit in the {preset.platform} UI overlay zone "
                f"and may be covered: {', '.join(violations[:5])}"
            )

    small_text = []
    for eid, role, box, scale in critical:
        if role not in _TEXT_ROLES:
            continue
        elem = by_id.get(eid)
        # For PSD type layers use the original layer height scaled; otherwise the box.
        height = box[3] - box[1]
        if elem is not None and elem.layer_type == "type":
            height = int(elem.bbox.height * scale)
        if 0 < height < _MIN_TEXT_HEIGHT_PX:
            small_text.append(eid)
    qa["small_text"] = small_text
    if small_text:
        warnings.append(
            f"Text may be illegible at {preset.width}x{preset.height} "
            f"(< {_MIN_TEXT_HEIGHT_PX}px tall): {', '.join(small_text[:5])}"
        )

    qa["quality_gates_passed"] = bool(result.gates_passed)
    if result.used_fallback:
        warnings.append("Generative candidate failed quality gates; deterministic layout used")
    return qa, warnings


class RenderService:
    """Thread-safe facade over the layout engines."""

    def __init__(self, use_ai: bool = False, max_upload_bytes: int | None = None) -> None:
        self.use_ai = use_ai
        self.max_upload_bytes = max_upload_bytes
        self._classifier: SemanticClassifier | None = None
        self._classifier_lock = threading.Lock()

    # -- helpers --------------------------------------------------------------------
    def _shared_classifier(self) -> SemanticClassifier:
        with self._classifier_lock:
            if self._classifier is None:
                self._classifier = SemanticClassifier(use_ai=self.use_ai)
            return self._classifier

    def _new_engine(self) -> ReLayoutEngine:
        return ReLayoutEngine(use_ai=self.use_ai, classifier=self._shared_classifier())

    def load(self, file_path: str | Path, *, trusted: bool = False) -> ReLayoutEngine:
        """Validate and parse a source file into a fresh engine."""
        path = str(file_path)
        if not trusted:
            validate_upload(path, self.max_upload_bytes)
        engine = self._new_engine()
        engine.load_file(path)
        return engine

    # -- public API -------------------------------------------------------------------
    def analyze(self, file_path: str | Path, *, display_name: str | None = None) -> dict[str, Any]:
        """Parse and classify a design without rendering."""
        engine = self.load(file_path)
        w, h = engine.source_size
        return {
            "file": display_name or Path(file_path).name,
            "width": w,
            "height": h,
            "source_type": _source_type(engine.elements),
            "layers": len(engine.elements),
            "elements": describe_elements(engine),
        }

    def render(
        self,
        file_path: str | Path,
        request: RenderRequest,
        *,
        display_name: str | None = None,
        progress: ProgressCallback | None = None,
        engine: ReLayoutEngine | None = None,
    ) -> RenderReport:
        """Render ``request.targets`` from the design at ``file_path``."""
        request.validate()
        started = time.perf_counter()
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        engine = engine or self.load(file_path)
        for elem_id, role in request.role_overrides.items():
            if not engine.update_element_role(elem_id, role):
                raise ValidationError(f"Unknown element id '{elem_id}' in role overrides")

        anchors: list[dict[str, int | str]] | None = None
        if request.manual_anchors:
            anchors = validate_manual_anchors(request.manual_anchors, engine.source_size)
        elif request.anchor_preset == "flat_banner_3anchors":
            anchors = flat_banner_anchor_preset(engine.source_size)

        w, h = engine.source_size
        source = {
            "file": display_name or Path(file_path).name,
            "width": w,
            "height": h,
            "sha256": _sha256(file_path),
            "source_type": _source_type(engine.elements),
            "layers": len(engine.elements),
        }

        assets: list[RenderedAsset] = []
        used_names: set[str] = set()
        total = len(request.targets)
        for idx, preset in enumerate(request.targets):
            if progress:
                progress(idx, total, preset.name)
            assets.append(
                self._render_one(engine, preset, request, anchors, used_names)
            )
        if progress:
            progress(total, total, "done")

        return RenderReport(
            source=source,
            mode=request.mode,
            assets=assets,
            created_at=created_at,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def _render_one(
        self,
        engine: ReLayoutEngine,
        preset: SizePreset,
        request: RenderRequest,
        anchors: list[dict[str, int | str]] | None,
        used_names: set[str],
    ) -> RenderedAsset:
        t0 = time.perf_counter()
        opts = request.export
        if opts.max_kb is None and request.respect_platform_limits and preset.max_kb:
            opts = ExportOptions(format=opts.format, quality=opts.quality, max_kb=preset.max_kb)

        base = f"{safe_filename(preset.id)}_{preset.width}x{preset.height}"
        name = base
        n = 2
        while name in used_names:
            name = f"{base}_{n}"
            n += 1
        used_names.add(name)
        filename = f"{name}.{opts.extension}"

        safe_rect = None
        if request.enforce_safe_zones and not preset.safe_zone.is_empty:
            safe_rect = preset.safe_zone.to_pixels(preset.width, preset.height)

        try:
            if request.mode == "phase3":
                result = engine.relayout_redesign(
                    preset.size, manual_anchors=anchors, safe_rect=safe_rect
                )
            else:
                result = engine.relayout(preset.size, safe_rect=safe_rect)
            encoded = encode_image(result.image, opts)
        except AutoBannerError as e:
            logger.warning("Render %s failed: %s", preset.id, e)
            return RenderedAsset(
                preset, filename, "failed", error=str(e),
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        except Exception as e:  # keep the rest of the batch alive
            logger.exception("Unexpected error rendering %s", preset.id)
            return RenderedAsset(
                preset, filename, "failed", error=f"Internal error: {type(e).__name__}",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )

        qa, warnings = evaluate_qa(preset, result, engine.elements, request.mode)
        if not encoded.within_budget:
            warnings.append(
                f"Could not fit under {encoded.max_kb} KB (smallest: {encoded.size_kb} KB); "
                "try WebP/JPEG or simplify the design"
            )
        return RenderedAsset(
            preset,
            filename,
            "ok",
            encoded=encoded,
            warnings=warnings,
            qa=qa,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            used_fallback=bool(result.used_fallback),
        )


def _source_type(elements: list[DesignElement]) -> str:
    if len(elements) == 1 and elements[0].effects.get("_source_type") == "flat_image":
        return "flat_image"
    return "layered"
