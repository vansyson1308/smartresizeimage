"""Orchestrates the quality contract v2 for one rendered output."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field

import numpy as np
import PIL
from PIL import Image

from ..enums import ElementRole
from ..models import CompositionResult, DesignElement, LayoutResult
from .contract import (
    CONTRACT_VERSION,
    CheckResult,
    CheckStatus,
    QualityReport,
    derive_verdict,
    summarize,
)
from .rendered import (
    LegibilityThresholds,
    OcrFn,
    VisibilityThresholds,
    element_visibility_checks,
    ocr_engine_status,
    text_legibility_checks,
)
from .structural import (
    REQUIRED_ROLES_DEFAULT,
    StructuralThresholds,
    canvas_bounds_checks,
    decomposition_checks,
    export_dimensions_check,
    min_text_size_checks,
    required_roles_check,
    text_overlap_checks,
)


@dataclass(frozen=True)
class QualityConfig:
    """All tunables of the evaluator, recorded verbatim in every report."""

    visibility: VisibilityThresholds = field(default_factory=VisibilityThresholds)
    legibility: LegibilityThresholds = field(default_factory=LegibilityThresholds)
    structural: StructuralThresholds = field(default_factory=StructuralThresholds)
    required_roles: frozenset[ElementRole] = REQUIRED_ROLES_DEFAULT
    run_ocr: bool = True

    def to_dict(self) -> dict:
        return {
            "visibility": asdict(self.visibility),
            "legibility": asdict(self.legibility),
            "structural": asdict(self.structural),
            "required_roles": sorted(r.value for r in self.required_roles),
            "run_ocr": self.run_ocr,
        }


def environment_fingerprint() -> dict:
    ocr = ocr_engine_status()
    return {
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "ocr_engine": "tesseract" if ocr["available"] else None,
        "ocr_version": ocr["version"],
    }


def evaluate_composition(
    *,
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    image: Image.Image,
    target_size: tuple[int, int],
    config: QualityConfig | None = None,
    measured_font_px: dict[str, int] | None = None,
    allowed_overlaps: set[tuple[str, str]] | None = None,
    ocr: OcrFn | None = None,
    extra_context: dict | None = None,
) -> QualityReport:
    """Run every implemented check and derive a verdict.

    Args:
        elements: Source elements (with pixel data where available).
        layout_results: Placement used to render ``image``.
        image: The rendered output to judge.
        target_size: Requested output size.
        config: Thresholds; defaults are recorded in the report.
        measured_font_px: Element id -> rendered font size when known.
        allowed_overlaps: Pairs of element ids whose overlap is intentional.
        ocr: Optional OCR function override (tests); default uses tesseract.
        extra_context: Free-form generation configuration to record.
    """
    cfg = config or QualityConfig()
    checks: list[CheckResult] = []

    checks.append(export_dimensions_check(image, target_size))
    checks.extend(decomposition_checks(elements))
    checks.extend(required_roles_check(elements, layout_results, cfg.required_roles))
    checks.extend(canvas_bounds_checks(elements, layout_results, target_size, cfg.structural))
    checks.extend(text_overlap_checks(elements, layout_results, allowed_overlaps, cfg.structural))
    checks.extend(min_text_size_checks(elements, layout_results, measured_font_px))

    visibility = element_visibility_checks(elements, layout_results, image, cfg.visibility)
    checks.extend(visibility)
    visibility_by_id = {c.subject_id: c for c in visibility if c.subject_id}

    if cfg.run_ocr:
        checks.extend(
            text_legibility_checks(
                elements,
                layout_results,
                image,
                ocr=ocr,
                visibility_by_id=visibility_by_id,
                thresholds=cfg.legibility,
            )
        )
    else:
        for elem in elements:
            if elem.text_content and elem.id in visibility_by_id:
                checks.append(
                    CheckResult(
                        check_id="text_legible",
                        status=CheckStatus.NOT_CHECKED,
                        severity=visibility_by_id[elem.id].severity,
                        subject_id=elem.id,
                        message="OCR legibility check disabled by configuration",
                        details={"reason": "run_ocr=false"},
                    )
                )

    verdict = derive_verdict(checks)
    return QualityReport(
        contract_version=CONTRACT_VERSION,
        verdict=verdict,
        checks=checks,
        config={
            "thresholds": cfg.to_dict(),
            "environment": environment_fingerprint(),
            "generation": dict(extra_context or {}),
        },
        summary=summarize(checks),
    )


def evaluate_result(
    result: CompositionResult,
    elements: list[DesignElement],
    target_size: tuple[int, int],
    **kwargs,
) -> QualityReport:
    """Convenience wrapper for a ``CompositionResult``."""
    return evaluate_composition(
        elements=elements,
        layout_results=result.layout_results,
        image=result.image,
        target_size=target_size,
        **kwargs,
    )
