"""Structural checks on the layout and export contract."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import BoundingBox, DesignElement, LayoutResult
from .contract import CheckResult, CheckStatus, Severity
from .rendered import TEXT_ROLES, severity_for_role

REQUIRED_ROLES_DEFAULT = frozenset({ElementRole.HEADLINE, ElementRole.CTA, ElementRole.LOGO})

# Minimum estimated glyph height (px) per role. Estimation from bbox height is
# uncertain, so estimated values below the floor produce NEEDS_REVIEW, while
# measured font sizes (from typography debug) produce FAIL.
MIN_FONT_PX = {
    ElementRole.HEADLINE: 20,
    ElementRole.SUBHEADLINE: 16,
    ElementRole.BODY_TEXT: 12,
    ElementRole.CTA: 12,
    ElementRole.LABEL: 10,
    ElementRole.BADGE: 10,
}


@dataclass(frozen=True)
class StructuralThresholds:
    max_outside_fraction: float = 0.02
    text_overlap_fail: float = 0.10
    text_overlap_review: float = 0.02


def export_dimensions_check(image: Image.Image, target_size: tuple[int, int]) -> CheckResult:
    ok = image.size == tuple(target_size)
    return CheckResult(
        check_id="export_dimensions",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        severity=Severity.CRITICAL,
        message=(
            "Output has the requested dimensions"
            if ok
            else f"Output is {image.size[0]}x{image.size[1]}, expected "
            f"{target_size[0]}x{target_size[1]}"
        ),
        details={"actual": list(image.size), "expected": list(target_size)},
    )


def required_roles_check(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    required_roles: frozenset[ElementRole] = REQUIRED_ROLES_DEFAULT,
) -> list[CheckResult]:
    """Every required role that exists in the source must remain visible."""
    layout_map = {r.element_id: r for r in layout_results}
    results: list[CheckResult] = []
    for elem in elements:
        if elem.role not in required_roles:
            continue
        layout = layout_map.get(elem.id)
        present = layout is not None and layout.visible
        results.append(
            CheckResult(
                check_id="required_element_present",
                status=CheckStatus.PASS if present else CheckStatus.FAIL,
                severity=Severity.CRITICAL,
                subject_id=elem.id,
                message=(
                    f"{elem.role.value.capitalize()} is placed"
                    if present
                    else f"{elem.role.value.capitalize()} '{elem.name}' was dropped from the layout"
                ),
                details={"role": elem.role.value},
            )
        )
    return results


def canvas_bounds_checks(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    target_size: tuple[int, int],
    thresholds: StructuralThresholds | None = None,
) -> list[CheckResult]:
    th = thresholds or StructuralThresholds()
    w, h = target_size
    layout_map = {r.element_id: r for r in layout_results}
    results: list[CheckResult] = []
    for elem in elements:
        if elem.role in BACKGROUND_ROLES:
            continue
        layout = layout_map.get(elem.id)
        if layout is None or not layout.visible:
            continue
        b = layout.new_bbox
        area = max(1, b.area)
        inter = _intersection_area(b, BoundingBox(0, 0, w, h))
        outside = 1.0 - inter / area
        status = CheckStatus.PASS if outside <= th.max_outside_fraction else CheckStatus.FAIL
        results.append(
            CheckResult(
                check_id="inside_canvas",
                status=status,
                severity=severity_for_role(elem.role),
                subject_id=elem.id,
                message=(
                    f"{elem.role.value.capitalize()} is inside the canvas"
                    if status == CheckStatus.PASS
                    else f"{elem.role.value.capitalize()} '{elem.name}' extends "
                    f"{outside * 100:.0f}% outside the canvas"
                ),
                details={"outside_fraction": round(outside, 4)},
            )
        )
    return results


def text_overlap_checks(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    allowed_overlaps: set[tuple[str, str]] | None = None,
    thresholds: StructuralThresholds | None = None,
) -> list[CheckResult]:
    """Text must not be covered by other content unless explicitly allowed."""
    th = thresholds or StructuralThresholds()
    allowed = allowed_overlaps or set()
    layout_map = {r.element_id: r for r in layout_results}
    content = [
        (e, layout_map[e.id])
        for e in elements
        if e.role not in BACKGROUND_ROLES
        and e.role != ElementRole.DECORATION
        and e.id in layout_map
        and layout_map[e.id].visible
    ]
    results: list[CheckResult] = []
    for elem, layout in content:
        if elem.role not in TEXT_ROLES:
            continue
        area = max(1, layout.new_bbox.area)
        worst = 0.0
        worst_id = None
        for other, other_layout in content:
            if other.id == elem.id:
                continue
            if (elem.id, other.id) in allowed or (other.id, elem.id) in allowed:
                continue
            frac = _intersection_area(layout.new_bbox, other_layout.new_bbox) / area
            if frac > worst:
                worst, worst_id = frac, other.id
        if worst > th.text_overlap_fail:
            status = CheckStatus.FAIL
            message = (
                f"{elem.role.value.capitalize()} box overlaps '{worst_id}' by {worst * 100:.0f}%"
            )
        elif worst > th.text_overlap_review:
            status = CheckStatus.NEEDS_REVIEW
            message = f"{elem.role.value.capitalize()} box touches '{worst_id}'"
        else:
            status = CheckStatus.PASS
            message = f"{elem.role.value.capitalize()} box is clear of other content"
        results.append(
            CheckResult(
                check_id="text_overlap",
                status=status,
                severity=Severity.MAJOR,
                subject_id=elem.id,
                message=message,
                details={"overlap_fraction": round(worst, 4), "other_id": worst_id},
            )
        )
    return results


def min_text_size_checks(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    measured_font_px: dict[str, int] | None = None,
) -> list[CheckResult]:
    """Text must not shrink below the role's minimum glyph size."""
    layout_map = {r.element_id: r for r in layout_results}
    measured = measured_font_px or {}
    results: list[CheckResult] = []
    for elem in elements:
        floor = MIN_FONT_PX.get(elem.role)
        if floor is None:
            continue
        layout = layout_map.get(elem.id)
        if layout is None or not layout.visible:
            continue
        if elem.id in measured:
            px = int(measured[elem.id])
            estimated = False
        else:
            lines = max(1, (elem.text_content or "").count("\n") + 1)
            px = int(layout.new_bbox.height / lines * 0.5)
            estimated = True
        ok = px >= floor
        if ok:
            status = CheckStatus.PASS
        elif estimated:
            status = CheckStatus.NEEDS_REVIEW
        else:
            status = CheckStatus.FAIL
        results.append(
            CheckResult(
                check_id="min_text_size",
                status=status,
                severity=Severity.MAJOR,
                subject_id=elem.id,
                message=(
                    f"{elem.role.value.capitalize()} text size is adequate"
                    if ok
                    else f"{elem.role.value.capitalize()} text is only ~{px}px "
                    f"(minimum {floor}px)"
                ),
                details={"font_px": px, "min_px": floor, "estimated": estimated},
            )
        )
    return results


def decomposition_checks(elements: list[DesignElement]) -> list[CheckResult]:
    """A flat image that was never decomposed cannot be verified semantically."""
    if len(elements) == 1 and elements[0].effects.get("_source_type") == "flat_image":
        return [
            CheckResult(
                check_id="design_understood",
                status=CheckStatus.NEEDS_REVIEW,
                severity=Severity.CRITICAL,
                subject_id=elements[0].id,
                message=(
                    "Flat image was adapted as a single picture; text, logo and product "
                    "placement were not verified. Review before use."
                ),
                details={"reason": "flat_image_not_decomposed"},
            )
        ]
    return []


def _intersection_area(a: BoundingBox, b: BoundingBox) -> int:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix1 < ix2 and iy1 < iy2:
        return (ix2 - ix1) * (iy2 - iy1)
    return 0
