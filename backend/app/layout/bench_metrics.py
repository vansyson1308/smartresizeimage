"""Metrics and evaluation for Phase 2.1 layout benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import DesignElement, LayoutResult
from .bench_thresholds import DEFAULT_THRESHOLDS, BenchThresholds
from .constraints import validate_layout
from .profiles import pick_profile
from .scoring import score_layout


@dataclass(frozen=True)
class BenchMetrics:
    overlap_area_ratio: float
    outside_margin_ratio: float
    min_font_size_ok: bool
    min_font_fail_roles: list[str] = field(default_factory=list)
    font_px_by_role: dict[str, list[int]] = field(default_factory=dict)
    font_unit: str = "px"
    hero_prominence_ratio: float = 0.0
    text_plate_applied_when_busy: bool = True
    text_plate_busy_score: float = 0.0
    text_plate_busy_threshold: float = 0.0
    total_score: float = 0.0
    violations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchEvaluation:
    metrics: BenchMetrics
    passed: bool
    fail_reasons: list[str]


def evaluate_bench_run(
    *,
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    target_size: tuple[int, int],
    text_plate_meta: dict | None,
    busy_expected: bool,
    thresholds: BenchThresholds = DEFAULT_THRESHOLDS,
) -> BenchEvaluation:
    """Evaluate one benchmark run using deterministic Phase 2.1 metrics."""
    role_by_id = {e.id: e.role for e in elements}

    overlap_area_ratio = _overlap_area_ratio(layout_results, role_by_id, target_size)
    profile = pick_profile(*target_size)
    violations = validate_layout(layout_results, profile, target_size, role_by_id)

    outside_margin_count = len([v for v in violations if v.startswith("outside_margin:")])
    content_count = max(1, _content_visible_count(layout_results, role_by_id))
    outside_margin_ratio = outside_margin_count / content_count

    min_font_size_ok, min_font_fail_roles, font_px_by_role = _min_font_size_ok(
        elements,
        layout_results,
    )
    hero_prominence_ratio = _hero_prominence_ratio(layout_results, role_by_id, target_size)

    plate_meta = text_plate_meta or {}
    plate_applied = bool(plate_meta.get("applied", False))
    busy_score = float(plate_meta.get("avg_busy", 0.0))
    busy_threshold = float(plate_meta.get("busy_threshold", thresholds.text_plate_busy_threshold))
    busy_observed = busy_score >= busy_threshold
    requires_plate = bool(busy_expected or busy_observed)
    text_plate_applied_when_busy = (not requires_plate) or plate_applied

    total_score = score_layout(layout_results, profile, target_size, role_by_id)

    metrics = BenchMetrics(
        overlap_area_ratio=overlap_area_ratio,
        outside_margin_ratio=outside_margin_ratio,
        min_font_size_ok=min_font_size_ok,
        min_font_fail_roles=min_font_fail_roles,
        font_px_by_role=font_px_by_role,
        font_unit="px",
        hero_prominence_ratio=hero_prominence_ratio,
        text_plate_applied_when_busy=text_plate_applied_when_busy,
        text_plate_busy_score=busy_score,
        text_plate_busy_threshold=busy_threshold,
        total_score=total_score,
        violations=violations,
    )

    fail_reasons: list[str] = []
    if overlap_area_ratio > thresholds.max_overlap_area_ratio:
        fail_reasons.append("overlap_area_ratio")
    if outside_margin_ratio > thresholds.max_outside_margin_ratio:
        fail_reasons.append("outside_margin_ratio")
    if not min_font_size_ok:
        fail_reasons.append("min_font_size")
    if hero_prominence_ratio < thresholds.min_hero_prominence_ratio:
        fail_reasons.append("hero_prominence")
    if not text_plate_applied_when_busy:
        fail_reasons.append("text_plate")
    if total_score < thresholds.min_total_score:
        fail_reasons.append("total_score")

    return BenchEvaluation(metrics=metrics, passed=not fail_reasons, fail_reasons=fail_reasons)


def _content_visible_count(
    layout_results: list[LayoutResult],
    role_by_id: dict[str, ElementRole],
) -> int:
    return sum(
        1
        for r in layout_results
        if r.visible and role_by_id.get(r.element_id, ElementRole.UNKNOWN) not in BACKGROUND_ROLES
    )


def _overlap_area_ratio(
    layout_results: list[LayoutResult],
    role_by_id: dict[str, ElementRole],
    target_size: tuple[int, int],
) -> float:
    visible = [
        r
        for r in layout_results
        if r.visible and role_by_id.get(r.element_id, ElementRole.UNKNOWN) not in BACKGROUND_ROLES
    ]
    overlap = 0
    for i in range(len(visible)):
        for j in range(i + 1, len(visible)):
            a = visible[i].new_bbox
            b = visible[j].new_bbox
            ix1 = max(a.x, b.x)
            iy1 = max(a.y, b.y)
            ix2 = min(a.x2, b.x2)
            iy2 = min(a.y2, b.y2)
            if ix1 < ix2 and iy1 < iy2:
                overlap += (ix2 - ix1) * (iy2 - iy1)
    canvas = max(1, target_size[0] * target_size[1])
    return overlap / canvas


def _hero_prominence_ratio(
    layout_results: list[LayoutResult],
    role_by_id: dict[str, ElementRole],
    target_size: tuple[int, int],
) -> float:
    hero_area = sum(
        r.new_bbox.area
        for r in layout_results
        if r.visible and role_by_id.get(r.element_id) == ElementRole.HERO_IMAGE
    )
    return hero_area / max(1, target_size[0] * target_size[1])


def _min_font_size_ok(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
) -> tuple[bool, list[str], dict[str, list[int]]]:
    layout_by_id = {r.element_id: r for r in layout_results}
    text_roles = {
        ElementRole.HEADLINE: 20,
        ElementRole.SUBHEADLINE: 16,
        ElementRole.CTA: 12,
    }

    fail_roles: set[str] = set()
    font_px_by_role: dict[str, list[int]] = {"headline": [], "subheadline": [], "cta": []}

    for elem in elements:
        min_font = text_roles.get(elem.role)
        if min_font is None:
            continue
        lr = layout_by_id.get(elem.id)
        if lr is None or not lr.visible:
            continue

        est_font = int(lr.new_bbox.height * 0.48)
        if elem.role == ElementRole.CTA:
            est_font = int(lr.new_bbox.height * 0.55)

        key = elem.role.value
        font_px_by_role.setdefault(key, []).append(est_font)
        if est_font < min_font:
            fail_roles.add(key)

    return not fail_roles, sorted(fail_roles), font_px_by_role
