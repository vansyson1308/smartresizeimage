"""Scoring for adaptive layout candidates."""

from __future__ import annotations

from ..enums import ElementRole
from ..models import LayoutResult
from .constraints import validate_layout
from .profiles import LayoutProfile


def score_layout(
    layout_results: list[LayoutResult],
    profile: LayoutProfile,
    target_size: tuple[int, int],
    role_by_id: dict[str, ElementRole],
) -> float:
    """Score candidate layout on a normalized 0..100 scale (higher is better)."""
    width, height = target_size
    violations = validate_layout(layout_results, profile, target_size, role_by_id)

    score = 100.0

    outside = len([v for v in violations if v.startswith("outside_margin:")])
    overlap_v = len([v for v in violations if v.startswith("overlap:")])
    min_size_v = len([v for v in violations if v.startswith("min_size:")])
    max_size_v = len([v for v in violations if v.startswith("max_size:")])
    hero_small = 1 if "hero_too_small" in violations else 0

    score -= outside * 35.0
    score -= overlap_v * 2.5
    score -= min_size_v * 0.6
    score -= max_size_v * 0.4
    score -= hero_small * 8.0

    overlap_area = _overlap_area_ratio(layout_results, target_size)
    score -= min(50.0, overlap_area * 140.0)


    col_w = width / max(1, profile.grid_cols)
    row_h = height / max(1, profile.grid_rows)

    visible = [r for r in layout_results if r.visible]

    # Modest reward for grid alignment (bounded).
    align_bonus = 0.0
    for r in visible:
        b = r.new_bbox
        grid_dx = abs((b.x / col_w) - round(b.x / col_w))
        grid_dy = abs((b.y / row_h) - round(b.y / row_h))
        align_bonus += max(0.0, 1.5 - 2.0 * (grid_dx + grid_dy))
    score += min(8.0, align_bonus)

    # Modest reward for hierarchy ordering.
    ranked = sorted(
        [r for r in visible if role_by_id.get(r.element_id) in profile.role_priority],
        key=lambda r: profile.role_priority.get(
            role_by_id.get(r.element_id, ElementRole.UNKNOWN),
            0,
        ),
        reverse=True,
    )
    ordered = 0
    for i in range(1, len(ranked)):
        prev = ranked[i - 1].new_bbox
        cur = ranked[i].new_bbox
        if prev.y <= cur.y:
            ordered += 1
    score += min(6.0, ordered * 1.5)

    return max(0.0, min(100.0, score))


def _overlap_area_ratio(
    layout_results: list[LayoutResult],
    target_size: tuple[int, int],
) -> float:
    visible = [r for r in layout_results if r.visible]
    if not visible:
        return 0.0

    overlap = 0
    canvas = max(1, target_size[0] * target_size[1])

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
    return overlap / canvas
