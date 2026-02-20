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
    """Score candidate layout: higher is better."""
    width, height = target_size
    violations = validate_layout(layout_results, profile, target_size, role_by_id)

    score = 1000.0 - 120.0 * len(violations)

    col_w = width / max(1, profile.grid_cols)
    row_h = height / max(1, profile.grid_rows)

    visible = [r for r in layout_results if r.visible]

    # Reward grid alignment.
    for r in visible:
        b = r.new_bbox
        grid_dx = abs((b.x / col_w) - round(b.x / col_w))
        grid_dy = abs((b.y / row_h) - round(b.y / row_h))
        score += max(0.0, 8.0 - 10.0 * (grid_dx + grid_dy))

    # Reward hierarchy ordering (higher priority visually earlier/higher).
    ranked = sorted(
        [r for r in visible if role_by_id.get(r.element_id) in profile.role_priority],
        key=lambda r: profile.role_priority.get(
            role_by_id.get(r.element_id, ElementRole.UNKNOWN),
            0,
        ),
        reverse=True,
    )
    for i in range(1, len(ranked)):
        prev = ranked[i - 1].new_bbox
        cur = ranked[i].new_bbox
        if prev.y <= cur.y:
            score += 6.0
        else:
            score -= 6.0

    # Reward consistent spacing.
    if len(visible) > 1:
        gaps: list[float] = []
        sorted_y = sorted(visible, key=lambda r: r.new_bbox.y)
        for i in range(1, len(sorted_y)):
            gap = sorted_y[i].new_bbox.y - sorted_y[i - 1].new_bbox.y2
            if gap >= 0:
                gaps.append(float(gap))
        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            score += max(0.0, 20.0 - variance * 0.02)

    return score
