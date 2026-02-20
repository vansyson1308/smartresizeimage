"""Constraint checks for adaptive layout candidates."""

from __future__ import annotations

from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import LayoutResult
from .profiles import LayoutProfile


def validate_layout(
    layout_results: list[LayoutResult],
    profile: LayoutProfile,
    target_size: tuple[int, int],
    role_by_id: dict[str, ElementRole],
) -> list[str]:
    """Return list of human-readable violations for a candidate layout."""
    width, height = target_size
    violations: list[str] = []

    margin_x = int(profile.margin_pct * width)
    margin_y = int(profile.margin_pct * height)

    visible = [r for r in layout_results if r.visible]
    content_visible = [
        r
        for r in visible
        if role_by_id.get(r.element_id, ElementRole.UNKNOWN) not in BACKGROUND_ROLES
    ]

    # Bounds & min-size checks
    for r in content_visible:
        b = r.new_bbox
        role = role_by_id.get(r.element_id, ElementRole.UNKNOWN)

        if b.x < margin_x or b.y < margin_y or b.x2 > width - margin_x or b.y2 > height - margin_y:
            violations.append(f"outside_margin:{r.element_id}")

        min_size = profile.min_sizes.get(role)
        if min_size is not None:
            min_w = int(min_size[0] * width)
            min_h = int(min_size[1] * height)
            if b.width < min_w or b.height < min_h:
                violations.append(f"min_size:{r.element_id}")

        max_size = profile.max_sizes.get(role)
        if max_size is not None:
            max_w = int(max_size[0] * width)
            max_h = int(max_size[1] * height)
            if b.width > max_w or b.height > max_h:
                violations.append(f"max_size:{r.element_id}")

    # Overlap checks
    for i in range(len(content_visible)):
        for j in range(i + 1, len(content_visible)):
            a = content_visible[i].new_bbox
            b = content_visible[j].new_bbox
            ix1 = max(a.x, b.x)
            iy1 = max(a.y, b.y)
            ix2 = min(a.x2, b.x2)
            iy2 = min(a.y2, b.y2)
            if ix1 < ix2 and iy1 < iy2:
                violations.append(
                    f"overlap:{content_visible[i].element_id}:{content_visible[j].element_id}"
                )

    # Hero prominence check
    hero_area = 0
    for r in visible:
        if role_by_id.get(r.element_id) == ElementRole.HERO_IMAGE:
            hero_area += r.new_bbox.area

    canvas_area = width * height
    if canvas_area > 0 and hero_area / canvas_area < profile.target_hero_ratio * 0.65:
        violations.append("hero_too_small")

    return violations
