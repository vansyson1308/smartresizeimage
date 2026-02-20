"""Deterministic layout repair helpers for hard-margin constraints."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import BoundingBox, DesignElement, LayoutResult
from .profiles import LayoutProfile
from .typography import fit_text_block


@dataclass
class RepairResult:
    layout: list[LayoutResult]
    repair_applied: bool
    steps: list[str]


def rescale_to_fit(bbox: BoundingBox, max_w: int, max_h: int) -> BoundingBox:
    """Rescale bbox to fit within max bounds, preserving aspect ratio."""
    max_w = max(1, max_w)
    max_h = max(1, max_h)
    if bbox.width <= max_w and bbox.height <= max_h:
        return BoundingBox(bbox.x, bbox.y, bbox.width, bbox.height)

    scale = min(max_w / max(1, bbox.width), max_h / max(1, bbox.height))
    new_w = max(1, int(bbox.width * scale))
    new_h = max(1, int(bbox.height * scale))
    cx, cy = bbox.center
    return BoundingBox(cx - new_w // 2, cy - new_h // 2, new_w, new_h)


def clamp_bbox_to_margins(
    bbox: BoundingBox,
    margins: tuple[int, int],
    canvas_w: int,
    canvas_h: int,
) -> BoundingBox:
    """Return bbox guaranteed to be inside margin-safe region."""
    margin_x, margin_y = margins
    allowed_w = max(1, canvas_w - 2 * margin_x)
    allowed_h = max(1, canvas_h - 2 * margin_y)

    fitted = rescale_to_fit(bbox, allowed_w, allowed_h)

    min_x = margin_x
    min_y = margin_y
    max_x = max(min_x, canvas_w - margin_x - fitted.width)
    max_y = max(min_y, canvas_h - margin_y - fitted.height)

    x = max(min_x, min(max_x, fitted.x))
    y = max(min_y, min(max_y, fitted.y))
    return BoundingBox(x, y, fitted.width, fitted.height)


def apply_repair(
    layout_results: list[LayoutResult],
    elements_by_id: dict[str, DesignElement],
    profile: LayoutProfile,
    target_size: tuple[int, int],
) -> RepairResult:
    """Apply deterministic repair pass to enforce margin-safe bboxes."""
    width, height = target_size
    margin_x = int(profile.margin_pct * width)
    margin_y = int(profile.margin_pct * height)
    steps: list[str] = []
    repaired: list[LayoutResult] = []
    applied = False

    for r in layout_results:
        b = r.new_bbox
        elem = elements_by_id.get(r.element_id)
        role = elem.role if elem else ElementRole.UNKNOWN

        if role in BACKGROUND_ROLES:
            repaired.append(r)
            continue

        role_max = profile.max_sizes.get(role)
        if role_max is not None:
            max_w = min(width - 2 * margin_x, int(role_max[0] * width))
            max_h = min(height - 2 * margin_y, int(role_max[1] * height))
            capped = rescale_to_fit(b, max_w, max_h)
            if (capped.width, capped.height) != (b.width, b.height):
                steps.append(f"cap_max:{r.element_id}")
                b = capped
                applied = True

        b = clamp_bbox_to_margins(b, (margin_x, margin_y), width, height)

        if elem and elem.role in {ElementRole.HEADLINE, ElementRole.SUBHEADLINE, ElementRole.CTA}:
            text = elem.text_content or elem.name
            font_min, font_max, max_lines = _typography_bounds_for_role(elem.role)

            fit = fit_text_block(
                text=text,
                font_family=None,
                max_font=font_max,
                min_font=font_min,
                max_width=max(1, b.width),
                max_lines=max_lines,
            )

            role_max_h = height - 2 * margin_y
            if role_max is not None:
                role_max_h = min(role_max_h, int(role_max[1] * height))

            min_h_for_font = int(round(font_min / 0.45))
            target_h = max(min_h_for_font, fit.bbox[1], b.height)
            new_h = min(role_max_h, max(1, target_h))
            new_w = min(max(1, fit.bbox[0]), b.width)

            if new_h != b.height or new_w != b.width:
                cx, cy = b.center
                b = BoundingBox(cx - new_w // 2, cy - new_h // 2, new_w, new_h)
                b = clamp_bbox_to_margins(b, (margin_x, margin_y), width, height)
                steps.append(f"reflow:{r.element_id}:font_px={fit.font_size}")
                applied = True

        repaired.append(
            LayoutResult(
                element_id=r.element_id,
                new_bbox=b,
                scale_factor=r.scale_factor,
                visible=r.visible,
            )
        )

    resolved, overlap_changed = resolve_overlaps(
        repaired,
        elements_by_id,
        profile,
        target_size,
    )
    if overlap_changed:
        steps.append("resolve_overlaps")
        applied = True

    return RepairResult(layout=resolved, repair_applied=applied, steps=steps)


def resolve_overlaps(
    layout_results: list[LayoutResult],
    elements_by_id: dict[str, DesignElement],
    profile: LayoutProfile,
    target_size: tuple[int, int],
    rounds: int = 8,
) -> tuple[list[LayoutResult], bool]:
    """Resolve overlaps by moving/shrinking lower-priority elements deterministically."""
    width, height = target_size
    margin_x = int(profile.margin_pct * width)
    margin_y = int(profile.margin_pct * height)
    min_gap = max(4, int(profile.baseline_spacing_pct * height))

    out = [
        LayoutResult(
            r.element_id,
            BoundingBox(
                r.new_bbox.x,
                r.new_bbox.y,
                r.new_bbox.width,
                r.new_bbox.height,
            ),
            r.scale_factor,
            r.visible,
        )
        for r in layout_results
    ]

    changed = False
    for _ in range(rounds):
        any_change = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a = out[i]
                b = out[j]
                if not a.visible or not b.visible:
                    continue
                ea = elements_by_id.get(a.element_id)
                eb = elements_by_id.get(b.element_id)
                ra = ea.role if ea else ElementRole.UNKNOWN
                rb = eb.role if eb else ElementRole.UNKNOWN
                if ra in BACKGROUND_ROLES or rb in BACKGROUND_ROLES:
                    continue

                ia = a.new_bbox
                ib = b.new_bbox
                ix1 = max(ia.x, ib.x)
                iy1 = max(ia.y, ib.y)
                ix2 = min(ia.x2, ib.x2)
                iy2 = min(ia.y2, ib.y2)
                if ix1 >= ix2 or iy1 >= iy2:
                    continue

                pri_a = profile.role_priority.get(ra, 10)
                pri_b = profile.role_priority.get(rb, 10)
                target = b if pri_a >= pri_b else a
                target_role = rb if target is b else ra
                tb = target.new_bbox

                push = (iy2 - iy1) + min_gap
                down = BoundingBox(tb.x, tb.y + push, tb.width, tb.height)
                down = clamp_bbox_to_margins(down, (margin_x, margin_y), width, height)

                # If movement doesn't separate enough, shrink lower priority element.
                if _boxes_overlap(down, ia if target is b else ib):
                    min_size = profile.min_sizes.get(target_role)
                    min_w = int(min_size[0] * width) if min_size else 1
                    min_h = int(min_size[1] * height) if min_size else 1
                    shrink_w = max(min_w, int(tb.width * 0.92))
                    shrink_h = max(min_h, int(tb.height * 0.92))
                    cx, cy = tb.center
                    down = BoundingBox(cx - shrink_w // 2, cy - shrink_h // 2, shrink_w, shrink_h)
                    down = clamp_bbox_to_margins(down, (margin_x, margin_y), width, height)

                if (down.x, down.y, down.width, down.height) != (tb.x, tb.y, tb.width, tb.height):
                    target.new_bbox = down
                    any_change = True
                    changed = True
        if not any_change:
            break

    return out, changed


def _boxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    return max(a.x, b.x) < min(a.x2, b.x2) and max(a.y, b.y) < min(a.y2, b.y2)


def _typography_bounds_for_role(role: ElementRole) -> tuple[int, int, int]:
    if role == ElementRole.HEADLINE:
        return 20, 64, 4
    if role == ElementRole.SUBHEADLINE:
        return 16, 42, 5
    if role == ElementRole.CTA:
        return 14, 30, 2
    return 12, 28, 5
