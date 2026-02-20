"""Iterative collision solver and debug helpers for adaptive layout."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from ..enums import ElementRole
from ..models import BoundingBox, LayoutResult
from .profiles import LayoutProfile
from .repair import clamp_bbox_to_margins


def solve_layout(
    placements: list[LayoutResult],
    target_size: tuple[int, int],
    profile: LayoutProfile,
    role_by_id: dict[str, ElementRole],
    iterations: int = 24,
) -> tuple[list[LayoutResult], dict[str, float]]:
    """Iteratively reduce overlaps, snap to grid, and enforce vertical rhythm."""
    width, height = target_size
    margin_x = int(profile.margin_pct * width)
    margin_y = int(profile.margin_pct * height)
    col_w = width / max(1, profile.grid_cols)
    row_h = height / max(1, profile.grid_rows)
    min_gap = int(profile.baseline_spacing_pct * height)

    results = [
        LayoutResult(
            element_id=r.element_id,
            new_bbox=BoundingBox(r.new_bbox.x, r.new_bbox.y, r.new_bbox.width, r.new_bbox.height),
            scale_factor=r.scale_factor,
            visible=r.visible,
        )
        for r in placements
    ]

    for _ in range(iterations):
        # 1) push apart overlaps by role priority
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a = results[i]
                b = results[j]
                if not a.visible or not b.visible:
                    continue

                ia = a.new_bbox
                ib = b.new_bbox
                ix1 = max(ia.x, ib.x)
                iy1 = max(ia.y, ib.y)
                ix2 = min(ia.x2, ib.x2)
                iy2 = min(ia.y2, ib.y2)
                if ix1 >= ix2 or iy1 >= iy2:
                    continue

                overlap_h = iy2 - iy1
                push = max(2, overlap_h // 2)

                pri_a = profile.role_priority.get(
                    role_by_id.get(a.element_id, ElementRole.UNKNOWN),
                    10,
                )
                pri_b = profile.role_priority.get(
                    role_by_id.get(b.element_id, ElementRole.UNKNOWN),
                    10,
                )

                if pri_a >= pri_b:
                    ib.y += push
                else:
                    ia.y -= push

                a.new_bbox = ia
                b.new_bbox = ib

        # 2) snap to grid
        for r in results:
            if not r.visible:
                continue
            b = r.new_bbox
            snapped_x = int(round(b.x / col_w) * col_w)
            snapped_y = int(round(b.y / row_h) * row_h)
            b.x = snapped_x
            b.y = snapped_y
            r.new_bbox = b

        # 3) enforce vertical rhythm (sorted by y)
        ordered = sorted([r for r in results if r.visible], key=lambda rr: rr.new_bbox.y)
        for idx in range(1, len(ordered)):
            prev = ordered[idx - 1].new_bbox
            cur = ordered[idx].new_bbox
            min_y = prev.y2 + min_gap
            if cur.y < min_y:
                cur.y = min_y

        # 4) hard clamp to margins/safe area
        for r in results:
            if not r.visible:
                continue
            b = r.new_bbox
            r.new_bbox = clamp_bbox_to_margins(b, (margin_x, margin_y), width, height)

    overlap = total_overlap_area(results)
    return results, {"overlap_area": float(overlap)}


def total_overlap_area(placements: list[LayoutResult]) -> int:
    """Compute total pairwise overlap area for visible placements."""
    area = 0
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            a = placements[i]
            b = placements[j]
            if not a.visible or not b.visible:
                continue
            ia = a.new_bbox
            ib = b.new_bbox
            ix1 = max(ia.x, ib.x)
            iy1 = max(ia.y, ib.y)
            ix2 = min(ia.x2, ib.x2)
            iy2 = min(ia.y2, ib.y2)
            if ix1 < ix2 and iy1 < iy2:
                area += (ix2 - ix1) * (iy2 - iy1)
    return area


def render_layout_debug_overlay(
    target_size: tuple[int, int],
    layout_results: list[LayoutResult],
    role_by_id: dict[str, ElementRole],
) -> Image.Image:
    """Create debug overlay image with bbox and role labels."""
    image = Image.new("RGBA", target_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    for r in layout_results:
        if not r.visible:
            continue
        b = r.new_bbox
        role = role_by_id.get(r.element_id, ElementRole.UNKNOWN)
        color = (255, 0, 0, 220) if role == ElementRole.HEADLINE else (0, 128, 255, 220)
        draw.rectangle((b.x, b.y, b.x2, b.y2), outline=color, width=2)
        draw.text((b.x + 2, b.y + 2), f"{role.value}:{r.element_id}", fill=color)

    return image


def export_layout_debug_json(path: Path, payload: dict) -> None:
    """Export layout debug metadata JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
