"""Anchor extraction and protected-mask utilities for Phase 3 target-first redesign."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import BoundingBox, DesignElement, LayoutResult

_PROTECTED_ROLES = {
    ElementRole.HEADLINE,
    ElementRole.SUBHEADLINE,
    ElementRole.BODY_TEXT,
    ElementRole.CTA,
    ElementRole.LOGO,
    ElementRole.LABEL,
    ElementRole.BADGE,
}


@dataclass(frozen=True)
class Anchor:
    element_id: str
    role: ElementRole
    image: Image.Image
    source_bbox: BoundingBox
    target_bbox: BoundingBox
    protected: bool


@dataclass(frozen=True)
class AnchorBundle:
    anchors: list[Anchor]
    protected_mask: np.ndarray


def extract_anchors(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    target_size: tuple[int, int],
    mask_padding: int = 4,
) -> AnchorBundle:
    """Extract anchors and build protected mask from semantic roles."""
    layout_by_id = {r.element_id: r for r in layout_results}
    anchors: list[Anchor] = []

    for elem in elements:
        if elem.role in BACKGROUND_ROLES:
            continue
        if elem.image is None:
            continue
        lr = layout_by_id.get(elem.id)
        if lr is None or not lr.visible:
            continue

        protected = elem.role in _PROTECTED_ROLES
        anchors.append(
            Anchor(
                element_id=elem.id,
                role=elem.role,
                image=elem.image.convert("RGBA"),
                source_bbox=elem.bbox,
                target_bbox=lr.new_bbox,
                protected=protected,
            )
        )

    mask = build_protected_mask(anchors, target_size, padding=mask_padding)
    return AnchorBundle(anchors=anchors, protected_mask=mask)


def extract_anchors_from_boxes(
    source_image: Image.Image,
    boxes: list[dict[str, int | str]],
    target_layout: list[LayoutResult],
    target_size: tuple[int, int],
    mask_padding: int = 4,
) -> AnchorBundle:
    """Extract anchors for flat images from user-provided boxes."""
    layout_by_id = {r.element_id: r for r in target_layout}
    anchors: list[Anchor] = []

    for item in boxes:
        eid = str(item.get("id", "anchor"))
        role_raw = str(item.get("role", ElementRole.LOGO.value))
        valid_roles = {r.value for r in ElementRole}
        role = (
            ElementRole(role_raw)
            if role_raw in valid_roles
            else ElementRole.LOGO
        )
        src = BoundingBox(
            int(item.get("x", 0)),
            int(item.get("y", 0)),
            int(item.get("width", 1)),
            int(item.get("height", 1)),
        )
        lr = layout_by_id.get(eid)
        if lr is None:
            sx = target_size[0] / max(1, source_image.size[0])
            sy = target_size[1] / max(1, source_image.size[1])
            lr = LayoutResult(
                element_id=eid,
                new_bbox=BoundingBox(
                    int(src.x * sx),
                    int(src.y * sy),
                    max(1, int(src.width * sx)),
                    max(1, int(src.height * sy)),
                ),
                scale_factor=min(sx, sy),
                visible=True,
            )
        crop = source_image.convert("RGBA").crop((src.x, src.y, src.x2, src.y2))
        anchors.append(
            Anchor(
                element_id=eid,
                role=role,
                image=crop,
                source_bbox=src,
                target_bbox=lr.new_bbox,
                protected=True,
            )
        )

    mask = build_protected_mask(anchors, target_size, padding=mask_padding)
    return AnchorBundle(anchors=anchors, protected_mask=mask)


def build_protected_mask(
    anchors: list[Anchor],
    target_size: tuple[int, int],
    padding: int = 4,
) -> np.ndarray:
    """Build boolean protected mask from anchor alpha masks."""
    w, h = target_size
    mask = np.zeros((h, w), dtype=bool)

    for anchor in anchors:
        if not anchor.protected:
            continue
        b = anchor.target_bbox
        x1 = max(0, b.x - padding)
        y1 = max(0, b.y - padding)
        x2 = min(w, b.x2 + padding)
        y2 = min(h, b.y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = True

    return mask
