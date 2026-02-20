"""Protected/editable mask generation utilities for anchored redesign."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from ..enums import ElementRole
from ..models import DesignElement, LayoutResult

logger = logging.getLogger("autobanner.generative.masks")


@dataclass
class Masks:
    """Container for protected/editable masks."""

    protected_mask: np.ndarray  # bool, shape (H, W)
    editable_mask: np.ndarray  # bool, shape (H, W)
    text_mask: np.ndarray  # bool, shape (H, W)
    hero_mask: np.ndarray  # bool, shape (H, W)

    @property
    def protected_ratio(self) -> float:
        return float(self.protected_mask.mean())

    @property
    def editable_ratio(self) -> float:
        return float(self.editable_mask.mean())


_PROTECTED_ROLES = {
    ElementRole.LOGO,
    ElementRole.HEADLINE,
    ElementRole.SUBHEADLINE,
    ElementRole.BODY_TEXT,
    ElementRole.CTA,
    ElementRole.BADGE,
    ElementRole.LABEL,
    ElementRole.HERO_IMAGE,
}


def build_masks(parsed_elements: list[DesignElement], image: Image.Image) -> Masks:
    """Build protected/editable masks for PSD and flat-image inputs.

    For PSD-like inputs: role- and text-driven bbox protection.
    For flat images: OCR-derived text mask + saliency hero mask.
    """

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image size must be positive")

    protected = np.zeros((height, width), dtype=bool)
    text_mask = np.zeros((height, width), dtype=bool)
    hero_mask = np.zeros((height, width), dtype=bool)

    is_flat = _is_flat_source(parsed_elements)
    if is_flat:
        text_mask = _build_ocr_text_mask(image)
        hero_mask = _build_saliency_hero_mask(image)
        protected |= text_mask | hero_mask
    else:
        for elem in parsed_elements:
            if _is_protected_element(elem):
                _paint_bbox(protected, elem.bbox.x, elem.bbox.y, elem.bbox.width, elem.bbox.height)

    editable = ~protected

    masks = Masks(
        protected_mask=protected,
        editable_mask=editable,
        text_mask=text_mask,
        hero_mask=hero_mask,
    )

    logger.info(
        "Mask stats: size=%dx%d protected=%d (%.2f%%) editable=%d (%.2f%%)",
        width,
        height,
        int(protected.sum()),
        masks.protected_ratio * 100,
        int(editable.sum()),
        masks.editable_ratio * 100,
    )

    return masks


def _is_flat_source(parsed_elements: list[DesignElement]) -> bool:
    if len(parsed_elements) != 1:
        return False
    return parsed_elements[0].effects.get("_source_type") == "flat_image"


def _is_protected_element(element: DesignElement) -> bool:
    if element.role in _PROTECTED_ROLES:
        return True
    if element.text_content:
        return True
    if element.effects.get("protected") is True:
        return True

    name = element.name.lower()
    return any(token in name for token in ("logo", "mascot", "headline", "title"))


def _paint_bbox(mask: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    if w <= 0 or h <= 0:
        return
    height, width = mask.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = True


def _build_ocr_text_mask(image: Image.Image) -> np.ndarray:
    width, height = image.size
    mask = np.zeros((height, width), dtype=bool)
    boxes = _extract_text_boxes(image)

    for x, y, w, h in boxes:
        _paint_bbox(mask, x, y, w, h)

    return mask


def _extract_text_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Best-effort OCR text box extraction; returns empty list on missing deps."""
    try:
        import pytesseract  # type: ignore

        data = pytesseract.image_to_data(
            image.convert("RGB"),
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR unavailable; text mask fallback to empty: %s", exc)
        return []

    boxes: list[tuple[int, int, int, int]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf = float(data.get("conf", ["-1"])[i])
        if not text or conf < 0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        boxes.append((x, y, w, h))

    return boxes


def _build_saliency_hero_mask(image: Image.Image) -> np.ndarray:
    """Simple deterministic saliency mask for flat image protection."""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float32)

    threshold = float(np.percentile(arr, 88.0))
    hero = arr >= threshold

    # Deterministic one-pass dilation to avoid tiny holes.
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        hero |= np.roll(np.roll(hero, dy, axis=0), dx, axis=1)

    # Remove wrapped borders introduced by roll.
    hero[0, :] = False
    hero[-1, :] = False
    hero[:, 0] = False
    hero[:, -1] = False

    return hero


def build_layout_masks(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    target_size: tuple[int, int],
) -> Masks:
    """Build masks using layout bboxes in target canvas coordinates."""
    width, height = target_size
    protected = np.zeros((height, width), dtype=bool)
    text_mask = np.zeros((height, width), dtype=bool)
    hero_mask = np.zeros((height, width), dtype=bool)

    layout_map = {r.element_id: r for r in layout_results}

    for elem in elements:
        layout = layout_map.get(elem.id)
        if layout is None or not layout.visible:
            continue
        if not _is_protected_element(elem):
            continue

        bbox = layout.new_bbox
        _paint_bbox(protected, bbox.x, bbox.y, bbox.width, bbox.height)

    editable = ~protected
    masks = Masks(
        protected_mask=protected,
        editable_mask=editable,
        text_mask=text_mask,
        hero_mask=hero_mask,
    )

    logger.info(
        "Layout mask stats: size=%dx%d protected=%d (%.2f%%) editable=%d (%.2f%%)",
        width,
        height,
        int(protected.sum()),
        masks.protected_ratio * 100,
        int(editable.sum()),
        masks.editable_ratio * 100,
    )
    return masks
