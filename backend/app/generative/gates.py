"""Quality gates for generative outputs with explicit fallback reasons."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..enums import ElementRole
from ..models import DesignElement, LayoutResult

logger = logging.getLogger("autobanner.generative.gates")

TextBoxesExtractor = Callable[[Image.Image], list[tuple[int, int, int, int]]]


@dataclass
class GateReport:
    """Quality gate result payload."""

    gates_passed: bool
    fail_reasons: list[str] = field(default_factory=list)
    used_fallback: bool = False


def evaluate_quality_gates(
    baseline: Image.Image,
    candidate: Image.Image,
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    protected_mask: np.ndarray,
    ocr_extractor: TextBoxesExtractor | None = None,
) -> GateReport:
    """Run quality gates and return explicit pass/fail reasons."""
    reasons: list[str] = []

    base = np.asarray(baseline.convert("RGBA"), dtype=np.int16)
    cand = np.asarray(candidate.convert("RGBA"), dtype=np.int16)

    if base.shape != cand.shape:
        reasons.append("size_mismatch")
        logger.warning(
            "Gate fail: baseline/candidate size mismatch %s vs %s",
            base.shape,
            cand.shape,
        )

    editable_mask = ~protected_mask

    if not _ocr_gate(candidate, editable_mask, ocr_extractor):
        reasons.append("ocr_gate_failed")

    logo_mask = _build_role_mask(elements, layout_results, protected_mask.shape, {ElementRole.LOGO})
    if np.any(logo_mask) and not _similarity_ok(base, cand, logo_mask, threshold=8.0):
        reasons.append("logo_similarity_failed")

    mascot_roles = {ElementRole.HERO_IMAGE, ElementRole.PHOTO, ElementRole.ILLUSTRATION}
    mascot_mask = _build_role_mask(elements, layout_results, protected_mask.shape, mascot_roles)
    if np.any(mascot_mask) and not _similarity_ok(base, cand, mascot_mask, threshold=10.0):
        reasons.append("mascot_similarity_failed")

    if not _color_drift_ok(base, cand, protected_mask, max_delta=10.0):
        reasons.append("color_drift_failed")

    passed = len(reasons) == 0
    if passed:
        logger.info("Quality gates passed")
    else:
        logger.warning("Quality gates failed: %s", ", ".join(reasons))

    return GateReport(gates_passed=passed, fail_reasons=reasons, used_fallback=not passed)


def _ocr_gate(
    candidate: Image.Image,
    editable_mask: np.ndarray,
    ocr_extractor: TextBoxesExtractor | None,
) -> bool:
    extractor = ocr_extractor or _extract_text_boxes
    boxes = extractor(candidate)

    overlaps = 0
    for x, y, w, h in boxes:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(editable_mask.shape[1], x + w)
        y2 = min(editable_mask.shape[0], y + h)
        if x1 >= x2 or y1 >= y2:
            continue
        if bool(np.any(editable_mask[y1:y2, x1:x2])):
            overlaps += 1

    if overlaps > 0:
        logger.warning("OCR gate detected %d suspicious text boxes in editable area", overlaps)
    return overlaps == 0


def _similarity_ok(base: np.ndarray, cand: np.ndarray, mask: np.ndarray, threshold: float) -> bool:
    base_rgb = base[:, :, :3]
    cand_rgb = cand[:, :, :3]
    diff = np.abs(base_rgb[mask] - cand_rgb[mask]).astype(np.float32)
    mae = float(diff.mean()) if diff.size > 0 else 0.0
    return mae <= threshold


def _color_drift_ok(
    base: np.ndarray,
    cand: np.ndarray,
    protected_mask: np.ndarray,
    max_delta: float,
) -> bool:
    # Brand-safe: protected zones should not drift perceptibly.
    if not np.any(protected_mask):
        return True

    base_rgb = base[:, :, :3][protected_mask].astype(np.float32)
    cand_rgb = cand[:, :, :3][protected_mask].astype(np.float32)
    mean_delta = float(np.abs(base_rgb - cand_rgb).mean())
    return mean_delta <= max_delta


def _build_role_mask(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    shape: tuple[int, int],
    roles: set[ElementRole],
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    layout_map = {r.element_id: r for r in layout_results}

    for elem in elements:
        if elem.role not in roles:
            continue

        layout = layout_map.get(elem.id)
        if layout is None or not layout.visible:
            continue

        bbox = layout.new_bbox
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + bbox.height)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = True

    return mask


def _extract_text_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    try:
        import pytesseract  # type: ignore

        data = pytesseract.image_to_data(
            image.convert("RGB"),
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("OCR extractor unavailable for gates: %s", exc)
        return []

    boxes: list[tuple[int, int, int, int]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf = float(data.get("conf", ["-1"])[i])
        if not text or conf < 0:
            continue
        boxes.append(
            (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            )
        )

    return boxes
