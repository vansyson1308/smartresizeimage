"""Optional decor synthesis for BG_PLUS_DECOR policy."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from PIL import Image

from ..layout.templates import TEMPLATES
from .engine import GenerativeOutpaintEngine

logger = logging.getLogger("autobanner.generative.decor")

TextBoxesExtractor = Callable[[Image.Image], list[tuple[int, int, int, int]]]


def build_decor_zone_mask(
    target_size: tuple[int, int],
    protected_mask: np.ndarray,
) -> np.ndarray:
    """Build decor zones from template negative space minus protected mask."""
    width, height = target_size
    template = _select_template(width, height)

    occupied = np.zeros((height, width), dtype=bool)
    for zone in template["zones"]:
        x1 = max(0, int(zone["x"] * width))
        y1 = max(0, int(zone["y"] * height))
        x2 = min(width, int((zone["x"] + zone["w"]) * width))
        y2 = min(height, int((zone["y"] + zone["h"]) * height))
        if x1 < x2 and y1 < y2:
            occupied[y1:y2, x1:x2] = True

    negative_space = ~occupied
    decor_mask = negative_space & ~protected_mask

    logger.info(
        "Decor zone stats: decor_px=%d (%.2f%%)",
        int(decor_mask.sum()),
        float(decor_mask.mean()) * 100,
    )
    return decor_mask


def apply_optional_decor_synthesis(
    base_canvas: Image.Image,
    protected_mask: np.ndarray,
    policy: str,
    seed: int,
    outpaint_engine: GenerativeOutpaintEngine,
    ocr_extractor: TextBoxesExtractor | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    """Apply decor generation only in decor zones and run OCR-negative gate."""
    if policy != "BG_PLUS_DECOR":
        return base_canvas, {
            "policy": policy,
            "applied": False,
            "fallback_reason": "policy_off",
            "ocr_gate_ran": False,
        }

    decor_mask = build_decor_zone_mask(base_canvas.size, protected_mask)
    if not np.any(decor_mask):
        return base_canvas, {
            "policy": policy,
            "applied": False,
            "fallback_reason": "empty_decor_zone",
            "ocr_gate_ran": False,
        }

    candidate = outpaint_engine.outpaint_background(
        base_canvas=base_canvas,
        editable_mask=decor_mask,
        policy="BG_PLUS_DECOR",
        seed=seed,
    )

    gate_ok, gate_meta = run_ocr_negative_gate(candidate, decor_mask, ocr_extractor)
    if not gate_ok:
        logger.warning("Decor OCR-negative gate failed; discarding decor result")
        return base_canvas, {
            "policy": policy,
            "applied": False,
            "fallback_reason": "ocr_gate_failed",
            "ocr_gate_ran": True,
            **gate_meta,
        }

    return candidate, {
        "policy": policy,
        "applied": True,
        "decor_ratio": float(decor_mask.mean()),
        "ocr_gate_ran": True,
        **gate_meta,
    }


def run_ocr_negative_gate(
    image: Image.Image,
    decor_mask: np.ndarray,
    ocr_extractor: TextBoxesExtractor | None = None,
) -> tuple[bool, dict[str, object]]:
    """Fail if OCR detects text within generated decor zones."""
    extractor = ocr_extractor or _extract_text_boxes
    boxes = extractor(image)

    overlap_count = 0
    for x, y, w, h in boxes:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(decor_mask.shape[1], x + w)
        y2 = min(decor_mask.shape[0], y + h)
        if x1 >= x2 or y1 >= y2:
            continue
        if bool(np.any(decor_mask[y1:y2, x1:x2])):
            overlap_count += 1

    return overlap_count == 0, {"ocr_boxes": len(boxes), "ocr_overlap_boxes": overlap_count}


def _extract_text_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Best-effort OCR boxes; returns empty list when OCR deps are unavailable."""
    try:
        import pytesseract  # type: ignore

        data = pytesseract.image_to_data(
            image.convert("RGB"),
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("OCR gate extractor unavailable, defaulting to no text boxes: %s", exc)
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


def _select_template(width: int, height: int) -> dict:
    aspect = width / height if height > 0 else 1.0
    for template in TEMPLATES.values():
        min_a, max_a = template["aspect_range"]
        if min_a <= aspect < max_a:
            return template
    return TEMPLATES["landscape"]
