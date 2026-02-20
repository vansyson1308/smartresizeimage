"""Text-safe background plate utilities.

Applies readability plates behind text boxes on busy backgrounds while respecting
protected/avoid masks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

logger = logging.getLogger("autobanner.generative.text_plate")


@dataclass(frozen=True)
class TextPlateConfig:
    """Parameters controlling text-safe plate generation."""

    enabled: bool = True
    style: str = "blur"  # blur | gradient | solid
    busy_threshold: float = 0.22
    padding: int = 12
    feather_radius: int = 10
    opacity: int = 110
    corner_radius: int = 10


def compute_busy_score(image: Image.Image, bbox: tuple[int, int, int, int]) -> float:
    """Compute clutter score in [0,1] from variance + edge density."""
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return 0.0

    region = image.convert("L").crop((x, y, x + w, y + h))
    arr = np.array(region, dtype=np.float32) / 255.0
    if arr.size == 0:
        return 0.0

    variance = float(np.var(arr))
    gx = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    edge_density = float((gx + gy) * 0.5)

    score = min(1.0, variance * 4.0 * 0.6 + edge_density * 3.0 * 0.4)
    return max(0.0, score)


def apply_text_safe_plates(
    background: Image.Image,
    text_boxes: list[tuple[int, int, int, int]],
    avoid_mask: np.ndarray | None,
    config: TextPlateConfig,
) -> tuple[Image.Image, dict[str, float | int | str | bool]]:
    """Apply readability plates behind text zones where background is busy.

    Plate drawing is restricted by ``avoid_mask`` (True means protected/blocked).
    """
    rgba = background.convert("RGBA")
    w, h = rgba.size

    if not config.enabled or not text_boxes:
        return rgba, {"applied": False, "plates": 0, "avg_busy": 0.0, "style": config.style}

    blocked = np.zeros((h, w), dtype=bool)
    if avoid_mask is not None and avoid_mask.shape == (h, w):
        blocked = avoid_mask.copy()

    applied = 0
    busy_scores: list[float] = []

    for box in text_boxes:
        x, y, bw, bh = _expand_box(box, config.padding, w, h)
        busy = compute_busy_score(rgba, (x, y, bw, bh))
        busy_scores.append(busy)
        if busy < config.busy_threshold:
            continue

        patch = _build_plate_patch(rgba, (x, y, bw, bh), config)
        alpha = np.array(patch.split()[3], dtype=np.uint8)

        allow = (~blocked[y : y + bh, x : x + bw]).astype(np.uint8) * 255
        alpha = np.minimum(alpha, allow)
        patch.putalpha(Image.fromarray(alpha, mode="L"))

        rgba.alpha_composite(patch, dest=(x, y))
        applied += 1

    logger.info(
        "text-plate: applied=%d boxes=%d avg_busy=%.3f style=%s",
        applied,
        len(text_boxes),
        (sum(busy_scores) / len(busy_scores)) if busy_scores else 0.0,
        config.style,
    )

    return rgba, {
        "applied": applied > 0,
        "plates": applied,
        "avg_busy": (sum(busy_scores) / len(busy_scores)) if busy_scores else 0.0,
        "style": config.style,
    }


def _expand_box(
    box: tuple[int, int, int, int],
    pad: int,
    canvas_w: int,
    canvas_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(canvas_w, x + w + pad)
    y2 = min(canvas_h, y + h + pad)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _build_plate_patch(
    background: Image.Image,
    box: tuple[int, int, int, int],
    config: TextPlateConfig,
) -> Image.Image:
    x, y, w, h = box
    if config.style == "gradient":
        return _gradient_plate((w, h), config)
    if config.style == "solid":
        return _solid_plate((w, h), config)
    return _blur_plate(background, box, config)


def _blur_plate(
    background: Image.Image,
    box: tuple[int, int, int, int],
    config: TextPlateConfig,
) -> Image.Image:
    x, y, w, h = box
    patch = background.crop((x, y, x + w, y + h)).convert("RGBA")
    blurred = patch.filter(ImageFilter.GaussianBlur(radius=max(1, config.feather_radius // 2)))

    brighten = Image.new("RGBA", (w, h), (255, 255, 255, min(140, config.opacity + 20)))
    blurred.alpha_composite(brighten)

    mask = _rounded_mask((w, h), config.corner_radius)
    if config.feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=config.feather_radius / 2))
    blurred.putalpha(mask)
    return blurred


def _gradient_plate(size: tuple[int, int], config: TextPlateConfig) -> Image.Image:
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for row in range(h):
        t = abs((row / max(1, h - 1)) - 0.5) * 2.0
        alpha = int(max(0, (1.0 - t * 0.85) * config.opacity))
        arr[row, :, :] = (255, 255, 255, alpha)

    plate = Image.fromarray(arr, mode="RGBA")
    mask = _rounded_mask(size, config.corner_radius)
    if config.feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=config.feather_radius / 2))
    plate.putalpha(ImageChops.multiply(mask, plate.split()[3]))
    return plate


def _solid_plate(size: tuple[int, int], config: TextPlateConfig) -> Image.Image:
    w, h = size
    plate = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(plate)
    draw.rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=max(1, config.corner_radius),
        fill=(255, 255, 255, config.opacity),
    )
    if config.feather_radius > 0:
        alpha = plate.split()[3].filter(ImageFilter.GaussianBlur(radius=config.feather_radius / 2))
        plate.putalpha(alpha)
    return plate


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    w, h = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=max(1, radius), fill=255)
    return mask
