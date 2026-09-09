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
    busy_threshold: float = 0.10
    padding: int = 12
    feather_radius: int = 10
    opacity: int = 110
    corner_radius: int = 10


_BUSY_SAMPLE_WIDTH = 320


def _clutter(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    variance = float(np.var(arr))
    gx = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    edge_density = float((gx + gy) * 0.5)
    return max(0.0, min(1.0, variance * 4.0 * 0.6 + edge_density * 3.0 * 0.4))


def compute_busy_score(image: Image.Image, bbox: tuple[int, int, int, int]) -> float:
    """Compute clutter score in [0,1] from variance + edge density.

    Texture at glyph-stroke scale is what hurts legibility, so the region is
    sampled at native resolution (a bounded central window keeps cost flat)
    and at half resolution; the larger score wins so both fine noise and
    coarse patterns count.
    """
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return 0.0

    region = image.convert("L").crop((x, y, x + w, y + h))
    if region.width > _BUSY_SAMPLE_WIDTH or region.height > _BUSY_SAMPLE_WIDTH:
        cw = min(region.width, _BUSY_SAMPLE_WIDTH)
        chh = min(region.height, _BUSY_SAMPLE_WIDTH)
        left = (region.width - cw) // 2
        top = (region.height - chh) // 2
        region = region.crop((left, top, left + cw, top + chh))
    arr = np.array(region, dtype=np.float32) / 255.0
    native = _clutter(arr)
    half = 0.0
    if region.width >= 4 and region.height >= 4:
        small = region.resize((region.width // 2, region.height // 2), Image.Resampling.BOX)
        half = _clutter(np.array(small, dtype=np.float32) / 255.0)
    return max(native, half)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def low_contrast_fraction(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    text_rgb: tuple[int, int, int],
    min_ratio: float = 3.0,
) -> float:
    """Fraction of background pixels whose WCAG contrast with the text is below ``min_ratio``."""
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return 0.0
    region = image.convert("RGB").crop((x, y, x + w, y + h))
    if region.width > _BUSY_SAMPLE_WIDTH:
        scale = _BUSY_SAMPLE_WIDTH / region.width
        region = region.resize(
            (_BUSY_SAMPLE_WIDTH, max(1, int(region.height * scale))), Image.Resampling.BOX
        )
    arr = np.asarray(region, dtype=np.float32) / 255.0
    lin = np.where(arr <= 0.03928, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    lt = _relative_luminance(text_rgb)
    hi = np.maximum(lum, lt) + 0.05
    lo = np.minimum(lum, lt) + 0.05
    ratio = hi / lo
    return float(np.mean(ratio < min_ratio))


def apply_text_safe_plates(
    background: Image.Image,
    text_boxes: list[tuple[int, int, int, int]],
    avoid_mask: np.ndarray | None,
    config: TextPlateConfig,
    text_colors: list[tuple[int, int, int] | None] | None = None,
) -> tuple[Image.Image, dict[str, float | int | str | bool]]:
    """Apply readability plates behind text zones where background is busy or
    lacks contrast with the text colour.

    Plate drawing is restricted by ``avoid_mask`` (True means protected/blocked).
    When ``text_colors`` are known, the plate colour is chosen to contrast with
    the text (light plate under dark text, dark scrim under light text).
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
    contrast_scores: list[float] = []
    colors = list(text_colors or [])

    # Neighbouring text boxes (a headline/sub/CTA stack) share one plate so the
    # result reads as a single panel instead of a patchwork.
    clusters = _cluster_boxes(text_boxes, config.padding, w, h)

    for members in clusters:
        box = _union_box([text_boxes[i] for i in members])
        member_colors = [colors[i] for i in members if i < len(colors) and colors[i]]
        idx = members[0]
        x, y, bw, bh = _expand_box(box, config.padding, w, h)
        busy = compute_busy_score(rgba, (x, y, bw, bh))
        busy_scores.append(busy)
        fallback_rgb = colors[idx] if idx < len(colors) else None
        text_rgb = member_colors[0] if member_colors else fallback_rgb
        low_contrast = (
            low_contrast_fraction(rgba, (x, y, bw, bh), text_rgb) if text_rgb else 0.0
        )
        contrast_scores.append(low_contrast)
        if busy < config.busy_threshold and low_contrast < 0.25:
            continue

        # Busier / lower-contrast backgrounds get a stronger plate, bounded at 230.
        severity = max(max(0.0, busy - config.busy_threshold) * 300.0, low_contrast * 160.0)
        boost = int(min(230, config.opacity + severity))
        dark_plate = text_rgb is not None and _relative_luminance(text_rgb) > 0.4
        cfg = TextPlateConfig(
            enabled=config.enabled,
            style=config.style if (busy < 0.5 and low_contrast < 0.5) else "solid",
            busy_threshold=config.busy_threshold,
            padding=config.padding,
            feather_radius=config.feather_radius,
            opacity=boost,
            corner_radius=config.corner_radius,
        )
        patch = _build_plate_patch(rgba, (x, y, bw, bh), cfg, dark=dark_plate)
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
        "busy_scores": [round(float(v), 6) for v in busy_scores],
        "low_contrast": [round(float(v), 4) for v in contrast_scores],
        "busy_threshold": float(config.busy_threshold),
        "style": config.style,
    }


def _cluster_boxes(
    boxes: list[tuple[int, int, int, int]], pad: int, canvas_w: int, canvas_h: int
) -> list[list[int]]:
    """Group boxes whose padded rectangles touch (transitively)."""
    expanded = [_expand_box(b, pad * 2, canvas_w, canvas_h) for b in boxes]
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = expanded[i], expanded[j]
            overlap_x = a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            overlap_y = a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
            if overlap_x and overlap_y:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return x1, y1, x2 - x1, y2 - y1


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
    dark: bool = False,
) -> Image.Image:
    x, y, w, h = box
    if config.style == "gradient":
        return _gradient_plate((w, h), config, dark=dark)
    if config.style == "solid":
        return _solid_plate((w, h), config, dark=dark)
    return _blur_plate(background, box, config, dark=dark)


def _plate_rgb(dark: bool) -> tuple[int, int, int]:
    return (18, 18, 22) if dark else (255, 255, 255)


def _blur_plate(
    background: Image.Image,
    box: tuple[int, int, int, int],
    config: TextPlateConfig,
    dark: bool = False,
) -> Image.Image:
    x, y, w, h = box
    patch = background.crop((x, y, x + w, y + h)).convert("RGBA")
    blurred = patch.filter(ImageFilter.GaussianBlur(radius=max(1, config.feather_radius // 2)))

    tint = Image.new("RGBA", (w, h), (*_plate_rgb(dark), min(200, config.opacity + 20)))
    blurred.alpha_composite(tint)

    mask = _rounded_mask((w, h), config.corner_radius)
    if config.feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=config.feather_radius / 2))
    blurred.putalpha(mask)
    return blurred


def _gradient_plate(
    size: tuple[int, int], config: TextPlateConfig, dark: bool = False
) -> Image.Image:
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    rgb = _plate_rgb(dark)
    for row in range(h):
        t = abs((row / max(1, h - 1)) - 0.5) * 2.0
        alpha = int(max(0, (1.0 - t * 0.85) * config.opacity))
        arr[row, :, :] = (*rgb, alpha)

    plate = Image.fromarray(arr, mode="RGBA")
    mask = _rounded_mask(size, config.corner_radius)
    if config.feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=config.feather_radius / 2))
    plate.putalpha(ImageChops.multiply(mask, plate.split()[3]))
    return plate


def _solid_plate(
    size: tuple[int, int], config: TextPlateConfig, dark: bool = False
) -> Image.Image:
    w, h = size
    rgb = _plate_rgb(dark)
    plate = Image.new("RGBA", size, (*rgb, 0))
    draw = ImageDraw.Draw(plate)
    draw.rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=max(1, config.corner_radius),
        fill=(*rgb, config.opacity),
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
