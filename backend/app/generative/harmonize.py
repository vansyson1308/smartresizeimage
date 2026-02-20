"""Safe harmonization utilities for Phase 2 PR-C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from ..models import DesignElement, LayoutResult


@dataclass
class ShadowSpec:
    """Grounding shadow parameters."""

    offset_x: int = 4
    offset_y: int = 6
    blur_radius: float = 8.0
    opacity: float = 0.22
    color: tuple[int, int, int] = (0, 0, 0)


def apply_color_grading_safe(
    image: Image.Image,
    protected_mask: np.ndarray,
    strength: float = 0.08,
    gain_limits: tuple[float, float] = (0.95, 1.05),
) -> Image.Image:
    """Apply slight background-only color harmonization while keeping protected pixels exact."""
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32)
    rgb = arr[:, :, :3]

    editable = ~protected_mask
    if not np.any(editable):
        return rgba

    prot = protected_mask
    target_mean = rgb[prot].mean(axis=0) if np.any(prot) else rgb[editable].mean(axis=0)

    src_mean = np.maximum(rgb[editable].mean(axis=0), 1.0)
    raw_gain = target_mean / src_mean
    lo, hi = gain_limits
    gain = np.clip(raw_gain, lo, hi)
    gain = 1.0 + (gain - 1.0) * float(np.clip(strength, 0.0, 1.0))

    graded = rgb.copy()
    graded[editable] = np.clip(graded[editable] * gain[None, :], 0.0, 255.0)

    out = arr.copy()
    out[:, :, :3] = graded
    # Force protected pixels unchanged exactly.
    out[prot, :3] = arr[prot, :3]

    return Image.fromarray(out.astype(np.uint8), mode="RGBA")


def create_grounding_shadow_layer(
    size: tuple[int, int],
    mascot_masks: list[np.ndarray],
    spec: ShadowSpec | None = None,
) -> Image.Image:
    """Build a combined grounding shadow layer from mascot alpha masks."""
    if spec is None:
        spec = ShadowSpec()

    w, h = size
    shadow_alpha = np.zeros((h, w), dtype=np.float32)

    for mask in mascot_masks:
        if mask.shape != (h, w):
            continue
        shadow_alpha = np.maximum(shadow_alpha, mask.astype(np.float32))

    shadow_img = Image.fromarray(np.clip(shadow_alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=spec.blur_radius))

    # Offset shadow
    offset_layer = Image.new("L", size, 0)
    offset_layer.paste(shadow_img, (spec.offset_x, spec.offset_y))

    alpha = np.asarray(offset_layer, dtype=np.float32) / 255.0
    alpha *= float(np.clip(spec.opacity, 0.0, 1.0))

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = np.array(spec.color, dtype=np.uint8)
    out[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def apply_grounding_shadow_safe(
    image: Image.Image,
    mascot_masks: list[np.ndarray],
    protected_mask: np.ndarray,
    spec: ShadowSpec | None = None,
) -> Image.Image:
    """Composite grounding shadow under mascot and keep protected pixels untouched."""
    base = image.convert("RGBA")
    base_arr = np.array(base, dtype=np.uint8, copy=True)

    shadow_layer = create_grounding_shadow_layer(base.size, mascot_masks, spec)
    combined = Image.alpha_composite(base, shadow_layer)
    out_arr = np.array(combined, dtype=np.uint8, copy=True)

    # Ensure protected region pixels remain exactly the same.
    out_arr[protected_mask] = base_arr[protected_mask]

    return Image.fromarray(out_arr, mode="RGBA")


def extract_mascot_masks(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    """Extract mascot/hero alpha masks projected into target canvas coordinates."""
    layout_map = {r.element_id: r for r in layout_results}
    w, h = target_size
    masks: list[np.ndarray] = []

    for elem in elements:
        layout = layout_map.get(elem.id)
        if layout is None or not layout.visible or elem.image is None:
            continue

        is_mascot = elem.role.value == "hero_image" or "mascot" in elem.name.lower()
        if not is_mascot:
            continue

        bbox = layout.new_bbox
        if bbox.width <= 0 or bbox.height <= 0:
            continue

        alpha = elem.image.convert("RGBA").split()[3].resize(
            (bbox.width, bbox.height), Image.Resampling.LANCZOS
        )
        alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0

        mask = np.zeros((h, w), dtype=bool)
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + bbox.height)
        if x1 >= x2 or y1 >= y2:
            continue

        sx1 = x1 - bbox.x
        sy1 = y1 - bbox.y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        mask[y1:y2, x1:x2] = alpha_arr[sy1:sy2, sx1:sx2] > 0.05
        masks.append(mask)

    return masks
