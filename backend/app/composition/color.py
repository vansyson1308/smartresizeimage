"""Color and alpha utilities for linear-light premultiplied compositing."""

from __future__ import annotations

from enum import Enum

import numpy as np
from PIL import Image


class BlendMode(str, Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"


_BLEND_ALIASES: dict[str, BlendMode] = {
    "normal": BlendMode.NORMAL,
    "pass through": BlendMode.NORMAL,
    "pass_through": BlendMode.NORMAL,
    "blendmode.normal": BlendMode.NORMAL,
    "multiply": BlendMode.MULTIPLY,
    "blendmode.multiply": BlendMode.MULTIPLY,
    "screen": BlendMode.SCREEN,
    "blendmode.screen": BlendMode.SCREEN,
    "overlay": BlendMode.OVERLAY,
    "blendmode.overlay": BlendMode.OVERLAY,
}


def parse_blend_mode(mode: str | None) -> tuple[BlendMode, bool]:
    """Parse blend mode text to supported BlendMode.

    Returns tuple of (mode, is_supported).
    """
    if not mode:
        return BlendMode.NORMAL, True

    key = mode.strip().lower()
    parsed = _BLEND_ALIASES.get(key)
    if parsed is None:
        return BlendMode.NORMAL, False
    return parsed, True


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0..1] to linear-light [0..1]."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear-light [0..1] to sRGB [0..1]."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055)


def premultiply_alpha(rgb_linear: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Premultiply linear RGB by alpha."""
    return rgb_linear * alpha


def unpremultiply_alpha(rgb_premult_linear: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Unpremultiply linear RGB by alpha with safe handling for alpha=0."""
    out = np.zeros_like(rgb_premult_linear, dtype=np.float32)
    mask = alpha > 1e-8
    out[mask[..., 0]] = rgb_premult_linear[mask[..., 0]] / alpha[mask[..., 0]]
    return np.clip(out, 0.0, 1.0)


def rgba_u8_to_linear_premult(arr: np.ndarray) -> np.ndarray:
    """Convert uint8 RGBA to float32 linear-premult RGBA."""
    rgba = arr.astype(np.float32) / 255.0
    alpha = rgba[:, :, 3:4]
    rgb_linear = srgb_to_linear(rgba[:, :, :3])
    rgb_pm = premultiply_alpha(rgb_linear, alpha)
    return np.concatenate([rgb_pm, alpha], axis=2).astype(np.float32)


def linear_premult_to_rgba_u8(arr: np.ndarray) -> np.ndarray:
    """Convert float32 linear-premult RGBA to uint8 RGBA."""
    alpha = np.clip(arr[:, :, 3:4], 0.0, 1.0)
    rgb_linear = unpremultiply_alpha(arr[:, :, :3], alpha)
    rgb_srgb = linear_to_srgb(rgb_linear)
    out = np.concatenate([rgb_srgb, alpha], axis=2)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def alpha_composite_linear_premult(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Composite src over dst in linear-premult space."""
    sa = src[:, :, 3:4]
    da = dst[:, :, 3:4]
    out_a = sa + da * (1.0 - sa)
    out_rgb = src[:, :, :3] + dst[:, :, :3] * (1.0 - sa)
    return np.concatenate([out_rgb, out_a], axis=2).astype(np.float32)


def _blend_rgb(cb: np.ndarray, cs: np.ndarray, mode: BlendMode) -> np.ndarray:
    """Blend source/backdrop RGB in linear-light (both straight)."""
    if mode == BlendMode.MULTIPLY:
        return cb * cs
    if mode == BlendMode.SCREEN:
        return 1.0 - (1.0 - cb) * (1.0 - cs)
    if mode == BlendMode.OVERLAY:
        return np.where(
            cb <= 0.5,
            2.0 * cb * cs,
            1.0 - 2.0 * (1.0 - cb) * (1.0 - cs),
        )
    return cs


def alpha_blend_mode_linear_premult(
    dst: np.ndarray,
    src: np.ndarray,
    mode: BlendMode,
) -> np.ndarray:
    """Composite src over dst using blend mode in linear-premult space."""
    if mode == BlendMode.NORMAL:
        return alpha_composite_linear_premult(dst, src)

    sa = src[:, :, 3:4]
    da = dst[:, :, 3:4]

    cb = unpremultiply_alpha(dst[:, :, :3], da)
    cs = unpremultiply_alpha(src[:, :, :3], sa)
    blended = _blend_rgb(cb, cs, mode)

    out_a = sa + da - sa * da
    out_rgb = ((1.0 - sa) * dst[:, :, :3]) + ((1.0 - da) * src[:, :, :3]) + (
        sa * da * blended
    )

    return np.concatenate([out_rgb, out_a], axis=2).astype(np.float32)


def composite_pil_over(
    canvas: Image.Image,
    layer: Image.Image,
    position: tuple[int, int],
    use_linear: bool = True,
    blend_mode: BlendMode = BlendMode.NORMAL,
) -> Image.Image:
    """Composite a layer over a canvas at position with optional linear pipeline."""
    x, y = position
    base = canvas.convert("RGBA")
    src = layer.convert("RGBA")

    canvas_arr = np.array(base, dtype=np.uint8)
    layer_arr = np.array(src, dtype=np.uint8)

    h, w = canvas_arr.shape[:2]
    lh, lw = layer_arr.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + lw)
    y2 = min(h, y + lh)
    if x1 >= x2 or y1 >= y2:
        return base

    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    dst_patch = canvas_arr[y1:y2, x1:x2]
    src_patch = layer_arr[sy1:sy2, sx1:sx2]

    if use_linear:
        dst_lp = rgba_u8_to_linear_premult(dst_patch)
        src_lp = rgba_u8_to_linear_premult(src_patch)
        out_patch = linear_premult_to_rgba_u8(
            alpha_blend_mode_linear_premult(dst_lp, src_lp, blend_mode)
        )
    else:
        src_a = src_patch[:, :, 3:4].astype(np.float32) / 255.0
        dst_a = dst_patch[:, :, 3:4].astype(np.float32) / 255.0
        src_rgb = src_patch[:, :, :3].astype(np.float32) / 255.0
        dst_rgb = dst_patch[:, :, :3].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        out_rgb = src_rgb * src_a + dst_rgb * (1.0 - src_a)
        out = np.concatenate([out_rgb, out_a], axis=2)
        out_patch = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)

    canvas_arr[y1:y2, x1:x2] = out_patch
    return Image.fromarray(canvas_arr, mode="RGBA")
