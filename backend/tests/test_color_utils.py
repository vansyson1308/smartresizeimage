"""Tests for linear compositing and premultiplied alpha helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.composition.color import (
    BlendMode,
    alpha_blend_mode_linear_premult,
    composite_pil_over,
    linear_premult_to_rgba_u8,
    parse_blend_mode,
    premultiply_alpha,
    rgba_u8_to_linear_premult,
    srgb_to_linear,
    unpremultiply_alpha,
)


def test_premultiply_unpremultiply_roundtrip_rgb_channels() -> None:
    rgb_srgb = np.array([[[0.8, 0.3, 0.1]]], dtype=np.float32)
    alpha = np.array([[[0.25]]], dtype=np.float32)

    rgb_linear = srgb_to_linear(rgb_srgb)
    premult = premultiply_alpha(rgb_linear, alpha)
    recovered = unpremultiply_alpha(premult, alpha)

    assert np.allclose(recovered, rgb_linear, atol=1e-6)


def test_unpremultiply_zero_alpha_safe() -> None:
    premult = np.array([[[0.2, 0.1, 0.4]]], dtype=np.float32)
    alpha = np.array([[[0.0]]], dtype=np.float32)

    recovered = unpremultiply_alpha(premult, alpha)
    assert np.array_equal(recovered, np.zeros_like(recovered))


def test_linear_pipeline_preserves_alpha_channel() -> None:
    rgba = Image.new("RGBA", (2, 2), (255, 64, 32, 123))
    arr = np.array(rgba, dtype=np.uint8)
    lp = rgba_u8_to_linear_premult(arr)
    out = linear_premult_to_rgba_u8(lp)

    assert np.array_equal(out[:, :, 3], arr[:, :, 3])


def test_linear_compositing_reduces_halo_vs_legacy() -> None:
    bg = Image.new("RGBA", (64, 64), (25, 30, 200, 255))
    fg = Image.new("RGBA", (40, 40), (250, 240, 30, 140))

    linear = composite_pil_over(bg, fg, (12, 12), use_linear=True)
    legacy = composite_pil_over(bg, fg, (12, 12), use_linear=False)

    lin_px = np.array(linear)[12, 12, :3].astype(np.float32)
    old_px = np.array(legacy)[12, 12, :3].astype(np.float32)

    assert not np.array_equal(lin_px, old_px)


def test_parse_blend_mode_aliases_and_fallback() -> None:
    assert parse_blend_mode("multiply") == (BlendMode.MULTIPLY, True)
    assert parse_blend_mode("BlendMode.Screen") == (BlendMode.SCREEN, True)
    assert parse_blend_mode("unknown_weird_mode") == (BlendMode.NORMAL, False)


def test_alpha_blend_mode_linear_premult_shapes() -> None:
    dst = rgba_u8_to_linear_premult(
        np.array([[[20, 40, 180, 255]]], dtype=np.uint8)
    )
    src = rgba_u8_to_linear_premult(
        np.array([[[240, 220, 30, 160]]], dtype=np.uint8)
    )

    for mode in [BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.SCREEN, BlendMode.OVERLAY]:
        out = alpha_blend_mode_linear_premult(dst, src, mode)
        assert out.shape == dst.shape
        assert np.all(out[:, :, 3] >= 0.0)
        assert np.all(out[:, :, 3] <= 1.0)
