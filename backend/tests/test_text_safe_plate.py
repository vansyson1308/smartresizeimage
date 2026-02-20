"""Tests for text-safe background plate generation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.generative.text_plate import (
    TextPlateConfig,
    apply_text_safe_plates,
    compute_busy_score,
)


def _noisy_bg(w: int, h: int, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate([arr, alpha], axis=2), mode="RGBA")


def test_busy_score_higher_for_noisy_region() -> None:
    noisy = _noisy_bg(180, 120, seed=1)
    flat = Image.new("RGBA", (180, 120), (120, 120, 120, 255))

    noisy_score = compute_busy_score(noisy, (20, 20, 100, 60))
    flat_score = compute_busy_score(flat, (20, 20, 100, 60))

    assert noisy_score > flat_score
    assert 0.0 <= noisy_score <= 1.0


def test_plate_improves_readability_proxy_and_respects_protected() -> None:
    bg = _noisy_bg(240, 160, seed=3)
    text_box = (60, 50, 110, 40)

    protected = np.zeros((160, 240), dtype=bool)
    # protected mascot/logo zone must stay unchanged
    protected[40:100, 10:50] = True

    cfg = TextPlateConfig(enabled=True, style="blur", busy_threshold=0.1, opacity=120)
    out, meta = apply_text_safe_plates(bg, [text_box], protected, cfg)

    before = compute_busy_score(bg, text_box)
    after = compute_busy_score(out, text_box)
    assert after < before
    assert meta["applied"] is True

    bg_arr = np.array(bg)
    out_arr = np.array(out)
    assert np.array_equal(out_arr[protected], bg_arr[protected])


def test_plate_can_be_disabled() -> None:
    bg = _noisy_bg(220, 140, seed=4)
    text_box = (50, 40, 120, 50)
    protected = np.zeros((140, 220), dtype=bool)

    out, meta = apply_text_safe_plates(
        bg,
        [text_box],
        protected,
        TextPlateConfig(enabled=False),
    )

    assert meta["applied"] is False
    assert np.array_equal(np.array(out), np.array(bg))
