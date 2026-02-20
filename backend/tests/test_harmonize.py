"""Golden-style tests for safe harmonization and grounding shadow."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageDraw

from backend.app.generative.harmonize import (
    ShadowSpec,
    apply_color_grading_safe,
    apply_grounding_shadow_safe,
)


def _hash_rgba(img: Image.Image) -> str:
    return hashlib.sha256(img.convert("RGBA").tobytes()).hexdigest()


def test_shadow_grounding_mvp_golden_hash() -> None:
    canvas = Image.new("RGBA", (120, 100), (210, 220, 235, 255))

    mascot = Image.new("RGBA", (120, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mascot)
    draw.ellipse((38, 24, 84, 76), fill=(255, 160, 60, 255))

    mascot_mask = np.asarray(mascot.split()[3], dtype=np.float32) > 0
    protected = mascot_mask.copy()

    out = apply_grounding_shadow_safe(
        canvas,
        mascot_masks=[mascot_mask],
        protected_mask=protected,
        spec=ShadowSpec(offset_x=4, offset_y=7, blur_radius=7.0, opacity=0.25),
    )

    assert _hash_rgba(out) == "ed3271391293e243c7e06d32d1813b94c589e25ee1c3d10cde9978021620e3e9"


def test_color_grading_keeps_protected_logo_text_unchanged() -> None:
    base = Image.new("RGBA", (80, 60), (50, 90, 160, 255))
    draw = ImageDraw.Draw(base)
    draw.rectangle((8, 8, 28, 20), fill=(220, 20, 20, 255))  # logo-like region
    draw.rectangle((8, 24, 36, 34), fill=(240, 240, 240, 255))  # text-like region

    protected = np.zeros((60, 80), dtype=bool)
    protected[8:21, 8:29] = True
    protected[24:35, 8:37] = True

    before = np.asarray(base)
    out = apply_color_grading_safe(base, protected, strength=0.2)
    after = np.asarray(out)

    # protected pixels must remain exactly unchanged
    assert np.array_equal(after[protected], before[protected])

    # harmonization should affect editable area
    editable = ~protected
    delta = np.abs(after[editable].astype(np.int16) - before[editable].astype(np.int16))
    assert delta.mean() > 0.0
