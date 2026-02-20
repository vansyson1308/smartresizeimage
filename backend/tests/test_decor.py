"""Tests for optional decor synthesis (BG_PLUS_DECOR)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.generative.decor import (
    apply_optional_decor_synthesis,
    build_decor_zone_mask,
)
from backend.app.generative.engine import GenerativeOutpaintEngine


class _MockDecorBackend:
    model_id = "mock-decor-v1"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        base_canvas: Image.Image,
        editable_mask: np.ndarray,
        seed: int,
    ) -> Image.Image:
        _ = editable_mask, seed
        return Image.new("RGBA", base_canvas.size, (255, 0, 180, 255))


def test_policy_off_behaves_like_no_decor() -> None:
    engine = GenerativeOutpaintEngine(backend=_MockDecorBackend())
    base = Image.new("RGBA", (120, 100), (20, 30, 40, 255))
    protected = np.zeros((100, 120), dtype=bool)
    protected[20:40, 30:60] = True

    out, meta = apply_optional_decor_synthesis(
        base_canvas=base,
        protected_mask=protected,
        policy="OFF",
        seed=1,
        outpaint_engine=engine,
    )

    assert np.array_equal(np.asarray(out), np.asarray(base))
    assert meta["applied"] is False
    assert meta["fallback_reason"] == "policy_off"


def test_policy_on_adds_decor_only_in_allowed_area_and_never_protected() -> None:
    engine = GenerativeOutpaintEngine(backend=_MockDecorBackend())
    base = Image.new("RGBA", (120, 100), (10, 20, 30, 255))
    protected = np.zeros((100, 120), dtype=bool)
    protected[10:35, 15:50] = True

    decor_mask = build_decor_zone_mask(base.size, protected)
    out, meta = apply_optional_decor_synthesis(
        base_canvas=base,
        protected_mask=protected,
        policy="BG_PLUS_DECOR",
        seed=7,
        outpaint_engine=engine,
        ocr_extractor=lambda _img: [],
    )

    before = np.asarray(base)
    after = np.asarray(out)
    changed = np.any(after != before, axis=2)

    assert meta["applied"] is True
    assert changed.any()  # decor appears
    assert not np.any(changed & protected)  # protected untouched
    assert np.all(changed <= decor_mask)  # only decor zone modified


def test_ocr_negative_gate_runs_and_blocks_false_text() -> None:
    engine = GenerativeOutpaintEngine(backend=_MockDecorBackend())
    base = Image.new("RGBA", (120, 100), (80, 90, 100, 255))
    protected = np.zeros((100, 120), dtype=bool)

    # Fake OCR finds text in decor zone -> should fail gate.
    out, meta = apply_optional_decor_synthesis(
        base_canvas=base,
        protected_mask=protected,
        policy="BG_PLUS_DECOR",
        seed=9,
        outpaint_engine=engine,
        ocr_extractor=lambda _img: [(2, 2, 20, 12)],
    )

    assert np.array_equal(np.asarray(out), np.asarray(base))
    assert meta["applied"] is False
    assert meta["ocr_gate_ran"] is True
    assert meta["ocr_overlap_boxes"] >= 1
