"""Tests for generative BG_ONLY outpainting engine."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.generative.engine import GenerativeOutpaintEngine


class _MockBackend:
    model_id = "mock-diffusion-v1"

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        base_canvas: Image.Image,
        editable_mask: np.ndarray,
        seed: int,
    ) -> Image.Image:
        # intentionally overwrite whole frame to verify mask enforcement
        _ = editable_mask, seed
        return Image.new("RGBA", base_canvas.size, (255, 0, 0, 255))


def test_outpaint_background_fallback_when_backend_unavailable() -> None:
    engine = GenerativeOutpaintEngine(backend=_MockBackend(available=False))
    base = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    editable = np.ones((8, 8), dtype=bool)

    out = engine.outpaint_background(base, editable, policy="BG_ONLY", seed=123)

    assert np.array_equal(np.asarray(out), np.asarray(base))
    assert engine.last_run_metadata["backend_used"] is False
    assert engine.last_run_metadata["fallback_reason"] == "backend_unavailable"


def test_outpaint_background_preserves_protected_pixels_when_mocked_backend() -> None:
    engine = GenerativeOutpaintEngine(backend=_MockBackend(available=True))

    base = Image.new("RGBA", (6, 6), (0, 0, 255, 255))
    editable = np.ones((6, 6), dtype=bool)
    editable[2:4, 2:4] = False  # protected region

    before = np.asarray(base).copy()
    out = engine.outpaint_background(base, editable, policy="BG_ONLY", seed=7)
    after = np.asarray(out)

    assert out.size == (6, 6)
    # protected region pixel-equal before/after
    assert np.array_equal(after[2:4, 2:4], before[2:4, 2:4])
    # editable region can change
    assert (after[0, 0] == np.array([255, 0, 0, 255])).all()

    assert engine.last_run_metadata["backend_used"] is True
    assert engine.last_run_metadata["policy"] == "BG_ONLY"
    assert engine.last_run_metadata["seed"] == 7
    assert engine.last_run_metadata["model_id"] == "mock-diffusion-v1"
