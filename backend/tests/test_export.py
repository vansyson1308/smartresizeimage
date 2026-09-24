"""Tests for format encoding and file-size budgets."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from backend.app.exceptions import ValidationError
from backend.app.export import ExportOptions, encode_image


def _noisy(size=(600, 400), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _smooth(size=(600, 400)) -> Image.Image:
    x = np.linspace(0, 255, size[0], dtype=np.float32)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :, 0] = x.astype(np.uint8)
    arr[:, :, 2] = 255 - x.astype(np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.mark.parametrize("fmt", ["png", "jpeg", "jpg", "webp"])
def test_roundtrip_formats(fmt):
    out = encode_image(_smooth(), ExportOptions(format=fmt))
    decoded = Image.open(io.BytesIO(out.data))
    assert decoded.size == (600, 400)
    assert out.within_budget
    assert out.extension in {"png", "jpg", "webp"}
    assert out.mime_type.startswith("image/")


def test_jpeg_budget_search_lowers_quality_to_fit():
    img = _noisy()
    unconstrained = encode_image(img, ExportOptions(format="jpeg", quality=90))
    budget_kb = int(unconstrained.size_kb * 0.6)
    out = encode_image(img, ExportOptions(format="jpeg", quality=90, max_kb=budget_kb))
    assert out.within_budget
    assert len(out.data) <= budget_kb * 1024
    assert out.quality is not None and out.quality < 90


def test_impossible_budget_reports_failure_with_smallest_attempt():
    out = encode_image(_noisy(), ExportOptions(format="webp", max_kb=1))
    assert not out.within_budget
    assert out.quality == 35


def test_png_budget_falls_back_to_palette():
    img = _smooth((800, 800))
    lossless = encode_image(img, ExportOptions(format="png"))
    out = encode_image(img, ExportOptions(format="png", max_kb=max(1, int(lossless.size_kb / 3))))
    assert out.palette_colors is not None or not out.within_budget
    assert len(out.data) <= len(lossless.data)


def test_rgba_input_to_jpeg_is_flattened():
    img = Image.new("RGBA", (50, 50), (10, 20, 30, 128))
    out = encode_image(img, ExportOptions(format="jpeg"))
    assert Image.open(io.BytesIO(out.data)).mode == "RGB"


@pytest.mark.parametrize(
    "kwargs", [{"format": "gif"}, {"quality": 0}, {"quality": 101}, {"max_kb": 0}]
)
def test_invalid_options(kwargs):
    with pytest.raises(ValidationError):
        ExportOptions(**kwargs)
