"""Role-aware stack layout engine."""

from __future__ import annotations

import pytest
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.layout.stack import StackLayoutEngine
from backend.app.models import BoundingBox, DesignElement
from backend.app.presets import PRESETS
from backend.tools.flat_banner_samples import make_sample

SIZES = sorted({p.size for p in PRESETS.values()}) + [(1080, 1920), (1200, 628)]


def _layout(sample_idx: int, size: tuple[int, int]):
    sample = make_sample(sample_idx)
    elements = sample.design_elements()
    engine = StackLayoutEngine()
    results = engine.calculate(elements, sample.image.size, size)
    return elements, {r.element_id: r for r in results}, engine.last_debug


def _overlap(a: BoundingBox, b: BoundingBox) -> int:
    ix = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    iy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return ix * iy


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
@pytest.mark.parametrize("sample_idx", [0, 1, 2])
def test_no_overlap_inside_canvas_and_undistorted(sample_idx, size):
    elements, results, _ = _layout(sample_idx, size)
    tw, th = size
    fg = [
        e for e in elements
        if e.role != ElementRole.BACKGROUND and results[e.id].visible
    ]
    for e in fg:
        b = results[e.id].new_bbox
        assert b.x >= 0 and b.y >= 0 and b.x2 <= tw and b.y2 <= th, (e.id, b)
        src_ratio = e.image.width / e.image.height
        assert abs(b.width / b.height - src_ratio) / src_ratio < 0.12, e.id
    for i, a in enumerate(fg):
        for b in fg[i + 1:]:
            ov = _overlap(results[a.id].new_bbox, results[b.id].new_bbox)
            assert ov <= 0.02 * min(results[a.id].new_bbox.area, results[b.id].new_bbox.area), (
                a.id, b.id,
            )


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_headline_and_cta_are_never_dropped(size):
    _, results, _ = _layout(0, size)
    assert results["headline"].visible
    assert results["cta"].visible


def test_hierarchy_is_preserved():
    for size in [(1080, 1080), (1080, 1920), (1200, 628), (300, 600)]:
        _, results, _ = _layout(0, size)
        if results["subheadline"].visible:
            assert results["headline"].new_bbox.height > results["subheadline"].new_bbox.height


def test_tiny_strip_drops_secondary_copy_first():
    _, results, debug = _layout(1, (320, 50))
    assert debug.mode == "strip"
    assert not results["subheadline"].visible
    assert results["headline"].visible and results["cta"].visible
    assert "subheadline" in debug.dropped
    # a roomier strip keeps it, at a legible size
    _, kept, _ = _layout(1, (728, 90))
    assert kept["subheadline"].visible and kept["subheadline"].new_bbox.height >= 9


def test_modes_follow_aspect_ratio():
    assert _layout(0, (728, 90))[2].mode == "strip"
    assert _layout(0, (1200, 628))[2].mode == "landscape"
    assert _layout(0, (1080, 1920))[2].mode == "vertical"


def test_landscape_keeps_hero_side_from_source():
    # sample 1 has the hero on the left
    _, results, _ = _layout(1, (1200, 628))
    assert results["hero_image"].new_bbox.x < results["headline"].new_bbox.x


def test_returns_none_without_foreground():
    bg = DesignElement(
        "bg", "bg", "pixel", BoundingBox(0, 0, 100, 100),
        image=Image.new("RGBA", (100, 100)), role=ElementRole.BACKGROUND,
    )
    assert StackLayoutEngine().calculate([bg], (100, 100), (50, 50)) is None


def test_fit_aspect_never_stretches():
    box = BoundingBox(10, 10, 400, 100)
    fitted = box.fit_aspect(200, 200)
    assert (fitted.width, fitted.height) == (100, 100)
    assert fitted.x == 160 and fitted.y == 10
    assert box.fit_aspect(800, 200) is box
