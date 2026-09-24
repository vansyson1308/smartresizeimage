"""Tests for safe-zone group fitting."""

from __future__ import annotations

from backend.app.layout.safe_zone import fit_group_into_safe_rect
from backend.app.models import BoundingBox


def _inside(b, safe):
    return b.x >= safe[0] and b.y >= safe[1] and b.x2 <= safe[2] and b.y2 <= safe[3]


def test_noop_when_already_inside():
    boxes = {"a": BoundingBox(100, 300, 200, 100)}
    out, scale = fit_group_into_safe_rect(boxes, {"a"}, (0, 200, 1080, 1600))
    assert scale == 1.0
    assert out == boxes


def test_noop_without_critical_elements():
    boxes = {"hero": BoundingBox(0, 0, 100, 100)}
    out, scale = fit_group_into_safe_rect(boxes, set(), (10, 10, 90, 90))
    assert (out, scale) == (boxes, 1.0)


def test_group_is_translated_preserving_arrangement():
    boxes = {"headline": BoundingBox(100, 50, 400, 100), "cta": BoundingBox(100, 200, 200, 80)}
    safe = (0, 269, 1080, 1536)
    out, scale = fit_group_into_safe_rect(boxes, {"headline", "cta"}, safe)
    assert scale == 1.0
    assert all(_inside(b, safe) for b in out.values())
    # Relative offsets unchanged -> no new overlaps.
    assert out["cta"].y - out["headline"].y == 150
    assert out["cta"].x - out["headline"].x == 0


def test_group_is_shrunk_when_taller_than_safe_rect():
    boxes = {"headline": BoundingBox(0, 0, 500, 900), "cta": BoundingBox(0, 1000, 300, 900)}
    safe = (0, 269, 1080, 1536)
    out, scale = fit_group_into_safe_rect(boxes, {"headline", "cta"}, safe)
    assert scale < 1.0
    assert all(_inside(b, safe) for b in out.values())
    assert out["headline"].width == round(500 * scale)
