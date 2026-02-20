"""Tests for drop shadow effect schema and renderer."""

from __future__ import annotations

from PIL import Image

from backend.app.composition.effects import parse_drop_shadow_effect, render_drop_shadow


def test_parse_drop_shadow_absent_returns_none() -> None:
    assert parse_drop_shadow_effect({}) is None
    assert parse_drop_shadow_effect({"drop_shadow": None}) is None


def test_parse_drop_shadow_defaults_when_missing_fields() -> None:
    params = parse_drop_shadow_effect({"drop_shadow": {}})
    assert params is not None
    assert params.blur_radius >= 0


def test_render_drop_shadow_returns_image_and_offset() -> None:
    src = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
    params = parse_drop_shadow_effect(
        {"drop_shadow": {"offset_x": 4, "offset_y": 5, "blur_radius": 3, "opacity": 0.5}}
    )
    assert params is not None
    shadow, offset = render_drop_shadow(src, params)
    assert shadow.mode == "RGBA"
    assert shadow.size[0] > src.size[0]
    assert shadow.size[1] > src.size[1]
    assert isinstance(offset, tuple)
