"""Tests for typography reflow engine."""

from __future__ import annotations

import hashlib

from PIL import Image, ImageDraw

from backend.app.enums import ElementRole
from backend.app.layout.typography import (
    fit_text_block,
    load_font,
    measure_text,
    wrap_text_to_width,
)


def test_measure_text_returns_positive_dims() -> None:
    font = load_font(None, 24)
    w, h = measure_text(font, "Hello World")
    assert w > 0
    assert h > 0


def test_wrap_text_to_width_preserves_characters() -> None:
    font = load_font(None, 20)
    text = "This is a long headline with multiple words"
    lines = wrap_text_to_width(text, font, max_width=180)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


def test_fit_text_block_long_headline() -> None:
    result = fit_text_block(
        text="Ultra Long Headline For Portrait Story Layout",
        font_family=None,
        max_font=56,
        min_font=18,
        max_width=420,
        max_lines=4,
    )
    assert result.font_size >= 18
    assert len(result.lines) <= 4
    assert result.bbox[0] <= 420


def test_fit_text_block_long_subheadline() -> None:
    result = fit_text_block(
        text="This subheadline needs wrapping but should remain readable and complete",
        font_family=None,
        max_font=40,
        min_font=14,
        max_width=360,
        max_lines=5,
    )
    assert result.font_size >= 14
    assert len(result.lines) <= 5
    assert result.bbox[0] <= 360


def test_fit_text_block_long_cta() -> None:
    result = fit_text_block(
        text="LIMITED TIME OFFER CLICK TO SHOP NOW",
        font_family=None,
        max_font=28,
        min_font=12,
        max_width=280,
        max_lines=2,
    )
    assert result.font_size >= 12
    assert len(result.lines) <= 2
    assert result.bbox[0] <= 280


def test_fit_text_block_does_not_drop_characters() -> None:
    text = "Headline with exact characters 1234 !"
    result = fit_text_block(text, None, 44, 16, 300, 3)
    joined = " ".join(result.lines)
    assert "".join(joined.split()) == "".join(text.split())


def test_golden_text_block_render_hash_regression() -> None:
    text = "Designer Like Re Layout Typography"
    result = fit_text_block(text, None, 44, 16, 320, 3)

    font = load_font(None, result.font_size)
    img = Image.new("L", (340, 140), 255)
    draw = ImageDraw.Draw(img)
    y = 8
    for line in result.lines:
        draw.text((10, y), line, font=font, fill=0)
        _, lh = measure_text(font, line)
        y += lh + 2

    digest = hashlib.sha256(img.tobytes()).hexdigest()
    assert digest == "8cfd82efbde536174490beb5538353d68c08a161f481d0acabd5d6433b5293a2"


def test_layout_engine_typography_logs_choice(caplog) -> None:
    from backend.app.config import Config
    from backend.app.layout.engine import LayoutEngine
    from backend.app.models import BoundingBox, DesignElement

    prev = Config.LAYOUT_PROFILE_SCORING_ENABLED
    Config.LAYOUT_PROFILE_SCORING_ENABLED = True
    caplog.set_level("INFO")
    try:
        engine = LayoutEngine()
        elements = [
            DesignElement(
                id="bg",
                name="bg",
                layer_type="pixel",
                bbox=BoundingBox(0, 0, 1200, 628),
                role=ElementRole.BACKGROUND,
                priority=9,
            ),
            DesignElement(
                id="hl",
                name="headline",
                layer_type="type",
                bbox=BoundingBox(80, 60, 600, 100),
                text_content="A very long adaptive headline to test auto font and wrapping",
                role=ElementRole.HEADLINE,
                priority=1,
            ),
        ]
        _ = engine.calculate_layout(elements, (1200, 628), (1080, 1920))
    finally:
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev

    assert "typography element=hl" in caplog.text
    assert "font=" in caplog.text
    assert "lines=" in caplog.text
