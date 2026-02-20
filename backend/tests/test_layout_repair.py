"""Tests for hard-margin repair pass."""

from __future__ import annotations

from backend.app.config import Config
from backend.app.enums import ElementRole
from backend.app.layout.constraints import validate_layout
from backend.app.layout.engine import LayoutEngine
from backend.app.layout.profiles import pick_profile
from backend.app.layout.repair import clamp_bbox_to_margins
from backend.app.models import BoundingBox, DesignElement


def test_clamp_bbox_to_margins_keeps_bbox_inside() -> None:
    bbox = BoundingBox(x=-120, y=-50, width=900, height=700)
    clamped = clamp_bbox_to_margins(bbox, margins=(54, 108), canvas_w=1080, canvas_h=1920)

    assert clamped.x >= 54
    assert clamped.y >= 108
    assert clamped.x2 <= 1080 - 54
    assert clamped.y2 <= 1920 - 108


def test_layout_engine_repair_removes_outside_margin_violations() -> None:
    prev = Config.LAYOUT_PROFILE_SCORING_ENABLED
    Config.LAYOUT_PROFILE_SCORING_ENABLED = True
    try:
        engine = LayoutEngine()
        source_size = (1200, 628)
        target_size = (1080, 1920)

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
                id="headline",
                name="headline",
                layer_type="type",
                bbox=BoundingBox(20, 20, 980, 260),
                text_content="CROWDED HEADLINE SALE SALE SALE",
                role=ElementRole.HEADLINE,
                priority=1,
            ),
            DesignElement(
                id="sub",
                name="sub",
                layer_type="type",
                bbox=BoundingBox(40, 180, 900, 240),
                text_content="Long subheadline that tends to overlap and overflow",
                role=ElementRole.SUBHEADLINE,
                priority=2,
            ),
            DesignElement(
                id="cta",
                name="cta",
                layer_type="type",
                bbox=BoundingBox(50, 260, 700, 180),
                text_content="SHOP NOW",
                role=ElementRole.CTA,
                priority=2,
            ),
            DesignElement(
                id="logo",
                name="logo",
                layer_type="pixel",
                bbox=BoundingBox(960, 12, 300, 140),
                role=ElementRole.LOGO,
                priority=1,
            ),
            DesignElement(
                id="hero",
                name="hero",
                layer_type="pixel",
                bbox=BoundingBox(720, 90, 520, 560),
                role=ElementRole.HERO_IMAGE,
                priority=2,
            ),
        ]

        layout = engine.calculate_layout(elements, source_size, target_size)
        profile = pick_profile(*target_size)
        role_by_id = {e.id: e.role for e in elements}
        violations = validate_layout(layout, profile, target_size, role_by_id)

        assert all(not v.startswith("outside_margin:") for v in violations)
        assert engine.last_layout_debug.get("profile_name") == profile.name
        assert "repair_applied" in engine.last_layout_debug
        assert "repair_steps" in engine.last_layout_debug
        assert "fallback_used" in engine.last_layout_debug
    finally:
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev
