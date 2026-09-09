"""Regression tests for layout engine defects found while repairing the evaluator."""

from __future__ import annotations

from PIL import Image, ImageDraw

from backend.app.config import Config
from backend.app.enums import ElementRole
from backend.app.layout.engine import LayoutEngine
from backend.app.layout.profiles import pick_profile
from backend.app.layout.solver import solve_layout, total_overlap_area
from backend.app.models import BoundingBox, DesignElement, LayoutResult


def _raster(size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((0, 0, size[0] - 1, size[1] - 1), fill=(10, 10, 10, 255))
    return img


def _elements() -> list[DesignElement]:
    return [
        DesignElement(
            id="bg",
            name="Background",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, 1200, 628),
            image=Image.new("RGBA", (1200, 628), (80, 120, 160, 255)),
            role=ElementRole.BACKGROUND,
            priority=9,
        ),
        DesignElement(
            id="headline",
            name="Headline",
            layer_type="type",
            bbox=BoundingBox(70, 50, 620, 120),
            image=_raster((620, 120)),
            text_content="SUMMER SUPER SALE",
            role=ElementRole.HEADLINE,
            priority=1,
        ),
        DesignElement(
            id="sub",
            name="Subheadline",
            layer_type="type",
            bbox=BoundingBox(70, 190, 620, 90),
            image=_raster((620, 90)),
            text_content="Up to 50% off selected items",
            role=ElementRole.SUBHEADLINE,
            priority=2,
        ),
        DesignElement(
            id="cta",
            name="CTA",
            layer_type="type",
            bbox=BoundingBox(70, 305, 350, 95),
            image=_raster((350, 95)),
            text_content="SHOP NOW",
            role=ElementRole.CTA,
            priority=2,
        ),
        DesignElement(
            id="logo",
            name="Logo",
            layer_type="pixel",
            bbox=BoundingBox(960, 24, 180, 80),
            image=_raster((180, 80)),
            role=ElementRole.LOGO,
            priority=1,
        ),
        DesignElement(
            id="hero",
            name="Hero",
            layer_type="pixel",
            bbox=BoundingBox(740, 120, 390, 430),
            image=_raster((390, 430)),
            role=ElementRole.HERO_IMAGE,
            priority=2,
        ),
    ]


def test_solver_never_moves_or_clamps_background() -> None:
    target = (1080, 1920)
    profile = pick_profile(*target)
    role_by_id = {e.id: e.role for e in _elements()}
    placements = [
        LayoutResult("bg", BoundingBox(0, 0, 1080, 1920), 1.0),
        LayoutResult("headline", BoundingBox(200, 400, 700, 60), 1.0),
        LayoutResult("hero", BoundingBox(150, 900, 780, 860), 1.0),
    ]
    solved, meta = solve_layout(placements, target, profile, role_by_id, iterations=12)
    bg = next(r for r in solved if r.element_id == "bg").new_bbox
    assert (bg.x, bg.y, bg.width, bg.height) == (0, 0, 1080, 1920)
    # Content overlap must not be inflated by the full-canvas background.
    assert meta["overlap_area"] == total_overlap_area(solved, role_by_id)
    assert total_overlap_area(solved, role_by_id) == 0


def test_solver_keeps_side_by_side_elements_side_by_side() -> None:
    """Vertical rhythm must only apply to elements sharing a column."""
    target = (1200, 628)
    profile = pick_profile(*target)
    role_by_id = {e.id: e.role for e in _elements()}
    placements = [
        LayoutResult("bg", BoundingBox(0, 0, 1200, 628), 1.0),
        LayoutResult("headline", BoundingBox(60, 80, 480, 100), 1.0),
        LayoutResult("hero", BoundingBox(700, 60, 400, 440), 1.0),
    ]
    solved, _ = solve_layout(placements, target, profile, role_by_id, iterations=12)
    hero = next(r for r in solved if r.element_id == "hero").new_bbox
    headline = next(r for r in solved if r.element_id == "headline").new_bbox
    # The hero starts near the top; it must not be pushed below the headline.
    assert hero.y < headline.y2


def test_template_layout_stacks_zone_members_without_overlap() -> None:
    prev = Config.LAYOUT_PROFILE_SCORING_ENABLED
    Config.LAYOUT_PROFILE_SCORING_ENABLED = False
    try:
        engine = LayoutEngine()
        elements = _elements()
        role_by_id = {e.id: e.role for e in elements}
        for target in [(1200, 628), (1080, 1080), (1080, 1920)]:
            layout = engine.calculate_layout(elements, (1200, 628), target)
            by_id = {r.element_id: r.new_bbox for r in layout}
            # headline and subheadline share a zone in every template: no overlap.
            h, s = by_id["headline"], by_id["sub"]
            assert h.y2 <= s.y or s.y2 <= h.y, (target, h, s)
            assert total_overlap_area(layout, role_by_id) < 0.02 * target[0] * target[1]
    finally:
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev


def test_raster_text_keeps_aspect_ratio() -> None:
    """Raster text must be scaled uniformly, never squashed to a native-fit bbox."""
    prev = Config.LAYOUT_PROFILE_SCORING_ENABLED
    Config.LAYOUT_PROFILE_SCORING_ENABLED = False
    try:
        engine = LayoutEngine()
        elements = _elements()
        layout = engine.calculate_layout(elements, (1200, 628), (1080, 1920))
        headline = next(r for r in layout if r.element_id == "headline").new_bbox
        src_aspect = 620 / 120
        assert abs(headline.width / headline.height - src_aspect) < 0.08
        assert headline.x > 0 and headline.x2 <= 1080
    finally:
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev


def test_adaptive_layout_no_longer_collapses_portrait_to_bottom() -> None:
    """Historical failure: every element stacked under the hero at the bottom."""
    prev = Config.LAYOUT_PROFILE_SCORING_ENABLED
    Config.LAYOUT_PROFILE_SCORING_ENABLED = True
    try:
        engine = LayoutEngine()
        elements = _elements()
        role_by_id = {e.id: e.role for e in elements}
        layout = engine.calculate_layout(elements, (1200, 628), (1080, 1920))
        reason = engine.last_layout_debug.get("fallback_reason")
        assert reason != "catastrophic_overlap_after_repair"
        by_id = {r.element_id: r.new_bbox for r in layout}
        assert by_id["headline"].y < by_id["hero"].y
        assert total_overlap_area(layout, role_by_id) < 0.05 * 1080 * 1920
    finally:
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev
