"""Tests for adaptive layout profiles, constraints, and scoring."""

from __future__ import annotations

from backend.app.config import Config
from backend.app.enums import ElementRole
from backend.app.layout.constraints import validate_layout
from backend.app.layout.profiles import pick_profile
from backend.app.layout.scoring import score_layout
from backend.app.models import BoundingBox, DesignElement, LayoutResult


def test_pick_profile_thresholds() -> None:
    assert pick_profile(1920, 1080).name == "LANDSCAPE"
    assert pick_profile(1080, 1080).name == "SQUARE"
    assert pick_profile(1080, 1920).name == "PORTRAIT"


def test_constraints_property_like_inside_margins_and_min_size() -> None:
    profile = pick_profile(1080, 1920)
    role_by_id = {
        "headline": ElementRole.HEADLINE,
        "hero": ElementRole.HERO_IMAGE,
    }

    # property-style loop over a few safe placements
    for y in (120, 180, 240, 300):
        layout = [
            LayoutResult("headline", BoundingBox(90, y, 760, 220), 1.0),
            LayoutResult("hero", BoundingBox(120, 780, 840, 960), 1.0),
        ]
        violations = validate_layout(layout, profile, (1080, 1920), role_by_id)
        assert all(not v.startswith("outside_margin") for v in violations)
        assert all(not v.startswith("min_size") for v in violations)


def test_scoring_prefers_non_overlapping_and_margin_respecting_layout() -> None:
    profile = pick_profile(1080, 1920)
    role_by_id = {
        "headline": ElementRole.HEADLINE,
        "sub": ElementRole.SUBHEADLINE,
        "hero": ElementRole.HERO_IMAGE,
    }

    good = [
        LayoutResult("headline", BoundingBox(80, 120, 800, 220), 1.0),
        LayoutResult("sub", BoundingBox(80, 370, 700, 130), 1.0),
        LayoutResult("hero", BoundingBox(130, 760, 820, 980), 1.0),
    ]

    bad = [
        LayoutResult("headline", BoundingBox(10, 20, 1000, 450), 1.0),  # near margin break
        LayoutResult("sub", BoundingBox(60, 300, 900, 220), 1.0),  # overlap
        LayoutResult("hero", BoundingBox(40, 360, 980, 900), 1.0),  # heavy overlap
    ]

    good_score = score_layout(good, profile, (1080, 1920), role_by_id)
    bad_score = score_layout(bad, profile, (1080, 1920), role_by_id)

    assert good_score > bad_score


def test_layout_engine_logs_profile_scoring(caplog) -> None:
    from backend.app.layout.engine import LayoutEngine

    caplog.set_level("INFO")

    prev = Config.LAYOUT_PROFILE_SCORING_ENABLED
    Config.LAYOUT_PROFILE_SCORING_ENABLED = True
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
                id="headline",
                name="headline",
                layer_type="type",
                bbox=BoundingBox(100, 60, 600, 120),
                role=ElementRole.HEADLINE,
                priority=1,
            ),
            DesignElement(
                id="hero",
                name="hero",
                layer_type="pixel",
                bbox=BoundingBox(700, 80, 420, 520),
                role=ElementRole.HERO_IMAGE,
                priority=2,
            ),
        ]
        _ = engine.calculate_layout(elements, (1200, 628), (1080, 1920))
    finally:
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev

    assert "profile=PORTRAIT" in caplog.text
    assert "candidates=" in caplog.text
    assert "best_score=" in caplog.text
