"""Tests for collision solver, snapping, and debug export."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from backend.app.config import Config
from backend.app.enums import ElementRole
from backend.app.layout.profiles import pick_profile
from backend.app.layout.solver import solve_layout, total_overlap_area
from backend.app.models import BoundingBox, DesignElement, LayoutResult
from backend.app.relayout import ReLayoutEngine


def test_solver_resolves_crowded_five_elements_within_margins() -> None:
    profile = pick_profile(1080, 1920)
    role_by_id = {
        "h": ElementRole.HEADLINE,
        "s": ElementRole.SUBHEADLINE,
        "c": ElementRole.CTA,
        "l": ElementRole.LOGO,
        "d": ElementRole.DECORATION,
    }

    crowded = [
        LayoutResult("h", BoundingBox(80, 140, 640, 220), 1.0),
        LayoutResult("s", BoundingBox(100, 170, 600, 180), 1.0),
        LayoutResult("c", BoundingBox(120, 200, 520, 120), 1.0),
        LayoutResult("l", BoundingBox(140, 210, 260, 100), 1.0),
        LayoutResult("d", BoundingBox(160, 240, 620, 300), 1.0),
    ]

    solved, _meta = solve_layout(
        crowded,
        target_size=(1080, 1920),
        profile=profile,
        role_by_id=role_by_id,
        iterations=30,
    )

    overlap = total_overlap_area(solved)
    assert overlap <= 2000  # threshold for dense pack

    margin_x = int(profile.margin_pct * 1080)
    margin_y = int(profile.margin_pct * 1920)
    for r in solved:
        b = r.new_bbox
        assert b.x >= margin_x
        assert b.y >= margin_y
        assert b.x2 <= 1080 - margin_x
        assert b.y2 <= 1920 - margin_y


def test_layout_debug_toggle_exports_overlay_and_json(tmp_path, monkeypatch) -> None:
    prev_enabled = Config.LAYOUT_DEBUG_ENABLED
    prev_dir = Config.LAYOUT_DEBUG_DIR
    prev_scoring = Config.LAYOUT_PROFILE_SCORING_ENABLED

    monkeypatch.setattr(Config, "LAYOUT_DEBUG_ENABLED", True)
    monkeypatch.setattr(Config, "LAYOUT_DEBUG_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "LAYOUT_PROFILE_SCORING_ENABLED", True)

    try:
        engine = ReLayoutEngine(use_ai=False)
        engine.source_size = (100, 100)
        engine.elements = [
            DesignElement(
                id="bg",
                name="bg",
                layer_type="pixel",
                bbox=BoundingBox(0, 0, 100, 100),
                image=Image.new("RGBA", (100, 100), (220, 220, 220, 255)),
                role=ElementRole.BACKGROUND,
                priority=9,
            ),
            DesignElement(
                id="h",
                name="headline",
                layer_type="type",
                bbox=BoundingBox(10, 10, 60, 20),
                text_content="Test headline",
                image=Image.new("RGBA", (60, 20), (0, 0, 0, 0)),
                role=ElementRole.HEADLINE,
                priority=1,
            ),
        ]

        _ = engine.relayout((1080, 1920))
        overlay = Path(tmp_path) / "layout_debug_overlay.png"
        payload = Path(tmp_path) / "layout_debug.json"
        assert overlay.exists()
        assert payload.exists()
    finally:
        Config.LAYOUT_DEBUG_ENABLED = prev_enabled
        Config.LAYOUT_DEBUG_DIR = prev_dir
        Config.LAYOUT_PROFILE_SCORING_ENABLED = prev_scoring
