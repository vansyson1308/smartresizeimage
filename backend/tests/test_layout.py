"""Tests for layout engine."""

import pytest
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.layout.engine import LayoutEngine
from backend.app.models import BoundingBox, DesignElement


class TestTemplateSelection:
    def setup_method(self):
        self.engine = LayoutEngine()

    def test_portrait_tall_selected_for_9_16(self):
        template = self.engine._select_template(9 / 16)  # 0.5625
        assert template["aspect_range"] == (0.5, 0.65)

    def test_square_selected_for_1_1(self):
        template = self.engine._select_template(1.0)
        assert template["aspect_range"] == (0.85, 1.15)

    def test_landscape_selected_for_16_9(self):
        template = self.engine._select_template(16 / 9)  # ~1.78
        assert template["aspect_range"] == (1.15, 2.0)

    def test_landscape_wide_selected_for_3_1(self):
        template = self.engine._select_template(3.0)
        assert template["aspect_range"] == (2.0, 4.0)

    def test_default_to_landscape_for_extreme_ratio(self):
        template = self.engine._select_template(10.0)
        # Should fall back to landscape
        assert "zones" in template


class TestCalculateLayout:
    def setup_method(self):
        self.engine = LayoutEngine()

    def _make_elements(self):
        elements = []

        bg = DesignElement(
            id="bg_001", name="Background", layer_type="pixel",
            bbox=BoundingBox(0, 0, 1080, 1080),
            image=Image.new("RGBA", (1080, 1080)),
            role=ElementRole.BACKGROUND, priority=9,
        )
        elements.append(bg)

        headline = DesignElement(
            id="hl_001", name="Headline", layer_type="type",
            bbox=BoundingBox(50, 100, 500, 80),
            image=Image.new("RGBA", (500, 80)),
            role=ElementRole.HEADLINE, priority=1,
        )
        elements.append(headline)

        hero = DesignElement(
            id="hero_001", name="Hero", layer_type="pixel",
            bbox=BoundingBox(300, 300, 400, 400),
            image=Image.new("RGBA", (400, 400)),
            role=ElementRole.HERO_IMAGE, priority=3,
        )
        elements.append(hero)

        return elements

    def test_layout_returns_results_for_all_elements(self):
        elements = self._make_elements()
        results = self.engine.calculate_layout(
            elements, (1080, 1080), (1080, 1920)
        )
        result_ids = {r.element_id for r in results}
        element_ids = {e.id for e in elements}
        assert element_ids == result_ids

    def test_background_covers_full_canvas(self):
        elements = self._make_elements()
        results = self.engine.calculate_layout(
            elements, (1080, 1080), (500, 500)
        )
        bg_result = next(r for r in results if r.element_id == "bg_001")
        assert bg_result.new_bbox.width == 500
        assert bg_result.new_bbox.height == 500

    def test_all_results_visible_by_default(self):
        elements = self._make_elements()
        results = self.engine.calculate_layout(
            elements, (1080, 1080), (1080, 1080)
        )
        for r in results:
            assert r.visible is True


class TestZoneAssignment:
    def setup_method(self):
        self.engine = LayoutEngine()

    def test_zone_overflow_prevented(self):
        """Creating many elements of the same role should not overflow a zone."""
        elements = []
        for i in range(5):
            elements.append(
                DesignElement(
                    id=f"hl_{i}", name=f"Headline {i}", layer_type="type",
                    bbox=BoundingBox(0, 0, 200, 50),
                    image=Image.new("RGBA", (200, 50)),
                    role=ElementRole.HEADLINE, priority=1,
                )
            )

        results = self.engine.calculate_layout(
            elements, (1080, 1080), (1080, 1080)
        )
        # All elements should have results (assigned or unassigned fallback)
        assert len(results) == 5

    def test_invalid_element_id_handled_gracefully(self):
        """Layout should not crash even with edge cases."""
        elements = [
            DesignElement(
                id="only_one", name="Test", layer_type="pixel",
                bbox=BoundingBox(0, 0, 100, 100),
                image=Image.new("RGBA", (100, 100)),
                role=ElementRole.DECORATION, priority=8,
            ),
        ]
        # Should not raise
        results = self.engine.calculate_layout(
            elements, (1080, 1080), (500, 500)
        )
        assert len(results) >= 1
