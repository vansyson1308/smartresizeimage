"""Tests for semantic classifier."""

import pytest
from PIL import Image

from backend.app.classifier.semantic_classifier import SemanticClassifier
from backend.app.constants import ROLE_PRIORITIES
from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, DesignElement


class TestRuleBased:
    def setup_method(self):
        self.classifier = SemanticClassifier(use_ai=False)
        self.canvas_size = (1080, 1080)

    def _make_element(self, name: str, layer_type: str = "pixel", **kwargs):
        return DesignElement(
            id=f"test_{name}",
            name=name,
            layer_type=layer_type,
            bbox=kwargs.get("bbox", BoundingBox(x=0, y=0, width=100, height=100)),
            image=kwargs.get("image", Image.new("RGBA", (100, 100))),
            **{k: v for k, v in kwargs.items() if k not in ("bbox", "image")},
        )

    def test_headline_classification(self):
        elem = self._make_element("headline_main")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.HEADLINE

    def test_title_classification(self):
        elem = self._make_element("title_text")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.HEADLINE

    def test_cta_classification(self):
        elem = self._make_element("cta_button")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.CTA

    def test_button_classification(self):
        elem = self._make_element("shop_now_btn")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.CTA

    def test_logo_classification(self):
        elem = self._make_element("brand_logo")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.LOGO

    def test_background_classification(self):
        elem = self._make_element("bg_main")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.BACKGROUND

    def test_hero_classification(self):
        elem = self._make_element("hero_product")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.HERO_IMAGE

    def test_group_layer_returns_group(self):
        elem = self._make_element("any_name", layer_type="group")
        role = self.classifier.classify(elem, self.canvas_size)
        assert role == ElementRole.GROUP

    def test_unknown_name_returns_unknown_or_heuristic(self):
        elem = self._make_element("xyz_123", layer_type="unknown")
        role = self.classifier.classify(elem, self.canvas_size)
        assert isinstance(role, ElementRole)


class TestAllRolesHavePriorities:
    def test_every_role_has_priority(self):
        """Every ElementRole member must have a priority defined."""
        for role in ElementRole:
            assert role in ROLE_PRIORITIES, (
                f"ElementRole.{role.name} missing from ROLE_PRIORITIES"
            )

    def test_priorities_are_valid_range(self):
        """All priorities should be between 1 and 9."""
        for role, priority in ROLE_PRIORITIES.items():
            assert 1 <= priority <= 9, (
                f"Priority for {role.name} is {priority}, expected 1-9"
            )


class TestHeuristics:
    def setup_method(self):
        self.classifier = SemanticClassifier(use_ai=False)

    def test_large_pixel_layer_classified_as_background(self):
        """A pixel layer covering >50% of canvas should be BACKGROUND."""
        canvas_size = (1000, 1000)
        elem = DesignElement(
            id="big_img",
            name="unnamed_layer",
            layer_type="pixel",
            bbox=BoundingBox(x=0, y=0, width=800, height=800),
            image=Image.new("RGBA", (800, 800)),
        )
        role = self.classifier.classify(elem, canvas_size)
        assert role == ElementRole.BACKGROUND

    def test_small_pixel_layer_classified_as_icon(self):
        """A pixel layer covering <2% of canvas should be ICON."""
        canvas_size = (1000, 1000)
        elem = DesignElement(
            id="tiny_img",
            name="unnamed_layer",
            layer_type="pixel",
            bbox=BoundingBox(x=0, y=0, width=30, height=30),
            image=Image.new("RGBA", (30, 30)),
        )
        role = self.classifier.classify(elem, canvas_size)
        assert role == ElementRole.ICON

    def test_text_layer_at_top_classified_as_headline(self):
        """A text layer near the top with short content should be HEADLINE via heuristics."""
        canvas_size = (1000, 1000)
        elem = DesignElement(
            id="text_top",
            name="layer_42",  # Name that doesn't match any rule pattern
            layer_type="type",
            bbox=BoundingBox(x=50, y=50, width=300, height=60),
            text_content="Big Sale",
        )
        role = self.classifier.classify(elem, canvas_size)
        assert role == ElementRole.HEADLINE

    def test_text_layer_with_long_text_classified_as_body(self):
        """A text layer with >50 chars should be BODY_TEXT."""
        canvas_size = (1000, 1000)
        elem = DesignElement(
            id="text_long",
            name="some_text",
            layer_type="type",
            bbox=BoundingBox(x=50, y=500, width=300, height=100),
            text_content="A" * 60,  # 60 characters
        )
        role = self.classifier.classify(elem, canvas_size)
        assert role == ElementRole.BODY_TEXT


class TestClassifyAll:
    def test_classify_all_updates_roles(self):
        classifier = SemanticClassifier(use_ai=False)
        elements = [
            DesignElement(
                id="h1", name="headline_big", layer_type="type",
                bbox=BoundingBox(0, 0, 100, 50),
                image=Image.new("RGBA", (100, 50)),
            ),
            DesignElement(
                id="bg1", name="background_main", layer_type="pixel",
                bbox=BoundingBox(0, 0, 1080, 1080),
                image=Image.new("RGBA", (1080, 1080)),
            ),
        ]
        result = classifier.classify_all(elements, (1080, 1080))
        assert result[0].role == ElementRole.HEADLINE
        assert result[1].role == ElementRole.BACKGROUND
        assert result[0].priority <= result[1].priority
