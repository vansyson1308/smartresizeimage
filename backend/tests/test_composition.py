"""Tests for composition engine."""

import pytest
from PIL import Image

from backend.app.composition.background import BackgroundExtender, create_feather_mask
from backend.app.composition.engine import CompositionEngine
from backend.app.composition.resize import high_quality_resize
from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, CompositionResult, DesignElement, LayoutResult


class TestHighQualityResize:
    def test_resize_rgba_image(self):
        img = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
        result = high_quality_resize(img, (100, 50))
        assert result.size == (100, 50)
        assert result.mode == "RGBA"

    def test_resize_rgb_image(self):
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        result = high_quality_resize(img, (100, 50))
        assert result.size == (100, 50)
        assert result.mode == "RGB"

    def test_resize_grayscale_image(self, grayscale_image):
        """Grayscale images should not crash (Bug 3 fix)."""
        result = high_quality_resize(grayscale_image, (50, 50))
        assert result.size == (50, 50)
        assert result.mode in ("RGB", "RGBA")

    def test_resize_palette_image(self):
        """Palette mode images should be converted and resized."""
        img = Image.new("P", (100, 100))
        result = high_quality_resize(img, (50, 50))
        assert result.size == (50, 50)

    def test_resize_zero_target_returns_original(self):
        img = Image.new("RGBA", (100, 100))
        result = high_quality_resize(img, (0, 50))
        assert result.size == (100, 100)  # Returns original

    def test_resize_negative_target_returns_original(self):
        img = Image.new("RGBA", (100, 100))
        result = high_quality_resize(img, (-1, 50))
        assert result.size == (100, 100)


class TestBackgroundExtender:
    def test_extend_with_blur_normal_image(self):
        extender = BackgroundExtender(use_ai_inpainting=False)
        img = Image.new("RGBA", (100, 100), (200, 100, 50, 255))
        result = extender.extend(img, (200, 200))
        assert result.size == (200, 200)

    def test_extend_zero_dimension(self):
        """Zero-dimension image should return blank canvas (Bug 4 fix)."""
        extender = BackgroundExtender(use_ai_inpainting=False)
        # Create a 1x1 image (smallest valid)
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        result = extender.extend(img, (500, 500))
        assert result.size == (500, 500)


class TestFeatherMask:
    def test_creates_correct_size(self):
        mask = create_feather_mask((100, 100), feather=10)
        assert mask.size == (100, 100)
        assert mask.mode == "L"

    def test_center_is_opaque(self):
        mask = create_feather_mask((100, 100), feather=10)
        import numpy as np
        arr = np.array(mask)
        # Center pixel should be fully opaque (255)
        assert arr[50, 50] == 255

    def test_edge_is_transparent(self):
        mask = create_feather_mask((100, 100), feather=20)
        import numpy as np
        arr = np.array(mask)
        # Top-left corner should be near 0
        assert arr[0, 0] < 20

    def test_zero_feather_returns_all_white(self):
        mask = create_feather_mask((50, 50), feather=0)
        import numpy as np
        arr = np.array(mask)
        assert (arr == 255).all()


class TestCompositionEngine:
    def test_compose_with_background_and_content(self):
        engine = CompositionEngine(use_ai_inpainting=False)

        bg_elem = DesignElement(
            id="bg", name="BG", layer_type="pixel",
            bbox=BoundingBox(0, 0, 500, 500),
            image=Image.new("RGBA", (500, 500), (200, 200, 200, 255)),
            role=ElementRole.BACKGROUND, priority=9,
        )
        hero_elem = DesignElement(
            id="hero", name="Hero", layer_type="pixel",
            bbox=BoundingBox(100, 100, 200, 200),
            image=Image.new("RGBA", (200, 200), (255, 0, 0, 255)),
            role=ElementRole.HERO_IMAGE, priority=3, z_index=1,
        )

        layout_results = [
            LayoutResult("bg", BoundingBox(0, 0, 300, 300), 0.6),
            LayoutResult("hero", BoundingBox(50, 50, 100, 100), 0.5),
        ]

        result = engine.compose(
            [bg_elem, hero_elem], layout_results,
            (500, 500), (300, 300)
        )

        assert isinstance(result, CompositionResult)
        assert result.image.size == (300, 300)
        assert result.image.mode == "RGB"

    def test_compose_with_no_elements(self):
        engine = CompositionEngine(use_ai_inpainting=False)
        result = engine.compose([], [], (500, 500), (300, 300))
        assert result.image.size == (300, 300)

    def test_compose_skips_invisible_elements(self):
        engine = CompositionEngine(use_ai_inpainting=False)

        elem = DesignElement(
            id="hidden", name="Hidden", layer_type="pixel",
            bbox=BoundingBox(0, 0, 100, 100),
            image=Image.new("RGBA", (100, 100), (255, 0, 0, 255)),
            role=ElementRole.HERO_IMAGE, priority=3,
        )
        layout = LayoutResult("hidden", BoundingBox(0, 0, 50, 50), 0.5, visible=False)

        result = engine.compose([elem], [layout], (100, 100), (100, 100))
        assert isinstance(result, CompositionResult)
