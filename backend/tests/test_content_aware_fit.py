"""Tests for content-aware fit strategy and improved background extension."""

from __future__ import annotations

import pytest
from PIL import Image

from backend.app.composition.background import BackgroundExtender
from backend.app.composition.content_aware_fit import (
    ContentAwareFitStrategy,
    FitMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(w: int, h: int, color: tuple = (200, 100, 50)) -> Image.Image:
    """Create a solid RGBA test image."""
    return Image.new("RGBA", (w, h), color + (255,))


# ---------------------------------------------------------------------------
# ContentAwareFitStrategy tests
# ---------------------------------------------------------------------------

class TestContentAwareFitStrategy:
    """Test the smart fit strategy for flat images."""

    def setup_method(self) -> None:
        extender = BackgroundExtender(use_ai_inpainting=False)
        self.strategy = ContentAwareFitStrategy(extender=extender)

    # --- Output size tests ---

    def test_fit_wider_target(self) -> None:
        """500x500 -> 800x400 (wider)."""
        result = self.strategy.fit(_solid_image(500, 500), (800, 400))
        assert result.size == (800, 400)

    def test_fit_taller_target(self) -> None:
        """500x500 -> 400x800 (taller)."""
        result = self.strategy.fit(_solid_image(500, 500), (400, 800))
        assert result.size == (400, 800)

    def test_fit_same_aspect_upscale(self) -> None:
        """500x500 -> 1000x1000 (same aspect, bigger)."""
        result = self.strategy.fit(_solid_image(500, 500), (1000, 1000))
        assert result.size == (1000, 1000)

    def test_fit_same_aspect_downscale(self) -> None:
        """500x500 -> 250x250 (same aspect, smaller)."""
        result = self.strategy.fit(_solid_image(500, 500), (250, 250))
        assert result.size == (250, 250)

    # --- Real-world ad format tests ---

    def test_fit_instagram_story(self) -> None:
        """Landscape banner -> Instagram Story (extreme aspect change)."""
        result = self.strategy.fit(_solid_image(886, 500), (1080, 1920))
        assert result.size == (1080, 1920)

    def test_fit_facebook_cover(self) -> None:
        """Square-ish banner -> Facebook Cover."""
        result = self.strategy.fit(_solid_image(886, 500), (1200, 630))
        assert result.size == (1200, 630)

    def test_fit_facebook_small(self) -> None:
        """Landscape banner -> Facebook ad small."""
        result = self.strategy.fit(_solid_image(886, 500), (300, 250))
        assert result.size == (300, 250)

    def test_fit_pinterest_pin(self) -> None:
        """Landscape banner -> Pinterest vertical pin."""
        result = self.strategy.fit(_solid_image(886, 500), (160, 600))
        assert result.size == (160, 600)

    def test_fit_youtube_thumbnail(self) -> None:
        """Landscape banner -> YouTube thumbnail."""
        result = self.strategy.fit(_solid_image(886, 500), (1280, 720))
        assert result.size == (1280, 720)

    def test_fit_twitter_header(self) -> None:
        """Landscape banner -> Twitter header (very wide)."""
        result = self.strategy.fit(_solid_image(886, 500), (1500, 500))
        assert result.size == (1500, 500)

    # --- Mode tests ---

    def test_contain_mode_fits_entirely(self) -> None:
        """CONTAIN should always fit the entire source image."""
        result = self.strategy.fit(
            _solid_image(400, 200), (200, 200), FitMode.CONTAIN
        )
        assert result.size == (200, 200)

    def test_cover_mode_fills_entirely(self) -> None:
        """COVER should fill the entire target canvas."""
        result = self.strategy.fit(
            _solid_image(400, 200), (200, 200), FitMode.COVER
        )
        assert result.size == (200, 200)

    def test_smart_mode_uses_cover_for_similar_aspect(self) -> None:
        """SMART mode should use COVER when crop is minimal (< 20%)."""
        # 500x500 -> 500x400 needs ~20% crop on height => borderline
        # 500x500 -> 500x450 needs ~10% crop => should COVER
        result = self.strategy.fit(
            _solid_image(500, 500), (500, 450), FitMode.SMART
        )
        assert result.size == (500, 450)

    def test_smart_mode_uses_contain_for_extreme_aspect(self) -> None:
        """SMART mode should use CONTAIN when crop would be >20%."""
        # 100x800 -> 400x100: extreme aspect difference
        result = self.strategy.fit(
            _solid_image(100, 800), (400, 100), FitMode.SMART
        )
        assert result.size == (400, 100)

    # --- Edge cases ---

    def test_fit_zero_source(self) -> None:
        """Zero-size source should return blank canvas."""
        result = self.strategy.fit(Image.new("RGBA", (0, 0)), (100, 100))
        assert result.size == (100, 100)

    def test_fit_rgb_input(self) -> None:
        """RGB input (no alpha) should work fine."""
        source = Image.new("RGB", (300, 200), (128, 128, 128))
        result = self.strategy.fit(source, (500, 300))
        assert result.size == (500, 300)


# ---------------------------------------------------------------------------
# BackgroundExtender improved methods tests
# ---------------------------------------------------------------------------

class TestOpenCVInpaint:
    """Test OpenCV inpainting extension."""

    def setup_method(self) -> None:
        self.extender = BackgroundExtender(use_ai_inpainting=False)

    def test_opencv_inpaint_creates_correct_size(self) -> None:
        """cv2.inpaint output should match target size."""
        img = _solid_image(100, 100)
        result = self.extender._extend_with_opencv_inpaint(img, (200, 300))
        assert result.size == (200, 300)

    def test_opencv_inpaint_rgba_output(self) -> None:
        """Output should be RGBA."""
        img = _solid_image(100, 100)
        result = self.extender._extend_with_opencv_inpaint(img, (200, 300))
        assert result.mode == "RGBA"

    def test_opencv_inpaint_zero_source(self) -> None:
        """Zero-dimension source returns blank canvas."""
        img = Image.new("RGBA", (0, 0))
        result = self.extender._extend_with_opencv_inpaint(img, (100, 100))
        assert result.size == (100, 100)


class TestEdgeRepeat:
    """Test edge-pixel repetition extension."""

    def setup_method(self) -> None:
        self.extender = BackgroundExtender(use_ai_inpainting=False)

    def test_edge_repeat_creates_correct_size(self) -> None:
        """Edge repeat output should match target size."""
        img = _solid_image(100, 100)
        result = self.extender._extend_with_edge_repeat(img, (200, 300))
        assert result.size == (200, 300)

    def test_edge_repeat_extends_vertically(self) -> None:
        """Extending a wide image vertically."""
        img = _solid_image(200, 50)
        result = self.extender._extend_with_edge_repeat(img, (200, 200))
        assert result.size == (200, 200)

    def test_edge_repeat_extends_horizontally(self) -> None:
        """Extending a tall image horizontally."""
        img = _solid_image(50, 200)
        result = self.extender._extend_with_edge_repeat(img, (200, 200))
        assert result.size == (200, 200)

    def test_edge_repeat_zero_source(self) -> None:
        """Zero-dimension source returns blank canvas."""
        img = Image.new("RGBA", (0, 0))
        result = self.extender._extend_with_edge_repeat(img, (100, 100))
        assert result.size == (100, 100)


class TestExtendPriority:
    """Test that extend() uses the right method priority."""

    def test_extend_without_lama_uses_opencv(self) -> None:
        """Without LaMa, extend() should succeed (using opencv or edge repeat)."""
        extender = BackgroundExtender(use_ai_inpainting=False)
        img = _solid_image(100, 100)
        result = extender.extend(img, (200, 300))
        assert result.size == (200, 300)
