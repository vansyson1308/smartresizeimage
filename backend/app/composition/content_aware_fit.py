"""Content-aware fit strategy for flat image relayout."""

from __future__ import annotations

import logging
from enum import Enum

from PIL import Image

from ..config import Config
from .background import BackgroundExtender
from .resize import high_quality_resize

logger = logging.getLogger("autobanner.composition.content_aware_fit")


class FitMode(Enum):
    """How to fit source image into target canvas."""

    CONTAIN = "contain"  # Fit entire image, extend edges where needed
    COVER = "cover"  # Fill entire canvas, crop excess
    SMART = "smart"  # Auto-decide based on aspect ratio difference


class ContentAwareFitStrategy:
    """Intelligent scaling and fitting for flat (single-layer) images.

    Design principles:
    1. Never crop more than MAX_CROP_PERCENT of source content
    2. Prefer scaling + edge extension over aggressive cropping
    3. Use high-quality inpainting/edge-repeat for extension
    4. Keep text and important content visible at all sizes

    Usage:
        strategy = ContentAwareFitStrategy()
        result = strategy.fit(source_image, (1080, 1920))  # Instagram Story
    """

    def __init__(self, extender: BackgroundExtender | None = None) -> None:
        self.extender = extender or BackgroundExtender(use_ai_inpainting=False)

    def fit(
        self,
        source_image: Image.Image,
        target_size: tuple[int, int],
        mode: FitMode = FitMode.SMART,
    ) -> Image.Image:
        """Fit source image to target size intelligently.

        Args:
            source_image: Source PIL image (any mode, will be converted to RGBA).
            target_size: Target (width, height).
            mode: Fit mode — CONTAIN, COVER, or SMART (default).

        Returns:
            Result image at exactly target_size dimensions.
        """
        target_w, target_h = target_size
        src_w, src_h = source_image.size

        # Edge cases
        if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
            return Image.new("RGBA", target_size, (255, 255, 255, 255))

        image = source_image.convert("RGBA")

        # Calculate which strategy to use
        use_cover = self._should_use_cover(
            (src_w, src_h), target_size, mode
        )

        if use_cover:
            return self._apply_cover(image, target_size)
        return self._apply_contain(image, target_size)

    def _should_use_cover(
        self,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        mode: FitMode,
    ) -> bool:
        """Determine whether to use COVER (crop) or CONTAIN (extend)."""
        if mode == FitMode.COVER:
            return True
        if mode == FitMode.CONTAIN:
            return False

        # SMART mode: calculate how much cropping COVER would need
        src_w, src_h = source_size
        target_w, target_h = target_size

        scale_cover = max(target_w / src_w, target_h / src_h)
        cover_w = int(src_w * scale_cover)
        cover_h = int(src_h * scale_cover)

        # Calculate crop percentage on each axis
        crop_x = max(0, cover_w - target_w)
        crop_y = max(0, cover_h - target_h)
        crop_pct_x = crop_x / cover_w if cover_w > 0 else 0
        crop_pct_y = crop_y / cover_h if cover_h > 0 else 0
        max_crop_pct = max(crop_pct_x, crop_pct_y)

        max_allowed = Config.MAX_CROP_PERCENT

        logger.debug(
            "Smart fit: crop would be %.1f%% (max allowed %.1f%%)",
            max_crop_pct * 100,
            max_allowed * 100,
        )

        return max_crop_pct <= max_allowed

    def _apply_cover(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """Scale to fill target entirely, center-crop excess.

        Used when the aspect ratio change is small enough that we only
        need to crop a small percentage of content.
        """
        target_w, target_h = target_size
        src_w, src_h = image.size

        # Scale to cover
        scale = max(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        scaled = high_quality_resize(image, (new_w, new_h))

        # Center crop
        crop_x = (new_w - target_w) // 2
        crop_y = (new_h - target_h) // 2
        cropped = scaled.crop((
            crop_x, crop_y,
            crop_x + target_w, crop_y + target_h,
        ))

        return cropped

    def _apply_contain(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """Scale to fit within target, extend edges for remaining area.

        Used when the aspect ratio difference is large and cropping would
        lose too much content.
        """
        target_w, target_h = target_size
        src_w, src_h = image.size

        # Scale to contain (fit within target)
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        scaled = high_quality_resize(image, (new_w, new_h))

        # If scaled matches target, return directly
        if new_w == target_w and new_h == target_h:
            return scaled

        # Use background extender to fill remaining area
        return self.extender.extend(scaled, target_size)
