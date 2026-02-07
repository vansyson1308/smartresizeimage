"""Content-aware fit strategy for flat image relayout."""

from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageFilter

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
    5. Place content at visual center (top-biased for vertical targets)

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
        """Scale to fit within target, extend edges using inpainting.

        This method scales the image to fill the target width (or height),
        then extends the missing dimension using edge-aware inpainting.
        The original content is NEVER cropped — only new pixels are generated.

        For vertical targets (portrait): scales to fill width, extends top/bottom.
        For horizontal targets (wide): scales to fill height, extends left/right.
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

        # Place image on canvas and extend edges — WITHOUT re-scaling
        return self._extend_edges_inplace(scaled, target_size)

    def _extend_edges_inplace(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """Place image on canvas and extend edges via inpainting.

        Unlike BackgroundExtender.extend() which re-scales the image,
        this method keeps the image at its current size and only fills
        the remaining area with generated/extended pixels.

        Content placement uses a slight top bias (40% from top) for
        portrait targets, which is more visually natural for banners.
        """
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Calculate placement — center horizontally, slight top bias vertically
        x_off = (target_w - img_w) // 2

        gap_h = target_h - img_h
        if gap_h > 0 and target_h > target_w:
            # Portrait target: place content at ~40% from top (more natural)
            y_off = int(gap_h * 0.4)
        else:
            y_off = gap_h // 2

        # Try OpenCV inpainting first (best quality for edge extension)
        try:
            return self._inpaint_extend(image, target_size, x_off, y_off)
        except Exception as e:
            logger.warning("Inpaint extend failed: %s, using edge repeat", e)

        # Fallback to edge-pixel repetition
        return self._edge_repeat_extend(image, target_size, x_off, y_off)

    def _inpaint_extend(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
        x_off: int,
        y_off: int,
    ) -> Image.Image:
        """Extend image using OpenCV inpainting at fixed position."""

        target_w, target_h = target_size
        img_w, img_h = image.size

        # Convert source to OpenCV BGR
        src_rgb = np.array(image.convert("RGB"))
        src_cv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)

        # Create canvas and mask
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        mask = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # Calculate safe regions
        dst_x1 = max(0, x_off)
        dst_y1 = max(0, y_off)
        dst_x2 = min(target_w, x_off + img_w)
        dst_y2 = min(target_h, y_off + img_h)

        src_x1 = max(0, -x_off)
        src_y1 = max(0, -y_off)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        # Place image on canvas
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = src_cv[src_y1:src_y2, src_x1:src_x2]
        mask[dst_y1:dst_y2, dst_x1:dst_x2] = 0

        # Pre-fill extended area with edge colors for better inpainting
        # This gives the inpainting algorithm a better starting point
        self._prefill_edges(canvas, dst_x1, dst_y1, dst_x2, dst_y2, target_w, target_h)

        if mask.max() == 0:
            result_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb).convert("RGBA")

        # Dilate mask slightly for smoother blending at edges
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)

        # Apply inpainting
        inpaint_radius = Config.OPENCV_INPAINT_RADIUS
        inpainted = cv2.inpaint(canvas, mask_dilated, inpaint_radius, cv2.INPAINT_TELEA)

        # Blend: keep sharp original, smooth transition to inpainted
        result = self._blend_with_feather(
            inpainted, src_cv, dst_x1, dst_y1, dst_x2, dst_y2,
            src_x1, src_y1, target_w, target_h,
            feather_px=8,
        )

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb).convert("RGBA")

    @staticmethod
    def _prefill_edges(
        canvas: np.ndarray,
        dx1: int, dy1: int, dx2: int, dy2: int,
        tw: int, th: int,
    ) -> None:
        """Pre-fill extended area with edge pixel colors.

        This significantly improves inpainting quality by providing
        a color-coherent starting point instead of black pixels.
        """
        # Top extension
        if dy1 > 0:
            top_row = canvas[dy1:dy1 + 1, dx1:dx2]
            canvas[:dy1, dx1:dx2] = np.tile(top_row, (dy1, 1, 1))

        # Bottom extension
        if dy2 < th:
            bottom_row = canvas[dy2 - 1:dy2, dx1:dx2]
            canvas[dy2:, dx1:dx2] = np.tile(bottom_row, (th - dy2, 1, 1))

        # Left extension
        if dx1 > 0:
            left_col = canvas[:, dx1:dx1 + 1]
            canvas[:, :dx1] = np.tile(left_col, (1, dx1, 1))

        # Right extension
        if dx2 < tw:
            right_col = canvas[:, dx2 - 1:dx2]
            canvas[:, dx2:] = np.tile(right_col, (1, tw - dx2, 1))

    @staticmethod
    def _blend_with_feather(
        inpainted: np.ndarray,
        original_cv: np.ndarray,
        dx1: int, dy1: int, dx2: int, dy2: int,
        sx1: int, sy1: int,
        tw: int, th: int,
        feather_px: int = 8,
    ) -> np.ndarray:
        """Blend original image back onto inpainted result with feathered edges.

        This ensures the original content is pixel-perfect while the transition
        to inpainted areas is smooth.
        """
        result = inpainted.copy()

        # Paste original back (sharp, no quality loss)
        ow = dx2 - dx1
        oh = dy2 - dy1
        result[dy1:dy2, dx1:dx2] = original_cv[sy1:sy1 + oh, sx1:sx1 + ow]

        # Create feather blend at boundaries
        if feather_px <= 0:
            return result

        # Feather mask: 1.0 = original, 0.0 = inpainted
        mask = np.zeros((th, tw), dtype=np.float32)
        mask[dy1:dy2, dx1:dx2] = 1.0

        # Apply Gaussian blur to create smooth transition
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_px)

        # Blend
        mask_3d = mask[:, :, np.newaxis]
        original_layer = np.zeros_like(result, dtype=np.float32)
        original_layer[dy1:dy2, dx1:dx2] = original_cv[sy1:sy1 + oh, sx1:sx1 + ow].astype(np.float32)

        blended = (original_layer * mask_3d + inpainted.astype(np.float32) * (1.0 - mask_3d))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _edge_repeat_extend(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
        x_off: int,
        y_off: int,
    ) -> Image.Image:
        """Extend image using edge-pixel repetition with light blur."""
        target_w, target_h = target_size
        img_w, img_h = image.size

        arr = np.array(image.convert("RGBA"))

        # Create canvas
        canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255

        # Safe regions
        dx1 = max(0, x_off)
        dy1 = max(0, y_off)
        dx2 = min(target_w, x_off + img_w)
        dy2 = min(target_h, y_off + img_h)

        sx1 = max(0, -x_off)
        sy1 = max(0, -y_off)
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)

        # Place image
        canvas[dy1:dy2, dx1:dx2] = arr[sy1:sy2, sx1:sx2]

        # Extend edges
        if dy1 > 0:
            top_row = arr[sy1:sy1 + 1, sx1:sx2]
            canvas[:dy1, dx1:dx2] = np.tile(top_row, (dy1, 1, 1))
        if dy2 < target_h:
            bottom_row = arr[sy2 - 1:sy2, sx1:sx2]
            canvas[dy2:, dx1:dx2] = np.tile(bottom_row, (target_h - dy2, 1, 1))
        if dx1 > 0:
            left_col = canvas[dy1:dy2, dx1:dx1 + 1]
            canvas[dy1:dy2, :dx1] = np.tile(left_col, (1, dx1, 1))
        if dx2 < target_w:
            right_col = canvas[dy1:dy2, dx2 - 1:dx2]
            canvas[dy1:dy2, dx2:] = np.tile(right_col, (1, target_w - dx2, 1))

        # Fill corners
        if dy1 > 0 and dx1 > 0:
            canvas[:dy1, :dx1] = arr[sy1, sx1]
        if dy1 > 0 and dx2 < target_w:
            canvas[:dy1, dx2:] = arr[sy1, sx2 - 1]
        if dy2 < target_h and dx1 > 0:
            canvas[dy2:, :dx1] = arr[sy2 - 1, sx1]
        if dy2 < target_h and dx2 < target_w:
            canvas[dy2:, dx2:] = arr[sy2 - 1, sx2 - 1]

        # Blur only extended regions
        result = Image.fromarray(canvas)
        blurred = result.filter(ImageFilter.GaussianBlur(radius=3))
        final = blurred.copy()
        sharp = result.crop((dx1, dy1, dx2, dy2))
        final.paste(sharp, (dx1, dy1))

        return final
