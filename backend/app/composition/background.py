"""Background extension and generation for AutoBanner."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageFilter

from ..config import Config
from .resize import high_quality_resize

logger = logging.getLogger("autobanner.composition.background")

# Optional LaMa import
try:
    from simple_lama_inpainting import SimpleLama

    HAS_LAMA = True
except ImportError:
    HAS_LAMA = False
    logger.info("LaMa not available. Using OpenCV inpainting as primary method.")


class BackgroundExtender:
    """Handle background extension using AI inpainting, OpenCV inpaint, or edge repeat."""

    def __init__(self, use_ai_inpainting: bool = True) -> None:
        self.use_ai_inpainting = use_ai_inpainting and HAS_LAMA
        self._lama = None
        self._lama_loaded = False

    def _ensure_lama_loaded(self) -> None:
        """Lazy-load LaMa inpainting model."""
        if self._lama_loaded or not self.use_ai_inpainting:
            return
        self._lama_loaded = True
        try:
            logger.info("Loading LaMa inpainting model...")
            self._lama = SimpleLama()
            logger.info("LaMa model loaded")
        except Exception as e:
            logger.warning("Could not load LaMa: %s", e)
            self.use_ai_inpainting = False

    def extend(
        self, image: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """Extend image to fill target size.

        Priority:
        1. LaMa AI inpainting (if available)
        2. OpenCV inpainting (cv2.inpaint TELEA) — much better than blur
        3. Edge-pixel repetition (fallback)

        Args:
            image: Source image.
            target_size: Target (width, height).

        Returns:
            Extended image.
        """
        # Try LaMa first
        if self.use_ai_inpainting:
            self._ensure_lama_loaded()
            if self._lama:
                return self._extend_with_lama(image, target_size)

        # Try OpenCV inpainting (primary method)
        try:
            return self._extend_with_opencv_inpaint(image, target_size)
        except Exception as e:
            logger.warning("OpenCV inpaint failed: %s, using edge repeat", e)

        # Fallback to edge-pixel repetition
        return self._extend_with_edge_repeat(image, target_size)

    def _extend_with_lama(
        self, image: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """Extend image using LaMa AI inpainting."""
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Scale image to cover as much of target as possible first
        scale = max(target_w / img_w, target_h / img_h)
        scaled_w = max(1, int(img_w * scale))
        scaled_h = max(1, int(img_h * scale))
        scaled = high_quality_resize(image, (scaled_w, scaled_h))

        # Create extended canvas
        canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))

        # Center scaled image
        x_off = (target_w - scaled_w) // 2
        y_off = (target_h - scaled_h) // 2
        canvas.paste(scaled, (x_off, y_off))

        # Create mask for inpainting (white = areas to inpaint)
        mask = Image.new("L", target_size, 255)
        mask.paste(0, (x_off, y_off, x_off + scaled_w, y_off + scaled_h))

        # Check if there's anything to inpaint
        if np.array(mask).max() == 0:
            return canvas

        try:
            canvas_rgb = canvas.convert("RGB")
            result = self._lama(canvas_rgb, mask)
            return result.convert("RGBA")
        except Exception as e:
            logger.warning("LaMa inpainting failed: %s, falling back to OpenCV", e)
            try:
                return self._extend_with_opencv_inpaint(image, target_size)
            except Exception:
                return self._extend_with_edge_repeat(image, target_size)

    def _extend_with_opencv_inpaint(
        self, image: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """Extend image using OpenCV inpainting (TELEA algorithm).

        Much better than blur — generates natural-looking texture for extended areas.

        Args:
            image: Source image.
            target_size: Target (width, height).

        Returns:
            Extended image with inpainted edges.
        """
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Guard against zero dimensions
        if img_w <= 0 or img_h <= 0:
            return Image.new("RGBA", target_size, (255, 255, 255, 255))

        # Scale image to cover as much of the target as possible
        scale = max(target_w / img_w, target_h / img_h)
        scaled_w = max(1, int(img_w * scale))
        scaled_h = max(1, int(img_h * scale))
        scaled = high_quality_resize(image, (scaled_w, scaled_h))

        # Convert to OpenCV BGR format
        scaled_rgb = np.array(scaled.convert("RGB"))
        scaled_cv = cv2.cvtColor(scaled_rgb, cv2.COLOR_RGB2BGR)

        # Create canvas and mask
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        mask = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # Calculate placement (center the scaled image)
        x_off = (target_w - scaled_w) // 2
        y_off = (target_h - scaled_h) // 2

        # Calculate source/destination regions (handle negative offsets)
        dst_x1 = max(0, x_off)
        dst_y1 = max(0, y_off)
        dst_x2 = min(target_w, x_off + scaled_w)
        dst_y2 = min(target_h, y_off + scaled_h)

        src_x1 = max(0, -x_off)
        src_y1 = max(0, -y_off)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        # Place scaled image on canvas
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_cv[src_y1:src_y2, src_x1:src_x2]
        mask[dst_y1:dst_y2, dst_x1:dst_x2] = 0  # Black = keep

        # Check if there's anything to inpaint
        if mask.max() == 0:
            result_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb).convert("RGBA")

        # Apply cv2.inpaint with TELEA algorithm
        inpaint_radius = Config.OPENCV_INPAINT_RADIUS
        inpainted = cv2.inpaint(canvas, mask, inpaint_radius, cv2.INPAINT_TELEA)

        # Convert back to PIL RGBA
        result_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb).convert("RGBA")

    def _extend_with_edge_repeat(
        self, image: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """Extend image by repeating edge pixels with light blur.

        Better than heavy Gaussian blur — creates seamless edge extensions.

        Args:
            image: Source image.
            target_size: Target (width, height).

        Returns:
            Extended image.
        """
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Guard against zero dimensions
        if img_w <= 0 or img_h <= 0:
            return Image.new("RGBA", target_size, (255, 255, 255, 255))

        # Scale image to cover as much of the target as possible
        scale = max(target_w / img_w, target_h / img_h)
        scaled_w = max(1, int(img_w * scale))
        scaled_h = max(1, int(img_h * scale))
        scaled = high_quality_resize(image, (scaled_w, scaled_h))

        arr = np.array(scaled.convert("RGBA"))

        # Calculate placement
        x_off = (target_w - scaled_w) // 2
        y_off = (target_h - scaled_h) // 2

        # Create output canvas
        canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255  # Fully opaque

        # Calculate safe regions
        dst_x1 = max(0, x_off)
        dst_y1 = max(0, y_off)
        dst_x2 = min(target_w, x_off + scaled_w)
        dst_y2 = min(target_h, y_off + scaled_h)

        src_x1 = max(0, -x_off)
        src_y1 = max(0, -y_off)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        # Place scaled image
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]

        # Extend edges: top
        if dst_y1 > 0:
            top_row = arr[src_y1 : src_y1 + 1, src_x1:src_x2]
            canvas[:dst_y1, dst_x1:dst_x2] = np.tile(top_row, (dst_y1, 1, 1))

        # Extend edges: bottom
        if dst_y2 < target_h:
            bottom_row = arr[src_y2 - 1 : src_y2, src_x1:src_x2]
            remaining = target_h - dst_y2
            canvas[dst_y2:, dst_x1:dst_x2] = np.tile(bottom_row, (remaining, 1, 1))

        # Extend edges: left
        if dst_x1 > 0:
            left_col = canvas[dst_y1:dst_y2, dst_x1 : dst_x1 + 1]
            canvas[dst_y1:dst_y2, :dst_x1] = np.tile(left_col, (1, dst_x1, 1))

        # Extend edges: right
        if dst_x2 < target_w:
            right_col = canvas[dst_y1:dst_y2, dst_x2 - 1 : dst_x2]
            remaining = target_w - dst_x2
            canvas[dst_y1:dst_y2, dst_x2:] = np.tile(right_col, (1, remaining, 1))

        # Fill corners using nearest edge pixel
        if dst_y1 > 0 and dst_x1 > 0:
            canvas[:dst_y1, :dst_x1] = arr[src_y1, src_x1]
        if dst_y1 > 0 and dst_x2 < target_w:
            canvas[:dst_y1, dst_x2:] = arr[src_y1, src_x2 - 1]
        if dst_y2 < target_h and dst_x1 > 0:
            canvas[dst_y2:, :dst_x1] = arr[src_y2 - 1, src_x1]
        if dst_y2 < target_h and dst_x2 < target_w:
            canvas[dst_y2:, dst_x2:] = arr[src_y2 - 1, src_x2 - 1]

        # Apply light Gaussian blur only to extended regions for smoothing
        result = Image.fromarray(canvas)
        blurred = result.filter(ImageFilter.GaussianBlur(radius=3))

        # Composite: keep sharp center, use blurred edges
        final = blurred.copy()
        # Paste sharp original back
        sharp_region = result.crop((dst_x1, dst_y1, dst_x2, dst_y2))
        final.paste(sharp_region, (dst_x1, dst_y1))

        return final


def create_feather_mask(
    size: tuple[int, int], feather: int = 30
) -> Image.Image:
    """Create a feathered mask for smooth blending.

    Args:
        size: Mask (width, height).
        feather: Feather radius in pixels.

    Returns:
        Grayscale mask image.
    """
    w, h = size
    mask = Image.new("L", size, 255)

    if feather <= 0:
        return mask

    # Clamp feather to not exceed half the smallest dimension
    feather = min(feather, w // 2, h // 2)
    if feather <= 0:
        return mask

    arr = np.array(mask, dtype=np.float32)

    for i in range(feather):
        alpha = int(255 * (i / feather))
        arr[i, :] = np.minimum(arr[i, :], alpha)
        arr[h - 1 - i, :] = np.minimum(arr[h - 1 - i, :], alpha)
        arr[:, i] = np.minimum(arr[:, i], alpha)
        arr[:, w - 1 - i] = np.minimum(arr[:, w - 1 - i], alpha)

    return Image.fromarray(arr.astype(np.uint8))
