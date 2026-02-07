"""Background extension and generation for AutoBanner."""

from __future__ import annotations

import logging

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
    logger.info("LaMa not available. Using basic inpainting only.")


class BackgroundExtender:
    """Handle background extension using AI inpainting or blur-based methods."""

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

        Tries AI inpainting first, falls back to blur-based extension.

        Args:
            image: Source image.
            target_size: Target (width, height).

        Returns:
            Extended image.
        """
        if self.use_ai_inpainting:
            self._ensure_lama_loaded()
            if self._lama:
                return self._extend_with_inpainting(image, target_size)

        return self._extend_with_blur(image, target_size)

    def _extend_with_inpainting(
        self, image: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """Extend image using AI inpainting."""
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Create extended canvas
        canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))

        # Center original image
        x_off = (target_w - img_w) // 2
        y_off = (target_h - img_h) // 2
        canvas.paste(image, (x_off, y_off))

        # Create mask for inpainting
        mask = Image.new("L", target_size, 255)  # White = inpaint
        mask.paste(0, (x_off, y_off, x_off + img_w, y_off + img_h))  # Black = keep

        try:
            canvas_rgb = canvas.convert("RGB")
            result = self._lama(canvas_rgb, mask)
            return result.convert("RGBA")
        except Exception as e:
            logger.warning("Inpainting failed: %s, falling back to blur", e)
            return self._extend_with_blur(image, target_size)

    def _extend_with_blur(
        self, image: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """Extend image using blur-based method."""
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Guard against zero dimensions
        if img_w <= 0 or img_h <= 0:
            logger.warning(
                "Image has zero dimension (%dx%d), returning blank canvas", img_w, img_h
            )
            return Image.new("RGBA", target_size, (255, 255, 255, 255))

        # Scale image to cover at least one dimension
        scale = max(target_w / img_w, target_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        scaled = high_quality_resize(image, (scaled_w, scaled_h))

        # Create blurred version for extension
        blurred = scaled.filter(ImageFilter.GaussianBlur(radius=Config.BLUR_RADIUS))

        # Create canvas with blurred background
        canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))

        # Tile/center blurred version
        x_off = (target_w - scaled_w) // 2
        y_off = (target_h - scaled_h) // 2

        # Fill with edge colors first
        if scaled_w < target_w or scaled_h < target_h:
            arr = np.array(scaled)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                top_color = tuple(arr[0, :, :3].mean(axis=0).astype(int))
                bottom_color = tuple(arr[-1, :, :3].mean(axis=0).astype(int))
                left_color = tuple(arr[:, 0, :3].mean(axis=0).astype(int))
                right_color = tuple(arr[:, -1, :3].mean(axis=0).astype(int))
                avg_color = tuple(
                    (
                        (
                            np.array(top_color)
                            + np.array(bottom_color)
                            + np.array(left_color)
                            + np.array(right_color)
                        )
                        // 4
                    ).astype(int)
                )
                canvas = Image.new("RGBA", target_size, avg_color + (255,))

        # Paste blurred
        canvas.paste(blurred, (x_off, y_off))

        # Paste original sharp in center
        center_x = (target_w - img_w) // 2
        center_y = (target_h - img_h) // 2

        # Create gradient mask for blending
        mask = create_feather_mask(image.size, feather=50)
        canvas.paste(image, (center_x, center_y), mask)

        return canvas


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
