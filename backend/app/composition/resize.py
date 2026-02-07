"""High-quality image resize with gamma correction."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from ..config import Config

logger = logging.getLogger("autobanner.composition.resize")


def high_quality_resize(
    image: Image.Image, target_size: tuple[int, int]
) -> Image.Image:
    """High-quality resize with gamma correction.

    Performs resize in linear color space for more accurate results.

    Args:
        image: Source PIL image.
        target_size: Target (width, height).

    Returns:
        Resized image.
    """
    if target_size[0] <= 0 or target_size[1] <= 0:
        return image

    # Ensure image is in a supported mode
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    # Convert to numpy for gamma correction
    arr = np.array(image).astype(np.float32) / 255.0

    # Gamma decode (to linear)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4] if arr.shape[2] == 4 else None

    linear = np.power(np.clip(rgb, 0, 1), Config.GAMMA)

    if alpha is not None:
        linear = np.concatenate([linear, alpha], axis=2)

    # Resize
    pil_linear = Image.fromarray(
        (linear * 255).astype(np.uint8),
        mode="RGBA" if alpha is not None else "RGB",
    )
    resized = pil_linear.resize(target_size, Config.RESIZE_QUALITY)

    # Gamma encode (back to sRGB)
    arr_resized = np.array(resized).astype(np.float32) / 255.0
    rgb_resized = arr_resized[:, :, :3]

    encoded = np.power(np.clip(rgb_resized, 0, 1), 1.0 / Config.GAMMA)

    if alpha is not None:
        alpha_resized = arr_resized[:, :, 3:4]
        encoded = np.concatenate([encoded, alpha_resized], axis=2)
        result = Image.fromarray((encoded * 255).astype(np.uint8), mode="RGBA")
    else:
        result = Image.fromarray((encoded * 255).astype(np.uint8), mode="RGB")

    return result
