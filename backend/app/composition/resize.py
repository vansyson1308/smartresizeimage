"""High-quality image resize with gamma correction."""

from __future__ import annotations

import logging
from functools import lru_cache

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

    # Gamma decode -> resize in linear light -> gamma encode. Both transfer
    # functions map uint8 -> uint8, so they are applied as 256-entry lookup
    # tables (bit-identical to the float pipeline, ~10x faster).
    decode, encode, alpha = _gamma_luts(float(Config.GAMMA))
    has_alpha = image.mode == "RGBA"
    lut_decode = decode * 3 + (alpha if has_alpha else [])
    lut_encode = encode * 3 + (alpha if has_alpha else [])

    linear = image.point(lut_decode)
    resized = linear.resize(target_size, Config.RESIZE_QUALITY)
    return resized.point(lut_encode)


@lru_cache(maxsize=4)
def _gamma_luts(gamma: float) -> tuple[list[int], list[int], list[int]]:
    # Same float32 arithmetic (x / 255 -> f -> * 255 -> truncate) as the
    # original per-pixel implementation, so results are bit-identical.
    values = np.arange(256, dtype=np.float32) / np.float32(255.0)
    decode = (np.power(np.clip(values, 0, 1), gamma) * 255).astype(np.uint8)
    encode = (np.power(np.clip(values, 0, 1), 1.0 / gamma) * 255).astype(np.uint8)
    alpha = (values * 255).astype(np.uint8)
    return decode.tolist(), encode.tolist(), alpha.tolist()
