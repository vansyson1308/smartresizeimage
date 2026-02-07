"""Input validation for AutoBanner."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .constants import SUPPORTED_EXTENSIONS
from .exceptions import ValidationError


def validate_dimensions(width: int, height: int) -> None:
    """Validate target dimensions.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.

    Raises:
        ValidationError: If dimensions are invalid.
    """
    if not isinstance(width, int | float) or not isinstance(height, int | float):
        raise ValidationError(
            f"Dimensions must be numbers, got {type(width).__name__} and {type(height).__name__}"
        )

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValidationError(f"Dimensions must be positive, got {width}x{height}")
    if width > Config.MAX_IMAGE_SIZE or height > Config.MAX_IMAGE_SIZE:
        raise ValidationError(
            f"Dimensions exceed maximum {Config.MAX_IMAGE_SIZE}, got {width}x{height}"
        )
    if width < Config.MIN_ELEMENT_SIZE or height < Config.MIN_ELEMENT_SIZE:
        raise ValidationError(
            f"Dimensions below minimum {Config.MIN_ELEMENT_SIZE}, got {width}x{height}"
        )


def validate_file_path(path: str) -> None:
    """Validate input file exists and has supported extension.

    Args:
        path: Path to the input file.

    Raises:
        ValidationError: If file doesn't exist or format is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"File not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported format '{p.suffix}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
