"""Global configuration for AutoBanner."""

from __future__ import annotations

from PIL import Image


class Config:
    """Global configuration."""

    # Processing limits
    MAX_IMAGE_SIZE = 4096
    MIN_ELEMENT_SIZE = 10

    # Layout
    MARGIN_PERCENT = 0.05  # 5% margin from edges
    MIN_SPACING = 20  # Minimum spacing between elements

    # Background
    INPAINT_RADIUS = 5
    BLUR_RADIUS = 50
    OPENCV_INPAINT_RADIUS = 5  # Radius for cv2.inpaint TELEA

    # Content-aware fit (flat PNG/JPG relayout)
    MAX_CROP_PERCENT = 0.20  # Never crop more than 20% of source content

    # Quality
    RESIZE_QUALITY = Image.Resampling.LANCZOS
    GAMMA = 2.2

    # AI
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    CONFIDENCE_THRESHOLD = 0.7

    # Zone assignment
    MAX_ELEMENTS_PER_ZONE = 2
