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
    HERO_PROMINENCE_TARGET = 0.18
    SMART_CROP_MAX_ZOOM = 1.6
    SMART_CROP_SAFE_PADDING = 12

    # Quality
    RESIZE_QUALITY = Image.Resampling.LANCZOS
    GAMMA = 2.2
    USE_LINEAR_COMPOSITING = True

    # AI
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    CONFIDENCE_THRESHOLD = 0.7

    # Zone assignment
    MAX_ELEMENTS_PER_ZONE = 2
    LAYOUT_PROFILE_SCORING_ENABLED = False
    LAYOUT_SOLVER_MAX_ITERS = 24
    LAYOUT_DEBUG_ENABLED = False
    LAYOUT_DEBUG_DIR = "backend/tests/fixtures/outputs"

    # Text-safe background plate (Phase 2.1-E)
    TEXT_SAFE_PLATE_ENABLED = True
    TEXT_SAFE_PLATE_STYLE = "blur"  # blur | gradient | solid
    TEXT_SAFE_BUSY_THRESHOLD = 0.20
    TEXT_SAFE_PLATE_PADDING = 12
    TEXT_SAFE_PLATE_FEATHER = 10
    TEXT_SAFE_PLATE_OPACITY = 110
    TEXT_SAFE_PLATE_RADIUS = 10

    # Generative background outpainting (Phase 2)
    GENERATIVE_BG_ENABLED = False
    GENERATIVE_BG_POLICY = "BG_ONLY"
    GENERATIVE_BG_SEED = 42
    GENERATIVE_BG_MODEL_ID = "none"

    # Optional decor synthesis (Phase 2 PR-D)
    GENERATIVE_DECOR_POLICY = "OFF"  # OFF | BG_PLUS_DECOR
    GENERATIVE_DECOR_SEED = 123
