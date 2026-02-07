"""Shared pytest fixtures for AutoBanner tests."""

from __future__ import annotations

import pytest
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, DesignElement


@pytest.fixture
def sample_element() -> DesignElement:
    """Create a basic DesignElement for testing."""
    img = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
    return DesignElement(
        id="test_elem_001",
        name="Test Element",
        layer_type="pixel",
        bbox=BoundingBox(x=10, y=20, width=200, height=100),
        image=img,
        role=ElementRole.HERO_IMAGE,
        priority=3,
    )


@pytest.fixture
def sample_bg_element() -> DesignElement:
    """Create a background DesignElement."""
    img = Image.new("RGBA", (1080, 1080), (200, 200, 200, 255))
    return DesignElement(
        id="bg_elem_001",
        name="Background",
        layer_type="pixel",
        bbox=BoundingBox(x=0, y=0, width=1080, height=1080),
        image=img,
        role=ElementRole.BACKGROUND,
        priority=9,
    )


@pytest.fixture
def grayscale_image() -> Image.Image:
    """Create a grayscale PIL image."""
    return Image.new("L", (100, 100), 128)


@pytest.fixture
def rgba_image() -> Image.Image:
    """Create an RGBA PIL image."""
    return Image.new("RGBA", (200, 150), (100, 150, 200, 255))


@pytest.fixture
def small_canvas_elements() -> list[DesignElement]:
    """Create a set of elements for layout testing."""
    elements = []

    # Background
    bg_img = Image.new("RGBA", (1080, 1080), (240, 240, 240, 255))
    elements.append(
        DesignElement(
            id="bg_001",
            name="Background",
            layer_type="pixel",
            bbox=BoundingBox(x=0, y=0, width=1080, height=1080),
            image=bg_img,
            role=ElementRole.BACKGROUND,
            priority=9,
        )
    )

    # Headline
    headline_img = Image.new("RGBA", (500, 80), (255, 0, 0, 255))
    elements.append(
        DesignElement(
            id="headline_001",
            name="Main Headline",
            layer_type="type",
            bbox=BoundingBox(x=50, y=100, width=500, height=80),
            image=headline_img,
            text_content="Sale Now On",
            role=ElementRole.HEADLINE,
            priority=1,
        )
    )

    # CTA
    cta_img = Image.new("RGBA", (200, 60), (0, 255, 0, 255))
    elements.append(
        DesignElement(
            id="cta_001",
            name="Shop Now Button",
            layer_type="shape",
            bbox=BoundingBox(x=100, y=500, width=200, height=60),
            image=cta_img,
            role=ElementRole.CTA,
            priority=2,
        )
    )

    # Hero Image
    hero_img = Image.new("RGBA", (400, 400), (0, 0, 255, 255))
    elements.append(
        DesignElement(
            id="hero_001",
            name="Hero Product",
            layer_type="pixel",
            bbox=BoundingBox(x=340, y=340, width=400, height=400),
            image=hero_img,
            role=ElementRole.HERO_IMAGE,
            priority=3,
        )
    )

    return elements
