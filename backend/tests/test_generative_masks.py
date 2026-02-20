"""Tests for protected/editable mask generation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.generative.masks import build_masks
from backend.app.models import BoundingBox, DesignElement


def test_build_masks_psd_like_protects_role_bboxes() -> None:
    image = Image.new("RGBA", (100, 80), (255, 255, 255, 255))
    elements = [
        DesignElement(
            id="logo",
            name="brand_logo",
            layer_type="pixel",
            bbox=BoundingBox(10, 12, 20, 10),
            role=ElementRole.LOGO,
        ),
        DesignElement(
            id="decor",
            name="sparkles",
            layer_type="pixel",
            bbox=BoundingBox(60, 20, 15, 15),
            role=ElementRole.DECORATION,
        ),
        DesignElement(
            id="headline",
            name="headline",
            layer_type="type",
            bbox=BoundingBox(5, 50, 40, 12),
            role=ElementRole.HEADLINE,
        ),
    ]

    masks = build_masks(elements, image)

    # Protected regions
    assert masks.protected_mask[12:22, 10:30].all()
    assert masks.protected_mask[50:62, 5:45].all()

    # Decoration should remain editable
    assert not masks.protected_mask[20:35, 60:75].any()


def test_build_masks_flat_ocr_text_mask(monkeypatch) -> None:
    image = Image.new("RGBA", (90, 60), (120, 120, 120, 255))
    flat = DesignElement(
        id="flat",
        name="flat_input",
        layer_type="pixel",
        bbox=BoundingBox(0, 0, 90, 60),
        role=ElementRole.UNKNOWN,
        effects={"_source_type": "flat_image"},
    )

    def fake_ocr(_img: Image.Image):
        return [(8, 10, 24, 12)]

    monkeypatch.setattr("backend.app.generative.masks._extract_text_boxes", fake_ocr)

    masks = build_masks([flat], image)

    assert masks.text_mask[10:22, 8:32].all()
    assert masks.protected_mask[10:22, 8:32].all()


def test_editable_mask_is_canvas_minus_protected() -> None:
    image = Image.new("RGBA", (40, 30), (0, 0, 0, 255))
    elements = [
        DesignElement(
            id="text",
            name="title",
            layer_type="type",
            bbox=BoundingBox(10, 5, 15, 10),
            role=ElementRole.HEADLINE,
        )
    ]

    masks = build_masks(elements, image)

    assert np.array_equal(masks.editable_mask, ~masks.protected_mask)
    assert np.count_nonzero(masks.protected_mask & masks.editable_mask) == 0
    assert np.count_nonzero(masks.protected_mask | masks.editable_mask) == 40 * 30


def test_build_masks_logs_area_ratios(caplog) -> None:
    image = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
    elements = [
        DesignElement(
            id="logo",
            name="logo",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, 10, 10),
            role=ElementRole.LOGO,
        )
    ]

    with caplog.at_level("INFO", logger="autobanner.generative.masks"):
        masks = build_masks(elements, image)

    assert masks.protected_ratio == 0.25
    assert "Mask stats:" in caplog.text
