"""Regression tests for layout/composition review findings."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from backend.app.composition.engine import CompositionEngine
from backend.app.enums import ElementRole
from backend.app.generative.masks import build_layout_masks
from backend.app.layout.stack import StackLayoutEngine
from backend.app.models import BoundingBox, DesignElement, LayoutResult
from backend.app.parser.auto_layers import decompose_flat_image
from backend.app.presets import custom_preset, get_preset
from backend.app.service import RenderRequest, RenderService
from backend.tools.flat_banner_samples import make_sample


def _bg(size=(400, 200)) -> DesignElement:
    return DesignElement(
        "bg", "bg", "pixel", BoundingBox(0, 0, *size),
        image=Image.new("RGBA", size, (255, 255, 255, 255)), role=ElementRole.BACKGROUND,
    )


def test_aspect_fitted_element_is_drawn_centred_where_the_mask_says():
    logo = DesignElement(
        "logo", "logo", "pixel", BoundingBox(0, 0, 100, 100),
        image=Image.new("RGBA", (100, 100), (255, 0, 0, 255)), role=ElementRole.LOGO,
    )
    layout = [
        LayoutResult("bg", BoundingBox(0, 0, 400, 200), 1.0),
        LayoutResult("logo", BoundingBox(0, 0, 300, 100), 1.0),  # wider than the logo
    ]
    img = CompositionEngine(use_ai_inpainting=False).compose(
        [_bg(), logo], layout, (400, 200), (400, 200)
    ).image
    arr = np.asarray(img)
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 60)
    ys, xs = np.nonzero(red)
    assert xs.min() == 100 and xs.max() == 199  # centred in the 300px box
    mask = build_layout_masks([_bg(), logo], layout, (400, 200)).protected_mask
    assert mask[50, 150] and not mask[50, 20]


def test_manual_anchors_take_precedence_over_auto_layers(tmp_path):
    path = tmp_path / "flat.png"
    make_sample(0).image.save(path)
    report = RenderService().render(
        path,
        RenderRequest(
            targets=[custom_preset(300, 300)], mode="phase3",
            anchor_preset="flat_banner_3anchors", auto_layers=True,
        ),
    )
    assert report.source["source_type"] == "flat_image"
    assert report.assets[0].ok


def test_dropped_elements_are_reported_in_qa(tmp_path):
    sample = make_sample(1)
    service = RenderService()
    engine = service._new_engine()
    engine.elements = sample.design_elements()
    engine.source_size = sample.image.size
    path = tmp_path / "x.png"
    sample.image.save(path)
    report = service.render(
        path, RenderRequest(targets=[get_preset("iab-mobile-banner")]), engine=engine
    )
    asset = report.assets[0]
    assert {"id": "subheadline", "role": "subheadline"} in asset.qa["dropped"]
    assert any("Removed to stay legible" in w for w in asset.warnings)


def test_auto_layer_alpha_excludes_neighbouring_components():
    img = Image.new("RGB", (800, 420), (240, 240, 250))
    d = ImageDraw.Draw(img)
    d.ellipse((420, 60, 760, 400), fill=(220, 60, 60))  # hero
    d.rectangle((40, 150, 330, 230), fill=(20, 20, 20))  # plate / text-like block
    # small badge inside the hero's bounding rectangle but not touching the disc
    d.rectangle((424, 64, 470, 90), fill=(30, 140, 60))
    elements = decompose_flat_image(img)
    assert elements is not None
    hero = max(elements[1:], key=lambda e: e.bbox.area)
    alpha = np.asarray(hero.image.getchannel("A"))
    bx, by = 424 + 20 - hero.bbox.x, 64 + 12 - hero.bbox.y
    if 0 <= bx < alpha.shape[1] and 0 <= by < alpha.shape[0]:
        assert alpha[by, bx] == 0  # badge pixels are not part of the hero layer


def test_vertical_stack_is_vertically_centred():
    sample = make_sample(0)
    elements = sample.design_elements()
    results = StackLayoutEngine().calculate(elements, sample.image.size, (1080, 1080))
    boxes = [r.new_bbox for r in results if r.visible and r.element_id != "background"]
    top = min(b.y for b in boxes)
    bottom = 1080 - max(b.y2 for b in boxes)
    assert abs(top - bottom) <= 2


def test_phase3_layered_base_keeps_text_plates_without_ghosts(tmp_path):
    sample = make_sample(0)
    service = RenderService()
    engine = service._new_engine()
    engine.elements = sample.design_elements()
    engine.source_size = sample.image.size
    path = tmp_path / "x.png"
    sample.image.save(path)
    report = service.render(
        path, RenderRequest(targets=[get_preset("meta-story")], mode="phase3"), engine=engine
    )
    asset = report.assets[0]
    assert asset.ok and asset.qa["safe_zone"]["violations"] == []
