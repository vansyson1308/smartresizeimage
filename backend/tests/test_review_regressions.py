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


def test_dropped_decoration_is_recorded_but_not_warned():
    from backend.app.models import CompositionResult
    from backend.app.presets import get_preset as _gp
    from backend.app.service import evaluate_qa

    deco = DesignElement(
        "dot", "dot", "pixel", BoundingBox(0, 0, 10, 10),
        image=Image.new("RGBA", (10, 10), (255, 255, 255, 255)), role=ElementRole.DECORATION,
    )
    result = CompositionResult(
        image=Image.new("RGB", (300, 250)),
        layout_results=[LayoutResult("dot", BoundingBox(0, 0, 1, 1), 0.0, visible=False)],
    )
    qa, warnings = evaluate_qa(_gp("iab-medium-rectangle"), result, [deco], "phase21")
    assert qa["dropped"] == [{"id": "dot", "role": "decoration"}]
    assert not any("Removed" in w for w in warnings)


def test_hero_sticker_stays_attached_and_strip_logo_is_capped():
    from backend.tools.make_demo import tech_master

    master = tech_master()
    elements = master.elements()
    for size in [(1080, 1080), (1080, 1920), (728, 90), (300, 600)]:
        results = {r.element_id: r for r in StackLayoutEngine().calculate(
            elements, (1200, 628), size)}
        badge, hero = results["badge"], results["hero"]
        assert badge.visible and hero.visible
        b, h = badge.new_bbox, hero.new_bbox
        ix = max(0, min(b.x2, h.x2) - max(b.x, h.x))
        iy = max(0, min(b.y2, h.y2) - max(b.y, h.y))
        assert ix * iy > 0.2 * b.area, size  # still sits on the product
    strip = {r.element_id: r for r in StackLayoutEngine().calculate(
        elements, (1200, 628), (728, 90))}
    assert strip["logo"].new_bbox.width <= 0.16 * 728 + 2
    assert strip["logo"].new_bbox.height < strip["headline"].new_bbox.height


def test_auto_layers_keep_i_dots_in_the_text_layer():
    img = Image.new("RGB", (1200, 628), (250, 240, 225))
    d = ImageDraw.Draw(img)
    from backend.tools.make_demo import font

    d.text((80, 200), "Morning", font=font(110), fill=(70, 40, 30))
    d.rounded_rectangle((80, 420, 380, 500), radius=40, fill=(220, 100, 40))
    d.ellipse((760, 140, 1100, 480), fill=(200, 80, 60))
    elements = decompose_flat_image(img)
    assert elements is not None
    bg = np.asarray(elements[0].image.convert("RGB"), dtype=np.int32)
    ink = (np.abs(bg - np.array([70, 40, 30])).sum(axis=2) < 60)
    assert not ink[180:340, 60:620].any()  # no glyph remnants (i-dot) left in the plate


def test_orphan_fragment_goes_to_exactly_one_layer():
    from backend.app.parser.auto_layers import _assign_orphans, _Block

    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:40, 10:60] = 1   # block A
    labels[30:80, 40:90] = 2   # block B, overlapping A's box
    labels[34:36, 50:52] = 3   # tiny orphan inside both boxes
    a = _Block(10, 10, 60, 40, fill=1.0, area=1500, labels=frozenset({1}))
    b = _Block(40, 30, 90, 80, fill=1.0, area=2500, labels=frozenset({2}))
    owners = _assign_orphans(labels, [a, b])
    holders = [i for i, labs in owners.items() if 3 in labs]
    assert holders == [0]  # the smaller containing block only


def test_protruding_sticker_never_overlaps_strip_neighbours():
    bg = _bg((1200, 628))
    hero = DesignElement(
        "hero", "hero", "pixel", BoundingBox(700, 100, 300, 400),
        image=Image.new("RGBA", (300, 400), (200, 60, 60, 255)), role=ElementRole.HERO_IMAGE,
    )
    # sticker hanging well off the hero's right edge (still overlaps it by ~40%)
    badge = DesignElement(
        "badge", "badge", "pixel", BoundingBox(900, 60, 200, 200),
        image=Image.new("RGBA", (200, 200), (255, 64, 96, 255)), role=ElementRole.BADGE,
    )
    headline = DesignElement(
        "headline", "headline", "pixel", BoundingBox(60, 200, 500, 90),
        image=Image.new("RGBA", (500, 90), (0, 0, 0, 255)), role=ElementRole.HEADLINE,
    )
    cta = DesignElement(
        "cta", "cta", "pixel", BoundingBox(60, 380, 240, 70),
        image=Image.new("RGBA", (240, 70), (255, 200, 0, 255)), role=ElementRole.CTA,
    )
    elements = [bg, hero, badge, headline, cta]
    for size in [(728, 90), (970, 250), (320, 50), (1080, 1080)]:
        res = {r.element_id: r for r in StackLayoutEngine().calculate(
            elements, (1200, 628), size)}
        b = res["badge"].new_bbox
        for other in ("headline", "cta"):
            o = res[other].new_bbox
            ix = max(0, min(b.x2, o.x2) - max(b.x, o.x))
            iy = max(0, min(b.y2, o.y2) - max(b.y, o.y))
            assert ix * iy == 0, (size, other)
