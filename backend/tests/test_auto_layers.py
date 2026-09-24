"""Auto-layering of flat banners into pseudo-layers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.parser.auto_layers import decompose_flat_image
from backend.app.presets import get_preset
from backend.app.relayout import ReLayoutEngine
from backend.app.service import RenderRequest, RenderService
from backend.tools.flat_banner_samples import make_sample


def _iou(box: dict, b) -> float:
    ix = max(0, min(box["x"] + box["width"], b.x2) - max(box["x"], b.x))
    iy = max(0, min(box["y"] + box["height"], b.y2) - max(box["y"], b.y))
    inter = ix * iy
    union = box["width"] * box["height"] + b.area - inter
    return inter / union if union else 0.0


def test_detects_elements_with_correct_roles():
    correct = total = 0
    for i in range(8):
        sample = make_sample(i)
        elements = decompose_flat_image(sample.image)
        assert elements is not None
        assert elements[0].role == ElementRole.BACKGROUND
        for role, box in sample.boxes.items():
            total += 1
            best = max(elements[1:], key=lambda e: _iou(box, e.bbox))
            # wordmark logos on an invisible plate only match the text part
            if best.role.value == role and _iou(box, best.bbox) > (0.2 if role == "logo" else 0.5):
                correct += 1
    assert correct / total >= 0.95, f"{correct}/{total}"


def test_headline_is_the_tallest_text_and_cta_detected():
    elements = decompose_flat_image(make_sample(0).image)
    roles = [e.role for e in elements]
    assert roles.count(ElementRole.HEADLINE) == 1
    assert roles.count(ElementRole.CTA) == 1
    assert roles.count(ElementRole.LOGO) <= 1


def test_background_plate_has_foreground_removed():
    sample = make_sample(0)
    elements = decompose_flat_image(sample.image)
    bg = np.asarray(elements[0].image.convert("RGB"), dtype=np.float32)
    hb = sample.boxes["headline"]
    patch = bg[hb["y"]:hb["y"] + hb["height"], hb["x"]:hb["x"] + hb["width"]]
    # the dark headline ink is gone from the plate: the patch is near-uniform
    assert patch.std(axis=(0, 1)).max() < 20


@pytest.mark.parametrize(
    "image",
    [
        Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8), "RGB"
        ),
        Image.new("RGB", (400, 200), (120, 130, 140)),
    ],
    ids=["noise_photo", "blank"],
)
def test_declines_images_without_a_design_structure(image):
    assert decompose_flat_image(image) is None


def test_auto_layers_turn_flat_leaderboard_into_a_real_relayout(tmp_path):
    sample = make_sample(0)
    path = tmp_path / "banner.png"
    sample.image.save(path)
    flat = ReLayoutEngine(use_ai=False)
    flat.load_file(str(path))
    auto = ReLayoutEngine(use_ai=False)
    auto.load_file(str(path), auto_layers=True)
    assert len(auto.elements) > 1

    result = auto.relayout((728, 90))
    boxes = {r.element_id: r.new_bbox for r in result.layout_results if r.visible}
    headline = next(e for e in auto.elements if e.role == ElementRole.HEADLINE)
    # In the flat fit the whole 1200x628 design shrinks to ~172x90; the headline
    # would be ~10px tall. Auto layers give it most of the strip height.
    assert boxes[headline.id].height >= 30


def test_service_reports_auto_layer_source(tmp_path):
    path = tmp_path / "banner.png"
    make_sample(1).image.save(path)
    service = RenderService()
    info = service.analyze(path, auto_layers=True)
    assert info["source_type"] == "auto_layers"
    report = service.render(
        path, RenderRequest(targets=[get_preset("iab-leaderboard")], auto_layers=True)
    )
    asset = report.assets[0]
    assert asset.ok and asset.qa["evaluated"]
