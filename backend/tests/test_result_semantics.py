"""Semantics of pipeline results: skipped checks, fallbacks and mocks are explicit."""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, CompositionResult, DesignElement, LayoutResult
from backend.app.redesign.anchors import Anchor
from backend.app.redesign.generator import GenerativeFillAdapter, make_generator
from backend.app.redesign.selector import _compose_anchors
from backend.app.redesign.validators import OCRTextValidator
from backend.app.relayout import ReLayoutEngine


def test_composition_result_is_not_evaluated_by_default() -> None:
    result = CompositionResult(image=Image.new("RGB", (10, 10)), layout_results=[])
    assert result.gates_passed is None
    assert result.quality is None
    assert result.verdict == "not_evaluated"


def test_ocr_placeholder_validator_is_not_a_pass() -> None:
    res = OCRTextValidator().validate(Image.new("RGBA", (10, 10)), [])
    assert res.status == "not_checked"
    assert res.passed is False
    assert res.reason == "ocr_not_checked"


def test_generative_adapter_declares_itself_a_mock(monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_ENABLE_GENERATIVE_REDESIGN", "true")
    gen, name = make_generator()
    assert name == "generative_adapter"
    assert isinstance(gen, GenerativeFillAdapter)
    from backend.app.redesign.planner import RedesignPlan

    fill = np.ones((32, 32), dtype=bool)
    plan = RedesignPlan(fill, fill, fill, (10, 14), (8, 16), [])
    _, meta = gen.generate(
        Image.new("RGBA", (32, 32), (100, 100, 100, 255)),
        (32, 32),
        fill,
        fill,
        1,
        0,
        plan,
        "background_only",
    )
    assert meta["is_mock"] is True
    assert meta["provider"] == "none"


def test_anchors_composite_in_z_order() -> None:
    red = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    blue = Image.new("RGBA", (10, 10), (0, 0, 255, 255))
    anchors = [
        Anchor("top", ElementRole.HEADLINE, red, BoundingBox(0, 0, 10, 10),
               BoundingBox(0, 0, 10, 10), True, z_index=5),
        Anchor("bottom", ElementRole.HERO_IMAGE, blue, BoundingBox(0, 0, 10, 10),
               BoundingBox(0, 0, 10, 10), False, z_index=1),
    ]
    out = _compose_anchors(Image.new("RGBA", (10, 10), (0, 0, 0, 255)), anchors)
    # Higher z_index is drawn last regardless of list order.
    assert out.getpixel((5, 5)) == (255, 0, 0, 255)


def test_production_relayout_attaches_quality_report() -> None:
    img = Image.new("RGB", (200, 150), (0, 255, 0))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        path = f.name
    try:
        engine = ReLayoutEngine(use_ai=False)
        engine.load_file(path)
        result = engine.relayout((300, 300))
        assert result.quality is not None
        assert result.metadata["quality"]["contract_version"] == result.quality.contract_version
        # A flat image is never silently accepted: it was not decomposed.
        assert result.verdict == "needs_review"
        assert result.gates_passed is False
    finally:
        os.unlink(path)


def test_load_elements_uses_production_path_and_verdict() -> None:
    bg = DesignElement(
        id="bg", name="bg", layer_type="pixel", bbox=BoundingBox(0, 0, 400, 200),
        image=Image.new("RGBA", (400, 200), (90, 120, 150, 255)),
        role=ElementRole.BACKGROUND, priority=9,
    )
    logo = DesignElement(
        id="logo", name="logo", layer_type="pixel", bbox=BoundingBox(300, 20, 80, 40),
        image=Image.new("RGBA", (80, 40), (250, 250, 250, 255)),
        role=ElementRole.LOGO, priority=1, z_index=2,
    )
    engine = ReLayoutEngine(use_ai=False)
    engine.load_elements([bg, logo], (400, 200))
    result = engine.relayout((400, 400))
    assert result.quality is not None
    assert result.verdict in {"accepted", "needs_review", "failed"}
    assert isinstance(result.layout_results[0], LayoutResult)
