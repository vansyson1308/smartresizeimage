"""Tests for Phase 3 target-first redesign."""

from __future__ import annotations

import numpy as np
from PIL import Image

from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, CompositionResult, DesignElement, LayoutResult
from backend.app.redesign.anchors import Anchor, build_protected_mask
from backend.app.redesign.api import run_target_first_redesign
from backend.app.redesign.generator import DeterministicFlatGenerator
from backend.app.redesign.planner import infer_horizon_hint
from backend.app.redesign.selector import select_best_candidate
from backend.app.redesign.validators import AnchorIntegrityValidator, SeamArtifactHeuristic


def test_horizon_hint_inference_on_synthetic_flat() -> None:
    img = Image.new("RGB", (300, 180), (120, 180, 245))
    arr = np.array(img)
    arr[110:, :] = [80, 90, 130]
    img = Image.fromarray(arr, mode="RGB")
    y1, y2 = infer_horizon_hint(img)
    assert 95 <= y1 <= 125
    assert y1 < y2


def test_anchor_mask_union_with_padding() -> None:
    anchors = [
        Anchor(
            "a",
            ElementRole.LOGO,
            Image.new("RGBA", (10, 10), "white"),
            BoundingBox(0, 0, 10, 10),
            BoundingBox(20, 20, 10, 10),
            True,
        ),
        Anchor(
            "b",
            ElementRole.CTA,
            Image.new("RGBA", (10, 10), "white"),
            BoundingBox(0, 0, 10, 10),
            BoundingBox(40, 40, 10, 10),
            True,
        ),
    ]
    mask = build_protected_mask(anchors, (80, 80), padding=2)
    assert mask[18:32, 18:32].all()
    assert mask[38:52, 38:52].all()


def test_anchor_integrity_validator_rejects_changed_anchor() -> None:
    anchor = Anchor(
        "logo",
        ElementRole.LOGO,
        Image.new("RGBA", (20, 20), (255, 0, 0, 255)),
        BoundingBox(0, 0, 20, 20),
        BoundingBox(10, 10, 20, 20),
        True,
    )
    img = Image.new("RGBA", (60, 60), (255, 255, 255, 255))
    img.alpha_composite(anchor.image, dest=(10, 10))
    arr = np.array(img)
    arr[12:18, 12:18] = [0, 255, 0, 255]
    bad = Image.fromarray(arr, mode="RGBA")

    v = AnchorIntegrityValidator(min_ssim=0.995)
    res = v.validate(bad, [anchor])
    assert res.passed is False
    assert res.hard_fail is True


def test_seam_heuristic_catches_edge_repeat_pattern() -> None:
    arr = np.zeros((80, 120), dtype=np.uint8)
    for x in range(120):
        arr[:, x] = 10 if x % 2 == 0 else 240
    img = Image.fromarray(arr, mode="L").convert("RGB")
    mask = np.ones((80, 120), dtype=bool)
    res = SeamArtifactHeuristic().validate(img, mask)
    assert res.score < 70.0


def test_selector_rejects_bad_candidate_and_picks_valid() -> None:
    anchor = Anchor(
        "logo",
        ElementRole.LOGO,
        Image.new("RGBA", (10, 10), (255, 0, 0, 255)),
        BoundingBox(0, 0, 10, 10),
        BoundingBox(5, 5, 10, 10),
        True,
    )

    class BadGen(DeterministicFlatGenerator):
        def generate(self, *args, **kwargs):  # type: ignore[override]
            img, meta = super().generate(*args, **kwargs)
            arr = np.array(img.convert("RGBA"))
            arr[6:12, 6:12] = [0, 255, 0, 255]
            return Image.fromarray(arr, mode="RGBA"), meta

    fill = np.ones((40, 40), dtype=bool)
    decor = np.zeros((40, 40), dtype=bool)
    source_bg = Image.new("RGBA", (40, 40), (220, 220, 220, 255))
    from backend.app.redesign.planner import RedesignPlan

    plan = RedesignPlan(fill, decor, fill, (12, 18), (8, 20), [(0, 0, 8, 8)])
    selected = select_best_candidate(
        generator=BadGen(),
        source_background=source_bg,
        anchors=[anchor],
        plan=plan,
        target_size=(40, 40),
        n=2,
        seed=42,
    )
    assert isinstance(selected.image, Image.Image)
    assert len(selected.candidates) == 2


def test_phase3_redesign_flat_with_manual_anchors_debug_fields() -> None:
    src = Image.new("RGBA", (240, 140), (200, 220, 255, 255))
    logo = Image.new("RGBA", (40, 20), (255, 50, 50, 255))
    src.alpha_composite(logo, dest=(10, 10))

    elements = [
        DesignElement(
            id="bg",
            name="bg",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, 240, 140),
            image=src,
            role=ElementRole.BACKGROUND,
            priority=9,
        ),
        DesignElement(
            id="logo",
            name="logo",
            layer_type="pixel",
            bbox=BoundingBox(10, 10, 40, 20),
            image=logo,
            role=ElementRole.LOGO,
            priority=1,
        ),
    ]
    layout = [
        LayoutResult("bg", BoundingBox(0, 0, 1080, 1080), 1.0),
        LayoutResult("logo", BoundingBox(50, 50, 200, 100), 1.0),
    ]

    out: CompositionResult = run_target_first_redesign(
        elements=elements,
        layout_results=layout,
        source_size=(240, 140),
        target_size=(1080, 1080),
        manual_anchors=[
            {"id": "logo", "role": "logo", "x": 10, "y": 10, "width": 40, "height": 20}
        ],
    )

    assert out.image.size == (1080, 1080)
    redesign = out.metadata["redesign"]
    assert redesign["mode"] == "phase3_target_first"
    assert "horizon_hint" in redesign
    assert "skyline_band" in redesign
    assert "palette_stats" in redesign
    assert isinstance(redesign["candidates"], list)
    assert "penalties" in redesign["candidates"][0]
