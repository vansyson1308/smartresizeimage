"""Tests for the quality contract v2 (rendered-output verification).

The first test preserves the historical false positive: a layout whose text is
completely covered by the hero was reported as PASS by the legacy
layout-metadata benchmark (score >= 32, overlap ratio < 0.10). Contract v2 must
fail it for the right reason: the text is not visible in the rendered pixels.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.app.composition.engine import CompositionEngine
from backend.app.enums import ElementRole
from backend.app.layout.bench_metrics import evaluate_bench_run
from backend.app.models import BoundingBox, DesignElement, LayoutResult
from backend.app.quality import (
    CONTRACT_VERSION,
    CheckResult,
    CheckStatus,
    QualityConfig,
    Severity,
    Verdict,
    evaluate_composition,
)
from backend.app.quality.contract import derive_verdict
from backend.app.quality.rendered import normalize_text, ocr_engine_status, text_similarity


def _text_image(size: tuple[int, int], text: str) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, int(size[1] * 0.6)))
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(20, 20, 20, 255), font=font)
    return img


def _hero_image(size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((2, 2, size[0] - 2, size[1] - 2), fill=(236, 88, 88, 255))
    return img


CANVAS = (600, 300)
PORTRAIT = (1080, 1920)


def _elements() -> list[DesignElement]:
    return [
        DesignElement(
            id="bg",
            name="Background",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, 600, 300),
            image=Image.new("RGBA", (600, 300), (86, 126, 164, 255)),
            role=ElementRole.BACKGROUND,
            priority=9,
            z_index=0,
        ),
        DesignElement(
            id="headline",
            name="Headline",
            layer_type="type",
            bbox=BoundingBox(20, 20, 300, 60),
            image=_text_image((300, 60), "SUMMER SALE"),
            text_content="SUMMER SALE",
            role=ElementRole.HEADLINE,
            priority=1,
            z_index=1,
        ),
        DesignElement(
            id="cta",
            name="CTA",
            layer_type="type",
            bbox=BoundingBox(20, 200, 200, 50),
            image=_text_image((200, 50), "SHOP NOW"),
            text_content="SHOP NOW",
            role=ElementRole.CTA,
            priority=2,
            z_index=1,
        ),
        DesignElement(
            id="logo",
            name="Logo",
            layer_type="pixel",
            bbox=BoundingBox(480, 20, 100, 50),
            image=Image.new("RGBA", (100, 50), (245, 245, 245, 255)),
            role=ElementRole.LOGO,
            priority=1,
            z_index=2,
        ),
        DesignElement(
            id="hero",
            name="Hero",
            layer_type="pixel",
            bbox=BoundingBox(340, 80, 220, 200),
            image=_hero_image((220, 200)),
            role=ElementRole.HERO_IMAGE,
            priority=3,
            z_index=3,
        ),
    ]


def _good_layout() -> list[LayoutResult]:
    return [
        LayoutResult("bg", BoundingBox(0, 0, 600, 300), 1.0),
        LayoutResult("headline", BoundingBox(20, 20, 300, 60), 1.0),
        LayoutResult("cta", BoundingBox(20, 200, 200, 50), 1.0),
        LayoutResult("logo", BoundingBox(480, 20, 100, 50), 1.0),
        LayoutResult("hero", BoundingBox(340, 80, 220, 200), 1.0),
    ]


def _occluded_layout() -> list[LayoutResult]:
    """Text stacked under the hero on a portrait canvas: the historical failure.

    Mirrors the preserved evidence (case_01 at 1080x1920, commit 9d913a0):
    headline/cta boxes sit inside the hero's box near the bottom of the canvas.
    """
    return [
        LayoutResult("bg", BoundingBox(0, 0, 1080, 1920), 1.0),
        LayoutResult("headline", BoundingBox(270, 1500, 696, 49), 1.0),
        LayoutResult("cta", BoundingBox(540, 1560, 169, 31), 1.0),
        LayoutResult("logo", BoundingBox(120, 120, 324, 144), 1.0),
        LayoutResult("hero", BoundingBox(236, 945, 780, 860), 1.0),
    ]


def _render(layout: list[LayoutResult], size: tuple[int, int] = CANVAS) -> Image.Image:
    engine = CompositionEngine(use_ai_inpainting=False)
    return engine.compose(_elements(), layout, (600, 300), size).image


def test_contract_v2_fails_historical_false_positive_for_the_right_reason() -> None:
    elements = _elements()
    layout = _occluded_layout()
    image = _render(layout, PORTRAIT)

    # Legacy evaluator only looks at layout metadata. Its overlap metric is
    # relative to the whole canvas, so fully hidden text on a large canvas stays
    # under the 0.10 threshold: this is the blind spot behind the historical
    # "PASS with 13 violations" result.
    legacy = evaluate_bench_run(
        elements=elements,
        layout_results=layout,
        target_size=PORTRAIT,
        text_plate_meta={"applied": True},
        busy_expected=False,
    )
    assert legacy.metrics.overlap_area_ratio < 0.10
    assert "overlap_area_ratio" not in legacy.fail_reasons

    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=PORTRAIT,
        config=QualityConfig(run_ocr=False),
    )
    assert report.contract_version == CONTRACT_VERSION
    assert report.verdict == Verdict.FAILED

    by_subject = {
        c.subject_id: c for c in report.checks if c.check_id == "element_visible"
    }
    assert by_subject["headline"].status == CheckStatus.FAIL
    assert by_subject["headline"].severity == Severity.CRITICAL
    assert "hidden behind another element" in by_subject["headline"].message
    assert by_subject["headline"].details["visible_fraction"] < 0.5
    assert by_subject["cta"].status == CheckStatus.FAIL
    # The hero is on top, so it is fully visible.
    assert by_subject["hero"].status == CheckStatus.PASS
    assert "element_visible:headline" in report.summary["critical_failures"]


def test_contract_v2_accepts_clean_render() -> None:
    elements = _elements()
    layout = _good_layout()
    image = _render(layout)
    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=(600, 300),
        ocr=lambda _img: "SUMMER SALE SHOP NOW",
    )
    assert report.verdict == Verdict.ACCEPTED, [c.message for c in report.issues()]
    vis = [c for c in report.checks if c.check_id == "element_visible"]
    assert all(c.status == CheckStatus.PASS for c in vis)
    assert all(c.details["visible_fraction"] >= 0.97 for c in vis)


def test_contract_v2_detects_canvas_clipping() -> None:
    elements = _elements()
    layout = _good_layout()
    layout[1] = LayoutResult("headline", BoundingBox(450, 20, 300, 60), 1.0)  # half off-canvas
    image = _render(layout)
    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=(600, 300),
        config=QualityConfig(run_ocr=False),
    )
    assert report.verdict == Verdict.FAILED
    inside = {c.subject_id: c for c in report.checks if c.check_id == "inside_canvas"}
    assert inside["headline"].status == CheckStatus.FAIL
    vis = {c.subject_id: c for c in report.checks if c.check_id == "element_visible"}
    assert vis["headline"].details["clipped_fraction"] > 0.3


def test_contract_v2_detects_dropped_required_element() -> None:
    elements = _elements()
    layout = [r for r in _good_layout() if r.element_id != "logo"]
    image = _render(layout)
    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=(600, 300),
        config=QualityConfig(run_ocr=False),
    )
    assert report.verdict == Verdict.FAILED
    required = {
        c.subject_id: c for c in report.checks if c.check_id == "required_element_present"
    }
    assert required["logo"].status == CheckStatus.FAIL
    assert "dropped" in required["logo"].message


def test_export_dimension_mismatch_is_critical() -> None:
    elements = _elements()
    layout = _good_layout()
    image = _render(layout)
    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=(640, 300),
        config=QualityConfig(run_ocr=False),
    )
    assert report.verdict == Verdict.FAILED
    dims = next(c for c in report.checks if c.check_id == "export_dimensions")
    assert dims.status == CheckStatus.FAIL


def test_skipped_check_is_not_evidence() -> None:
    """A critical check that did not run must leave the verdict at NEEDS_REVIEW."""
    checks = [
        CheckResult("a", CheckStatus.PASS, Severity.CRITICAL, "ok"),
        CheckResult("b", CheckStatus.NOT_CHECKED, Severity.CRITICAL, "ocr unavailable"),
    ]
    assert derive_verdict(checks) == Verdict.NEEDS_REVIEW
    assert derive_verdict(checks[:1]) == Verdict.ACCEPTED
    minor_fail = [CheckResult("c", CheckStatus.FAIL, Severity.MINOR, "polish")]
    assert derive_verdict(minor_fail) == Verdict.NEEDS_REVIEW
    major_fail = [CheckResult("d", CheckStatus.FAIL, Severity.MAJOR, "broken")]
    assert derive_verdict(major_fail) == Verdict.FAILED
    # A minor note (soft rule) is listed but does not send an otherwise clean output to review.
    minor_note = [CheckResult("e", CheckStatus.NEEDS_REVIEW, Severity.MINOR, "crowded")]
    assert derive_verdict(minor_note) == Verdict.ACCEPTED
    major_note = [CheckResult("f", CheckStatus.NEEDS_REVIEW, Severity.MAJOR, "partially hidden")]
    assert derive_verdict(major_note) == Verdict.NEEDS_REVIEW


def test_ocr_disabled_reports_not_checked_for_text() -> None:
    elements = _elements()
    layout = _good_layout()
    image = _render(layout)
    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=(600, 300),
        config=QualityConfig(run_ocr=False),
    )
    legible = [c for c in report.checks if c.check_id == "text_legible"]
    assert legible and all(c.status == CheckStatus.NOT_CHECKED for c in legible)
    # Text elements are critical: an unchecked legibility test blocks ACCEPTED
    # and is surfaced to reviewers as an issue.
    assert report.verdict == Verdict.NEEDS_REVIEW
    assert any(c.check_id == "text_legible" for c in report.issues())


def test_custom_ocr_function_drives_legibility_status() -> None:
    elements = _elements()
    layout = _good_layout()
    image = _render(layout)

    report_good = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=(600, 300),
        ocr=lambda _img: "SUMMER SALE SHOP NOW",
    )
    legible = {c.subject_id: c for c in report_good.checks if c.check_id == "text_legible"}
    assert legible["headline"].status == CheckStatus.PASS
    assert report_good.verdict == Verdict.ACCEPTED

    report_bad = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=image,
        target_size=(600, 300),
        ocr=lambda _img: "",
    )
    legible = {c.subject_id: c for c in report_bad.checks if c.check_id == "text_legible"}
    # Pixels are visible, OCR disagrees -> OCR is not an oracle -> NEEDS_REVIEW.
    assert legible["headline"].status == CheckStatus.NEEDS_REVIEW
    assert report_bad.verdict == Verdict.NEEDS_REVIEW


def test_text_similarity_normalization() -> None:
    assert normalize_text("  Up to 50% off, selected items! ") == "UP TO 50% OFF SELECTED ITEMS"
    ratio, recall = text_similarity("SUMMER SUPER SALE", "summer super sale")
    assert ratio == 1.0 and recall == 1.0
    ratio, recall = text_similarity("SUMMER SUPER SALE", "")
    assert ratio == 0.0 and recall == 0.0


@pytest.mark.skipif(not ocr_engine_status()["available"], reason="tesseract not installed")
def test_tesseract_reads_clean_headline() -> None:
    elements = _elements()
    layout = _good_layout()
    image = _render(layout)
    report = evaluate_composition(
        elements=elements, layout_results=layout, image=image, target_size=(600, 300)
    )
    legible = {c.subject_id: c for c in report.checks if c.check_id == "text_legible"}
    assert legible["headline"].status in (CheckStatus.PASS, CheckStatus.NEEDS_REVIEW)
    assert legible["headline"].details["engine"] == "tesseract"
    assert report.config["environment"]["ocr_engine"] == "tesseract"


def test_flat_image_is_flagged_for_review() -> None:
    img = Image.new("RGBA", (300, 200), (200, 100, 50, 255))
    elem = DesignElement(
        id="source_image_0",
        name="Source Image",
        layer_type="pixel",
        bbox=BoundingBox(0, 0, 300, 200),
        image=img,
        role=ElementRole.BACKGROUND,
        priority=9,
    )
    elem.effects["_source_type"] = "flat_image"
    layout = [LayoutResult("source_image_0", BoundingBox(0, 0, 600, 300), 2.0)]
    report = evaluate_composition(
        elements=[elem],
        layout_results=layout,
        image=Image.new("RGB", (600, 300)),
        target_size=(600, 300),
    )
    assert report.verdict == Verdict.NEEDS_REVIEW
    assert any(c.check_id == "design_understood" for c in report.checks)


def test_report_serializes_to_plain_dict() -> None:
    elements = _elements()
    layout = _good_layout()
    report = evaluate_composition(
        elements=elements,
        layout_results=layout,
        image=_render(layout),
        target_size=(600, 300),
        config=QualityConfig(run_ocr=False),
        extra_context={"mode": "unit"},
    )
    payload = report.to_dict()
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["verdict"] in {"accepted", "needs_review", "failed"}
    assert payload["config"]["generation"]["mode"] == "unit"
    assert isinstance(payload["checks"], list)
    import json

    json.dumps(payload)  # must be JSON-serializable


def test_phase3_generator_seed_is_stable_across_processes() -> None:
    """Built-in hash() is salted per process; seeds must not depend on it."""
    repo_root = str(Path(__file__).resolve().parents[2])
    script = r"""
import hashlib, sys
import numpy as np
from PIL import Image
sys.path.insert(0, REPO_ROOT)
from backend.app.redesign.generator import DeterministicFlatGenerator
from backend.app.redesign.planner import RedesignPlan
fill = np.ones((48, 64), dtype=bool); fill[10:20, 10:30] = False
decor = np.zeros((48, 64), dtype=bool); decor[2:10, 2:60] = True
plan = RedesignPlan(fill, decor, fill, (20, 28), (16, 32), [(0, 0, 8, 8)])
src = Image.new("RGBA", (64, 48), (120, 160, 200, 255))
gen = DeterministicFlatGenerator()
img, _ = gen.generate(src, (64, 48), fill, decor, 42, 1, plan, "strong_decor")
print(hashlib.sha256(img.tobytes()).hexdigest())
""".replace("REPO_ROOT", repr(repo_root))
    outputs = []
    for seed in ("1", "2"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1]
