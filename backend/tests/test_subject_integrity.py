"""Mascot/subject invariants with an independent pixel oracle (phase C2).

A subject or logo must keep its proportions and its pixels. The check compares the
rendered region with the master asset scaled to the planned box, so an occlusion,
recolouring or crop shows up on the rendered output regardless of what the planner
believed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from backend.app.design.adapter import elements_from_document
from backend.app.design.variant import (
    VariantBrief,
    generate_variant,
    subject_integrity_checks,
)
from backend.app.models import BoundingBox, LayoutResult
from backend.app.quality import QualityConfig
from backend.tests.test_planner import _doc


def _patterned_doc(tmp_path: Path):
    """The planner fixture with a textured hero so correlation is meaningful."""
    doc, store = _doc(tmp_path)
    hero = Image.new("RGBA", (390, 430), (236, 88, 88, 255))
    d = ImageDraw.Draw(hero)
    for i in range(0, 430, 40):
        d.rectangle((0, i, 390, i + 20), fill=(40, 40, 120, 255))
    d.ellipse((60, 100, 330, 330), fill=(250, 220, 60, 255))
    ref = store.put(hero, "hero")
    doc.element("hero").asset = ref
    return doc, store


def test_generated_variants_pass_the_integrity_check(tmp_path: Path) -> None:
    doc, store = _patterned_doc(tmp_path)
    for size in ((1200, 628), (300, 250), (1080, 1350)):
        result = generate_variant(
            doc, store, VariantBrief(*size, name="t"),
            quality_config=QualityConfig(run_ocr=False), planner="constraints",
        )
        checks = {c.subject_id: c for c in result.report.checks
                  if c.check_id == "subject_integrity"}
        assert set(checks) >= {"hero", "logo"}, size
        for c in checks.values():
            assert c.status.value == "pass", (size, c.message)
            assert c.details["correlation"] >= 0.9


def test_stretched_and_occluded_subjects_are_caught(tmp_path: Path) -> None:
    doc, store = _patterned_doc(tmp_path)
    elements = elements_from_document(doc, store)
    hero = next(e for e in elements if e.id == "hero")
    canvas = Image.new("RGBA", (600, 400), (255, 255, 255, 255))
    # honest placement: uniform scale, pixels present
    box = BoundingBox(300, 40, 195, 215)
    placed = hero.image.convert("RGBA").resize((box.width, box.height), Image.LANCZOS)
    canvas.alpha_composite(placed, (box.x, box.y))
    layout = [LayoutResult("hero", box, 0.5)]
    ok = subject_integrity_checks(doc, elements, layout, canvas)
    assert ok[0].status.value == "pass" and ok[0].details["correlation"] > 0.95
    # stretched: width doubled -> critical failure before any pixel comparison
    stretched = [LayoutResult("hero", BoundingBox(300, 40, 390, 215), 0.5)]
    bad = subject_integrity_checks(doc, elements, stretched, canvas)
    assert bad[0].status.value == "fail" and bad[0].severity.value == "critical"
    assert "stretched" in bad[0].message
    # occluded: a panel painted over most of the hero -> needs review with the evidence
    covered = canvas.copy()
    ImageDraw.Draw(covered).rectangle((300, 40, 495, 200), fill=(255, 255, 255, 255))
    occ = subject_integrity_checks(doc, elements, layout, covered)
    assert occ[0].status.value == "needs_review"
    assert occ[0].details["correlation"] < 0.9 and "occluded" in occ[0].message
    # recoloured: same shapes, inverted colours -> the oracle disagrees
    from PIL import ImageOps

    inverted = canvas.copy()
    region = ImageOps.invert(canvas.crop((box.x, box.y, box.x2, box.y2)).convert("RGB"))
    inverted.paste(region, (box.x, box.y))
    rec = subject_integrity_checks(doc, elements, layout, inverted)
    assert rec[0].status.value == "needs_review" and rec[0].severity.value == "major"
    # mostly off-canvas: not checked, never silently passed
    off = [LayoutResult("hero", BoundingBox(598, 398, 195, 215), 0.5)]
    nc = subject_integrity_checks(doc, elements, off, canvas)
    assert nc[0].status.value == "not_checked"
