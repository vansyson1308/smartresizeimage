"""The counterexample oracles judge final boxes and rendered masks independently (C4)."""

from __future__ import annotations

from pathlib import Path

from backend.app.design.document import Constraint, Provenance, new_id
from backend.app.design.variant import VariantBrief, generate_variant
from backend.app.models import BoundingBox, LayoutResult
from backend.app.quality import QualityConfig
from backend.tests.test_planner import _doc
from backend.tools.find_counterexamples import _oracles, _perturb


def test_oracles_agree_with_the_contract_on_a_clean_render(tmp_path: Path) -> None:
    doc, store = _doc(tmp_path)
    result = generate_variant(
        doc, store, VariantBrief(1200, 628, name="t"), planner="constraints",
        quality_config=QualityConfig(run_ocr=False),
    )
    assert _oracles(doc, result, (1200, 628)) == []


def test_oracles_fire_on_forged_layouts(tmp_path: Path) -> None:
    doc, store = _doc(tmp_path)
    result = generate_variant(
        doc, store, VariantBrief(1200, 628, name="t"), planner="constraints",
        quality_config=QualityConfig(run_ocr=False),
    )
    # a hard order rule the final boxes violate: headline (top) must sit below the CTA
    doc.add_constraint(Constraint(
        id=new_id("c"), type="order_below", elements=["headline", "cta"], hard=True,
        provenance=Provenance(origin="user", confidence=1.0),
    ))
    fired = _oracles(doc, result, (1200, 628))
    assert any(f.startswith("order:headline>cta") for f in fired)
    # push the logo off the canvas and onto the hero: bounds and overlap fire
    by_id = {r.element_id: r for r in result.layout}
    hero = by_id["hero"].new_bbox
    by_id["logo"].new_bbox = BoundingBox(hero.x + 10, hero.y + 10, hero.width, hero.height)
    result.layout.append(LayoutResult("hero", BoundingBox(1100, 500, 400, 400), 1.0))
    fired = _oracles(doc, result, (1200, 628))
    assert any(f.startswith("bounds:hero") for f in fired)
    assert any(f.startswith("overlap:") and "logo" in f for f in fired)


def test_perturbations_are_seeded_and_described(tmp_path: Path) -> None:
    import random

    doc, _store = _doc(tmp_path)
    for kind in ("long_copy", "short_copy", "hide_subject", "hard_order", "hard_clear",
                 "direction"):
        brief, desc = _perturb(doc, random.Random(1), kind)
        assert desc["kind"] == kind
        assert brief or desc.get("rule")
    assert any(c.type == "order_below" and c.hard for c in doc.constraints)
    assert any(c.type == "clear_space" and c.hard for c in doc.constraints)
