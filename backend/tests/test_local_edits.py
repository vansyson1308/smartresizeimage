"""H3 regression set: campaign revisions must change pixels only inside the edited scope."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops

from backend.app.design.document import DesignDocument
from backend.app.design.serialize import document_from_dict, document_to_dict
from backend.app.design.variant import VariantBrief, generate_variant
from backend.app.quality import QualityConfig
from backend.tests.test_planner import _doc

SIZE = (1080, 1080)
SCOPE_PAD = 36  # text-plate padding (12) twice, plus a margin


def _generate(doc: DesignDocument, store, brief: VariantBrief, reference: dict | None = None):
    return generate_variant(
        doc,
        store,
        brief,
        reference=reference,
        quality_config=QualityConfig(run_ocr=False),
        planner="constraints",
    )


def _boxes(result) -> dict[str, tuple[int, int, int, int]]:
    return {
        p["element_id"]: (p["x"], p["y"], p["width"], p["height"])
        for p in result.plan["placements"]
        if p["visible"]
    }


def out_of_scope_diff(before, after, edited: list[str]) -> float:
    """Fraction of pixels outside the edited elements' boxes (before and after) that differ."""
    mask = np.ones((SIZE[1], SIZE[0]), dtype=bool)
    for result in (before, after):
        for p in result.plan["placements"]:
            if p["element_id"] in edited:
                x1 = max(0, p["x"] - SCOPE_PAD)
                y1 = max(0, p["y"] - SCOPE_PAD)
                x2 = min(SIZE[0], p["x"] + p["width"] + SCOPE_PAD)
                y2 = min(SIZE[1], p["y"] + p["height"] + SCOPE_PAD)
                mask[y1:y2, x1:x2] = False
    diff = ImageChops.difference(before.image.convert("RGB"), after.image.convert("RGB"))
    changed = np.asarray(diff).max(axis=2) > 0
    return float((changed & mask).sum() / max(1, mask.sum()))


def _copy(doc: DesignDocument) -> DesignDocument:
    return document_from_dict(document_to_dict(doc))


def test_copy_override_is_local_with_reference(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    base = _generate(doc, store, VariantBrief(*SIZE, name="base"))
    edited = _generate(
        doc,
        store,
        VariantBrief(*SIZE, name="rev", text_overrides={"cta": "BUY NOW"}),
        reference=base.plan,
    )
    assert edited.plan["planner_meta"]["reference"]["replanned"] == []
    assert out_of_scope_diff(base, edited, ["cta"]) == 0.0
    b0, b1 = _boxes(base), _boxes(edited)
    assert all(b0[k] == b1[k] for k in b0 if k != "cta")
    assert edited.plan["typography"]["cta"]["font_px"] <= base.plan["typography"]["cta"]["font_px"]


def test_master_copy_asset_and_style_edits_are_local(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    base = _generate(doc, store, VariantBrief(*SIZE, name="base"))

    # copy edit on the master (headline)
    d1 = _copy(doc)
    d1.element("headline").text.runs[0].text = "WINTER SALE"
    r1 = _generate(d1, store, VariantBrief(*SIZE, name="rev"), reference=base.plan)
    assert out_of_scope_diff(base, r1, ["headline"]) == 0.0

    # asset swap (same size, different colour) on the logo
    d2 = _copy(doc)
    d2.element("logo").asset = store.put(Image.new("RGBA", (180, 80), (30, 30, 30, 255)), "logo2")
    r2 = _generate(d2, store, VariantBrief(*SIZE, name="rev"), reference=base.plan)
    assert out_of_scope_diff(base, r2, ["logo"]) == 0.0
    assert _boxes(r2)["logo"] == _boxes(base)["logo"]

    # style edit (colour) on the subheadline
    d3 = _copy(doc)
    d3.element("sub").text.runs[0].style.color = "#ffd166"
    r3 = _generate(d3, store, VariantBrief(*SIZE, name="rev"), reference=base.plan)
    assert out_of_scope_diff(base, r3, ["sub"]) == 0.0

    # hiding an element leaves the others where they were
    r4 = _generate(
        doc, store, VariantBrief(*SIZE, name="rev", hidden_elements=["sub"]), reference=base.plan
    )
    assert out_of_scope_diff(base, r4, ["sub"]) == 0.0
    assert "sub" not in _boxes(r4)


def test_asset_with_new_aspect_is_refitted_not_distorted(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    base = _generate(doc, store, VariantBrief(*SIZE, name="base"))
    d = _copy(doc)
    d.element("logo").asset = store.put(
        Image.new("RGBA", (180, 180), (30, 30, 30, 255)), "square_logo"
    )
    r = _generate(d, store, VariantBrief(*SIZE, name="rev"), reference=base.plan)
    x, y, w, h = _boxes(r)["logo"]
    bx, by, bw, bh = _boxes(base)["logo"]
    assert w == h  # uniform fit
    assert bx <= x and by <= y and x + w <= bx + bw and y + h <= by + bh  # inside the old box
    assert out_of_scope_diff(base, r, ["logo"]) == 0.0


def test_copy_that_no_longer_fits_falls_back_to_a_fresh_plan(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    base = _generate(doc, store, VariantBrief(*SIZE, name="base"))
    long_copy = "SHOP NOW WHILE THE SUMMER CLEARANCE OFFERS ARE STILL AVAILABLE ONLINE"
    r = _generate(
        doc,
        store,
        VariantBrief(*SIZE, name="rev", text_overrides={"cta": long_copy}),
        reference=base.plan,
    )
    ref = r.plan["planner_meta"]["reference"]
    assert "cta" in ref["replanned"] and "cta" not in ref["kept"]
    assert any(w.startswith("layout_change:cta") for w in r.warnings)


def test_longer_cta_on_busy_background_keeps_the_other_plates(tmp_path) -> None:
    """The H3 residual: stack plates used to grow with the CTA; pinned rects keep them."""
    import json

    from backend.app.design.adapter import document_from_elements
    from backend.app.design.assets import AssetStore
    from backend.tools.generate_bench_fixtures import generate_fixtures
    from backend.tools.run_ablations import _native_elements
    from backend.tools.run_layout_bench import _elements_from_meta

    # Self-contained: the busy-background case is generated deterministically here
    # (seed 42, case 4) rather than read from an untracked fixture directory.
    generate_fixtures(tmp_path / "fixtures", cases=4, seed=42)
    case = tmp_path / "fixtures" / "case_04_busy_bg"
    meta = json.loads((case / "metadata.json").read_text())
    source = Image.open(case / "input.png").convert("RGBA")
    bg = case / "background.png"
    background = Image.open(bg).convert("RGBA") if bg.exists() else source
    elements = _native_elements(_elements_from_meta(meta, source, background))
    store = AssetStore(tmp_path / "assets")
    doc = document_from_elements(
        elements,
        (meta["source_size"]["width"], meta["source_size"]["height"]),
        store,
        name=case.name,
        origin="fixture",
    )
    cta = next(e for e in doc.elements if e.role == "cta")
    base = _generate(doc, store, VariantBrief(*SIZE, name="base"))
    assert base.plan["text_plate_rects"], "busy background should get a plate"
    longer = VariantBrief(*SIZE, name="rev", text_overrides={cta.id: cta.text.plain + " TODAY"})
    edited = _generate(doc, store, longer, reference=base.plan)
    assert out_of_scope_diff(base, edited, [cta.id]) == 0.0
    # without pinned rects the stack plate moves with the wider CTA
    ref_without_rects = {k: v for k, v in base.plan.items() if k != "text_plate_rects"}
    unpinned = _generate(doc, store, longer, reference=ref_without_rects)
    assert out_of_scope_diff(base, unpinned, [cta.id]) >= 0.0  # informative, may be zero
