"""Tests for learning layout families from approved examples (H1)."""

from __future__ import annotations

from backend.app.design.adapter import elements_from_document
from backend.app.design.document import DesignDocument, Geometry
from backend.app.design.examples import infer_families, learned_families, match_elements
from backend.app.design.planner import choose_families, plan_layout
from backend.app.design.serialize import document_from_dict, document_to_dict
from backend.tests.test_planner import _doc


def _portrait_example(doc: DesignDocument, *, drift: int = 0) -> DesignDocument:
    """An 'approved' 1080x1920 variant: subject on top, text stack below, logo top-left."""
    ex = document_from_dict(document_to_dict(doc))
    ex.id = "example"
    ex.canvas_width, ex.canvas_height = 1080, 1920
    geo = {
        "bg": Geometry(0, 0, 1080, 1920),
        "logo": Geometry(76, 60, 300, 130),
        "hero": Geometry(140 + drift, 260, 800, 860),
        "headline": Geometry(90, 1180, 900, 200),
        "sub": Geometry(90, 1400, 900, 120),
        "cta": Geometry(90, 1560, 520, 130),
    }
    for e in ex.elements:
        e.geometry = geo[e.id]
    return ex


def test_match_elements_by_asset_text_and_role(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    ex = _portrait_example(doc)
    # rename the CTA text so it can only match by (unique) role; give the
    # subheadline a translation so text matching goes through translations
    for e in ex.elements:
        if e.id == "cta":
            e.text.runs[0].text = "BUY TODAY"
            e.id = "cta_renamed"
        if e.id == "sub":
            e.text.translations["vi"] = e.text.plain
            e.text.runs[0].text = "Giảm đến 50%"
            e.id = "sub_vi"
    matches = {m.master_id: m for m in match_elements(doc, ex)}
    assert matches["logo"].method == "asset" and matches["logo"].confidence > 0.9
    assert matches["hero"].method == "asset"
    assert matches["headline"].method == "text"
    assert matches["sub"].method == "text" and matches["sub"].example_id == "sub_vi"
    assert matches["cta"].method == "role" and matches["cta"].example_id == "cta_renamed"
    assert matches["cta"].confidence < matches["headline"].confidence
    assert "bg" not in matches  # backgrounds are never matched


def test_infer_families_reproduces_the_designers_portrait(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    inferred = infer_families(doc, [_portrait_example(doc)])
    assert set(inferred) == {"portrait"}
    fam = inferred["portrait"]
    assert fam.family.name == "learned_portrait"
    assert fam.examples == 1 and fam.confidence == 0.5
    assert any("single example" in n for n in fam.notes)
    assert fam.family.subject_first is True
    assert fam.family.text_align == "left"
    # slots contain the example's tight boxes, expand into free space and stay clear
    # of their neighbours (subject above the text stack here)
    text, subject, logo = fam.family.text, fam.family.subject, fam.family.logo
    assert text.x <= 90 / 1080 and text.y <= 1180 / 1920 <= 1690 / 1920 <= text.y + text.h
    assert text.x + text.w >= 990 / 1080 - 1e-3 and text.y + text.h > 1690 / 1920
    assert abs(subject.y - 260 / 1920) < 1e-6 and abs(subject.h - 860 / 1920) < 1e-6  # tight
    assert subject.y + subject.h <= text.y + 1e-6  # no overlap between the two slots
    assert logo.x <= 76 / 1080 and logo.y + logo.h >= 190 / 1920
    assert logo.y + logo.h <= subject.y + 1e-6
    # constraint proposals are reviewable, soft and provenance-marked
    kinds = {c.type: c for c in fam.constraints}
    assert kinds["anchor_edge"].params["edge"] == "top"
    assert kinds["anchor_edge"].hard is False
    assert kinds["anchor_edge"].provenance.origin == "recovered"
    assert kinds["scale_range"].params["min"] < 860 / 1920 < kinds["scale_range"].params["max"]
    d = fam.to_dict()
    assert d["family"] == "learned_portrait" and d["constraints"] == ["anchor_edge", "scale_range"]


def test_confidence_grows_with_agreeing_examples(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    one = infer_families(doc, [_portrait_example(doc)])["portrait"]
    examples = [_portrait_example(doc), _portrait_example(doc, drift=20)]
    two = infer_families(doc, examples)["portrait"]
    assert two.examples == 2
    assert two.confidence > one.confidence
    assert two.confidence <= 0.9


def test_choose_families_prefers_the_learned_family(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    elements = elements_from_document(doc, store)
    learned = learned_families(doc, [_portrait_example(doc)])
    targets = [(1080, 1920), (1080, 1350), (1200, 628)]
    chosen = choose_families(doc, elements, targets, learned=learned)
    assert chosen["portrait"].name == "learned_portrait"
    assert chosen["landscape"].name.startswith("landscape")  # no example: hand-written
    # the learned family plans the designer's composition on a held-out portrait size
    plan = plan_layout(doc, elements, (1080, 1350), families=[chosen["portrait"]])
    boxes = {r.element_id: r.new_bbox for r in plan.layout if r.visible}
    assert boxes["hero"].y2 <= boxes["headline"].y  # subject above the text stack
    assert boxes["logo"].y < boxes["hero"].y
    # without the bonus the hand-written portrait family still competes
    without = choose_families(doc, elements, targets, learned=learned, learned_bonus=-1000.0)
    assert without["portrait"].name != "learned_portrait"
