"""Tests for rules derived from correction history (H5)."""

from __future__ import annotations

from backend.app.design.corrections import RejectionSnapshot, derive_corrections
from backend.app.design.document import Constraint, Provenance
from backend.app.design.variant import _min_text_px_constraint
from backend.tests.test_planner import _doc


def _placements(**boxes):
    return [
        {"element_id": k, "x": x, "y": y, "width": w, "height": h, "visible": True}
        for k, (x, y, w, h) in boxes.items()
    ]


def _rejection(reason, placements, typography=None, ts="2026-09-09T10:00:00"):
    return RejectionSnapshot(
        variant_id="v1",
        name="S",
        width=1000,
        height=1000,
        reason=reason,
        ts=ts,
        placements=placements,
        typography=typography or {},
    )


def _approved(placements, typography=None, ts="2026-09-09T11:00:00"):
    rec = {"id": "v1", "width": 1000, "height": 1000, "updated_at": ts}
    return rec, {"placements": placements, "typography": typography or {}}


def test_logo_too_small_becomes_a_scale_range(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    rej = _rejection(
        "logo too small", _placements(logo=(50, 50, 90, 40), hero=(500, 200, 400, 440))
    )
    approved = _approved(_placements(logo=(50, 50, 180, 80), hero=(500, 200, 400, 440)))
    proposals, unresolved = derive_corrections(doc, [rej], [approved])
    assert unresolved == []
    assert len(proposals) == 1
    p = proposals[0]
    assert p.kind == "scale_range" and p.element_id == "logo"
    assert p.params == {"min": 0.08, "max": 1.0}  # keep at least what was approved
    c = p.to_constraint()
    assert c.hard is False and c.provenance.origin == "recovered" and c.provenance.confidence == 0.5
    assert "logo too small" in c.provenance.notes


def test_hard_to_read_cta_becomes_a_scaled_minimum_text_size(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    rej = _rejection(
        "CTA hard to read",
        _placements(cta=(70, 700, 200, 40)),
        {"cta": {"font_px": 18}},
    )
    approved = _approved(_placements(cta=(70, 700, 260, 52)), {"cta": {"font_px": 28}})
    proposals, unresolved = derive_corrections(doc, [rej], [approved])
    assert unresolved == []
    [p] = proposals
    assert p.kind == "min_text_size" and p.element_id == "cta"
    assert p.params == {"px": 28, "measured_on": [1000, 1000]}
    # the minimum scales with the target canvas
    c = p.to_constraint()
    assert _min_text_px_constraint([c], "cta", (500, 500)) == 14
    assert _min_text_px_constraint([c], "cta", (1000, 1000)) == 28
    assert _min_text_px_constraint([c], "cta") == 28


def test_overlap_reason_becomes_clear_space(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    rej = _rejection(
        "headline hidden behind the hero",
        _placements(headline=(400, 300, 400, 100), hero=(500, 200, 400, 440)),
    )
    approved = _approved(_placements(headline=(50, 300, 400, 100), hero=(500, 200, 400, 440)))
    proposals, _ = derive_corrections(doc, [rej], [approved])
    assert [(p.kind, p.element_id) for p in proposals] == [("clear_space", "headline")]


def test_unparseable_or_unpaired_rejections_are_reported_not_invented(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    rej = _rejection("meh", _placements(logo=(50, 50, 90, 40)))
    proposals, unresolved = derive_corrections(
        doc, [rej], [_approved(_placements(logo=(50, 50, 90, 40)))]
    )
    assert proposals == [] and len(unresolved) == 1 and "no rule derived" in unresolved[0].note
    # no approved variant after the rejection
    proposals, unresolved = derive_corrections(doc, [rej], [])
    assert proposals == [] and unresolved[0].approved_variant is None
    # an approved variant of another size does not pair
    rec, plan = _approved(_placements(logo=(50, 50, 180, 80)))
    rec["width"] = 1200
    proposals, unresolved = derive_corrections(
        doc, [_rejection("logo too small", _placements(logo=(50, 50, 90, 40)))], [(rec, plan)]
    )
    assert proposals == [] and unresolved[0].approved_variant is None


def test_existing_stronger_rule_is_not_proposed_again(tmp_path) -> None:
    doc, _ = _doc(tmp_path)
    doc.add_constraint(
        Constraint(
            id="c_have",
            type="scale_range",
            elements=["logo"],
            params={"min": 0.1, "max": 1.0},
            hard=False,
            provenance=Provenance(origin="user"),
        )
    )
    rej = _rejection("logo too small", _placements(logo=(50, 50, 90, 40)))
    proposals, unresolved = derive_corrections(
        doc, [rej], [_approved(_placements(logo=(50, 50, 180, 80)))]
    )
    assert proposals == [] and unresolved == []
