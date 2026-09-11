"""Hard rules are evaluated after every transformation (independent audit, finding 2).

A hard constraint that the final layout violates must block automatic acceptance,
whatever produced the layout: the planner, a kept reference plan (local edits) or a
repair step. An enabled hard rule that could not be evaluated is not evidence
either. Soft rules stay advisory.
"""

from __future__ import annotations

from backend.app.design.document import Constraint
from backend.app.design.variant import VariantBrief, constraint_checks, generate_variant
from backend.app.models import BoundingBox, LayoutResult
from backend.app.quality import CheckStatus, QualityConfig, Severity, Verdict
from backend.tests.test_planner import _doc

SIZE = (1080, 1080)


def _generate(doc, store, reference=None, **kw):
    return generate_variant(
        doc,
        store,
        VariantBrief(*SIZE, name="t"),
        reference=reference,
        quality_config=QualityConfig(run_ocr=False),
        planner="constraints",
        **kw,
    )


def _placed(result) -> dict[str, BoundingBox]:
    return {
        p["element_id"]: BoundingBox(p["x"], p["y"], p["width"], p["height"])
        for p in result.plan["placements"]
        if p["visible"]
    }


def _check(result, check_id: str, subject: str):
    return next(
        c for c in result.report.checks if c.check_id == check_id and c.subject_id == subject
    )


def test_new_hard_order_rule_is_enforced_on_a_kept_reference_layout(tmp_path) -> None:
    """The audit's case: base layout has the headline above the CTA; a new hard rule
    demands the opposite; regenerating with the reference plan must not be accepted
    while the rule is violated."""
    doc, store = _doc(tmp_path)
    base = _generate(doc, store)
    b = _placed(base)
    assert b["headline"].y < b["cta"].y
    doc.add_constraint(
        Constraint(id="c_order", type="order_below", elements=["headline", "cta"], hard=True)
    )
    again = _generate(doc, store, reference=base.plan)
    chk = _check(again, "constraint_order", "headline")
    assert chk.severity == Severity.CRITICAL
    a = _placed(again)
    # the rule is honoured on the final layout and nothing critical fails; the only
    # non-pass items are the OCR checks this test disables (never counted as evidence)
    assert chk.status == CheckStatus.PASS
    assert not any(
        c.status == CheckStatus.FAIL and c.severity == Severity.CRITICAL
        for c in again.report.checks
    )
    assert again.report.verdict != Verdict.FAILED
    # the kept-layout path replans the constrained elements instead of copying the violation
    ref = again.plan["planner_meta"]["reference"]
    assert "headline" in ref["replanned"]
    assert any(w.startswith("layout_change:headline") for w in again.warnings)
    assert a["headline"].y >= a["cta"].y


def test_soft_order_rule_is_advisory(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    base = _generate(doc, store)
    doc.add_constraint(
        Constraint(id="c_order", type="order_below", elements=["headline", "cta"], hard=False)
    )
    again = _generate(doc, store, reference=base.plan)
    chk = _check(again, "constraint_order", "headline")
    assert chk.severity == Severity.MINOR
    assert chk.status in (CheckStatus.PASS, CheckStatus.NEEDS_REVIEW)
    # a soft rule does not force a re-layout of the kept reference
    assert "headline" in again.plan["planner_meta"]["reference"]["kept"]


def _layout(boxes: dict[str, tuple[int, int, int, int]]) -> list[LayoutResult]:
    return [LayoutResult(eid, BoundingBox(*box), 1.0, visible=True) for eid, box in boxes.items()]


def _run(doc, layout, typography=None, target=(1000, 1000)):
    return constraint_checks(doc, layout, target, typography or {}, set())


def _one(checks, check_id):
    found = [c for c in checks if c.check_id == check_id]
    assert len(found) == 1, [c.check_id for c in checks]
    return found[0]


def test_every_hard_constraint_type_fails_when_violated(tmp_path) -> None:
    doc, _store = _doc(tmp_path)
    doc.constraints.clear()
    boxes = {
        "bg": (0, 0, 1000, 1000),
        "logo": (40, 40, 180, 80),
        "hero": (600, 100, 390, 430),
        "headline": (40, 200, 500, 120),
        "sub": (40, 340, 500, 80),
        "cta": (40, 900, 200, 60),
    }
    cases = {
        "order_below": (Constraint(id="c", type="order_below", elements=["headline", "cta"]),
                        "constraint_order"),
        "clear_space": (Constraint(id="c", type="clear_space", elements=["logo"],
                                   params={"ratio": 3.0}), "constraint_clear_space"),
        "keep_group": (Constraint(id="c", type="keep_group", elements=["sub", "cta"]),
                       "constraint_keep_group"),
        "anchor_edge": (Constraint(id="c", type="anchor_edge", elements=["logo"],
                                   params={"edge": "bottom"}), "constraint_anchor_edge"),
        "scale_range": (Constraint(id="c", type="scale_range", elements=["hero"],
                                   params={"min": 0.7, "max": 0.9}), "constraint_scale_range"),
        "min_text_size": (Constraint(id="c", type="min_text_size", elements=["cta"],
                                     params={"px": 40}), "constraint_min_text_size"),
        "keep_visible": (Constraint(id="c", type="keep_visible", elements=["sub"]),
                         "constraint_keep_visible"),
    }
    typography = {"cta": {"font_px": 20}}
    for ctype, (constraint, check_id) in cases.items():
        layout = _layout(boxes)
        if ctype == "keep_visible":
            next(r for r in layout if r.element_id == "sub").visible = False
        for hard in (True, False):
            constraint.hard = hard
            doc.constraints = [constraint]
            chk = _one(_run(doc, layout, typography), check_id)
            if hard:
                assert chk.status == CheckStatus.FAIL, ctype
                assert chk.severity == Severity.CRITICAL, ctype
            else:
                assert chk.status in (CheckStatus.FAIL, CheckStatus.NEEDS_REVIEW), ctype
                assert chk.severity != Severity.CRITICAL, ctype
            assert chk.details["constraint_id"] == "c"


def test_every_hard_constraint_type_passes_when_satisfied(tmp_path) -> None:
    doc, _store = _doc(tmp_path)
    boxes = {
        "bg": (0, 0, 1000, 1000),
        "logo": (40, 880, 180, 80),
        "hero": (560, 100, 390, 750),
        "headline": (40, 500, 400, 120),
        "sub": (40, 640, 400, 80),
        "cta": (40, 200, 200, 60),
    }
    doc.constraints = [
        Constraint(id="c1", type="order_below", elements=["headline", "cta"], hard=True),
        Constraint(id="c2", type="clear_space", elements=["logo"], params={"ratio": 0.3},
                   hard=True),
        Constraint(id="c3", type="keep_group", elements=["headline", "sub"], hard=True),
        Constraint(id="c4", type="anchor_edge", elements=["logo"], params={"edge": "bottom"},
                   hard=True),
        Constraint(id="c5", type="scale_range", elements=["hero"], params={"min": 0.7, "max": 0.8},
                   hard=True),
        Constraint(id="c6", type="min_text_size", elements=["cta"], params={"px": 20}, hard=True),
        Constraint(id="c7", type="keep_visible", elements=["sub"], hard=True),
    ]
    checks = _run(doc, _layout(boxes), {"cta": {"font_px": 24}})
    assert len(checks) == 7
    assert all(c.status == CheckStatus.PASS for c in checks), [
        (c.check_id, c.message) for c in checks
    ]


def test_hard_rule_that_cannot_be_evaluated_is_not_checked_critical(tmp_path) -> None:
    """Dropping a constrained element must not erase its rule."""
    doc, _store = _doc(tmp_path)
    boxes = {"bg": (0, 0, 1000, 1000), "cta": (40, 900, 200, 60), "logo": (40, 40, 180, 80)}
    doc.constraints = [
        Constraint(id="c1", type="order_below", elements=["headline", "cta"], hard=True),
        Constraint(id="c2", type="scale_range", elements=["hero"], params={"min": 0.1, "max": 0.9},
                   hard=True),
        Constraint(id="c3", type="min_text_size", elements=["cta"], params={"px": 20}, hard=True),
        Constraint(id="c4", type="anchor_edge", elements=["hero"], params={"edge": "top"},
                   hard=True),
    ]
    checks = _run(doc, _layout(boxes), {})  # no typography: cta size unknown
    assert len(checks) == 4
    for c in checks:
        assert c.status == CheckStatus.NOT_CHECKED, (c.check_id, c.message)
        assert c.severity == Severity.CRITICAL
    # the same rules as soft ones are simply skipped
    for c in doc.constraints:
        c.hard = False
    assert _run(doc, _layout(boxes), {}) == []


def test_repair_never_moves_an_element_against_a_hard_order_rule(tmp_path) -> None:
    """Found by the counterexample search: the planner honoured a hard 'subheadline below
    CTA' rule and the overlap repair then moved the CTA back below it. A repair move that
    would break a hard order rule is undone."""
    doc, store = _doc(tmp_path, long_copy=True)
    doc.add_constraint(
        Constraint(id="c_order2", type="order_below", elements=["sub", "cta"], hard=True)
    )
    for size in ((1080, 1080), (300, 250), (1080, 1920)):
        result = generate_variant(
            doc, store, VariantBrief(*size, name="t"), quality_config=QualityConfig(run_ocr=False),
            planner="constraints",
        )
        placed = _placed(result)
        if "sub" in placed and "cta" in placed:
            assert placed["sub"].y >= placed["cta"].y, (size, result.repair_steps)
        chk = _check(result, "constraint_order", "sub")
        assert chk.status != CheckStatus.FAIL, (size, chk.message, result.repair_steps)
