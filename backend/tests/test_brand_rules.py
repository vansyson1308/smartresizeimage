"""Tests for brand-level rule union and proposal (H5 follow-up)."""

from __future__ import annotations

from backend.app.design.brand import collect_brand_rules, propose_brand_rules
from backend.app.design.document import Constraint, Provenance
from backend.tests.test_planner import _doc


def _rule(doc, ctype, element, params, origin="user", enabled=True, notes=""):
    doc.add_constraint(
        Constraint(
            id=f"c_{ctype}_{element}_{origin}",
            type=ctype,
            elements=[element],
            params=params,
            hard=False,
            enabled=enabled,
            provenance=Provenance(origin=origin, confidence=1.0, notes=notes),
        )
    )


def test_union_keeps_the_stricter_rule_and_ignores_unconfirmed(tmp_path) -> None:
    a, _ = _doc(tmp_path / "a")
    b, _ = _doc(tmp_path / "b")
    _rule(a, "scale_range", "logo", {"min": 0.08, "max": 1.0})
    _rule(
        b,
        "scale_range",
        "logo",
        {"min": 0.12, "max": 0.5},
        origin="recovered",
        notes="from rejection 'logo too small'",
    )
    _rule(a, "min_text_size", "cta", {"px": 20, "measured_on": [1000, 1000]}, origin="recovered")
    _rule(
        b, "min_text_size", "cta", {"px": 14, "measured_on": [500, 500]}
    )  # 28 per 1000 -> stricter
    _rule(a, "clear_space", "logo", {"ratio": 0.4})
    _rule(b, "clear_space", "logo", {"ratio": 0.6}, origin="generated")  # unconfirmed: ignored
    _rule(b, "keep_visible", "hero", {})  # not a carried type
    _rule(a, "scale_range", "hero", {"min": 0.3, "max": 0.9}, enabled=False)  # disabled: ignored
    rules = {(r.type, r.role): r for r in collect_brand_rules([("A", a), ("B", b)])}
    assert set(rules) == {
        ("scale_range", "logo"),
        ("min_text_size", "cta"),
        ("clear_space", "logo"),
    }
    assert rules[("scale_range", "logo")].params == {"min": 0.12, "max": 0.5}
    assert rules[("scale_range", "logo")].sources == ["A", "B"]
    assert rules[("scale_range", "logo")].notes == ["from rejection 'logo too small'"]
    assert rules[("min_text_size", "cta")].params == {"px": 14, "measured_on": [500, 500]}
    assert rules[("clear_space", "logo")].params == {"ratio": 0.4} and rules[
        ("clear_space", "logo")
    ].sources == ["A"]


def test_proposals_land_on_matching_roles_once(tmp_path) -> None:
    a, _ = _doc(tmp_path / "a")
    _rule(a, "scale_range", "logo", {"min": 0.1, "max": 1.0})
    rules = collect_brand_rules([("A", a)])
    target, _ = _doc(tmp_path / "t")
    added = propose_brand_rules(target, rules, brand="Acme")
    assert [c.elements for c in added] == [["logo"]]
    assert added[0].provenance.origin == "generated" and added[0].hard is False
    assert "brand rule (Acme) from A" in added[0].provenance.notes
    assert propose_brand_rules(target, rules, brand="Acme") == []  # idempotent
    # a proposed (generated) rule is not carried onward
    assert collect_brand_rules([("T", target)]) == []
