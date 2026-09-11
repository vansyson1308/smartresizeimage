"""Layout grammar and creative directions (Mission V2, phase C).

The grammar composes layout families from a small vocabulary and records their
traits; a creative direction narrows the candidates (and reports what a format
cannot satisfy) instead of prompting anything. Everything stays deterministic and
judged by the same scoring and quality contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.design.adapter import elements_from_document
from backend.app.design.grammar import (
    apply_direction,
    direction_key,
    family_traits,
    grammar_families,
    parse_direction,
)
from backend.app.design.planner import (
    _handwritten_families,
    aspect_class,
    choose_families,
    families_for,
    plan_layout,
)
from backend.tests.test_api import _layered_project
from backend.tests.test_planner import _doc


def _placed(plan):
    return {r.element_id: r.new_bbox for r in plan.layout if r.visible}


@pytest.mark.parametrize("aspect,count", [(1200 / 628, 12), (1080 / 1350, 24), (1.0, 36)])
def test_grammar_families_are_well_formed(aspect: float, count: int) -> None:
    fams = grammar_families(aspect)
    assert len(fams) == count
    assert len({f.name for f in fams}) == count
    for f in fams:
        for region in (f.text, f.subject, f.logo):
            assert region.x >= 0.0 and region.x + region.w <= 1.0 + 1e-6, f.name
            assert region.y >= 0.0 and region.y + region.h <= 1.0 + 1e-6, f.name
            assert region.w > 0.1 and region.h > 0.05
        # text and subject never overlap: side by side or one above the other
        t, s = f.text, f.subject
        separated_x = t.x + t.w <= s.x + 1e-6 or s.x + s.w <= t.x + 1e-6
        separated_y = t.y + t.h <= s.y + 1e-6 or s.y + s.h <= t.y + 1e-6
        assert separated_x or separated_y, f.name
        assert f.traits["grammar"] is True and f.traits["arrangement"] in ("side", "stack")
    cls = aspect_class(aspect)
    arrangements = {f.traits["arrangement"] for f in fams}
    assert arrangements == ({"side"} if cls == "landscape" else
                            {"stack"} if cls == "portrait" else {"side", "stack"})


def test_hand_written_families_get_inferred_traits() -> None:
    traits = {f.name: family_traits(f) for f in _handwritten_families(1200 / 628)}
    assert traits["landscape_text_left"]["text_side"] == "left"
    assert traits["landscape_text_right"]["text_side"] == "right"
    portrait = {f.name: family_traits(f) for f in _handwritten_families(0.8)}
    assert portrait["portrait_subject_top"]["text_position"] == "bottom"
    assert portrait["portrait_centered"]["text_align"] == "center"
    assert all(t["grammar"] is False for t in portrait.values())


def test_families_for_includes_the_grammar_unless_disabled() -> None:
    assert len(families_for(1200 / 628, grammar=False)) == 3
    assert len(families_for(1200 / 628, grammar=True)) == 15


def test_parse_direction_tokens_and_objects() -> None:
    assert parse_direction("copy text-left") == {"emphasis": "copy", "text_side": "left"}
    assert parse_direction("Subject-top, center") == {"text_position": "bottom",
                                                      "text_align": "center"}
    assert parse_direction({"family": "g_landscape_side_left65_tl", "mood": "bold"}) == {
        "family": "g_landscape_side_left65_tl", "mood": "bold"}
    assert parse_direction("") is None and parse_direction({}) is None
    for bad in ("sideways", {"emphasis": "loud"}, {"nope": 1}, 42):
        with pytest.raises(ValueError):
            parse_direction(bad)
    assert direction_key({"emphasis": "copy", "mood": "x"}) == "emphasis=copy"
    assert direction_key(None) == ""


def test_apply_direction_filters_and_reports_unmet() -> None:
    fams = families_for(1200 / 628)
    kept, notes = apply_direction(fams, {"text_side": "right"})
    assert kept and not notes and all(family_traits(f)["text_side"] == "right" for f in kept)
    kept, notes = apply_direction(fams, {"text_position": "top"})  # landscape cannot stack
    assert kept == fams and notes == ["direction_unmet:text_position:top"]
    copy, _ = apply_direction(fams, {"emphasis": "copy"})
    subject, _ = apply_direction(fams, {"emphasis": "subject"})
    assert min(family_traits(f)["text_share"] for f in copy) >= max(
        family_traits(f)["text_share"] for f in subject)
    pinned, notes = apply_direction(fams, {"family": "landscape_wide_text"})
    assert [f.name for f in pinned] == ["landscape_wide_text"] and not notes
    _, notes = apply_direction(fams, {"family": "no_such_family"})
    assert notes == ["direction_unmet:family:no_such_family"]


def test_plan_layout_honours_directions(tmp_path: Path) -> None:
    doc, store = _doc(tmp_path)
    elements = elements_from_document(doc, store)
    target = (1200, 628)
    free = plan_layout(doc, elements, target)
    assert free.candidates == 15 and free.traits
    left = plan_layout(doc, elements, target, direction={"text_side": "left"})
    right = plan_layout(doc, elements, target, direction={"text_side": "right"})
    assert _placed(left)["headline"].x < _placed(left)["hero"].x
    assert _placed(right)["headline"].x > _placed(right)["hero"].x
    assert left.traits["text_side"] == "left" and right.traits["text_side"] == "right"
    copy = plan_layout(doc, elements, target, direction={"emphasis": "copy"})
    subject = plan_layout(doc, elements, target, direction={"emphasis": "subject"})
    assert copy.traits["text_share"] > subject.traits["text_share"]
    assert _placed(copy)["hero"].area <= _placed(subject)["hero"].area
    unmet = plan_layout(doc, elements, target, direction={"text_position": "top"})
    assert "direction_unmet:text_position:top" in unmet.decisions
    pinned = plan_layout(doc, elements, target, direction={"family": "landscape_wide_text"})
    assert pinned.family == "landscape_wide_text" and pinned.candidates == 1
    # the grammar never scores worse than the hand-written families alone (superset)
    hand = plan_layout(doc, elements, target, families=families_for(target[0] / target[1],
                                                                       grammar=False))
    assert free.score >= hand.score - 1e-6


def test_choose_families_applies_the_direction_per_orientation(tmp_path: Path) -> None:
    doc, store = _doc(tmp_path)
    elements = elements_from_document(doc, store)
    sizes = [(1200, 628), (1080, 1080), (1080, 1350)]
    chosen = choose_families(doc, elements, sizes, direction={"text_align": "center"})
    # landscape has no centred family: the direction is unmet there, met on the others
    assert family_traits(chosen["square"])["text_align"] == "center"
    assert family_traits(chosen["portrait"])["text_align"] == "center"
    assert "landscape" in chosen


def test_directions_flow_through_rows_csv_and_regeneration(tmp_path: Path, monkeypatch) -> None:
    from backend.app.api.server import create_app

    for name in ("AUTOBANNER_AUTH", "AUTOBANNER_API_KEYS", "AUTOBANNER_RATE_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as client:
        pid = _layered_project(client, tmp_path)["project"]["id"]
        csv = ("label,headline,direction\nA,ONE,copy text-left\nB,TWO,subject-top\n"
               "C,THREE,sideways\n")
        parsed = client.post(f"/api/projects/{pid}/campaign/rows", json={"csv": csv}).json()
        assert parsed["rows"][0]["direction"] == {"emphasis": "copy", "text_side": "left"}
        assert parsed["rows"][1]["direction"] == {"text_position": "bottom"}
        assert parsed["rows"][2]["direction"] is None
        assert any("sideways" in n for n in parsed["notes"])
        rows = [{"id": "a", "label": "A", "direction": "copy text-left"},
                {"id": "b", "label": "B", "direction": {"emphasis": "subject"}},
                {"id": "c", "label": "C"}]
        res = client.post(f"/api/projects/{pid}/variants", json={
            "targets": [{"width": 600, "height": 314, "name": "Wide"}],
            "rows": rows, "direction": "text-right",
        })
        assert res.status_code == 202, res.text
        service = client.app.state.service
        assert service.jobs.wait(res.json()["job"]["id"], timeout=300).status == "done"
        variants = {v["brief"]["row"]["id"]: v
                    for v in client.get(f"/api/projects/{pid}/variants").json()["variants"]}
        assert variants["a"]["brief"]["direction"] == {"emphasis": "copy", "text_side": "left"}
        assert variants["b"]["brief"]["direction"] == {"emphasis": "subject"}
        assert variants["c"]["brief"]["direction"] == {"text_side": "right"}  # job default
        plans = {k: client.get(f"/api/projects/{pid}/variants/{v['id']}").json()["plan"]
                 for k, v in variants.items()}
        assert plans["a"]["planner_meta"]["traits"]["text_side"] == "left"
        assert plans["c"]["planner_meta"]["traits"]["text_side"] == "right"
        assert plans["a"]["planner_meta"]["direction"] == {"emphasis": "copy",
                                                           "text_side": "left"}
        assert plans["a"]["planner_meta"]["traits"]["text_share"] >= \
            plans["b"]["planner_meta"]["traits"]["text_share"]
        # invalid directions are refused before anything is created
        res = client.post(f"/api/projects/{pid}/variants", json={
            "targets": [{"width": 600, "height": 314}], "direction": "loud"})
        assert res.status_code == 400 and "direction" in res.json()["detail"]
        # regeneration keeps the direction, and can change it
        vid = variants["c"]["id"]
        res = client.post(f"/api/projects/{pid}/variants/{vid}/regenerate",
                          json={"direction": "text-left", "keep_layout": False})
        assert service.jobs.wait(res.json()["job"]["id"], timeout=300).status == "done"
        plan = client.get(f"/api/projects/{pid}/variants/{vid}").json()["plan"]
        assert plan["planner_meta"]["traits"]["text_side"] == "left"
    app.state.service.shutdown()
