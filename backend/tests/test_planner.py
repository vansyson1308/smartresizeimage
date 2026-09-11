"""Tests for the constraint-aware planner."""

from __future__ import annotations

from PIL import Image

from backend.app.design.adapter import elements_from_document
from backend.app.design.assets import AssetStore
from backend.app.design.document import (
    Constraint,
    DesignDocument,
    Element,
    Geometry,
    Provenance,
    TextContent,
    TextRun,
    TextStyle,
)
from backend.app.design.planner import Family, Region, families_for, plan_layout
from backend.app.design.variant import VariantBrief, generate_variant
from backend.app.quality import Verdict


def _doc(tmp_path, long_copy: bool = False) -> tuple[DesignDocument, AssetStore]:
    store = AssetStore(tmp_path / "assets")
    doc = DesignDocument(id="doc", name="t", canvas_width=1200, canvas_height=628)
    doc.elements.append(
        Element(
            id="bg",
            kind="shape",
            name="Background",
            role="background",
            geometry=Geometry(0, 0, 1200, 628),
            z_index=0,
            shape={"type": "rect", "fill": "#2b4c7e"},
            provenance=Provenance(origin="fixture"),
        )
    )
    logo = store.put(Image.new("RGBA", (180, 80), (250, 250, 250, 255)), "logo")
    hero = store.put(Image.new("RGBA", (390, 430), (236, 88, 88, 255)), "hero")
    doc.elements.append(
        Element(
            id="logo",
            kind="image",
            name="Logo",
            role="logo",
            geometry=Geometry(960, 24, 180, 80),
            z_index=5,
            asset=logo,
            priority=1,
            provenance=Provenance(origin="fixture"),
        )
    )
    doc.elements.append(
        Element(
            id="hero",
            kind="image",
            name="Hero",
            role="hero_image",
            geometry=Geometry(740, 120, 390, 430),
            z_index=3,
            asset=hero,
            priority=2,
            provenance=Provenance(origin="fixture"),
        )
    )
    head = "MEGA CLEARANCE WEEKEND EVENT WITH EXTRA BONUS DISCOUNTS" if long_copy else "SUMMER SALE"
    doc.elements.append(
        Element(
            id="headline",
            kind="text",
            name="Headline",
            role="headline",
            geometry=Geometry(70, 50, 620, 120),
            z_index=4,
            priority=1,
            text=TextContent(
                runs=[TextRun(head, TextStyle(font_size=54, color="#ffffff", weight="bold"))]
            ),
            provenance=Provenance(origin="fixture"),
        )
    )
    doc.elements.append(
        Element(
            id="sub",
            kind="text",
            name="Sub",
            role="subheadline",
            geometry=Geometry(70, 190, 620, 90),
            z_index=4,
            priority=2,
            text=TextContent(
                runs=[
                    TextRun(
                        "Up to 50% off selected items", TextStyle(font_size=40, color="#ffffff")
                    )
                ]
            ),
            provenance=Provenance(origin="fixture"),
        )
    )
    doc.elements.append(
        Element(
            id="cta",
            kind="text",
            name="CTA",
            role="cta",
            geometry=Geometry(70, 305, 350, 95),
            z_index=4,
            priority=2,
            text=TextContent(
                runs=[TextRun("SHOP NOW", TextStyle(font_size=42, color="#ffd166", weight="bold"))]
            ),
            provenance=Provenance(origin="fixture"),
        )
    )
    doc.normalize_z()
    return doc, store


def _boxes(plan):
    return {r.element_id: r.new_bbox for r in plan.layout}


def test_families_cover_all_aspects() -> None:
    assert families_for(1.9)[0].name.startswith("landscape")
    assert families_for(0.56)[0].name.startswith("portrait")
    assert families_for(1.0)[0].name.startswith("square")


def test_plan_preserves_reading_order_hierarchy_and_bounds(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    elements = elements_from_document(doc, store)
    for target in [(1200, 628), (1080, 1080), (1080, 1920), (300, 250)]:
        plan = plan_layout(doc, elements, target)
        b = _boxes(plan)
        # reading order: headline above sub above cta
        assert b["headline"].y < b["sub"].y < b["cta"].y, target
        # hierarchy: the headline must be at least as large as the subheadline
        assert plan.text_px["headline"] >= plan.text_px["sub"], target
        # everything inside the canvas
        for eid, box in b.items():
            assert box.x >= 0 and box.y >= 0 and box.x2 <= target[0] and box.y2 <= target[1], (
                target,
                eid,
                box,
            )
        # logo and hero keep their aspect ratio (uniform scale only)
        assert abs(b["logo"].width / b["logo"].height - 180 / 80) < 0.05
        assert abs(b["hero"].width / b["hero"].height - 390 / 430) < 0.05
        # no text box overlaps the hero
        for eid in ("headline", "sub", "cta"):
            t, h = b[eid], b["hero"]
            assert t.x2 <= h.x or h.x2 <= t.x or t.y2 <= h.y or h.y2 <= t.y, (target, eid)


def test_order_below_and_anchor_edge_constraints_are_honoured(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    # force CTA above the headline (unusual) and logo anchored to the bottom edge
    doc.add_constraint(Constraint(id="c1", type="order_below", elements=["headline", "cta"]))
    doc.add_constraint(
        Constraint(id="c2", type="anchor_edge", elements=["logo"], params={"edge": "bottom"})
    )
    elements = elements_from_document(doc, store)
    plan = plan_layout(doc, elements, (1080, 1920))
    b = _boxes(plan)
    assert b["headline"].y >= b["cta"].y
    assert b["logo"].y2 >= 1920 - int(0.04 * 1080) - 1
    assert any(d.startswith("anchor:logo") for d in plan.decisions)


def test_scale_range_constraint_bounds_subject(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    doc.add_constraint(
        Constraint(id="c3", type="scale_range", elements=["hero"], params={"min": 0.5, "max": 0.6})
    )
    elements = elements_from_document(doc, store)
    plan = plan_layout(doc, elements, (1080, 1920))
    hero = _boxes(plan)["hero"]
    rel = hero.height / 1920
    assert 0.49 <= rel <= 0.61


def test_content_pressure_gives_copy_room_in_small_formats(tmp_path) -> None:
    doc, store = _doc(tmp_path, long_copy=True)
    elements = elements_from_document(doc, store)
    fam = Family(
        "square_text_top",
        Region(0.07, 0.15, 0.86, 0.34),
        Region(0.20, 0.52, 0.60, 0.44),
        Region(0.07, 0.04, 0.26, 0.09),
    )
    plan = plan_layout(doc, elements, (300, 250), families=[fam])
    b = _boxes(plan)
    assert any(d.startswith("content_pressure") for d in plan.decisions)
    # subject shrank but kept a floor, and no text sits on top of it
    assert b["hero"].height >= int(0.18 * 250) * 0.9
    for eid in ("headline", "sub", "cta"):
        t, h = b[eid], b["hero"]
        assert t.y2 <= h.y or h.y2 <= t.y or t.x2 <= h.x or h.x2 <= t.x


def test_variant_pipeline_with_constraint_planner_accepts_fixture(tmp_path) -> None:
    doc, store = _doc(tmp_path)
    for target in [(1200, 628), (1080, 1920)]:
        res = generate_variant(
            doc, store, VariantBrief(*target, name="t"), planner="constraints", quality_config=None
        )
        assert res.plan["planner"] == "constraints"
        assert res.plan["planner_meta"]["family"]
        assert res.report.verdict in (Verdict.ACCEPTED, Verdict.NEEDS_REVIEW), [
            c.message for c in res.report.issues()
        ]
        assert not res.report.failed


def test_choose_families_picks_one_family_per_orientation(tmp_path) -> None:
    from backend.app.design.planner import aspect_class, choose_families

    doc, store = _doc(tmp_path)
    elements = elements_from_document(doc, store)
    targets = [(1200, 628), (1500, 500), (1080, 1080), (1080, 1920), (1080, 1350)]
    chosen = choose_families(doc, elements, targets)
    assert set(chosen) == {"landscape", "square", "portrait"}
    assert "landscape" in chosen["landscape"].name
    assert "portrait" in chosen["portrait"].name
    # every target of a class plans with the chosen family
    for t in targets:
        fam = chosen[aspect_class(t[0] / t[1])]
        plan = plan_layout(doc, elements, t, families=[fam])
        assert plan.family == fam.name


def test_family_consistency_checks_flag_missing_and_drift() -> None:
    from backend.app.quality.family import VariantSnapshot, family_consistency_checks

    def snap(vid, target, family, px_head, px_cta, drop_logo=False):
        placements = [
            {
                "element_id": "headline",
                "x": 10,
                "y": 10,
                "width": 100,
                "height": 40,
                "visible": True,
            },
            {"element_id": "cta", "x": 10, "y": 80, "width": 80, "height": 30, "visible": True},
            {
                "element_id": "logo",
                "x": 200,
                "y": 10,
                "width": 60,
                "height": 30,
                "visible": not drop_logo,
            },
        ]
        return VariantSnapshot(
            variant_id=vid,
            target=target,
            family=family,
            placements=placements,
            typography={"headline": {"font_px": px_head}, "cta": {"font_px": px_cta}},
            roles={"headline": "headline", "cta": "cta", "logo": "logo"},
            master_px={"headline": 60.0, "cta": 30.0},
        )

    good = family_consistency_checks(
        [
            snap("a", (1200, 628), "landscape_text_left", 60, 30),
            snap("b", (1500, 500), "landscape_text_left", 40, 20),
            snap("c", (1080, 1920), "portrait_text_top", 80, 40),
        ]
    )
    assert all(c.status.value == "pass" for checks in good.values() for c in checks)

    bad = family_consistency_checks(
        [
            snap("a", (1200, 628), "landscape_text_left", 60, 30),
            snap("b", (1500, 500), "landscape_wide_text", 40, 40, drop_logo=True),  # drift + drop
        ]
    )
    by_id = {c.check_id: c for c in bad["b"]}
    assert by_id["family_identity"].status.value == "fail"
    assert by_id["family_hierarchy"].status.value == "needs_review"
    assert by_id["family_layout"].status.value == "needs_review"
    assert family_consistency_checks([snap("only", (1080, 1080), "square_text_left", 50, 25)]) == {
        "only": []
    }


def test_long_copy_on_a_small_format_prefers_a_wide_text_column(tmp_path) -> None:
    """Found by the counterexample search: a 75-word headline on 300x250 was planned into a
    narrow side column and ran three canvases tall. Boxes far outside the canvas now cost
    proportionally, so the planner picks the family that keeps the most copy on the canvas;
    what still does not fit is reported, never hidden."""
    from backend.app.design.grammar import family_traits
    from backend.app.design.planner import families_for

    doc, store = _doc(tmp_path)
    head = doc.element("headline")
    head.text.replace_text(head.text.plain + " and everything you need for the whole season "
                           "at prices you will remember")
    elements = elements_from_document(doc, store)
    plan = plan_layout(doc, elements, (300, 250))
    fam = next(f for f in families_for(300 / 250) if f.name == plan.family)
    traits = family_traits(fam)
    assert traits["arrangement"] == "stack" or traits["text_share"] >= 0.6, plan.family
    boxes = {r.element_id: r.new_bbox for r in plan.layout if r.visible}
    # the stack is not three canvases tall any more
    assert max(b.y2 for b in boxes.values()) < 2 * 250
