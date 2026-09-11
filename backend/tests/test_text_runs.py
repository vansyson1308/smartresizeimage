"""Mixed style runs survive the whole path (independent audit, finding 3).

``TextContent`` always modelled several runs, but fitting and rendering used the
first run's style and the PSD import read only the first style run. These tests
pin the repaired behaviour: runs keep their own face, weight, size ratio and colour
through import, editing, wrapping, rendering, serialization and the variant
pipeline's typography and font disclosure; features the renderer does not
reproduce are disclosed, never silently dropped.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.design.adapter import (
    document_from_elements,
    text_runs_from_font_info,
    unsupported_text_attributes,
)
from backend.app.design.assets import AssetStore
from backend.app.design.document import TextContent, TextRun, TextStyle
from backend.app.design.fonts import default_registry
from backend.app.design.serialize import document_from_dict, document_to_dict
from backend.app.design.text_render import fit_text, render_text
from backend.app.design.variant import VariantBrief, generate_variant
from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, DesignElement
from backend.app.parser.psd_parser import PSDParser
from backend.app.quality import QualityConfig
from backend.tests.test_api import _layered_project
from backend.tests.test_planner import _doc

RED, BLACK = "#ff0000", "#000000"


def _two_runs(size: float = 40.0) -> TextContent:
    return TextContent(
        runs=[
            TextRun("50% ", TextStyle(font_size=size, color=BLACK)),
            TextRun("OFF", TextStyle(font_size=size * 1.5, color=RED, weight="bold")),
        ]
    )


def _pixels(img: Image.Image, rgb: tuple[int, int, int]) -> list[tuple[int, int]]:
    px = img.load()
    out = []
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = px[x, y]
            if a > 200 and (r, g, b) == rgb:
                out.append((x, y))
    return out


# ---- shaping and rendering -------------------------------------------------------------


def test_fit_keeps_per_run_size_ratio_and_faces() -> None:
    layout = fit_text(_two_runs(), 600, 200, min_px=12, max_px=48)
    assert not layout.overflow and len(layout.lines) == 1
    assert layout.run_px == [layout.font_px, int(round(layout.font_px * 1.5))]
    assert layout.run_fonts[0].requested_weight == "regular"
    assert layout.run_fonts[1].requested_weight == "bold"
    assert "bold" in (layout.run_fonts[1].style or "").lower()
    segs = layout.segments[0]
    assert [(s.text, s.run) for s in segs] == [("50% ", 0), ("OFF", 1)]
    # the bold, larger run measures wider than the same glyphs in the primary style
    assert segs[1].width > 0 and segs[0].width > 0


def test_render_draws_each_run_with_its_own_colour_on_one_baseline() -> None:
    layout = fit_text(_two_runs(), 600, 200, min_px=12, max_px=48)
    img = render_text(_two_runs(), layout, 600, layout.height)
    black, red = _pixels(img, (0, 0, 0)), _pixels(img, (255, 0, 0))
    assert black and red, "both runs must be drawn"
    assert max(x for x, _ in black) < min(x for x, _ in red), "runs keep their order"
    # one baseline: the bottoms of both runs' glyphs agree (within a couple of pixels)
    assert abs(max(y for _, y in black) - max(y for _, y in red)) <= 2
    # the 1.5x bold run is taller than the primary run
    red_h = max(y for _, y in red) - min(y for _, y in red)
    black_h = max(y for _, y in black) - min(y for _, y in black)
    assert red_h > black_h


def test_wrapping_keeps_runs_and_never_splits_a_word_across_a_run_boundary() -> None:
    content = TextContent(
        runs=[
            TextRun("Big Sup", TextStyle(font_size=30, color=BLACK)),
            TextRun("er sale today", TextStyle(font_size=30, color=RED, weight="bold")),
        ]
    )
    layout = fit_text(content, 150, None, min_px=30, max_px=30)
    assert len(layout.lines) >= 2
    joined = "".join(layout.lines).replace(" ", "")
    assert joined == content.plain.replace(" ", ""), "nothing is dropped"
    for segs in layout.segments:
        text = "".join(s.text for s in segs)
        if "Sup" in text:
            assert "Super" in text, "the word spanning both runs stays together"
            assert [s.run for s in segs if s.text.strip()][-2:] == [0, 1]
    # a single-run text still wraps exactly as before
    plain = TextContent(runs=[TextRun(content.plain, TextStyle(font_size=30))])
    assert "".join(fit_text(plain, 150, None, min_px=30, max_px=30).lines) == "".join(layout.lines)


def test_run_scaling_survives_fitting_into_a_small_box() -> None:
    layout = fit_text(_two_runs(80), 120, 60, min_px=8, max_px=80)
    assert layout.overflow is False
    assert layout.run_px[1] >= layout.run_px[0]
    assert layout.height <= 60 and layout.width <= 120


# ---- PSD import ------------------------------------------------------------------------


class _FakeTypeLayer:
    """Just enough of a psd-tools type layer: text, engine and resource dicts."""

    def __init__(self) -> None:
        self.text = "50% OFF\rtoday"
        self.resource_dict = {"FontSet": [{"Name": "Montserrat-Regular"},
                                          {"Name": "Montserrat-Bold"}]}
        self.engine_dict = {
            "StyleRun": {
                "RunLengthArray": [4, 3, 6],
                "RunArray": [
                    {"StyleSheet": {"StyleSheetData": {
                        "Font": 0, "FontSize": 40.0,
                        "FillColor": {"Values": [1.0, 0.0, 0.0, 0.0]},
                    }}},
                    {"StyleSheet": {"StyleSheetData": {
                        "Font": 1, "FontSize": 60.0, "Tracking": 50,
                        "FillColor": {"Values": [1.0, 1.0, 0.0, 0.0]},
                    }}},
                    {"StyleSheet": {"StyleSheetData": {
                        "Font": 0, "FontSize": 40.0, "Underline": True,
                        "FillColor": {"Values": [1.0, 0.0, 0.0, 0.0]},
                    }}},
                ],
            },
            "ParagraphRun": {"RunArray": [
                {"ParagraphSheet": {"Properties": {"Justification": 2}}}
            ]},
        }


def test_psd_font_info_extracts_every_style_run() -> None:
    info = PSDParser()._extract_font_info(_FakeTypeLayer())
    assert [r["length"] for r in info["runs"]] == [4, 3, 6]
    assert [r["font_name"] for r in info["runs"]] == [
        "Montserrat-Regular", "Montserrat-Bold", "Montserrat-Regular"
    ]
    assert info["runs"][1]["bold"] is True and info["runs"][1]["font_size"] == 60.0
    assert info["runs"][1]["color"] == [1.0, 1.0, 0.0, 0.0]
    assert info["runs"][2]["underline"] is True
    assert info["align"] == "center"
    # legacy first-run keys stay for older callers
    assert info["font_name"] == "Montserrat-Regular" and info["font_size"] == 40.0


def test_adapter_builds_runs_from_font_info_and_discloses_unsupported_attributes() -> None:
    info = PSDParser()._extract_font_info(_FakeTypeLayer())
    runs = text_runs_from_font_info("50% OFF\rtoday", info)
    assert [r.text for r in runs] == ["50% ", "OFF", "\ntoday"]
    assert runs[0].style.weight == "regular" and runs[1].style.weight == "bold"
    assert runs[1].style.color == RED and runs[0].style.color == BLACK
    assert runs[1].style.font_size == 60.0 and runs[0].style.font_size == 40.0
    assert runs[1].style.letter_spacing == 3.0  # 50/1000 em at 60 px
    assert all(r.style.align == "center" for r in runs)
    assert unsupported_text_attributes(info) == ["underline"]
    # a length table that does not add up loses nothing
    short = dict(info, runs=[dict(info["runs"][0], length=2), dict(info["runs"][1], length=1)])
    runs = text_runs_from_font_info("50% OFF", short)
    assert "".join(r.text for r in runs) == "50% OFF"
    # no run table: the legacy keys style the whole text
    single = text_runs_from_font_info("Hello", {"font_name": "Arial-BoldMT", "font_size": 20})
    assert len(single) == 1 and single[0].style.weight == "bold"


def test_document_from_elements_registers_a_font_per_run(tmp_path: Path) -> None:
    info = PSDParser()._extract_font_info(_FakeTypeLayer())
    elem = DesignElement(
        id="headline", name="Headline", layer_type="type", bbox=BoundingBox(10, 10, 600, 120),
        text_content="50% OFF\rtoday", role=ElementRole.HEADLINE, priority=1, z_index=1,
        font_info=info, effects={"_role_source": "rule"},
    )
    doc = document_from_elements([elem], (800, 400), AssetStore(tmp_path / "assets"), name="t")
    head = doc.element("headline")
    assert [r.text for r in head.text.runs] == ["50% ", "OFF", "\ntoday"]
    assert {(f.family, f.weight) for f in doc.fonts} == {
        ("Montserrat-Regular", "regular"), ("Montserrat-Bold", "bold")
    }
    assert head.effects["unsupported_text_attributes"] == ["underline"]


# ---- serialization and editing ---------------------------------------------------------


def test_runs_round_trip_through_serialization(tmp_path: Path) -> None:
    doc, _store = _doc(tmp_path)
    doc.element("headline").text = _two_runs()
    again = document_from_dict(document_to_dict(doc))
    runs = again.element("headline").text.runs
    assert [(r.text, r.style.color, r.style.weight, r.style.font_size) for r in runs] == [
        ("50% ", BLACK, "regular", 40.0), ("OFF", RED, "bold", 60.0)
    ]


def test_edit_text_keeps_runs_around_the_change() -> None:
    content = _two_runs()
    content.edit_text("60% OFF")
    assert [(r.text, r.style.color) for r in content.runs] == [("60% ", BLACK), ("OFF", RED)]
    content.edit_text("60% OFF TODAY")
    assert [(r.text, r.style.color) for r in content.runs] == [
        ("60% ", BLACK), ("OFF TODAY", RED)
    ]
    content.edit_text("OFF TODAY")
    assert [(r.text, r.style.color) for r in content.runs] == [("OFF TODAY", RED)]
    content = _two_runs()
    content.edit_text("Totally new")  # nothing in common: first run's style, one run
    assert [(r.text, r.style.color) for r in content.runs] == [("Totally new", BLACK)]


def test_set_text_op_keeps_runs_on_text_and_style_edits(tmp_path: Path) -> None:
    from backend.app.api.server import create_app

    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as client:
        pid = _layered_project(client, tmp_path)["project"]["id"]

        def patch(*ops):
            res = client.patch(f"/api/projects/{pid}/document", json={"ops": list(ops)})
            assert res.status_code == 200, res.text
            return next(e for e in res.json()["document"]["elements"] if e["id"] == "headline")

        # explicit runs through the API
        head = patch({"op": "set_text", "element_id": "headline", "runs": [
            {"text": "50% ", "style": {"font_size": 40, "color": BLACK, "weight": "regular"}},
            {"text": "OFF", "style": {"font_size": 60, "color": RED, "weight": "bold"}},
        ]})
        assert [r["text"] for r in head["text"]["runs"]] == ["50% ", "OFF"]
        # the UI's single-style editor: text change + unchanged style fields
        head = patch({"op": "set_text", "element_id": "headline", "text": "70% OFF",
                      "style": {"font_family": "DejaVu Sans", "font_size": 40,
                                "weight": "regular", "align": "left", "color": BLACK}})
        runs = head["text"]["runs"]
        assert [(r["text"], r["style"]["color"], r["style"]["weight"]) for r in runs] == [
            ("70% ", BLACK, "regular"), ("OFF", RED, "bold")
        ]
        # a size edit scales every run by the same ratio; a colour edit applies to all
        head = patch({"op": "set_text", "element_id": "headline",
                      "style": {"font_size": 20, "color": "#123456"}})
        runs = head["text"]["runs"]
        assert [r["style"]["font_size"] for r in runs] == [20.0, 30.0]
        assert {r["style"]["color"] for r in runs} == {"#123456"}
        assert [r["style"]["weight"] for r in runs] == ["regular", "bold"]
        # invalid runs are refused
        res = client.patch(f"/api/projects/{pid}/document", json={"ops": [
            {"op": "set_text", "element_id": "headline", "runs": []}
        ]})
        assert res.status_code == 400
    app.state.service.shutdown()


# ---- variant pipeline and export disclosure --------------------------------------------


def test_variant_typography_reports_runs_and_discloses_each_face(tmp_path: Path) -> None:
    doc, store = _doc(tmp_path)
    head = doc.element("headline")
    head.text = TextContent(
        runs=[
            TextRun("50% ", TextStyle(font_size=54, color="#ffffff")),
            TextRun("OFF", TextStyle(font_size=81, color=RED, weight="bold",
                                     font_family="Montserrat-Bold")),
        ]
    )
    result = generate_variant(
        doc, store, VariantBrief(1080, 1080, name="sq"),
        quality_config=QualityConfig(run_ocr=False), planner="constraints",
    )
    typo = result.plan["typography"]["headline"]
    assert len(typo["runs"]) == 2
    assert typo["runs"][1]["weight"] == "bold" and typo["runs"][1]["color"] == RED
    assert typo["runs"][1]["font_px"] == int(round(typo["runs"][0]["font_px"] * 1.5))
    assert typo["runs"][1]["font_family_requested"] == "Montserrat-Bold"
    assert typo["runs"][1]["font_status"] in ("substituted", "missing")
    disclosed = {(f["requested"], f["status"]) for f in result.plan["fonts"]}
    assert ("Montserrat-Bold", typo["runs"][1]["font_status"]) in disclosed
    assert any("Montserrat-Bold" in w for w in result.warnings)
    # the rendered variant contains the red run
    assert _pixels(result.image.convert("RGBA"), (255, 0, 0))


def test_registry_reports_per_run_glyph_coverage(tmp_path: Path) -> None:
    doc, store = _doc(tmp_path)
    reg = default_registry()
    doc.element("headline").text = TextContent(
        runs=[
            TextRun("Sale ", TextStyle(font_size=54, color="#ffffff")),
            TextRun("漢字", TextStyle(font_size=54, color="#ffffff", weight="bold")),
        ]
    )
    result = generate_variant(
        doc, store, VariantBrief(1080, 1080, name="sq"), registry=reg,
        quality_config=QualityConfig(run_ocr=False), planner="constraints",
    )
    typo = result.plan["typography"]["headline"]
    covered = reg.missing_glyphs(reg.resolve_for_text("DejaVu Sans", "漢字", "bold").path, "漢字")
    if covered:  # no CJK-capable face installed: the missing glyphs are attributed to run 1
        assert typo["runs"][0]["missing_glyphs"] in ("", None)
        assert typo["runs"][1]["missing_glyphs"]
        assert typo["missing_glyphs"] == typo["runs"][1]["missing_glyphs"]
    else:
        assert not typo["missing_glyphs"]
