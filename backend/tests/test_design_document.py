"""Tests for the typed design document, serialization, fonts, text rendering and projects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from backend.app.design import (
    SCHEMA_VERSION,
    Constraint,
    DesignDocument,
    Element,
    Geometry,
    Provenance,
    TextContent,
    TextRun,
    TextStyle,
    document_from_dict,
    document_to_dict,
    migrate_document_dict,
)
from backend.app.design.adapter import document_from_elements, elements_from_document
from backend.app.design.assets import AssetStore
from backend.app.design.fonts import FontRegistry
from backend.app.design.project import Project
from backend.app.design.text_render import fit_text, render_text, wrap_lines
from backend.app.enums import ElementRole
from backend.app.models import BoundingBox, DesignElement


def _doc() -> DesignDocument:
    doc = DesignDocument(id="doc_test", name="Test", canvas_width=1200, canvas_height=628)
    doc.elements.append(
        Element(
            id="bg",
            kind="shape",
            name="Background",
            role="background",
            geometry=Geometry(0, 0, 1200, 628),
            z_index=0,
            shape={"type": "rect", "fill": "#4a6fa5"},
            provenance=Provenance(origin="fixture"),
        )
    )
    doc.elements.append(
        Element(
            id="headline",
            kind="text",
            name="Headline",
            role="headline",
            geometry=Geometry(70, 50, 620, 120),
            z_index=2,
            text=TextContent(
                runs=[
                    TextRun(
                        "SUMMER SUPER SALE",
                        TextStyle(font_family="DejaVu Sans", font_size=64, weight="bold",
                                  color="#ffffff"),
                    )
                ]
            ),
            priority=1,
            provenance=Provenance(origin="fixture", confidence=0.9),
        )
    )
    doc.elements.append(
        Element(
            id="cta",
            kind="text",
            name="CTA",
            role="cta",
            geometry=Geometry(70, 305, 350, 95),
            z_index=2,
            text=TextContent(runs=[TextRun("SHOP NOW", TextStyle(font_size=40, color="#ffffff"))]),
            priority=2,
            provenance=Provenance(origin="fixture"),
        )
    )
    doc.normalize_z()
    doc.add_constraint(
        Constraint(id="c1", type="order_below", elements=["cta", "headline"], hard=False)
    )
    doc.add_constraint(
        Constraint(id="c2", type="min_text_size", elements=["cta"], params={"px": 14})
    )
    return doc


def test_document_round_trip_and_validation() -> None:
    doc = _doc()
    payload = document_to_dict(doc)
    assert payload["schema_version"] == SCHEMA_VERSION
    restored = document_from_dict(json.loads(json.dumps(payload)))
    assert restored.element("headline").text.plain == "SUMMER SUPER SALE"
    assert restored.element("headline").text.primary_style.weight == "bold"
    assert restored.constraints[0].type == "order_below"
    assert restored.validate() == []


def test_document_rejects_bad_constraints_and_kinds() -> None:
    doc = _doc()
    with pytest.raises(ValueError):
        Constraint(id="x", type="not_a_type", elements=["cta"])
    with pytest.raises(KeyError):
        doc.add_constraint(Constraint(id="x", type="keep_visible", elements=["missing"]))
    with pytest.raises(ValueError):
        Element(id="e", kind="hologram", name="?", role="unknown", geometry=Geometry(0, 0, 1, 1))
    bad = document_to_dict(doc)
    bad["constraints"].append(
        {"id": "c9", "type": "clear_space", "elements": ["cta"], "params": {}}
    )
    with pytest.raises(ValueError, match="missing param"):
        document_from_dict(bad)


def test_migration_rejects_unknown_versions() -> None:
    with pytest.raises(ValueError):
        migrate_document_dict({"schema_version": "9.9"})
    legacy = document_to_dict(_doc())
    del legacy["schema_version"]
    assert migrate_document_dict(legacy)["schema_version"] == SCHEMA_VERSION


def test_font_registry_resolves_and_discloses_substitution(tmp_path: Path) -> None:
    reg = FontRegistry(scan_system=True)
    if "DejaVu Sans" not in reg.families:
        pytest.skip("DejaVu Sans not installed on this machine")
    ok = reg.resolve("DejaVu Sans", "bold", False)
    assert ok.status == "available" and ok.path and "Bold" in ok.path
    missing = reg.resolve("Definitely Not A Font", "regular", False)
    assert missing.status in ("substituted", "missing")
    if missing.status == "substituted":
        assert missing.family == "DejaVu Sans"
    # the bundled DejaVu Sans is always present (deterministic fallback on every
    # machine); other families are missing when only project-local dirs are scanned
    local = FontRegistry(extra_dirs=[tmp_path], scan_system=False)
    assert local.resolve("DejaVu Sans").status == "available"
    assert local.resolve("DejaVu Serif").status in ("substituted", "missing")


def test_wrap_and_fit_never_drop_characters() -> None:
    reg = FontRegistry()
    content = TextContent(runs=[TextRun("MEGA CLEARANCE WEEKEND EVENT", TextStyle(font_size=60))])
    layout = fit_text(content, 300, None, min_px=12, max_px=60, registry=reg, max_lines=3)
    assert " ".join(layout.lines) == "MEGA CLEARANCE WEEKEND EVENT"
    assert layout.width <= 300
    assert layout.overflow is False
    font = reg.load(layout.resolved_font, layout.font_px)
    assert "".join(wrap_lines("Supercalifragilistic", font, 40)).replace(" ", "") == (
        "Supercalifragilistic"
    )
    tight = fit_text(content, 60, 20, min_px=12, max_px=60, registry=reg, max_lines=1)
    assert tight.overflow is True


def test_render_text_produces_visible_glyphs_with_color() -> None:
    reg = FontRegistry()
    content = TextContent(
        runs=[TextRun("Giảm giá 50%", TextStyle(font_size=40, color="#ff0000", align="center"))],
        locale="vi",
    )
    layout = fit_text(content, 400, None, min_px=20, max_px=40, registry=reg)
    img = render_text(content, layout, 400, 80, registry=reg)
    assert img.size == (400, 80)
    px = img.getdata(band=3)
    assert sum(1 for a in px if a > 0) > 200
    reds = [p for p in img.getdata() if p[3] > 200]
    assert reds and all(p[0] > 200 and p[1] < 60 for p in reds)


def _engine_elements() -> list[DesignElement]:
    logo = Image.new("RGBA", (180, 80), (250, 250, 250, 255))
    ImageDraw.Draw(logo).text((10, 30), "LOGO", fill=(20, 20, 20, 255))
    bg = Image.new("RGBA", (1200, 628), (86, 126, 164, 255))
    return [
        DesignElement(
            id="bg", name="Background", layer_type="pixel", bbox=BoundingBox(0, 0, 1200, 628),
            image=bg, role=ElementRole.BACKGROUND, priority=9,
            effects={"_role_source": "rule"},
        ),
        DesignElement(
            id="headline", name="Headline", layer_type="type", bbox=BoundingBox(70, 50, 620, 120),
            text_content="SUMMER SUPER SALE", role=ElementRole.HEADLINE, priority=1, z_index=2,
            font_info={"font_name": "Montserrat-Bold", "font_size": 64,
                       "color": [1.0, 1.0, 1.0, 1.0]},
            effects={"_role_source": "heuristic"},
        ),
        DesignElement(
            id="logo", name="Logo", layer_type="pixel", bbox=BoundingBox(960, 24, 180, 80),
            image=logo, role=ElementRole.LOGO, priority=1, z_index=3,
            effects={"_role_source": "rule"},
        ),
    ]


def test_adapter_builds_document_with_provenance_and_assets(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "assets")
    doc = document_from_elements(_engine_elements(), (1200, 628), store, name="Case",
                                 source_ref="case.psd")
    assert doc.validate() == []
    head = doc.element("headline")
    assert head.kind == "text" and head.text.plain == "SUMMER SUPER SALE"
    assert head.role_confidence == 0.5  # heuristic
    assert head.text.primary_style.color == "#ffffff"
    logo = doc.element("logo")
    assert logo.kind == "image" and logo.asset is not None
    assert store.verify(logo.asset)
    assert logo.role_confidence == 0.9 and logo.allowed.scale_free is False
    fonts = {f.family: f for f in doc.fonts}
    assert fonts["Montserrat-Bold"].status in ("substituted", "missing")
    types = {c.type for c in doc.constraints}
    assert {"keep_visible", "clear_space"} <= types

    back = elements_from_document(doc, store)
    by_id = {e.id: e for e in back}
    assert by_id["headline"].image is None and by_id["headline"].layer_type == "type"
    assert by_id["logo"].image is not None and by_id["logo"].image.size == (180, 80)
    overridden = elements_from_document(doc, store, text_overrides={"headline": "WINTER SALE"})
    assert {e.id: e for e in overridden}["headline"].text_content == "WINTER SALE"


def test_asset_store_is_content_addressed(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    img = Image.new("RGBA", (20, 10), (1, 2, 3, 255))
    a = store.put(img, "logo.png")
    b = store.put(img.copy(), "again.png")
    assert a.asset_id == b.asset_id and a.content_hash == b.content_hash
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert store.get(a).size == (20, 10)


def test_project_save_reopen_history_and_undo(tmp_path: Path) -> None:
    doc = _doc()
    project = Project.create(tmp_path / "p1", doc, name="Campaign A", brand="Brand")
    assert (tmp_path / "p1" / "project.json").exists()
    v0 = project.document_version

    project.document.element("headline").text.replace_text("WINTER SALE")
    project.save(snapshot=True, label="edit headline")
    assert project.document_version == v0 + 1

    reopened = Project.load(tmp_path / "p1")
    assert reopened.document.element("headline").text.plain == "WINTER SALE"
    assert reopened.name == "Campaign A"
    assert [h["label"] for h in reopened.history()] == ["created", "edit headline"]

    assert reopened.undo() is True
    assert reopened.document.element("headline").text.plain == "SUMMER SUPER SALE"
    again = Project.load(tmp_path / "p1")
    assert again.document.element("headline").text.plain == "SUMMER SUPER SALE"
    assert again.history()[-1]["label"].startswith("undo")


def test_project_variant_records_persist(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "p2", _doc())
    rec = project.new_variant("Story", 1080, 1920, brief={"locale": "vi"})
    project.store_variant_output(
        rec.id, Image.new("RGB", (1080, 1920)), {"verdict": "accepted"}, {"x": 1},
        verdict="accepted",
    )
    project.set_approval(rec.id, "rejected", "logo too small")
    reopened = Project.load(tmp_path / "p2")
    got = reopened.variants[rec.id]
    assert got.status == "done" and got.verdict == "accepted"
    assert got.approval == "rejected" and got.approval_reason == "logo too small"
    assert reopened.variant_image(rec.id).size == (1080, 1920)
    assert reopened.variant_detail(rec.id)["plan"] == {"x": 1}


def test_translations_round_trip_and_locale_lookup() -> None:
    content = TextContent(runs=[TextRun("SUMMER SALE", TextStyle())],
                          translations={"vi": "GIẢM GIÁ HÈ", "de-DE": "SOMMERSCHLUSSVERKAUF"})
    assert content.text_for_locale(None) == "SUMMER SALE"
    assert content.text_for_locale("vi") == "GIẢM GIÁ HÈ"
    assert content.text_for_locale("vi-VN") == "GIẢM GIÁ HÈ"  # language prefix match
    assert content.text_for_locale("de") == "SOMMERSCHLUSSVERKAUF"
    assert content.text_for_locale("fr") == "SUMMER SALE"  # no approved copy -> master
    doc = _doc()
    doc.element("headline").text.translations = {"vi": "GIẢM GIÁ HÈ"}
    restored = document_from_dict(json.loads(json.dumps(document_to_dict(doc))))
    assert restored.element("headline").text.translations == {"vi": "GIẢM GIÁ HÈ"}


def test_locale_selects_translation_but_override_wins_unless_protected(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "assets")
    doc = _doc()
    doc.element("headline").text.translations = {"vi": "GIẢM GIÁ HÈ"}
    doc.element("cta").text.translations = {"vi": "MUA NGAY"}
    doc.element("cta").text.protected = True
    by_id = {e.id: e for e in elements_from_document(doc, store, locale="vi")}
    assert by_id["headline"].text_content == "GIẢM GIÁ HÈ"
    assert by_id["cta"].text_content == "MUA NGAY"
    by_id = {e.id: e for e in elements_from_document(
        doc, store, locale="vi", text_overrides={"headline": "FLASH", "cta": "NOPE"})}
    assert by_id["headline"].text_content == "FLASH"
    assert by_id["cta"].text_content == "MUA NGAY"  # protected copy ignores overrides


def test_font_coverage_falls_back_to_a_face_with_the_glyphs() -> None:
    reg = FontRegistry()
    if "DejaVu Sans" not in reg.families:
        pytest.skip("DejaVu Sans not installed")
    latin = reg.resolve_for_text("DejaVu Sans", "SUMMER SALE")
    assert latin.status == "available" and latin.family == "DejaVu Sans"
    missing = reg.missing_glyphs(latin.path, "漢字セール")
    if missing is None:
        pytest.skip("fontTools not available for coverage checks")
    assert missing
    cjk = reg.resolve_for_text("DejaVu Sans", "漢字セール")
    if reg.missing_glyphs(cjk.path, "漢字セール"):
        # Decided by glyph coverage, not by a font's name: a machine may have a
        # "Gothic" face without CJK glyphs, or CJK glyphs under any other name.
        pytest.skip("no installed font covers the CJK sample")
    assert cjk.status == "substituted" and cjk.family != "DejaVu Sans"
