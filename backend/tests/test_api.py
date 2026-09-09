"""End-to-end API tests: import -> edit -> save/reopen -> generate -> review -> export."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.app.api.server import create_app
from backend.app.design.document import Provenance
from backend.app.design.project import Project
from backend.app.design.serialize import document_from_dict


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


def _png_bytes(size=(600, 300)) -> bytes:
    img = Image.new("RGB", size, (80, 120, 160))
    d = ImageDraw.Draw(img)
    d.rectangle((400, 40, 560, 120), fill=(250, 250, 250))
    d.text((420, 70), "LOGO", fill=(20, 20, 20))
    d.text((40, 60), "SUMMER SALE", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _layered_project(client: TestClient, tmp_path: Path) -> dict:
    """Create a project from a flat upload, then replace its document with a layered one.

    PSD fixtures are not committed; the layered document exercises the same
    native-text path a PSD import produces.
    """
    res = client.post(
        "/api/projects",
        files={"file": ("master.png", _png_bytes(), "image/png")},
        data={"name": "Layered", "brand": "Acme"},
    )
    assert res.status_code == 201, res.text
    payload = res.json()
    project_id = payload["project"]["id"]
    service = client.app.state.service
    project = service.get_project(project_id)
    doc = project.document
    bg = doc.elements[0]
    bg.role = "background"
    # add a logo asset and two native text elements
    logo = Image.new("RGBA", (160, 80), (250, 250, 250, 255))
    ImageDraw.Draw(logo).text((10, 30), "LOGO", fill=(20, 20, 20, 255))
    ref = project.assets.put(logo, "logo.png")
    from backend.app.design.document import (
        AllowedTransforms,
        Element,
        Geometry,
        TextContent,
        TextRun,
        TextStyle,
    )

    doc.elements.append(
        Element(
            id="logo",
            kind="image",
            name="Logo",
            role="logo",
            geometry=Geometry(400, 40, 160, 80),
            z_index=3,
            asset=ref,
            priority=1,
            allowed=AllowedTransforms(scale_free=False),
            provenance=Provenance(origin="user"),
            role_confidence=1.0,
        )
    )
    doc.elements.append(
        Element(
            id="headline",
            kind="text",
            name="Headline",
            role="headline",
            geometry=Geometry(40, 40, 320, 80),
            z_index=2,
            priority=1,
            text=TextContent(
                runs=[
                    TextRun("SUMMER SALE", TextStyle(font_size=48, color="#ffffff", weight="bold"))
                ]
            ),
            provenance=Provenance(origin="user"),
            role_confidence=0.5,
        )
    )
    doc.elements.append(
        Element(
            id="cta",
            kind="text",
            name="CTA",
            role="cta",
            geometry=Geometry(40, 200, 200, 50),
            z_index=2,
            priority=2,
            text=TextContent(runs=[TextRun("SHOP NOW", TextStyle(font_size=28, color="#ffffff"))]),
            provenance=Provenance(origin="user"),
            role_confidence=0.9,
        )
    )
    doc.elements.append(
        Element(
            id="price",
            kind="text",
            name="Price",
            role="label",
            geometry=Geometry(40, 130, 200, 40),
            z_index=2,
            priority=3,
            text=TextContent(
                runs=[TextRun("$19.99", TextStyle(font_size=24, color="#ffffff"))], protected=True
            ),
            provenance=Provenance(origin="user"),
            role_confidence=0.9,
        )
    )
    doc.elements[0].provenance = Provenance(origin="user")
    doc.normalize_z()
    project.save(snapshot=True, label="layered fixture")
    return client.get(f"/api/projects/{project_id}").json()


def test_health_and_presets(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "ocr_engine" in h["environment"]
    p = client.get("/api/presets").json()
    assert p["presets"] and all("source" in x and "verified" in x for x in p["presets"])
    assert not any(x["verified"] for x in p["presets"])


def test_upload_rejects_bad_input(client: TestClient) -> None:
    res = client.post("/api/projects", files={"file": ("x.txt", b"hello", "text/plain")})
    assert res.status_code == 415
    res = client.post("/api/projects", files={"file": ("x.png", b"", "image/png")})
    assert res.status_code == 400
    res = client.post("/api/projects", files={"file": ("x.png", b"notapng", "image/png")})
    assert res.status_code == 400


def test_flat_import_is_honest(client: TestClient) -> None:
    res = client.post("/api/projects", files={"file": ("flat.png", _png_bytes(), "image/png")})
    assert res.status_code == 201
    payload = res.json()
    doc = payload["document"]
    # Every element of a flat import is either the (inferred) background or a recovered
    # element carrying a confidence below 1; nothing is presented as verified.
    for e in doc["elements"]:
        assert e["provenance"]["origin"] in ("flat_image", "recovered")
        if e["provenance"]["origin"] == "recovered" and e["role"] != "background":
            assert e["role_confidence"] < 0.95
    notes = doc["metadata"]["import_notes"]
    assert any("not separated" in n or "decomposed" in n for n in notes)
    preview = client.get(f"/api/projects/{payload['project']['id']}/preview.png")
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"


def test_document_edit_undo_and_reopen(client: TestClient, tmp_path: Path) -> None:
    payload = _layered_project(client, tmp_path)
    pid = payload["project"]["id"]
    v0 = payload["project"]["document_version"]

    res = client.patch(
        f"/api/projects/{pid}/document",
        json={
            "ops": [
                {"op": "set_text", "element_id": "headline", "text": "WINTER SALE"},
                {"op": "set_role", "element_id": "headline", "role": "headline"},
                {"op": "set_geometry", "element_id": "logo", "geometry": {"x": 380, "y": 30}},
                {
                    "op": "add_constraint",
                    "constraint": {"type": "allowed_overlap", "elements": ["headline", "cta"]},
                },
            ],
            "label": "edits",
        },
    )
    assert res.status_code == 200, res.text
    doc = res.json()["document"]
    head = next(e for e in doc["elements"] if e["id"] == "headline")
    assert head["text"]["runs"][0]["text"] == "WINTER SALE"
    assert head["role_confidence"] == 1.0
    assert res.json()["project"]["document_version"] == v0 + 1
    assert any(c["type"] == "allowed_overlap" for c in doc["constraints"])

    # locked elements refuse geometry edits
    client.patch(
        f"/api/projects/{pid}/document",
        json={"ops": [{"op": "set_flags", "element_id": "logo", "locked": True}]},
    )
    res = client.patch(
        f"/api/projects/{pid}/document",
        json={"ops": [{"op": "set_geometry", "element_id": "logo", "geometry": {"x": 0}}]},
    )
    assert res.status_code == 409

    res = client.post(f"/api/projects/{pid}/undo")
    assert res.json()["undone"] is True
    res = client.post(f"/api/projects/{pid}/undo")
    doc = res.json()["document"]
    head = next(e for e in doc["elements"] if e["id"] == "headline")
    assert head["text"]["runs"][0]["text"] == "SUMMER SALE"

    # reopen from disk in a fresh service instance
    reopened = Project.load(tmp_path / "data" / "projects" / pid)
    assert reopened.document.element("headline").text.plain == "SUMMER SALE"
    assert [h["label"] for h in reopened.history()][-1].startswith("undo")


def test_generate_review_export_round_trip(client: TestClient, tmp_path: Path) -> None:
    payload = _layered_project(client, tmp_path)
    pid = payload["project"]["id"]

    res = client.post(
        f"/api/projects/{pid}/variants",
        json={
            "preset_ids": ["ig_square"],
            "targets": [{"width": 1200, "height": 628, "name": "Wide"}],
            "text_overrides": {"cta": "BUY TODAY"},
            "locale": "en",
        },
        headers={"Idempotency-Key": "run-1"},
    )
    assert res.status_code == 202, res.text
    job_id = res.json()["job"]["id"]
    # idempotent resubmission returns the same job
    again = client.post(
        f"/api/projects/{pid}/variants",
        json={"preset_ids": ["ig_square"]},
        headers={"Idempotency-Key": "run-1"},
    )
    assert again.json()["job"]["id"] == job_id

    service = client.app.state.service
    job = service.jobs.wait(job_id, timeout=300)
    assert job is not None and job.status == "done", job.to_dict()

    variants = client.get(f"/api/projects/{pid}/variants").json()["variants"]
    assert len(variants) == 2
    assert all(v["status"] == "done" for v in variants)
    assert all(v["verdict"] in ("accepted", "needs_review", "failed") for v in variants)
    wide = next(v for v in variants if v["name"] == "Wide")
    detail = client.get(f"/api/projects/{pid}/variants/{wide['id']}").json()
    assert detail["quality"]["contract_version"].startswith("2.")
    assert detail["plan"]["brief"]["text_overrides"] == {"cta": "BUY TODAY"}
    assert "cta" in detail["plan"]["typography"]
    checks = {(c["check_id"], c["subject_id"]): c for c in detail["quality"]["checks"]}
    # protected price copy is rendered verbatim and verified
    assert ("text_legible", "price") in checks
    img = client.get(f"/api/projects/{pid}/variants/{wide['id']}/image.png")
    assert img.status_code == 200
    assert Image.open(io.BytesIO(img.content)).size == (1200, 628)

    # approve / reject
    res = client.post(
        f"/api/projects/{pid}/variants/{wide['id']}/approval", json={"approval": "rejected"}
    )
    assert res.status_code == 400  # reason required
    res = client.post(
        f"/api/projects/{pid}/variants/{wide['id']}/approval", json={"approval": "approved"}
    )
    assert res.json()["variant"]["approval"] == "approved"

    # export approved only
    res = client.get(f"/api/projects/{pid}/export?format=jpeg&only=approved")
    assert res.status_code == 200
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
    assert any(n.endswith(".jpg") for n in names)
    assert len(manifest["variants"]) == 1 and manifest["variants"][0]["approval"] == "approved"
    assert len(manifest["skipped"]) == 1
    assert "font_disclosure" in manifest

    # export editable project, import it back, and generate again
    res = client.get(f"/api/projects/{pid}/export/project")
    assert res.status_code == 200
    res = client.post(
        "/api/projects/import", files={"file": ("p.zip", res.content, "application/zip")}
    )
    assert res.status_code == 201, res.text
    new_pid = res.json()["project"]["id"]
    assert new_pid != pid
    assert len(res.json()["variants"]) == 2
    res = client.patch(
        f"/api/projects/{new_pid}/document",
        json={"ops": [{"op": "set_text", "element_id": "headline", "text": "MEGA SALE"}]},
    )
    assert res.status_code == 200
    res = client.post(
        f"/api/projects/{new_pid}/variants",
        json={"targets": [{"width": 300, "height": 250, "name": "MREC"}]},
    )
    job = service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job.status == "done", job.to_dict()
    mrec = next(
        v
        for v in client.get(f"/api/projects/{new_pid}/variants").json()["variants"]
        if v["name"] == "MREC"
    )
    detail = client.get(f"/api/projects/{new_pid}/variants/{mrec['id']}").json()
    assert detail["plan"]["target"] == {"width": 300, "height": 250}


def test_cancel_job(client: TestClient, tmp_path: Path) -> None:
    payload = _layered_project(client, tmp_path)
    pid = payload["project"]["id"]
    targets = [{"width": 1080, "height": 1920, "name": f"S{i}"} for i in range(6)]
    res = client.post(f"/api/projects/{pid}/variants", json={"targets": targets})
    job_id = res.json()["job"]["id"]
    res = client.post(f"/api/jobs/{job_id}/cancel")
    assert res.json()["cancelled"] is True
    service = client.app.state.service
    job = service.jobs.wait(job_id, timeout=300)
    assert job.status in ("cancelled", "partial")
    variants = client.get(f"/api/projects/{pid}/variants").json()["variants"]
    assert any(v["status"] == "cancelled" for v in variants)
    # cancelled variants never pretend to be done
    assert all(v["image_path"] is None for v in variants if v["status"] == "cancelled")


def test_api_key_protection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_API_KEY", "secret")
    app = create_app(tmp_path / "data2", max_workers=1)
    with TestClient(app) as c:
        assert c.get("/api/projects").status_code == 401
        assert c.get("/api/projects", headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/api/health").status_code == 200  # health stays public
    app.state.service.shutdown()


def test_restart_marks_running_variants_interrupted(client: TestClient, tmp_path: Path) -> None:
    payload = _layered_project(client, tmp_path)
    pid = payload["project"]["id"]
    project = client.app.state.service.get_project(pid)
    rec = project.new_variant("Stale", 300, 300)
    project.mark_variant(rec.id, "running")
    project.save()
    from backend.app.api.service import ProjectService

    fresh = ProjectService(tmp_path / "data", max_workers=1)
    got = fresh.get_project(pid).variants[rec.id]
    assert got.status == "failed" and "interrupted" in (got.error or "")
    fresh.shutdown()
    _ = document_from_dict  # keep import used for clarity of the reopen contract


def test_multi_size_job_plans_jointly_and_records_family_checks(
    client: TestClient, tmp_path: Path
) -> None:
    payload = _layered_project(client, tmp_path)
    pid = payload["project"]["id"]
    res = client.post(
        f"/api/projects/{pid}/variants",
        json={
            "targets": [
                {"width": 1200, "height": 628, "name": "L1"},
                {"width": 1500, "height": 500, "name": "L2"},
                {"width": 1080, "height": 1920, "name": "P1"},
            ]
        },
    )
    assert res.status_code == 202, res.text
    service = client.app.state.service
    job = service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job.status == "done", job.to_dict()
    variants = client.get(f"/api/projects/{pid}/variants").json()["variants"]
    details = {
        v["name"]: client.get(f"/api/projects/{pid}/variants/{v['id']}").json() for v in variants
    }
    fams = {n: d["plan"]["planner_meta"]["family"] for n, d in details.items()}
    assert fams["L1"] == fams["L2"] and fams["L1"].startswith("landscape")
    assert fams["P1"].startswith("portrait")
    assert all(d["plan"]["planner_meta"]["joint_family"] for d in details.values())
    for d in details.values():
        ids = {c["check_id"] for c in d["quality"]["checks"]}
        assert {
            "family_identity",
            "family_reading_order",
            "family_hierarchy",
            "family_layout",
        } <= ids
        assert d["quality"]["verdict"] == d["variant"]["verdict"]


def test_approved_variant_teaches_the_next_generation(client: TestClient, tmp_path: Path) -> None:
    """H1 in the product: approving a variant makes later runs of that orientation follow it."""
    payload = _layered_project(client, tmp_path)
    pid = payload["project"]["id"]
    assert client.get(f"/api/projects/{pid}/learned").json()["families"] == []
    res = client.post(
        f"/api/projects/{pid}/variants",
        json={"targets": [{"width": 1080, "height": 1920, "name": "P1"}]},
    )
    assert res.status_code == 202, res.text
    service = client.app.state.service
    job = service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job.status == "done", job.to_dict()
    p1 = client.get(f"/api/projects/{pid}/variants").json()["variants"][0]
    first = client.get(f"/api/projects/{pid}/variants/{p1['id']}").json()
    assert first["plan"]["planner_meta"]["from_examples"] is False
    client.post(f"/api/projects/{pid}/variants/{p1['id']}/approval", json={"approval": "approved"})

    learned = client.get(f"/api/projects/{pid}/learned").json()
    assert learned["examples"] == [p1["id"]]
    assert [f["aspect"] for f in learned["families"]] == ["portrait"]
    assert learned["families"][0]["examples"] == 1
    assert learned["families"][0]["confidence"] == 0.5

    res = client.post(
        f"/api/projects/{pid}/variants",
        json={"targets": [{"width": 1080, "height": 1350, "name": "P2"}]},
    )
    assert res.status_code == 202, res.text
    job = service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job.status == "done", job.to_dict()
    variants = client.get(f"/api/projects/{pid}/variants").json()["variants"]
    p2 = next(v for v in variants if v["name"] == "P2")
    second = client.get(f"/api/projects/{pid}/variants/{p2['id']}").json()
    meta = second["plan"]["planner_meta"]
    assert meta["family"] == "learned_portrait" and meta["from_examples"] is True

    # the learned composition keeps the approved reading order on the new size
    def order(detail: dict) -> list[str]:
        placed = sorted(detail["plan"]["placements"], key=lambda p: p["y"])
        return [p["element_id"] for p in placed if p["visible"]]

    assert order(first) == order(second)
