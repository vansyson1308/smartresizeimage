"""Brand profiles: the brand book as data, applied as proposals and as checks (phase B3).

A profile (colours, fonts, logo rules, minimum text size, tone) belongs to a workspace.
Projects of that brand get its logo/text rules as reviewable proposals; every rendered
variant is checked against the palette and fonts and lands in review when off-brand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.server import create_app
from backend.app.design.brand import color_in_palette, font_matches, normalize_profile
from backend.tests.test_api import _layered_project

A = {"X-API-Key": "ka"}
B = {"X-API-Key": "kb"}
V = {"X-API-Key": "kv"}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOBANNER_API_KEYS", "ka:acme:editor,kb:globex:editor,kv:acme:viewer")
    monkeypatch.delenv("AUTOBANNER_AUTH", raising=False)
    monkeypatch.delenv("AUTOBANNER_RATE_LIMIT", raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


PROFILE = {
    "colors": {"primary": "#1F3B63", "accent": "#ffd166", "palette": "#123456, #abc"},
    "fonts": {"headline": "Montserrat", "body": {"family": "DejaVu Sans", "weight": "regular"}},
    "logo": {"clear_space_ratio": 0.8, "min_height_ratio": 0.1, "always_visible": True},
    "text": {"min_px": 14},
    "tone": "Short and confident.",
    "do_not": "Logo on busy photos\nLowercase headlines",
}


def test_normalize_profile_validates_and_normalises() -> None:
    p = normalize_profile(PROFILE, brand="Acme")
    assert p["colors"]["primary"] == "#1f3b63" and p["colors"]["palette"] == ["#123456", "#aabbcc"]
    assert p["fonts"]["headline"] == {"family": "Montserrat", "weight": "regular"}
    assert p["logo"]["clear_space_ratio"] == 0.8 and p["text"]["min_px"] == 14
    assert p["do_not"] == ["Logo on busy photos", "Lowercase headlines"]
    for bad in (
        {"colors": {"primary": "blue"}},
        {"logo": {"clear_space_ratio": 9}},
        {"fonts": {"headline": {"weight": "bold"}}},
        {"text": {"min_px": "big"}},
    ):
        with pytest.raises(ValueError):
            normalize_profile(bad, brand="Acme")
    assert color_in_palette("#fefefe", ["#ffffff"]) and not color_in_palette("#ff0000", ["#ffffff"])
    assert font_matches("Montserrat-Bold", {"family": "Montserrat"}) is True
    assert font_matches("DejaVu Sans", {"family": "Montserrat"}) is False
    assert font_matches("DejaVu Sans", None) is None


def test_profiles_are_per_workspace_and_role_gated(client: TestClient) -> None:
    res = client.put("/api/brands/Acme", json=PROFILE, headers=A)
    assert res.status_code == 200, res.text
    assert res.json()["profile"]["updated_by"] == "acme"
    assert client.put("/api/brands/Acme", json={"colors": {"primary": "nope"}},
                      headers=A).status_code == 400
    assert client.put("/api/brands/Acme", json=PROFILE, headers=V).status_code == 403
    got = client.get("/api/brands/acme", headers=A).json()  # case-insensitive key
    assert got["profile"]["colors"]["primary"] == "#1f3b63"
    assert [b["brand"] for b in client.get("/api/brands", headers=A).json()["brands"]] == ["Acme"]
    # another workspace sees nothing
    assert client.get("/api/brands/Acme", headers=B).json()["profile"] is None
    assert client.get("/api/brands", headers=B).json()["brands"] == []
    assert client.delete("/api/brands/Acme", headers=B).status_code == 404
    assert client.delete("/api/brands/Acme", headers=A).status_code == 204
    assert client.get("/api/brands/Acme", headers=A).json()["profile"] is None


@pytest.fixture()
def open_client(tmp_path: Path, monkeypatch):
    for name in ("AUTOBANNER_API_KEYS", "AUTOBANNER_API_KEY", "AUTOBANNER_AUTH",
                 "AUTOBANNER_RATE_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


def test_profile_rules_are_proposed_and_variants_checked(open_client: TestClient, tmp_path: Path):
    client = open_client
    pid = _layered_project(client, tmp_path)["project"]["id"]  # brand "Acme"
    strict = dict(PROFILE, colors={"primary": "#1f3b63", "strict": True})
    assert client.put("/api/brands/Acme", json=strict).status_code == 200
    res = client.post(f"/api/projects/{pid}/variants",
                      json={"targets": [{"width": 600, "height": 314, "name": "Wide"}]})
    assert res.status_code == 202, res.text
    service = client.app.state.service
    assert service.jobs.wait(res.json()["job"]["id"], timeout=300).status == "done"
    doc = client.get(f"/api/projects/{pid}").json()["document"]
    proposed = [c for c in doc["constraints"]
                if "brand profile" in (c["provenance"]["notes"] or "")]
    kinds = {(c["type"], c["elements"][0]) for c in proposed}
    assert ("clear_space", "logo") in kinds and ("keep_visible", "logo") in kinds
    assert ("scale_range", "logo") in kinds
    assert ("min_text_size", "headline") in kinds and ("min_text_size", "cta") in kinds
    assert all(c["hard"] is False and c["provenance"]["origin"] == "generated" for c in proposed)
    clear = next(c for c in proposed if c["type"] == "clear_space")
    assert clear["params"]["ratio"] == 0.8
    # checks: white copy is off a strict palette, DejaVu is not the Montserrat headline font
    v = client.get(f"/api/projects/{pid}/variants").json()["variants"][0]
    detail = client.get(f"/api/projects/{pid}/variants/{v['id']}").json()
    checks = {(c["check_id"], c["subject_id"]): c for c in detail["quality"]["checks"]}
    assert checks[("brand_palette", "headline")]["status"] == "needs_review"
    assert "#ffffff" in checks[("brand_palette", "headline")]["message"]
    assert checks[("brand_font", "headline")]["status"] == "needs_review"
    assert checks[("brand_font", "cta")]["status"] == "pass"  # body font DejaVu allowed for CTA
    assert v["verdict"] in ("needs_review", "failed")
    # the document itself never stores the profile
    assert "brand_profile" not in (doc.get("metadata") or {})
    # an on-brand profile passes: white implied, DejaVu Sans everywhere
    relaxed = dict(PROFILE, colors={"primary": "#1f3b63"},
                   fonts={"headline": "DejaVu Sans", "body": "DejaVu Sans"})
    assert client.put("/api/brands/Acme", json=relaxed).status_code == 200
    res = client.post(f"/api/projects/{pid}/variants/{v['id']}/regenerate",
                      json={"keep_layout": False})
    assert service.jobs.wait(res.json()["job"]["id"], timeout=300).status == "done"
    detail = client.get(f"/api/projects/{pid}/variants/{v['id']}").json()
    brand_checks = [c for c in detail["quality"]["checks"] if c["check_id"].startswith("brand_")]
    assert brand_checks and all(c["status"] == "pass" for c in brand_checks)
    # no profile: no brand checks at all
    client.delete("/api/brands/Acme")
    res = client.post(f"/api/projects/{pid}/variants/{v['id']}/regenerate",
                      json={"keep_layout": False})
    assert service.jobs.wait(res.json()["job"]["id"], timeout=300).status == "done"
    detail = client.get(f"/api/projects/{pid}/variants/{v['id']}").json()
    assert not [c for c in detail["quality"]["checks"] if c["check_id"].startswith("brand_")]
