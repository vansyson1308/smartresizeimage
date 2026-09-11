"""Archive-controlled paths and metadata are untrusted (independent audit, finding 1).

An editable project archive is user-supplied data. Every path it carries (variant
images, reports, asset files) and every trust-bearing field (owner, approvals) must be
constrained at import and again at use, so a crafted archive can neither read nor
delete files outside its own project, nor manufacture approvals.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api.server import create_app
from backend.app.design.assets import AssetStore
from backend.app.design.document import DesignDocument
from backend.app.design.project import Project, safe_relative_path

A = {"X-API-Key": "k-admin"}
EDITOR = {"X-API-Key": "k-editor"}
APPROVER = {"X-API-Key": "k-approver"}
B = {"X-API-Key": "k-other"}


@pytest.fixture()
def clients(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "AUTOBANNER_API_KEYS",
        "k-admin:acme:admin, k-editor:acme:editor, k-approver:acme:approver, k-other:globex:admin",
    )
    monkeypatch.delenv("AUTOBANNER_API_KEY", raising=False)
    monkeypatch.delenv("AUTOBANNER_RATE_LIMIT", raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


def _archive_with(archive: bytes, edits: dict[str, bytes]) -> bytes:
    """Return ``archive`` with the named members replaced (or added)."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            if info.filename in edits:
                continue
            dst.writestr(info, src.read(info))
        for name, data in edits.items():
            dst.writestr(name, data)
    return out.getvalue()


def _blank_archive(c: TestClient, headers: dict) -> tuple[str, bytes]:
    res = c.post(
        "/api/projects/blank", json={"name": "Src", "width": 300, "height": 200}, headers=headers
    )
    assert res.status_code == 201, res.text
    pid = res.json()["project"]["id"]
    archive = c.get(f"/api/projects/{pid}/export/project", headers=headers).content
    return pid, archive


def _variant_index(**overrides) -> bytes:
    rec = {
        "id": "var_evil",
        "name": "Evil",
        "width": 300,
        "height": 200,
        "status": "done",
        "verdict": "accepted",
        "approval": "none",
        "approval_reason": "",
        "image_path": "variants/var_evil.png",
        "report_path": "variants/var_evil.json",
        "created_at": "2026-09-10T00:00:00+00:00",
        "updated_at": "2026-09-10T00:00:00+00:00",
        "brief": {},
        "document_version": 1,
    }
    rec.update(overrides)
    return json.dumps([rec]).encode()


# ---- safe_relative_path -------------------------------------------------------------------


def test_safe_relative_path_rejects_escapes(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "variants").mkdir(parents=True)
    inside = root / "variants" / "v.png"
    inside.write_bytes(b"x")
    assert safe_relative_path(root, "variants/v.png") == inside
    for bad in ("/etc/hostname", "../other/project.json", "variants/../../x", "", "C:\\x",
                "variants/\x00.png"):
        assert safe_relative_path(root, bad) is None, bad
    # an existing symlink inside the root that points outside is not followed
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"s")
    link = root / "variants" / "link.png"
    os.symlink(secret, link)
    assert safe_relative_path(root, "variants/link.png") is None


# ---- Project.load / use ---------------------------------------------------------------------


def test_project_load_neutralises_unsafe_variant_paths(tmp_path: Path) -> None:
    secret = tmp_path / "secret.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(secret)
    root = tmp_path / "proj"
    project = Project.create(root, DesignDocument(id="d", name="d", canvas_width=10,
                                                  canvas_height=10))
    (root / "variants" / "index.json").write_text(
        _variant_index(image_path=str(secret), report_path=str(tmp_path / "secret.json")).decode()
    )
    (tmp_path / "secret.json").write_text('{"leak": true}')
    loaded = Project.load(root)
    rec = loaded.variants["var_evil"]
    assert rec.image_path is None and rec.report_path is None
    assert rec.status == "failed" and "unsafe" in (rec.error or "")
    assert loaded.variant_image("var_evil") is None
    assert loaded.variant_detail("var_evil") is None
    assert loaded.delete_variant("var_evil") is True
    assert secret.exists() and (tmp_path / "secret.json").exists()
    assert project.root == root


def test_asset_store_rejects_paths_outside_the_store(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "assets")
    ref = store.put(Image.new("RGBA", (4, 4), (1, 2, 3, 255)), "logo")
    assert store.path(ref).exists()
    for bad in ("/etc/hostname", "../project.json", "sub/x.png", "..", ""):
        with pytest.raises(ValueError):
            store.path(bad)
        ref.path = bad
        with pytest.raises(ValueError):
            store.path(ref)
        assert not store.exists(ref)


# ---- import ---------------------------------------------------------------------------------


def test_import_rejects_absolute_and_traversal_paths_in_variant_index(
    clients: TestClient, tmp_path: Path
) -> None:
    c = clients
    secret = tmp_path / "secret.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(secret)
    _pid, archive = _blank_archive(c, A)
    # An archive whose index points at a file outside the project (absolute path)
    evil = _archive_with(archive, {"variants/index.json": _variant_index(image_path=str(secret))})
    res = c.post("/api/projects/import", files={"file": ("p.zip", evil, "application/zip")},
                 headers=EDITOR)
    assert res.status_code == 400, res.text
    assert "unsafe" in res.json()["detail"]
    assert secret.exists()
    # ... or at another project's files (traversal)
    other_pid, _ = _blank_archive(c, B)
    evil = _archive_with(
        archive,
        {"variants/index.json": _variant_index(
            report_path=f"../{other_pid}/project.json", image_path="../../secret.png")},
    )
    res = c.post("/api/projects/import", files={"file": ("p.zip", evil, "application/zip")},
                 headers=EDITOR)
    assert res.status_code == 400
    # no half-imported project is left behind
    ids = {p["id"] for p in c.get("/api/projects", headers=EDITOR).json()["projects"]}
    assert ids == {_pid}


def test_import_rejects_unsafe_asset_references(clients: TestClient, tmp_path: Path) -> None:
    c = clients
    _pid, archive = _blank_archive(c, A)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        payload = json.loads(zf.read("project.json"))
    payload["document"]["elements"].append({
        "id": "el_evil", "kind": "image", "name": "evil", "role": "logo",
        "geometry": {"x": 0, "y": 0, "width": 10, "height": 10},
        "asset": {"asset_id": "x", "content_hash": "0" * 64, "mime": "image/png",
                  "width": 8, "height": 8, "path": "../../../../etc/hostname"},
    })
    evil = _archive_with(archive, {"project.json": json.dumps(payload).encode()})
    res = c.post("/api/projects/import", files={"file": ("p.zip", evil, "application/zip")},
                 headers=EDITOR)
    assert res.status_code == 400, res.text
    assert "unsafe" in res.json()["detail"]


def test_import_rejects_archive_member_escapes_and_limits(clients: TestClient) -> None:
    c = clients
    _pid, archive = _blank_archive(c, A)
    for name in ("/abs.txt", "../escape.txt", "variants/../../x.txt"):
        evil = _archive_with(archive, {name: b"x"})
        res = c.post("/api/projects/import",
                     files={"file": ("p.zip", evil, "application/zip")}, headers=EDITOR)
        assert res.status_code == 400, name
    # a symlink member is refused rather than created
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info))
        link = zipfile.ZipInfo("variants/link.png")
        link.external_attr = (0o120777 << 16)
        dst.writestr(link, "/etc/hostname")
    res = c.post("/api/projects/import",
                 files={"file": ("p.zip", out.getvalue(), "application/zip")}, headers=EDITOR)
    assert res.status_code == 400
    # too many members
    many = _archive_with(archive, {f"history/junk_{i}.json": b"{}" for i in range(6000)})
    res = c.post("/api/projects/import",
                 files={"file": ("p.zip", many, "application/zip")}, headers=EDITOR)
    assert res.status_code == 400


def test_import_cannot_forge_owner_or_approvals(clients: TestClient) -> None:
    c = clients
    res = c.post("/api/projects/blank", json={"name": "Src", "width": 320, "height": 200},
                 headers=EDITOR)
    pid = res.json()["project"]["id"]
    res = c.post(f"/api/projects/{pid}/variants",
                 json={"targets": [{"width": 320, "height": 200, "name": "Same"}]},
                 headers=EDITOR)
    assert res.status_code == 202, res.text
    service = c.app.state.service
    job = service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job is not None and job.status == "done"
    vid = c.get(f"/api/projects/{pid}/variants", headers=EDITOR).json()["variants"][0]["id"]
    archive = c.get(f"/api/projects/{pid}/export/project", headers=EDITOR).content
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        payload = json.loads(zf.read("project.json"))
        index = json.loads(zf.read("variants/index.json"))
    payload["owner"] = "globex"
    payload["id"] = "proj_forged"
    index[0]["approval"] = "approved"
    index[0]["approval_reason"] = "forged"
    forged = _archive_with(archive, {
        "project.json": json.dumps(payload).encode(),
        "variants/index.json": json.dumps(index).encode(),
    })
    # an editor's import keeps the editor's workspace and drops the claimed approval
    res = c.post("/api/projects/import", files={"file": ("p.zip", forged, "application/zip")},
                 headers=EDITOR)
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["project"]["owner"] == "acme" and body["project"]["id"] != "proj_forged"
    assert c.get(f"/api/projects/{body['project']['id']}", headers=B).status_code == 404
    v = body["variants"][0]
    assert v["approval"] == "none" and "import" in v["approval_reason"]
    assert v["id"] == vid
    # an approver alone may not import (needs editor); an admin's import keeps approvals
    # it could have granted anyway
    res = c.post("/api/projects/import", files={"file": ("p.zip", forged, "application/zip")},
                 headers=APPROVER)
    assert res.status_code == 403
    res = c.post("/api/projects/import", files={"file": ("p.zip", forged, "application/zip")},
                 headers=A)
    assert res.status_code == 201, res.text
    assert res.json()["variants"][0]["approval"] == "approved"


def test_variant_image_endpoint_never_serves_outside_the_project(
    clients: TestClient, tmp_path: Path
) -> None:
    """Defence in depth: even a record that reached disk with a bad path is not served."""
    c = clients
    res = c.post("/api/projects/blank", json={"name": "Src", "width": 300, "height": 200},
                 headers=A)
    pid = res.json()["project"]["id"]
    service = c.app.state.service
    project = service.get_project(pid, "acme")
    secret = tmp_path / "secret.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(secret)
    (project.root / "variants" / "index.json").write_text(
        _variant_index(image_path=str(secret), report_path=str(secret)).decode()
    )
    service._cache.pop(pid, None)
    assert c.get(f"/api/projects/{pid}/variants/var_evil/image.png", headers=A).status_code == 404
    assert c.get(f"/api/projects/{pid}/variants/var_evil", headers=A).status_code == 200
    assert c.delete(f"/api/projects/{pid}/variants/var_evil", headers=A).status_code == 204
    assert secret.exists()
