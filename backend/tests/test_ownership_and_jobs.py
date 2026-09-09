"""Owner isolation, durable jobs, quotas and usage metering."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api.jobs import JobManager
from backend.app.api.server import create_app
from backend.app.api.service import ProjectService, Quota, ServiceError


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (400, 200), (30, 60, 90)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def two_owner_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOBANNER_API_KEYS", "key-a:owner-a, key-b:owner-b")
    monkeypatch.delenv("AUTOBANNER_API_KEY", raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


A = {"X-API-Key": "key-a"}
B = {"X-API-Key": "key-b"}


def test_projects_and_jobs_are_isolated_per_owner(two_owner_client: TestClient) -> None:
    c = two_owner_client
    assert c.get("/api/projects").status_code == 401
    res = c.post("/api/projects/blank", json={"name": "A1", "width": 600, "height": 300}, headers=A)
    assert res.status_code == 201
    pid = res.json()["project"]["id"]
    assert res.json()["project"]["owner"] == "owner-a"

    # owner B cannot see, read, mutate, export or delete A's project
    assert c.get("/api/projects", headers=B).json()["projects"] == []
    assert c.get(f"/api/projects/{pid}", headers=B).status_code == 404
    assert (
        c.patch(
            f"/api/projects/{pid}/document",
            json={"ops": [{"op": "rename", "name": "x"}]},
            headers=B,
        ).status_code
        == 404
    )
    assert c.get(f"/api/projects/{pid}/export", headers=B).status_code == 404
    assert c.delete(f"/api/projects/{pid}", headers=B).status_code == 404
    assert (
        c.post(
            f"/api/projects/{pid}/variants",
            json={"targets": [{"width": 300, "height": 250}]},
            headers=B,
        ).status_code
        == 404
    )

    # A's job is invisible to B
    res = c.post(
        f"/api/projects/{pid}/variants",
        json={"targets": [{"width": 300, "height": 250}]},
        headers=A,
    )
    assert res.status_code == 202
    job_id = res.json()["job"]["id"]
    assert c.get(f"/api/jobs/{job_id}", headers=B).status_code == 404
    assert c.post(f"/api/jobs/{job_id}/cancel", headers=B).status_code == 404
    assert c.get(f"/api/jobs/{job_id}", headers=A).status_code == 200
    c.app.state.service.jobs.wait(job_id, timeout=120)

    # A still sees its project; listing for A has exactly one project
    assert [p["id"] for p in c.get("/api/projects", headers=A).json()["projects"]] == [pid]
    # usage is per owner
    usage_a = c.get("/api/usage", headers=A).json()
    usage_b = c.get("/api/usage", headers=B).json()
    assert usage_a["totals"].get("variants") == 1 and usage_a["totals"].get("projects") == 1
    assert usage_b["totals"] == {}
    assert usage_a["storage_bytes"] > 0


def test_imported_project_belongs_to_importer(two_owner_client: TestClient) -> None:
    c = two_owner_client
    res = c.post("/api/projects/blank", json={"name": "A1", "width": 300, "height": 300}, headers=A)
    pid = res.json()["project"]["id"]
    archive = c.get(f"/api/projects/{pid}/export/project", headers=A).content
    res = c.post(
        "/api/projects/import", files={"file": ("p.zip", archive, "application/zip")}, headers=B
    )
    assert res.status_code == 201
    new_pid = res.json()["project"]["id"]
    assert res.json()["project"]["owner"] == "owner-b"
    assert c.get(f"/api/projects/{new_pid}", headers=A).status_code == 404


def test_durable_jobs_survive_restart_as_interrupted(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    manager = JobManager(max_workers=1, persist_dir=jobs_dir)

    def slow(job, item, progress):
        progress("render", 0.5)
        return {"ok": True}

    job = manager.submit("variants", "proj_x", [("v1", "V1")], slow, owner="owner-a")
    manager.wait(job.id, timeout=30)
    record = json.loads((jobs_dir / f"{job.id}.json").read_text())
    assert record["status"] == "done" and record["owner"] == "owner-a"

    # simulate a crash mid-run: rewrite the record as running and reload
    record["status"] = "running"
    record["items"][0]["status"] = "running"
    (jobs_dir / f"{job.id}.json").write_text(json.dumps(record))
    manager.shutdown()
    reloaded = JobManager(max_workers=1, persist_dir=jobs_dir)
    got = reloaded.get(job.id)
    assert got is not None and got.status == "interrupted"
    assert got.items[0].status == "interrupted"
    assert got.error and "interrupted" in got.error
    assert reloaded.cancel(job.id) is False
    on_disk = json.loads((jobs_dir / f"{job.id}.json").read_text())
    assert on_disk["status"] == "interrupted"
    reloaded.shutdown()


def test_quotas_block_over_use(tmp_path: Path) -> None:
    service = ProjectService(
        tmp_path / "data", max_workers=1, quota=Quota(variants_per_day=2, projects=1)
    )
    p = service.create_blank_project(name="one", width=300, height=300, owner="q")
    with pytest.raises(ServiceError) as exc:
        service.create_blank_project(name="two", width=300, height=300, owner="q")
    assert exc.value.status == 429
    job = service.request_variants(
        p.id, {"targets": [{"width": 300, "height": 250}, {"width": 300, "height": 300}]}, owner="q"
    )
    service.jobs.wait(job.id, timeout=120)
    with pytest.raises(ServiceError) as exc:
        service.request_variants(p.id, {"targets": [{"width": 300, "height": 250}]}, owner="q")
    assert exc.value.status == 429
    summary = service.usage_summary("q")
    assert summary["today"]["variants"] == 2 and summary["quota"]["variants_per_day"] == 2
    service.shutdown()


def test_open_mode_uses_local_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOBANNER_API_KEYS", raising=False)
    monkeypatch.delenv("AUTOBANNER_API_KEY", raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        res = c.post("/api/projects", files={"file": ("f.png", _png(), "image/png")})
        assert res.status_code == 201
        assert res.json()["project"]["owner"] == "local"
        assert c.get("/api/usage").json()["owner"] == "local"
        assert c.get("/api/health").json()["auth"] == "open"
    app.state.service.shutdown()


def test_upload_size_limit_is_enforced_before_buffering(tmp_path: Path) -> None:
    from backend.app.api import service as svc

    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        big = b"\x89PNG" + b"0" * (svc.MAX_UPLOAD_BYTES + 10)
        res = c.post("/api/projects", files={"file": ("big.png", big, "image/png")})
        assert res.status_code == 413
    app.state.service.shutdown()
