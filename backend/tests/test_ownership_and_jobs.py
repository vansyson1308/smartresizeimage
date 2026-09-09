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


def test_pilot_event_log_measures_acceptance(tmp_path: Path) -> None:
    from backend.app.api.events import EventLog

    log = EventLog(tmp_path / "events")
    log.record("o", "variant_done", project_id="p", variant_id="v1", verdict="accepted")
    log.record("o", "variant_done", project_id="p", variant_id="v2", verdict="needs_review")
    log.record("o", "approval", project_id="p", variant_id="v1", approval="approved", reason="")
    log.record("o", "variant_done", project_id="p", variant_id="v2", verdict="accepted")  # regen
    log.record("o", "approval", project_id="p", variant_id="v2", approval="approved", reason="")
    log.record("o", "document_ops", project_id="p", count=3, kinds=["set_text"])
    s = log.summary("o")
    assert s["variants_generated"] == 2 and s["variants_decided"] == 2
    assert s["approved"] == 2
    # v2 was regenerated before approval -> not first-pass
    assert s["first_pass_acceptance"] == 0.5
    assert s["corrections_per_project"] == {"p": 3}
    assert s["median_seconds_to_decision"] is not None
    off = EventLog(tmp_path / "events2", enabled=False)
    off.record("o", "approval", approval="approved")
    assert off.summary("o")["events"] == 0


def test_pilot_summary_endpoint_reflects_journey(two_owner_client: TestClient) -> None:
    c = two_owner_client
    res = c.post("/api/projects/blank", json={"name": "P", "width": 300, "height": 300}, headers=A)
    pid = res.json()["project"]["id"]
    res = c.post(
        f"/api/projects/{pid}/variants",
        json={"targets": [{"width": 300, "height": 250}]},
        headers=A,
    )
    c.app.state.service.jobs.wait(res.json()["job"]["id"], timeout=120)
    vid = c.get(f"/api/projects/{pid}/variants", headers=A).json()["variants"][0]["id"]
    c.post(f"/api/projects/{pid}/variants/{vid}/approval", json={"approval": "approved"}, headers=A)
    s = c.get("/api/pilot/summary", headers=A).json()
    assert s["variants_generated"] == 1 and s["approved"] == 1
    assert s["first_pass_acceptance"] == 1.0
    assert c.get("/api/pilot/summary", headers=B).json()["events"] == 0


# ---- roles, rate limiting, retention ----------------------------------------------------


@pytest.fixture()
def role_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "AUTOBANNER_API_KEYS",
        "k-admin:acme:admin, k-editor:acme:editor, k-approver:acme:approver, k-viewer:acme:viewer",
    )
    monkeypatch.delenv("AUTOBANNER_API_KEY", raising=False)
    monkeypatch.delenv("AUTOBANNER_RATE_LIMIT", raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


ADMIN = {"X-API-Key": "k-admin"}
EDITOR = {"X-API-Key": "k-editor"}
APPROVER = {"X-API-Key": "k-approver"}
VIEWER = {"X-API-Key": "k-viewer"}


def test_roles_within_an_owner(role_client: TestClient) -> None:
    c = role_client
    assert c.get("/api/health").json()["roles"] == ["admin", "approver", "editor", "viewer"]
    # viewer: read only
    assert (
        c.post(
            "/api/projects/blank", json={"name": "V", "width": 600, "height": 300}, headers=VIEWER
        ).status_code
        == 403
    )
    # editor creates and edits, approver and viewer see the same project (same owner)
    res = c.post(
        "/api/projects/blank", json={"name": "E", "width": 600, "height": 300}, headers=EDITOR
    )
    assert res.status_code == 201, res.text
    pid = res.json()["project"]["id"]
    assert c.get(f"/api/projects/{pid}", headers=VIEWER).status_code == 200
    assert c.get(f"/api/projects/{pid}", headers=APPROVER).status_code == 200
    ops = {"ops": [{"op": "rename", "name": "renamed"}]}
    assert c.patch(f"/api/projects/{pid}/document", json=ops, headers=VIEWER).status_code == 403
    assert c.patch(f"/api/projects/{pid}/document", json=ops, headers=APPROVER).status_code == 403
    assert c.patch(f"/api/projects/{pid}/document", json=ops, headers=EDITOR).status_code == 200
    # generation is an edit
    targets = {"targets": [{"width": 300, "height": 250, "name": "M"}]}
    assert (
        c.post(f"/api/projects/{pid}/variants", json=targets, headers=APPROVER).status_code == 403
    )
    res = c.post(f"/api/projects/{pid}/variants", json=targets, headers=EDITOR)
    assert res.status_code == 202, res.text
    job = c.app.state.service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job.status == "done"
    vid = c.get(f"/api/projects/{pid}/variants", headers=VIEWER).json()["variants"][0]["id"]
    # approvals: approver or admin only
    body = {"approval": "approved"}
    assert (
        c.post(
            f"/api/projects/{pid}/variants/{vid}/approval", json=body, headers=EDITOR
        ).status_code
        == 403
    )
    assert (
        c.post(
            f"/api/projects/{pid}/variants/{vid}/approval", json=body, headers=VIEWER
        ).status_code
        == 403
    )
    assert (
        c.post(
            f"/api/projects/{pid}/variants/{vid}/approval", json=body, headers=APPROVER
        ).status_code
        == 200
    )
    assert (
        c.post(
            f"/api/projects/{pid}/variants/{vid}/approval", json={"approval": "none"}, headers=ADMIN
        ).status_code
        == 200
    )
    # deletion is an edit; a bad key is still 401
    assert c.delete(f"/api/projects/{pid}", headers=VIEWER).status_code == 403
    assert c.delete(f"/api/projects/{pid}", headers={"X-API-Key": "nope"}).status_code == 401
    assert c.delete(f"/api/projects/{pid}", headers=EDITOR).status_code == 204


def test_unknown_role_is_rejected_at_startup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_API_KEYS", "k:acme:owner")
    with pytest.raises(ValueError):
        create_app(tmp_path / "data", max_workers=1)


def test_rate_limit_returns_429_with_retry_after(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_API_KEYS", "key-a:owner-a, key-b:owner-b")
    monkeypatch.setenv("AUTOBANNER_RATE_LIMIT", "2/minute")
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["rate_limit"] == "2/60s"
        body = {"name": "R", "width": 600, "height": 300}
        assert c.post("/api/projects/blank", json=body, headers=A).status_code == 201
        assert c.post("/api/projects/blank", json=body, headers=A).status_code == 201
        res = c.post("/api/projects/blank", json=body, headers=A)
        assert res.status_code == 429 and int(res.headers["Retry-After"]) >= 1
        # reads are never limited; other owners have their own bucket
        assert c.get("/api/projects", headers=A).status_code == 200
        assert c.post("/api/projects/blank", json=body, headers=B).status_code == 201
        events = app.state.service.events.read("owner-a")
        assert any(e["kind"] == "rate_limited" for e in events)
    app.state.service.shutdown()


def test_rate_parser_and_bucket() -> None:
    from backend.app.api.ratelimit import RateLimiter, parse_rate

    assert parse_rate(None) is None and parse_rate("") is None
    assert parse_rate("60/minute").per_seconds == 60 and parse_rate("600/hour").count == 600
    assert parse_rate("5/10s").per_seconds == 10
    with pytest.raises(ValueError):
        parse_rate("fast")
    now = [0.0]
    rl = RateLimiter(parse_rate("2/second"), clock=lambda: now[0])
    assert rl.allow("o") == (True, 0) and rl.allow("o") == (True, 0)
    allowed, retry = rl.allow("o")
    assert not allowed and retry >= 1
    now[0] += 1.0
    assert rl.allow("o")[0]


def test_retention_purges_untouched_projects(tmp_path: Path) -> None:
    service = ProjectService(tmp_path / "data", max_workers=1)
    try:
        old = service.create_blank_project(name="old", width=600, height=300, owner="local")
        fresh = service.create_blank_project(name="fresh", width=600, height=300, owner="local")
        # backdate the old project's last touch
        meta_path = service.projects_dir / old.id / "project.json"
        payload = json.loads(meta_path.read_text())
        payload["updated_at"] = "2020-01-01T00:00:00+00:00"
        meta_path.write_text(json.dumps(payload))
        service._cache.pop(old.id, None)
        deleted = service.purge_stale_projects(30)
        assert deleted == [old.id]
        assert not (service.projects_dir / old.id).exists()
        assert (service.projects_dir / fresh.id).exists()
        assert service.purge_stale_projects(0) == []
        events = service.events.read("local")
        assert any(e["kind"] == "project_purged" and e["project_id"] == old.id for e in events)
    finally:
        service.shutdown()
