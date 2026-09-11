"""Plan entitlements per workspace (phase B4).

A workspace is on a plan (free / team / business / unlimited, or operator-defined
plans); the plan caps variants per day, projects, members, campaign rows per job and
storage. Limits are shown in ``GET /api/usage`` and refused with HTTP 402 and a message
naming the plan. Global ``AUTOBANNER_QUOTA_*`` caps still apply on top.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.server import create_app

XRW = {"X-Requested-With": "fetch"}
A = {"X-API-Key": "ka"}
B = {"X-API-Key": "kb"}


def _blank(c: TestClient, name: str, headers: dict):
    return c.post("/api/projects/blank", json={"name": name, "width": 300, "height": 200},
                  headers={**headers, **XRW})


@pytest.fixture()
def keyed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOBANNER_API_KEYS", "ka:acme:editor,kb:globex:editor")
    monkeypatch.setenv("AUTOBANNER_WORKSPACE_PLANS", "acme:free")
    monkeypatch.setenv("AUTOBANNER_PLANS", '{"free": {"variants_per_day": 4}}')
    for name in ("AUTOBANNER_AUTH", "AUTOBANNER_RATE_LIMIT", "AUTOBANNER_DEFAULT_PLAN",
                 "AUTOBANNER_QUOTA_VARIANTS_PER_DAY", "AUTOBANNER_QUOTA_PROJECTS"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


def test_plan_limits_are_shown_and_enforced(keyed: TestClient) -> None:
    c = keyed
    usage = c.get("/api/usage", headers=A).json()
    assert usage["plan"]["name"] == "free"
    assert usage["plan"]["limits"]["projects"] == 3
    assert usage["plan"]["limits"]["variants_per_day"] == 4  # AUTOBANNER_PLANS override
    assert usage["plan"]["remaining"]["projects"] == 3
    plans = c.get("/api/plans", headers=A).json()
    assert plans["current"] == "free" and "team" in plans["plans"]
    # projects: the fourth is refused with 402 naming the plan
    for i in range(3):
        assert _blank(c, f"P{i}", A).status_code == 201
    res = _blank(c, "P3", A)
    assert res.status_code == 402 and "plan 'free'" in res.json()["detail"]
    assert c.get("/api/usage", headers=A).json()["plan"]["remaining"]["projects"] == 0
    # the other workspace is on the default (unlimited) plan
    for i in range(4):
        assert _blank(c, f"G{i}", B).status_code == 201
    assert c.get("/api/usage", headers=B).json()["plan"]["name"] == "unlimited"
    # variants per day: 5 targets exceed the plan's 4 before any record is created
    pid = c.get("/api/projects", headers=A).json()["projects"][0]["id"]
    targets = [{"width": 300, "height": 200 + i} for i in range(5)]
    res = c.post(f"/api/projects/{pid}/variants", json={"targets": targets}, headers=A)
    assert res.status_code == 402 and "variants per day" in res.json()["detail"]
    assert c.get(f"/api/projects/{pid}/variants", headers=A).json()["variants"] == []
    # campaign rows per job (free: 3)
    rows = [{"id": f"r{i}"} for i in range(4)]
    res = c.post(f"/api/projects/{pid}/variants",
                 json={"targets": targets[:1], "rows": rows}, headers=A)
    assert res.status_code == 402 and "campaign rows" in res.json()["detail"]


def test_operator_assigns_plans_with_the_setup_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_AUTH", "local")
    monkeypatch.setenv("AUTOBANNER_SETUP_TOKEN", "setup-secret")
    monkeypatch.setenv("AUTOBANNER_PBKDF2_ITERATIONS", "1000")
    for name in ("AUTOBANNER_API_KEYS", "AUTOBANNER_RATE_LIMIT", "AUTOBANNER_WORKSPACE_PLANS",
                 "AUTOBANNER_PLANS", "AUTOBANNER_DEFAULT_PLAN"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        res = c.post("/api/auth/setup", json={"username": "ada", "password": "correct horse",
                                              "workspace": "acme", "setup_token": "setup-secret"},
                     headers=XRW)
        assert res.status_code == 201
        assert c.get("/api/usage").json()["plan"]["name"] == "unlimited"
        # a workspace admin cannot change the plan; the operator's setup token can
        assert c.put("/api/plans/acme", json={"plan": "free"}, headers=XRW).status_code == 403
        res = c.put("/api/plans/acme", json={"plan": "free"},
                    headers={**XRW, "X-Setup-Token": "setup-secret"})
        assert res.status_code == 200 and res.json()["limits"]["members"] == 2
        assert c.put("/api/plans/acme", json={"plan": "gold"},
                     headers={**XRW, "X-Setup-Token": "setup-secret"}).status_code == 400
        usage = c.get("/api/usage").json()
        assert usage["plan"]["name"] == "free" and usage["plan"]["used"]["members"] == 1
        # members: the free plan allows two
        res = c.post("/api/auth/users", json={"username": "bob", "password": "password-1",
                                              "role": "editor"}, headers=XRW)
        assert res.status_code == 201
        res = c.post("/api/auth/users", json={"username": "cid", "password": "password-1",
                                              "role": "viewer"}, headers=XRW)
        assert res.status_code == 402 and "members" in res.json()["detail"]
        assert c.get("/api/usage").json()["plan"]["remaining"]["members"] == 0
    app.state.service.shutdown()
    # the assignment is durable
    app = create_app(tmp_path / "data", max_workers=1)
    assert app.state.service.entitlements.plan_name("acme") == "free"
    app.state.service.shutdown()


def test_global_quota_still_applies_on_top_of_a_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_API_KEYS", "ka:acme:editor")
    monkeypatch.setenv("AUTOBANNER_WORKSPACE_PLANS", "acme:team")
    monkeypatch.setenv("AUTOBANNER_QUOTA_PROJECTS", "1")
    for name in ("AUTOBANNER_AUTH", "AUTOBANNER_RATE_LIMIT", "AUTOBANNER_PLANS"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        assert _blank(c, "P0", A).status_code == 201
        res = _blank(c, "P1", A)
        assert res.status_code == 429  # the global cap, not the plan, is the stricter one
    app.state.service.shutdown()
