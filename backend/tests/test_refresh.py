"""Incremental refresh: after a design change, re-render only the variants it touched (C).

A variant is stale once the document version moves past the one it was rendered
from. Refreshing re-renders (keeping the layout, H3) the variants whose shown
elements, rules, fonts or canvas changed and marks the others current without a
render, so a campaign of many variants pays only for what the edit affected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.server import create_app
from backend.tests.test_api import _layered_project


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    for name in ("AUTOBANNER_AUTH", "AUTOBANNER_API_KEYS", "AUTOBANNER_RATE_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


def _wait(client: TestClient, job_id: str) -> None:
    job = client.app.state.service.jobs.wait(job_id, timeout=300)
    assert job.status == "done", job.to_dict()


def _variants(client: TestClient, pid: str) -> dict[str, dict]:
    return {v["name"]: v for v in client.get(f"/api/projects/{pid}/variants").json()["variants"]}


def test_refresh_re_renders_only_touched_variants(client: TestClient, tmp_path: Path) -> None:
    pid = _layered_project(client, tmp_path)["project"]["id"]
    # two variants: "Full" shows every element, "NoCTA" hides the CTA
    res = client.post(f"/api/projects/{pid}/variants", json={
        "targets": [{"width": 600, "height": 314, "name": "Full"}]})
    _wait(client, res.json()["job"]["id"])
    res = client.post(f"/api/projects/{pid}/variants", json={
        "targets": [{"width": 600, "height": 314, "name": "NoCTA"}], "hidden_elements": ["cta"]})
    _wait(client, res.json()["job"]["id"])
    before = _variants(client, pid)
    # nothing is stale yet
    res = client.post(f"/api/projects/{pid}/variants/refresh", json={})
    assert res.status_code == 202 and res.json()["job"] is None
    assert res.json()["refreshed"] == [] and res.json()["up_to_date"] == []
    assert set(res.json()["not_stale"]) == {before["Full"]["id"], before["NoCTA"]["id"]}
    # edit the CTA copy: only the variant that shows the CTA is affected
    res = client.patch(f"/api/projects/{pid}/document", json={"ops": [
        {"op": "set_text", "element_id": "cta", "text": "ORDER NOW"}]})
    assert res.status_code == 200
    version = res.json()["project"]["document_version"]
    res = client.post(f"/api/projects/{pid}/variants/refresh", json={"keep_layout": True})
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["refreshed"] == [before["Full"]["id"]]
    assert body["up_to_date"] == [before["NoCTA"]["id"]]
    assert "cta" in body["reasons"][before["Full"]["id"]]
    assert body["job"]["kind"] == "refresh" and body["job"]["meta"]["keep_layout"] is True
    _wait(client, body["job"]["id"])
    after = _variants(client, pid)
    assert after["Full"]["document_version"] == version
    assert after["NoCTA"]["document_version"] == version
    assert after["NoCTA"]["updated_at"] == before["NoCTA"]["updated_at"]  # not re-rendered
    assert after["Full"]["updated_at"] != before["Full"]["updated_at"]
    detail = client.get(f"/api/projects/{pid}/variants/{after['Full']['id']}").json()
    assert detail["plan"]["planner_meta"]["reference"]["kept"], "layout kept while the copy changed"
    # a rule change touches every variant
    res = client.patch(f"/api/projects/{pid}/document", json={"ops": [
        {"op": "add_constraint", "constraint": {"type": "clear_space", "elements": ["logo"],
                                                "params": {"ratio": 0.9}, "hard": False}}]})
    assert res.status_code == 200
    body = client.post(f"/api/projects/{pid}/variants/refresh").json()
    assert set(body["refreshed"]) == {after["Full"]["id"], after["NoCTA"]["id"]}
    assert all("rules" in r for r in body["reasons"].values())
    _wait(client, body["job"]["id"])
    # missing history counts as affected, never as silently current
    project = client.app.state.service.get_project(pid)
    for p in (project.root / "history").glob("*.json"):
        p.unlink()
    client.patch(f"/api/projects/{pid}/document", json={"ops": [
        {"op": "set_text", "element_id": "headline", "text": "AUTUMN SALE"}]})
    body = client.post(f"/api/projects/{pid}/variants/refresh").json()
    assert len(body["refreshed"]) == 2 and body["up_to_date"] == []
    assert all(r == "history unavailable" for r in body["reasons"].values())
    _wait(client, body["job"]["id"])
