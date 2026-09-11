"""Local user/session journey (independent audit, finding 4).

With authentication enabled the bundled UI must be able to sign in: first-run setup of
a workspace administrator (no external identity provider), login with an HttpOnly
session cookie that also authenticates images and downloads, roles surfaced per user,
personal API tokens for automation, and logout. Never solved by turning auth off.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.auth import LOCKOUT_FAILURES, UserStore
from backend.app.api.server import create_app

XRW = {"X-Requested-With": "fetch"}


@pytest.fixture()
def app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOBANNER_AUTH", "local")
    monkeypatch.setenv("AUTOBANNER_SETUP_TOKEN", "setup-secret")
    monkeypatch.setenv("AUTOBANNER_PBKDF2_ITERATIONS", "1000")
    monkeypatch.delenv("AUTOBANNER_API_KEYS", raising=False)
    monkeypatch.delenv("AUTOBANNER_API_KEY", raising=False)
    monkeypatch.delenv("AUTOBANNER_RATE_LIMIT", raising=False)
    application = create_app(tmp_path / "data", max_workers=1)
    yield application
    application.state.service.shutdown()


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c


def _setup(c: TestClient, username="ada", password="correct horse", workspace="acme"):
    res = c.post("/api/auth/setup", json={
        "username": username, "password": password, "workspace": workspace,
        "setup_token": "setup-secret",
    }, headers=XRW)
    assert res.status_code == 201, res.text
    return res


def test_setup_login_session_and_logout(client: TestClient) -> None:
    c = client
    status = c.get("/api/auth/status").json()
    assert status["mode"] == "local" and status["setup_required"] is True
    assert status["authenticated"] is False
    # anonymous requests are refused, not served as the local owner
    assert c.get("/api/projects").status_code == 401
    assert c.get("/api/health").json()["auth"] == "local"
    # setup needs the operator token and a real password
    res = c.post("/api/auth/setup", json={"username": "ada", "password": "correct horse",
                                          "setup_token": "nope"}, headers=XRW)
    assert res.status_code == 403
    res = c.post("/api/auth/setup", json={"username": "ada", "password": "short",
                                          "setup_token": "setup-secret"}, headers=XRW)
    assert res.status_code == 400
    res = _setup(c)
    assert res.json()["user"]["role"] == "admin"
    assert "ab_session" in res.cookies
    me = c.get("/api/auth/me").json()
    assert me["user"] == "ada" and me["role"] == "admin" and me["workspace"] == "acme"
    assert me["via"] == "session"
    # a second setup of the same workspace is refused; setup_required is now false
    assert c.post("/api/auth/setup", json={"username": "bob", "password": "correct horse",
                                           "workspace": "acme", "setup_token": "setup-secret"},
                  headers=XRW).status_code == 409
    assert c.get("/api/auth/status").json()["setup_required"] is False
    # the session authenticates ordinary work, images and downloads
    res = c.post("/api/projects/blank", json={"name": "P", "width": 300, "height": 200},
                 headers=XRW)
    assert res.status_code == 201, res.text
    pid = res.json()["project"]["id"]
    assert res.json()["project"]["owner"] == "acme"
    assert c.get(f"/api/projects/{pid}/preview.png").status_code == 200
    assert c.get(f"/api/projects/{pid}/export/project").status_code == 200
    # cookie-authenticated writes need the script header (CSRF protection); reads do not
    res = c.post("/api/projects/blank", json={"name": "P2", "width": 300, "height": 200})
    assert res.status_code == 403 and "CSRF" in res.json()["detail"]
    assert c.get("/api/projects").status_code == 200
    # logout ends the session
    assert c.post("/api/auth/logout", headers=XRW).status_code == 200
    assert c.get("/api/projects").status_code == 401
    # login: wrong password, then right
    res = c.post("/api/auth/login", json={"username": "ada", "password": "wrong"}, headers=XRW)
    assert res.status_code == 401
    res = c.post("/api/auth/login", json={"username": "ADA", "password": "correct horse"},
                 headers=XRW)
    assert res.status_code == 200 and "ab_session" in res.cookies
    assert c.get(f"/api/projects/{pid}").status_code == 200


def test_members_roles_and_tokens(client: TestClient) -> None:
    c = client
    _setup(c)
    for name, role in (("ed", "editor"), ("ap", "approver"), ("vi", "viewer")):
        res = c.post("/api/auth/users", json={"username": name, "password": "password-1",
                                              "role": role}, headers=XRW)
        assert res.status_code == 201, res.text
        assert res.json()["user"]["owner"] == "acme"
    assert c.post("/api/auth/users", json={"username": "x", "password": "password-1",
                                           "role": "king"}, headers=XRW).status_code == 400
    listed = {u["username"]: u["role"] for u in c.get("/api/auth/users").json()["users"]}
    assert listed == {"ada": "admin", "ed": "editor", "ap": "approver", "vi": "viewer"}
    # the last admin cannot be demoted or removed
    assert c.patch("/api/auth/users/ada", json={"role": "editor"}, headers=XRW).status_code == 409
    assert c.delete("/api/auth/users/ada", headers=XRW).status_code == 409
    assert c.post("/api/auth/logout", headers=XRW).status_code == 200

    def login(name: str) -> None:
        res = c.post("/api/auth/login", json={"username": name, "password": "password-1"},
                     headers=XRW)
        assert res.status_code == 200, res.text

    # viewer: read only, no member management
    login("vi")
    assert c.get("/api/auth/me").json()["role"] == "viewer"
    assert c.post("/api/projects/blank", json={"name": "V", "width": 300, "height": 200},
                  headers=XRW).status_code == 403
    assert c.get("/api/auth/users").status_code == 403
    c.post("/api/auth/logout", headers=XRW)
    # editor creates a project and a personal token for automation
    login("ed")
    res = c.post("/api/projects/blank", json={"name": "E", "width": 300, "height": 200},
                 headers=XRW)
    assert res.status_code == 201
    pid = res.json()["project"]["id"]
    res = c.post("/api/auth/tokens", json={"name": "ci"}, headers=XRW)
    assert res.status_code == 201
    raw, record = res.json()["token"], res.json()["record"]
    assert raw.startswith("abt_")
    c.post("/api/auth/logout", headers=XRW)
    assert c.get("/api/projects").status_code == 401
    # the token authenticates without a cookie and needs no CSRF header
    assert c.get("/api/projects", headers={"X-API-Key": raw}).json()["projects"][0]["id"] == pid
    res = c.post("/api/projects/blank", json={"name": "T", "width": 300, "height": 200},
                 headers={"X-API-Key": raw})
    assert res.status_code == 201
    assert c.get("/api/projects", headers={"X-API-Key": "abt_wrong"}).status_code == 401
    assert c.delete(f"/api/auth/tokens/{record['id']}",
                    headers={"X-API-Key": raw}).status_code == 204
    assert c.get("/api/projects", headers={"X-API-Key": raw}).status_code == 401
    # approver: may approve (checked before the 404 on a missing variant) but not edit
    login("ap")
    assert c.post(f"/api/projects/{pid}/variants/var_x/approval",
                  json={"approval": "approved"}, headers=XRW).status_code == 404
    assert c.patch(f"/api/projects/{pid}/document",
                   json={"ops": [{"op": "rename", "name": "x"}]}, headers=XRW).status_code == 403


def test_second_workspace_is_isolated(client: TestClient) -> None:
    c = client
    _setup(c)
    res = c.post("/api/projects/blank", json={"name": "A", "width": 300, "height": 200},
                 headers=XRW)
    pid = res.json()["project"]["id"]
    c.post("/api/auth/logout", headers=XRW)
    # the operator sets up a second workspace with the same setup token
    res = _setup(c, username="grace", workspace="globex")
    assert res.json()["user"]["owner"] == "globex"
    assert c.get("/api/projects").json()["projects"] == []
    assert c.get(f"/api/projects/{pid}").status_code == 404
    assert c.get(f"/api/projects/{pid}/preview.png").status_code == 404
    res = c.post("/api/auth/users", json={"username": "ada2", "password": "password-1",
                                          "role": "editor"}, headers=XRW)
    assert res.json()["user"]["owner"] == "globex"


def test_failed_logins_are_throttled(client: TestClient) -> None:
    c = client
    _setup(c)
    c.post("/api/auth/logout", headers=XRW)
    for _ in range(LOCKOUT_FAILURES):
        res = c.post("/api/auth/login", json={"username": "ada", "password": "wrong"},
                     headers=XRW)
        assert res.status_code == 401
    res = c.post("/api/auth/login", json={"username": "ada", "password": "correct horse"},
                 headers=XRW)
    assert res.status_code == 429


def test_static_keys_still_work_in_local_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_AUTH", "local")
    monkeypatch.setenv("AUTOBANNER_SETUP_TOKEN", "setup-secret")
    monkeypatch.setenv("AUTOBANNER_PBKDF2_ITERATIONS", "1000")
    monkeypatch.setenv("AUTOBANNER_API_KEYS", "k-ci:acme:editor")
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        res = c.post("/api/projects/blank", json={"name": "K", "width": 300, "height": 200},
                     headers={"X-API-Key": "k-ci"})
        assert res.status_code == 201 and res.json()["project"]["owner"] == "acme"
        assert c.get("/api/projects").status_code == 401
    app.state.service.shutdown()


def test_open_mode_is_explicit_and_becomes_local_once_a_user_exists(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("AUTOBANNER_AUTH", raising=False)
    monkeypatch.delenv("AUTOBANNER_API_KEYS", raising=False)
    monkeypatch.delenv("AUTOBANNER_API_KEY", raising=False)
    monkeypatch.setenv("AUTOBANNER_PBKDF2_ITERATIONS", "1000")
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        status = c.get("/api/auth/status").json()
        assert status["mode"] == "open" and status["authenticated"] is True
        assert status["workspace"] == "local" and status["role"] == "admin"
        assert c.post("/api/auth/login", json={}, headers=XRW).status_code == 409
    app.state.service.shutdown()
    # an operator who created users (e.g. with the store directly) gets local mode next start
    store = UserStore(tmp_path / "data" / "auth" / "users.json", iterations=1000)
    store.create_user("ada", "correct horse", "acme", "admin", created_by="cli")
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        assert c.get("/api/auth/status").json()["mode"] == "local"
        assert c.get("/api/projects").status_code == 401
    app.state.service.shutdown()


def test_store_keeps_only_digests(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.json", iterations=1000)
    store.create_user("ada", "correct horse", "acme", "admin", created_by="test")
    session = store.create_session("ada")
    raw, _rec = store.create_token("ada", "ci")
    text = (tmp_path / "users.json").read_text()
    assert session not in text and raw not in text and "correct horse" not in text
    assert store.resolve_session(session)["username"] == "ada"
    assert store.resolve_token(raw)["username"] == "ada"
    store.set_password("ada", "another password")
    assert store.resolve_session(session) is None  # sessions end on password change
    assert store.authenticate("ada", "another password")["username"] == "ada"
