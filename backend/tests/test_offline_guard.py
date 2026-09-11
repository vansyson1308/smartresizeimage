"""AUTOBANNER_OFFLINE: the installed product works with outbound network refused (B5).

The guard refuses every non-loopback connection at the socket level, so a hidden
dependency on the network would surface as an error rather than a silent call. The
API journey (blank canvas, variants, export) runs under the guard; the browser journey
in ``tests/e2e`` runs its whole server with it as well.
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api import offline
from backend.app.api.server import create_app


@pytest.fixture()
def guarded_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOBANNER_OFFLINE", "1")
    for name in ("AUTOBANNER_AUTH", "AUTOBANNER_API_KEYS", "AUTOBANNER_RATE_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(tmp_path / "data", max_workers=1)
    try:
        yield app
    finally:
        app.state.service.shutdown()
        offline.uninstall_offline_guard()


def test_outbound_connections_are_refused_but_loopback_works(guarded_app) -> None:
    assert offline.offline_guard_installed()
    with pytest.raises(OSError, match="AUTOBANNER_OFFLINE"):
        socket.create_connection(("192.0.2.1", 80), timeout=1)  # TEST-NET, never routable
    with pytest.raises(OSError, match="AUTOBANNER_OFFLINE"):
        s = socket.socket()
        try:
            s.connect_ex(("example.com", 443))
        finally:
            s.close()
    # loopback is allowed: a local listener accepts a connection
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = []

    def accept():
        conn, _ = server.accept()
        accepted.append(conn)

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    client = socket.create_connection(("127.0.0.1", port), timeout=5)
    t.join(5)
    assert accepted
    client.close()
    accepted[0].close()
    server.close()


def test_api_journey_runs_offline(guarded_app) -> None:
    with TestClient(guarded_app) as c:
        assert c.get("/api/health").json()["offline"] is True
        res = c.post("/api/projects/blank", json={"name": "Off", "width": 400, "height": 300})
        assert res.status_code == 201, res.text
        pid = res.json()["project"]["id"]
        res = c.post(f"/api/projects/{pid}/elements", data={
            "kind": "text", "name": "Headline", "role": "headline", "x": "20", "y": "20",
            "width": "300", "height": "80", "text": "OFFLINE OK",
        })
        assert res.status_code in (200, 201), res.text
        res = c.post(f"/api/projects/{pid}/variants",
                     json={"targets": [{"width": 300, "height": 250, "name": "MREC"}]})
        assert res.status_code == 202, res.text
        job = guarded_app.state.service.jobs.wait(res.json()["job"]["id"], timeout=300)
        assert job.status == "done", job.to_dict()
        assert c.get(f"/api/projects/{pid}/export?format=png&only=all").status_code == 200


def test_uninstall_restores_sockets() -> None:
    offline.install_offline_guard()
    offline.uninstall_offline_guard()
    assert not offline.offline_guard_installed()
    assert socket.socket.connect.__qualname__ == "socket.connect"
