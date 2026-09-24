"""HTTP API tests (FastAPI TestClient)."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api import Settings, create_app  # noqa: E402


def _png_bytes(size=(400, 210), color=(30, 90, 170)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _upload(name="banner.png", data=None):
    return {"file": (name, data if data is not None else _png_bytes(), "image/png")}


@pytest.fixture
def client():
    app = create_app(Settings(rate_limit_per_minute=0, workers=2))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def secured_client():
    app = create_app(Settings(api_keys=("k-alpha", "k-beta"), rate_limit_per_minute=0))
    with TestClient(app) as c:
        yield c


def test_health_ready_and_request_id(client):
    r = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.headers["x-request-id"] == "abc123"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/readyz").json()["status"] == "ready"


def test_presets_catalog(client):
    body = client.get("/v1/presets").json()
    ids = {p["id"] for p in body["presets"]}
    assert "meta-story" in ids
    assert "google-display" in body["packs"]
    iab = client.get("/v1/presets", params={"platform": "iab"}).json()["presets"]
    assert iab and all(p["platform"] == "iab" for p in iab)


def test_public_config(client):
    body = client.get("/v1/config").json()
    assert body["auth_required"] is False
    assert "phase3" in body["modes"]


def test_analyze(client):
    r = client.post("/v1/analyze", files=_upload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 400 and body["source_type"] == "flat_image"


def test_sync_render_returns_zip_with_manifest(client):
    options = {"presets": ["iab-medium-rectangle"], "sizes": ["320x480"], "format": "jpeg"}
    r = client.post("/v1/render", files=_upload(), data={"options": json.dumps(options)})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["x-autobanner-succeeded"] == "2"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    assert "iab-medium-rectangle_300x250.jpg" in names
    assert manifest["source"]["file"] == "banner.png"
    assert manifest["assets"][0]["export"]["max_kb"] == 150


def test_async_job_lifecycle(client):
    options = {"presets": ["iab-leaderboard", "meta-feed-square"], "format": "webp"}
    r = client.post("/v1/jobs", files=_upload(), data={"options": json.dumps(options)})
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]

    deadline = time.time() + 60
    body = {}
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert body["status"] == "succeeded", body
    assert body["progress"]["done"] == 2
    assert body["manifest"]["summary"]["succeeded"] == 2

    asset_url = next(iter(body["links"]["assets"].values()))
    asset = client.get(asset_url)
    assert asset.status_code == 200
    assert asset.headers["content-type"] == "image/webp"

    z = client.get(body["links"]["download"])
    assert z.status_code == 200
    assert zipfile.ZipFile(io.BytesIO(z.content)).testzip() is None

    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404


@pytest.mark.parametrize(
    ("options", "fragment"),
    [
        ("not json", "not valid JSON"),
        (json.dumps([1]), "JSON object"),
        (json.dumps({"presets": ["nope"]}), "Unknown preset"),
        (json.dumps({"sizes": ["12"]}), "Invalid size"),
        (json.dumps({"format": "gif"}), "format"),
        (json.dumps({"surprise": 1}), "surprise"),
        (json.dumps({}), "at least one"),
    ],
)
def test_bad_options_are_422(client, options, fragment):
    r = client.post("/v1/render", files=_upload(), data={"options": options})
    assert r.status_code == 422
    assert fragment in r.json()["error"]["message"]
    assert r.json()["error"]["request_id"]


def test_spoofed_file_rejected(client):
    r = client.post(
        "/v1/render",
        files=_upload("x.png", b"<svg xmlns='http://www.w3.org/2000/svg'/>"),
        data={"options": json.dumps({"sizes": ["100x100"]})},
    )
    assert r.status_code == 422
    assert "does not match" in r.json()["error"]["message"]


def test_unsupported_extension_is_415(client):
    r = client.post("/v1/analyze", files=_upload("x.gif", b"GIF89a"))
    assert r.status_code == 415


def test_upload_size_limit_is_413():
    app = create_app(Settings(max_upload_mb=1, rate_limit_per_minute=0))
    with TestClient(app) as c:
        r = c.post("/v1/analyze", files=_upload(data=b"\x89PNG\r\n\x1a\n" + b"0" * (1 << 21)))
    assert r.status_code == 413


def test_auth_required_and_job_isolation(secured_client):
    c = secured_client
    assert c.get("/v1/config").json()["auth_required"] is True
    assert c.get("/healthz").status_code == 200
    assert c.post("/v1/analyze", files=_upload()).status_code == 401
    bad = c.post("/v1/analyze", files=_upload(), headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401
    assert c.get("/metrics").status_code == 401

    ok = c.post(
        "/v1/jobs", files=_upload(), data={"options": json.dumps({"sizes": ["100x100"]})},
        headers={"Authorization": "Bearer k-alpha"},
    )
    assert ok.status_code == 202
    job_id = ok.json()["id"]
    other = c.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "k-beta"})
    assert other.status_code == 404
    mine = c.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "k-alpha"})
    assert mine.status_code == 200


def test_rate_limit_returns_429_with_retry_after():
    app = create_app(Settings(rate_limit_per_minute=2))
    with TestClient(app) as c:
        codes = [c.post("/v1/analyze", files=_upload()).status_code for _ in range(3)]
        assert codes[:2] == [200, 200]
        assert codes[2] == 429
        r = c.post("/v1/analyze", files=_upload())
        assert int(r.headers["retry-after"]) >= 1


def test_metrics_exposition(client):
    client.get("/healthz")
    text = client.get("/metrics").text
    assert "autobanner_http_requests_total" in text
    assert "# TYPE autobanner_http_request_duration_seconds histogram" in text


def test_studio_served_with_csp(client):
    r = client.get("/")
    if r.status_code == 404:
        pytest.skip("studio assets not bundled")
    assert "content-security-policy" in r.headers
    assert "AutoBanner" in r.text


# --- hardening regressions ---------------------------------------------------------------


def test_oversized_content_length_rejected_before_body_is_read():
    app = create_app(Settings(max_upload_mb=1, rate_limit_per_minute=0))
    with TestClient(app) as c:
        r = c.post(
            "/v1/analyze",
            content=b"x" * 16,
            headers={"content-length": str(50 * 1024 * 1024),
                     "content-type": "multipart/form-data; boundary=x"},
        )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_streamed_body_over_limit_is_413():
    app = create_app(Settings(max_upload_mb=1, rate_limit_per_minute=0))

    def chunks():
        for _ in range(40):
            yield b"0" * 100_000

    with TestClient(app) as c:
        r = c.post(
            "/v1/analyze", content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=x"},
        )
    assert r.status_code == 413


def test_auth_checked_before_body(secured_client):
    r = secured_client.post(
        "/v1/render",
        content=b"x",
        headers={"content-length": str(10 * 1024 ** 3),
                 "content-type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 401
    assert r.headers["x-request-id"]


def test_create_app_does_not_mutate_global_config():
    from backend.app.config import Config

    before = Config.MAX_UPLOAD_BYTES
    create_app(Settings(max_upload_mb=1))
    assert before == Config.MAX_UPLOAD_BYTES


def test_unhandled_error_keeps_request_id_and_headers():
    app = create_app(Settings(rate_limit_per_minute=0))

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom", headers={"X-Request-ID": "rid-500"})
        assert r.status_code == 500
        assert r.headers["x-request-id"] == "rid-500"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.json()["error"]["code"] == "internal_error"
        assert "kaboom" not in r.text
        assert 'status="500"' in c.get("/metrics").text


def test_unknown_route_uses_uniform_error_shape(client):
    r = client.get("/v1/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_job_results_are_served_from_disk_not_memory(client):
    r = client.post("/v1/jobs", files=_upload(),
                    data={"options": json.dumps({"sizes": ["120x120"]})})
    job_id = r.json()["id"]
    jobs = client.app.state.jobs
    deadline = time.time() + 60
    while time.time() < deadline and jobs.get(job_id, None).status not in ("succeeded", "failed"):
        time.sleep(0.05)
    job = jobs.get(job_id, None)
    assert job.status == "succeeded"
    assert not hasattr(job, "report")
    assert job.zip_path.is_file()
    (path, mime), = job.asset_files.values()
    assert path.is_file() and mime == "image/png"
    assert client.get(f"/v1/jobs/{job_id}/download").status_code == 200


def test_auto_layers_option_and_analyze_query(client):
    from backend.tools.flat_banner_samples import make_sample

    buf = io.BytesIO()
    make_sample(0).image.save(buf, "PNG")
    data = buf.getvalue()
    info = client.post(
        "/v1/analyze", params={"auto_layers": "true"}, files=_upload(data=data)
    ).json()
    assert info["source_type"] == "auto_layers"
    assert {"headline", "cta"} <= {e["role"] for e in info["elements"]}

    r = client.post(
        "/v1/render", files=_upload(data=data),
        data={"options": json.dumps({"presets": ["iab-leaderboard"], "auto_layers": True})},
    )
    assert r.status_code == 200
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("manifest.json"))
    assert manifest["source"]["source_type"] == "auto_layers"


def test_unwritable_data_dir_fails_fast(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")  # mkdir under a regular file fails even for root
    with pytest.raises(RuntimeError, match="AUTOBANNER_DATA_DIR"):
        create_app(Settings(data_dir=str(blocker / "data")))
