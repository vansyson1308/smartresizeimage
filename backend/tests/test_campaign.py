"""Campaign table: every content row in every size, in one job (Mission V2, phase B).

A campaign is a table of messages (rows) times a set of formats. Rows come from a
pasted CSV/TSV mapped onto the design's text elements or from the API directly;
each variant records its row, cross-variant consistency is judged within a row,
and the deliverables zip keeps one folder per row.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.server import create_app
from backend.app.api.service import MAX_CAMPAIGN_ROWS, MAX_VARIANTS_PER_JOB, parse_campaign_table
from backend.tests.test_api import _layered_project


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AUTOBANNER_AUTH", raising=False)
    monkeypatch.delenv("AUTOBANNER_API_KEYS", raising=False)
    monkeypatch.delenv("AUTOBANNER_API_KEY", raising=False)
    monkeypatch.delenv("AUTOBANNER_RATE_LIMIT", raising=False)
    app = create_app(tmp_path / "data", max_workers=2)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


CSV = """label,Headline,CTA,locale,Price,Mood
Week 1,SUMMER SALE,SHOP NOW,en,$5,sunny
Tuần 2,GIẢM GIÁ HÈ,MUA NGAY,vi,,
Week 3,LAST DAYS,,en,$1,
"""


def test_parse_campaign_table_maps_columns_to_text_elements(client: TestClient, tmp_path):
    pid = _layered_project(client, tmp_path)["project"]["id"]
    res = client.post(f"/api/projects/{pid}/campaign/rows", json={"csv": CSV})
    assert res.status_code == 200, res.text
    data = res.json()
    rows = data["rows"]
    assert [r["label"] for r in rows] == ["Week 1", "Tuần 2", "Week 3"]
    assert rows[0]["text_overrides"] == {"headline": "SUMMER SALE", "cta": "SHOP NOW"}
    assert rows[0]["locale"] == "en" and rows[1]["locale"] == "vi"
    assert rows[1]["text_overrides"] == {"headline": "GIẢM GIÁ HÈ", "cta": "MUA NGAY"}
    assert rows[2]["text_overrides"] == {"headline": "LAST DAYS"}  # empty cell keeps master
    # the verbatim price column and the unknown column are reported, never applied
    assert "price" not in json.dumps(rows[0]["text_overrides"])
    assert data["ignored_columns"] == ["Mood"]
    assert any("Verbatim" in n for n in data["notes"]) and any("Mood" in n for n in data["notes"])
    kinds = {c["header"]: c["kind"] for c in data["columns"]}
    assert kinds == {"label": "label", "Headline": "text", "CTA": "text", "locale": "locale",
                     "Price": "protected", "Mood": "ignore"}
    # tab separated, ids by element id, duplicate row ids get a suffix
    tsv = "row\theadline\nA\tOne\nA\tTwo\n"
    rows = client.post(f"/api/projects/{pid}/campaign/rows", json={"csv": tsv}).json()["rows"]
    assert [r["id"] for r in rows] == ["A", "A_2"]
    assert rows[1]["text_overrides"] == {"headline": "Two"}
    parse = f"/api/projects/{pid}/campaign/rows"
    assert client.post(parse, json={"csv": "  "}).status_code == 400
    assert client.post(parse, json={"csv": "a,b\n"}).status_code == 400
    assert client.post(parse, json={"nope": 1}).status_code == 400


def test_parse_campaign_table_limits(client: TestClient, tmp_path):
    pid = _layered_project(client, tmp_path)["project"]["id"]
    doc = client.app.state.service.get_project(pid).document
    big = "headline\n" + "\n".join(f"copy {i}" for i in range(MAX_CAMPAIGN_ROWS + 1))
    try:
        parse_campaign_table(doc, big)
    except Exception as exc:  # noqa: BLE001
        assert "too many rows" in str(exc)
    else:
        raise AssertionError("row limit not enforced")


def test_rows_times_targets_generate_review_and_export(client: TestClient, tmp_path):
    pid = _layered_project(client, tmp_path)["project"]["id"]
    rows = [
        {"id": "w1", "label": "Week 1", "text_overrides": {"headline": "SUMMER SALE"}},
        {"id": "w2", "label": "Tuần 2", "text_overrides": {"headline": "GIẢM GIÁ HÈ",
                                                          "cta": "MUA NGAY"}, "locale": "vi"},
        {"id": "w3", "label": "Week 3", "text_overrides": {}},
    ]
    targets = [{"width": 600, "height": 314, "name": "Wide"},
               {"width": 300, "height": 300, "name": "Square"}]
    res = client.post(f"/api/projects/{pid}/variants",
                      json={"targets": targets, "rows": rows, "text_overrides": {"cta": "SHOP"}})
    assert res.status_code == 202, res.text
    job = res.json()["job"]
    assert len(job["items"]) == 6
    assert job["items"][0]["label"] == "Week 1 · Wide"
    service = client.app.state.service
    done = service.jobs.wait(job["id"], timeout=600)
    assert done.status == "done", done.to_dict()
    variants = client.get(f"/api/projects/{pid}/variants").json()["variants"]
    assert len(variants) == 6 and all(v["status"] == "done" for v in variants)
    by_row = {}
    for v in variants:
        by_row.setdefault(v["brief"]["row"]["id"], []).append(v)
    assert {k: len(vs) for k, vs in by_row.items()} == {"w1": 2, "w2": 2, "w3": 2}
    # row copy overrides the job default; job defaults apply where the row is silent
    w1 = by_row["w1"][0]
    assert w1["brief"]["text_overrides"] == {"cta": "SHOP", "headline": "SUMMER SALE"}
    w2 = next(v for v in by_row["w2"] if v["name"] == "Wide")
    assert w2["brief"]["text_overrides"]["cta"] == "MUA NGAY" and w2["brief"]["locale"] == "vi"
    assert by_row["w3"][0]["brief"]["text_overrides"] == {"cta": "SHOP"}
    detail = client.get(f"/api/projects/{pid}/variants/{w2['id']}").json()
    assert detail["plan"]["brief"]["row"] == {"id": "w2", "label": "Tuần 2"}
    # a regeneration keeps the row
    res = client.post(f"/api/projects/{pid}/variants/{w2['id']}/regenerate",
                      json={"text_overrides": {"headline": "SIÊU GIẢM GIÁ"}})
    assert res.status_code == 202
    service.jobs.wait(res.json()["job"]["id"], timeout=300)
    again = client.get(f"/api/projects/{pid}/variants/{w2['id']}").json()["variant"]
    assert again["brief"]["row"]["id"] == "w2"
    assert again["brief"]["text_overrides"]["headline"] == "SIÊU GIẢM GIÁ"
    # export: one folder per row, the row in the manifest
    res = client.get(f"/api/projects/{pid}/export?format=png&only=all")
    assert res.status_code == 200
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
    folders = {n.split("/")[0] for n in names if n.endswith(".png")}
    assert folders == {"Week_1", "Tuần_2", "Week_3"}  # letters of any script are kept
    assert {e["row"]["id"] for e in manifest["variants"]} == {"w1", "w2", "w3"}
    assert all(e["name"] in ("Wide", "Square") for e in manifest["variants"])


def test_rows_are_validated(client: TestClient, tmp_path):
    pid = _layered_project(client, tmp_path)["project"]["id"]
    target = [{"width": 300, "height": 300}]

    def post(rows, **extra):
        return client.post(f"/api/projects/{pid}/variants",
                           json={"targets": target, "rows": rows, **extra})

    assert post([{"text_overrides": {"nope": "x"}}]).status_code == 400
    assert post([{"id": "../x"}]).status_code == 400
    assert post([{"id": "a"}, {"id": "a"}]).status_code == 400
    assert post("not a list").status_code == 400
    assert post([{"hidden_elements": ["ghost"]}]).status_code == 400
    too_many = [{"id": f"r{i}"} for i in range(MAX_VARIANTS_PER_JOB + 1)]
    res = post(too_many)
    assert res.status_code == 400 and "too many" in res.json()["detail"]
    # without rows the request behaves as before (no row on the brief)
    res = client.post(f"/api/projects/{pid}/variants", json={"targets": target})
    assert res.status_code == 202
    client.app.state.service.jobs.wait(res.json()["job"]["id"], timeout=300)
    v = client.get(f"/api/projects/{pid}/variants").json()["variants"][0]
    assert v["brief"].get("row") is None
