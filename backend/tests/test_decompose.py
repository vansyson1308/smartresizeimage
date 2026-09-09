"""Tests for honest flat-image decomposition and the recovered-element journey."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from backend.app.api.server import create_app
from backend.app.design.decompose import (
    _guess_text_role,
    _merge_lines_into_blocks,
    decompose_flat_image,
)
from backend.app.quality.rendered import ocr_engine_status


def _font(name: str, size: int):
    try:
        return ImageFont.truetype(name, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def flat_banner() -> Image.Image:
    w, h = 1200, 628
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (int(40 + 60 * x / w), int(80 + 40 * y / h), int(140 + 50 * x / w))
    d = ImageDraw.Draw(img)
    d.ellipse((760, 110, 1140, 530), fill=(236, 88, 88))
    d.ellipse((880, 230, 1020, 410), fill=(255, 220, 120))
    d.text(
        (70, 60), "SUMMER SUPER SALE", fill=(255, 255, 255), font=_font("DejaVuSans-Bold.ttf", 64)
    )
    d.text(
        (70, 170),
        "Up to 50% off selected items",
        fill=(235, 240, 250),
        font=_font("DejaVuSans.ttf", 34),
    )
    d.text((70, 300), "SHOP NOW", fill=(255, 209, 102), font=_font("DejaVuSans-Bold.ttf", 36))
    d.text((70, 420), "$19.99", fill=(255, 255, 255), font=_font("DejaVuSans.ttf", 34))
    return img


def test_role_guess_uses_confident_lines_only() -> None:
    blocks = [
        {"text": "[ese |", "line_h": 125, "conf": 0.5, "lines": 1},
        {"text": "SUMMER SUPER SALE", "line_h": 57, "conf": 0.9, "lines": 1},
        {"text": "Up to 50% off selected items", "line_h": 41, "conf": 0.9, "lines": 1},
        {"text": "SHOP NOW", "line_h": 34, "conf": 0.9, "lines": 1},
        {"text": "$19.99", "line_h": 39, "conf": 0.9, "lines": 1},
    ]
    roles = [_guess_text_role(b, blocks) for b in blocks]
    assert roles == ["logo", "headline", "subheadline", "cta", "label"]


def test_merge_lines_into_blocks_groups_wrapped_lines() -> None:
    lines = [
        {"x": 70, "y": 60, "w": 500, "h": 40, "text": "MEGA CLEARANCE", "conf": 0.9},
        {"x": 70, "y": 108, "w": 480, "h": 40, "text": "WEEKEND EVENT", "conf": 0.9},
        {"x": 70, "y": 300, "w": 200, "h": 30, "text": "SHOP NOW", "conf": 0.9},
    ]
    blocks = _merge_lines_into_blocks(lines)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "MEGA CLEARANCE\nWEEKEND EVENT" and blocks[0]["lines"] == 2


def test_decompose_without_ocr_still_finds_subject_and_marks_inference() -> None:
    res = decompose_flat_image(flat_banner().convert("RGBA"), run_ocr=False)
    kinds = [e.kind for e in res.elements]
    assert "subject" in kinds
    subject = next(e for e in res.elements if e.kind == "subject")
    x, y, w, h = subject.bbox
    # the red circle sits at (760..1140, 110..530)
    assert x <= 780 and x + w >= 1120 and y <= 130 and y + h >= 510
    assert 0.3 <= subject.confidence <= 0.9
    assert subject.image.mode == "RGBA"
    assert res.background.size == (1200, 628)
    assert any("inferred" in n for n in res.notes)
    assert res.method["ocr"] == "disabled"


@pytest.mark.skipif(not ocr_engine_status()["available"], reason="tesseract not installed")
def test_decompose_recovers_text_blocks_with_roles_and_colours() -> None:
    res = decompose_flat_image(flat_banner().convert("RGBA"))
    texts = {e.text: e for e in res.elements if e.kind == "text"}
    assert "SUMMER SUPER SALE" in texts
    assert texts["SUMMER SUPER SALE"].role_guess == "headline"
    assert texts["SHOP NOW"].role_guess == "cta"
    assert texts["$19.99"].role_guess == "label"
    assert texts["SHOP NOW"].text_color and texts["SHOP NOW"].text_color.startswith("#")
    assert all(0 < e.confidence <= 0.9 for e in texts.values())
    # the recovered background must not still contain crisp headline glyphs
    bg = res.background.convert("L").crop((70, 60, 780, 130))
    assert bg.getextrema()[1] - bg.getextrema()[0] < 120


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(tmp_path / "data", max_workers=1)
    with TestClient(app) as c:
        yield c
    app.state.service.shutdown()


@pytest.mark.skipif(not ocr_engine_status()["available"], reason="tesseract not installed")
def test_flat_import_recovers_elements_and_convert_to_text_journey(client: TestClient) -> None:
    buf = io.BytesIO()
    flat_banner().save(buf, format="PNG")
    res = client.post("/api/projects", files={"file": ("flat.png", buf.getvalue(), "image/png")})
    assert res.status_code == 201, res.text
    payload = res.json()
    pid = payload["project"]["id"]
    doc = payload["document"]
    recovered = [e for e in doc["elements"] if e["provenance"]["origin"] == "recovered"]
    assert len(recovered) >= 4
    headline = next(
        e for e in doc["elements"] if e["effects"].get("recovered_text") == "SUMMER SUPER SALE"
    )
    assert headline["kind"] == "image" and headline["role"] == "headline"
    assert 0 < headline["role_confidence"] <= 0.9
    assert any("decomposed" in n for n in doc["metadata"]["import_notes"])

    # variants from the recovered document keep raster text raster
    res = client.post(
        f"/api/projects/{pid}/variants",
        json={"targets": [{"width": 1080, "height": 1080, "name": "sq"}]},
    )
    job = client.app.state.service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job.status == "done", job.to_dict()

    # convert the headline to native text after review
    res = client.patch(
        f"/api/projects/{pid}/document",
        json={
            "ops": [
                {"op": "convert_to_text", "element_id": headline["id"], "text": "SUMMER SUPER SALE"}
            ]
        },
    )
    assert res.status_code == 200, res.text
    converted = next(e for e in res.json()["document"]["elements"] if e["id"] == headline["id"])
    assert converted["kind"] == "text"
    assert converted["text"]["runs"][0]["text"] == "SUMMER SUPER SALE"
    assert converted["text"]["runs"][0]["style"]["color"].startswith("#")
    assert converted["provenance"]["origin"] == "user"
    assert "recovered_text" not in converted["effects"]
    res = client.post(
        f"/api/projects/{pid}/variants",
        json={
            "targets": [{"width": 1080, "height": 1920, "name": "story"}],
            "text_overrides": {headline["id"]: "WINTER SALE"},
        },
    )
    job = client.app.state.service.jobs.wait(res.json()["job"]["id"], timeout=300)
    assert job.status == "done", job.to_dict()
    story = next(
        v
        for v in client.get(f"/api/projects/{pid}/variants").json()["variants"]
        if v["name"] == "story"
    )
    detail = client.get(f"/api/projects/{pid}/variants/{story['id']}").json()
    assert headline["id"] in detail["plan"]["typography"]
