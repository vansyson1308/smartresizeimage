"""End-to-end tests for the stateless render service."""

from __future__ import annotations

import io
import json
import threading
import zipfile

import pytest
from PIL import Image, ImageDraw

from backend.app.enums import ElementRole
from backend.app.exceptions import ValidationError
from backend.app.export import ExportOptions
from backend.app.presets import custom_preset, get_preset, resolve_targets
from backend.app.relayout import ReLayoutEngine
from backend.app.service import RenderRequest, RenderService, flat_banner_anchor_preset


@pytest.fixture
def banner(tmp_path) -> str:
    img = Image.new("RGB", (600, 314), (40, 90, 160))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 30, 300, 110), fill=(250, 250, 250))
    d.ellipse((360, 60, 560, 290), fill=(240, 90, 80))
    d.rectangle((40, 200, 200, 250), fill=(255, 200, 0))
    path = tmp_path / "banner.png"
    img.save(path)
    return str(path)


@pytest.fixture(scope="module")
def service() -> RenderService:
    return RenderService(use_ai=False)


def test_analyze_flat_image_keeps_background_role(service, banner):
    info = service.analyze(banner)
    assert info["source_type"] == "flat_image"
    assert info["width"] == 600 and info["height"] == 314
    assert info["elements"][0]["role"] == ElementRole.BACKGROUND.value


def test_render_multiple_sizes_with_manifest_and_zip(service, banner):
    request = RenderRequest(
        targets=[get_preset("iab-medium-rectangle"), custom_preset(320, 480, "Tall")],
        export=ExportOptions(format="webp"),
    )
    progress = []
    report = service.render(banner, request, progress=lambda i, t, n: progress.append((i, t)))

    assert [a.status for a in report.assets] == ["ok", "ok"]
    assert progress[-1] == (2, 2)
    for asset in report.assets:
        img = Image.open(io.BytesIO(asset.encoded.data))
        assert img.size == (asset.preset.width, asset.preset.height)
        assert asset.filename.endswith(".webp")

    manifest = report.manifest()
    assert manifest["summary"] == {"total": 2, "succeeded": 2, "failed": 0, "with_warnings": 0}
    assert len(manifest["source"]["sha256"]) == 64

    with zipfile.ZipFile(io.BytesIO(report.to_zip())) as zf:
        names = set(zf.namelist())
        assert names == {"iab-medium-rectangle_300x250.webp", "custom-320x480_320x480.webp",
                         "manifest.json"}
        assert json.loads(zf.read("manifest.json"))["mode"] == "phase21"


def test_platform_budget_applied_by_default(service, banner):
    report = service.render(banner, RenderRequest(targets=[get_preset("iab-leaderboard")]))
    export = report.assets[0].encoded
    assert export.max_kb == 150
    assert export.within_budget


def test_platform_budget_can_be_disabled(service, banner):
    request = RenderRequest(
        targets=[get_preset("iab-leaderboard")], respect_platform_limits=False
    )
    assert service.render(banner, request).assets[0].encoded.max_kb is None


def test_duplicate_targets_get_unique_filenames(service, banner):
    p = custom_preset(100, 100)
    report = service.render(banner, RenderRequest(targets=[p, p]))
    assert len({a.filename for a in report.assets}) == 2


def test_failure_of_one_size_does_not_abort_batch(service, banner, monkeypatch):
    original = ReLayoutEngine.relayout

    def flaky(self, size, safe_rect=None):
        if size == (123, 77):
            raise RuntimeError("boom")
        return original(self, size, safe_rect=safe_rect)

    monkeypatch.setattr(ReLayoutEngine, "relayout", flaky)
    report = service.render(
        banner, RenderRequest(targets=[custom_preset(123, 77), custom_preset(200, 200)])
    )
    assert [a.status for a in report.assets] == ["failed", "ok"]
    assert "RuntimeError" in report.assets[0].error
    assert report.manifest()["summary"]["failed"] == 1


def test_phase3_story_respects_safe_zone(service, banner):
    request = RenderRequest(
        targets=[get_preset("meta-story")], mode="phase3", anchor_preset="flat_banner_3anchors"
    )
    asset = service.render(banner, request).assets[0]
    assert asset.ok
    assert asset.qa["evaluated"]
    assert asset.qa["safe_zone"]["violations"] == []


def test_phase3_manual_anchors_are_not_stretched(banner):
    engine = ReLayoutEngine(use_ai=False)
    engine.load_file(banner)
    anchors = flat_banner_anchor_preset(engine.source_size)
    result = engine.relayout_redesign((300, 600), manual_anchors=anchors)
    for anchor, src in zip(result.metadata["redesign"]["anchors"], anchors, strict=True):
        b = anchor["bbox"]
        src_ratio = src["width"] / src["height"]
        assert abs(b["width"] / b["height"] - src_ratio) < 0.05


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"targets": []},
        {"targets": [custom_preset(100, 100)], "mode": "phase9"},
        {"targets": [custom_preset(100, 100)], "anchor_preset": "x"},
        {"targets": [custom_preset(100, 100)], "role_overrides": {"a": "wizard"}},
    ],
)
def test_invalid_requests(service, banner, request_kwargs):
    with pytest.raises(ValidationError):
        service.render(banner, RenderRequest(**request_kwargs))


def test_unknown_role_override_element(service, banner):
    request = RenderRequest(targets=[custom_preset(100, 100)], role_overrides={"x": "logo"})
    with pytest.raises(ValidationError, match="Unknown element"):
        service.render(banner, request)


def test_concurrent_renders_are_isolated(service, banner, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (200, 400), (0, 200, 0)).save(other)
    results = {}

    def run(path, key):
        report = service.render(path, RenderRequest(targets=resolve_targets(sizes=["150x150"])))
        results[key] = report.source["width"]

    threads = [
        threading.Thread(target=run, args=(banner, "a")),
        threading.Thread(target=run, args=(str(other), "b")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"a": 600, "b": 200}
