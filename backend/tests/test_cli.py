"""CLI tests."""

from __future__ import annotations

import json
import zipfile

import pytest
from PIL import Image

from backend.app.cli import main


@pytest.fixture
def two_inputs(tmp_path):
    paths = []
    for sub, color in (("a", (200, 30, 30)), ("b", (30, 200, 30))):
        d = tmp_path / sub
        d.mkdir()
        p = d / "input.png"
        Image.new("RGB", (300, 157), color).save(p)
        paths.append(str(p))
    return paths


def test_presets_json(capsys):
    assert main(["presets", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "meta-story" in {p["id"] for p in data["presets"]}


def test_presets_table(capsys):
    assert main(["presets", "--platform", "iab"]) == 0
    out = capsys.readouterr().out
    assert "iab-leaderboard" in out and "Packs:" not in out


def test_analyze(two_inputs, capsys):
    assert main(["analyze", two_inputs[0]]) == 0
    assert json.loads(capsys.readouterr().out)["width"] == 300


def test_render_multiple_inputs_with_same_stem(two_inputs, tmp_path):
    out = tmp_path / "out"
    code = main(["render", *two_inputs, "-s", "120x60", "-p", "iab-square",
                 "--format", "webp", "-o", str(out), "-q"])
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["input", "input_2"]
    for d in out.iterdir():
        files = sorted(f.name for f in d.iterdir())
        assert files == ["custom-120x60_120x60.webp", "iab-square_250x250.webp", "manifest.json"]
    red = Image.open(out / "input" / "custom-120x60_120x60.webp").convert("RGB").getpixel((60, 30))
    assert red[0] > red[1]


def test_render_zip(two_inputs, tmp_path):
    out = tmp_path / "zips"
    assert main(["render", two_inputs[0], "-s", "100x100", "--zip", "-o", str(out), "-q"]) == 0
    with zipfile.ZipFile(out / "input_autobanner.zip") as zf:
        assert "manifest.json" in zf.namelist()


def test_render_requires_targets(two_inputs, capsys):
    assert main(["render", two_inputs[0]]) == 2
    assert "at least one" in capsys.readouterr().err


def test_render_missing_file_is_partial_failure(tmp_path):
    assert main(["render", str(tmp_path / "nope.png"), "-s", "100x100", "-q"]) == 1


def test_bad_preset_is_usage_error(two_inputs, capsys):
    assert main(["render", two_inputs[0], "-p", "nope"]) == 2
    assert "Unknown preset" in capsys.readouterr().err
