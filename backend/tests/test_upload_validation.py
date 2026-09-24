"""Tests for untrusted-upload validation and anchor normalisation."""

from __future__ import annotations

import pytest
from PIL import Image

from backend.app.config import Config
from backend.app.exceptions import ValidationError
from backend.app.validators import (
    safe_filename,
    sniff_extension,
    validate_manual_anchors,
    validate_upload,
)


def _png(path, size=(64, 32)):
    Image.new("RGB", size, (1, 2, 3)).save(path, "PNG")
    return str(path)


def test_valid_png_passes(tmp_path):
    validate_upload(_png(tmp_path / "ok.png"))


def test_renamed_file_is_rejected(tmp_path):
    p = tmp_path / "evil.png"
    Image.new("RGB", (10, 10)).save(p, "JPEG")
    with pytest.raises(ValidationError, match="does not match"):
        validate_upload(str(p))


def test_jpeg_extension_alias(tmp_path):
    p = tmp_path / "photo.jpeg"
    Image.new("RGB", (10, 10)).save(p, "JPEG")
    validate_upload(str(p))


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(ValidationError, match="empty"):
        validate_upload(str(p))


def test_oversized_dimensions_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "MAX_SOURCE_DIMENSION", 50)
    with pytest.raises(ValidationError, match="exceeds"):
        validate_upload(_png(tmp_path / "big.png", (64, 32)))


def test_oversized_bytes_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValidationError, match="limit"):
        validate_upload(_png(tmp_path / "a.png"))


def test_psd_header_dimensions_checked(tmp_path, monkeypatch):
    header = b"8BPS" + b"\x00\x01" + b"\x00" * 6 + b"\x00\x03"
    header += (20000).to_bytes(4, "big") + (100).to_bytes(4, "big")
    p = tmp_path / "huge.psd"
    p.write_bytes(header + b"\x00" * 64)
    with pytest.raises(ValidationError):
        validate_upload(str(p))


def test_sniff_extension():
    assert sniff_extension(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == ".webp"
    assert sniff_extension(b"RIFF\x00\x00\x00\x00WAVEfmt ") is None
    assert sniff_extension(b"8BPS\x00\x01") == ".psd"
    assert sniff_extension(b"GIF89a") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "etc_passwd"),
        ("Instagram Story", "Instagram_Story"),
        ("", "output"),
        ("...", "output"),
        ("a/b\\c", "a_b_c"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_manual_anchors_are_clipped_and_sanitised():
    anchors = validate_manual_anchors(
        [{"id": "../x", "role": "logo", "x": -10, "y": 5, "width": 50, "height": 500}],
        (100, 100),
    )
    assert anchors == [
        {"id": "x", "role": "logo", "x": 0, "y": 5, "width": 40, "height": 95}
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {"not": "a list"},
        [1],
        [{"x": 0, "y": 0, "width": 10}],
        [{"x": 0, "y": 0, "width": 10, "height": 10, "role": "nope"}],
        [{"x": 500, "y": 500, "width": 10, "height": 10}],
    ],
)
def test_manual_anchors_reject_bad_input(bad):
    with pytest.raises(ValidationError):
        validate_manual_anchors(bad, (100, 100))


def test_duplicate_anchor_ids_are_made_unique():
    anchors = validate_manual_anchors(
        [
            {"id": "logo", "x": 0, "y": 0, "width": 10, "height": 10},
            {"id": "logo", "x": 20, "y": 0, "width": 10, "height": 10},
            {"id": "lo go", "x": 40, "y": 0, "width": 10, "height": 10},
            {"id": "lo_go", "x": 60, "y": 0, "width": 10, "height": 10},
        ],
        (100, 100),
    )
    assert [a["id"] for a in anchors] == ["logo", "logo_2", "lo_go", "lo_go_2"]


def test_explicit_upload_limit_overrides_config(tmp_path):
    path = _png(tmp_path / "a.png")
    with pytest.raises(ValidationError, match="limit"):
        validate_upload(path, max_bytes=10)
