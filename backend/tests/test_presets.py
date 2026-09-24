"""Tests for the output size catalog."""

from __future__ import annotations

import pytest

from backend.app.config import Config
from backend.app.exceptions import ValidationError
from backend.app.presets import (
    PACKS,
    PRESETS,
    SafeZone,
    expand_pack,
    get_preset,
    list_presets,
    parse_size,
    resolve_targets,
)


def test_all_presets_are_renderable_sizes():
    for preset in PRESETS.values():
        assert Config.MIN_ELEMENT_SIZE <= preset.width <= Config.MAX_IMAGE_SIZE
        assert Config.MIN_ELEMENT_SIZE <= preset.height <= Config.MAX_IMAGE_SIZE
        assert preset.max_kb is None or preset.max_kb > 0


def test_packs_reference_existing_presets():
    for pack, ids in PACKS.items():
        assert ids, pack
        for pid in ids:
            assert pid in PRESETS, (pack, pid)


def test_google_display_pack_respects_150kb_cap():
    for preset in expand_pack("google-display"):
        assert preset.max_kb == 150


def test_expand_all_returns_every_preset():
    assert len(expand_pack("all")) == len(PRESETS)


def test_unknown_pack_and_preset_raise():
    with pytest.raises(ValidationError):
        expand_pack("nope")
    with pytest.raises(ValidationError):
        get_preset("nope")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("1200x628", (1200, 628)), (" 300 X 250 ", (300, 250)), ("1080×1920", (1080, 1920))],
)
def test_parse_size(spec, expected):
    assert parse_size(spec) == expected


@pytest.mark.parametrize("spec", ["", "1200", "axb", "12x", "-1x5"])
def test_parse_size_rejects_garbage(spec):
    with pytest.raises(ValidationError):
        parse_size(spec)


def test_resolve_targets_dedupes_and_keeps_order():
    targets = resolve_targets(
        packs=["starter"], presets=["meta-feed-square", "iab-leaderboard"], sizes=["640x480"]
    )
    ids = [t.id for t in targets]
    assert ids == [
        "meta-feed-landscape",
        "meta-feed-square",
        "meta-story",
        "iab-leaderboard",
        "custom-640x480",
    ]


def test_safe_zone_pixels():
    zone = SafeZone(top=0.1, bottom=0.2, left=0.05, right=0.05)
    assert zone.to_pixels(1000, 2000) == (50, 200, 950, 1600)
    assert SafeZone().is_empty
    assert not zone.is_empty


def test_orientation_and_platform_filter():
    assert get_preset("meta-story").orientation == "portrait"
    assert get_preset("iab-leaderboard").orientation == "landscape"
    assert get_preset("meta-feed-square").orientation == "square"
    assert all(p.platform == "iab" for p in list_presets("iab"))
    assert get_preset("meta-story").to_dict()["orientation"] == "portrait"
