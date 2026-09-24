"""Smoke tests for the README demo generator (fast paths only)."""

from __future__ import annotations

from PIL import Image

from backend.app.enums import ElementRole
from backend.tools.make_demo import Tile, board, coffee_master, tech_master, travel_master


def test_masters_are_layered_designs_with_key_roles():
    for master in (coffee_master(), tech_master(), travel_master()):
        roles = {e.role for e in master.elements()}
        assert {ElementRole.BACKGROUND, ElementRole.HEADLINE, ElementRole.CTA,
                ElementRole.LOGO, ElementRole.HERO_IMAGE} <= roles
        assert master.flat().size == (1200, 628)


def test_board_lays_out_tiles():
    tiles = [Tile(Image.new("RGB", (300, 250), "red"), "Medium Rectangle 300×250", "5 KB")]
    img = board("Title", "Subtitle", [(tiles, 1.0), (tiles * 3, 0.5)])
    assert img.width == 1600 and img.height > 300
