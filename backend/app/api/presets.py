"""Channel size presets.

Every preset carries its provenance. These defaults are commonly used static
ad sizes; they are NOT an authoritative platform policy. Users can add or edit
presets per project and should verify against the channel's current spec
(`verified=False` until a user confirms a source).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

PRESET_SET_VERSION = "2026-09-09.1"


@dataclass(frozen=True)
class SizePreset:
    id: str
    name: str
    width: int
    height: int
    channel: str
    source: str = "AutoBanner default list; verify with the channel's current spec"
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PRESETS: list[SizePreset] = [
    SizePreset("ig_story", "Story / Reels (9:16)", 1080, 1920, "instagram"),
    SizePreset("ig_square", "Square post (1:1)", 1080, 1080, "instagram"),
    SizePreset("ig_portrait", "Portrait post (4:5)", 1080, 1350, "instagram"),
    SizePreset("fb_link", "Link / feed (1.91:1)", 1200, 628, "facebook"),
    SizePreset("li_post", "Feed post (1.91:1)", 1200, 627, "linkedin"),
    SizePreset("yt_thumb", "Thumbnail (16:9)", 1280, 720, "youtube"),
    SizePreset("pin_std", "Pin (2:3)", 1000, 1500, "pinterest"),
    SizePreset("x_header", "Header (3:1)", 1500, 500, "x"),
    SizePreset("gdn_mrec", "Medium rectangle", 300, 250, "display"),
    SizePreset("gdn_leader", "Leaderboard", 728, 90, "display"),
    SizePreset("gdn_skyscraper", "Wide skyscraper", 160, 600, "display"),
    SizePreset("gdn_half", "Half page", 300, 600, "display"),
]


def preset_catalog() -> dict:
    return {
        "version": PRESET_SET_VERSION,
        "presets": [p.to_dict() for p in DEFAULT_PRESETS],
        "note": (
            "Sizes are commonly used defaults, not platform policy. Each preset records its "
            "source; confirm with the channel's current specification before delivery."
        ),
    }


def find_preset(preset_id: str) -> SizePreset | None:
    for p in DEFAULT_PRESETS:
        if p.id == preset_id:
            return p
    return None
