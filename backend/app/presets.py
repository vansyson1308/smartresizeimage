"""Catalog of commercial output sizes (ad networks, social platforms, IAB).

Each :class:`SizePreset` carries the pixel size plus the platform constraints a
media buyer cares about: the maximum file weight accepted by the network and
the *safe zone* (insets that platform UI chrome may cover, e.g. the profile
header and reply bar on a Story). Presets are grouped into *packs* so a single
request can fan a master design out to every size a campaign needs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .exceptions import ValidationError


@dataclass(frozen=True)
class SafeZone:
    """Fractional insets (0..1) of the canvas that may be covered by platform UI."""

    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    left: float = 0.0

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the safe rectangle (x1, y1, x2, y2) in pixels."""
        return (
            int(round(width * self.left)),
            int(round(height * self.top)),
            int(round(width * (1.0 - self.right))),
            int(round(height * (1.0 - self.bottom))),
        )

    @property
    def is_empty(self) -> bool:
        return not any((self.top, self.right, self.bottom, self.left))


@dataclass(frozen=True)
class SizePreset:
    """One named output size."""

    id: str
    name: str
    width: int
    height: int
    platform: str
    max_kb: int | None = None
    safe_zone: SafeZone = field(default_factory=SafeZone)
    formats: tuple[str, ...] = ("png", "jpeg", "webp")

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def orientation(self) -> str:
        if self.width == self.height:
            return "square"
        return "landscape" if self.width > self.height else "portrait"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["orientation"] = self.orientation
        data["formats"] = list(self.formats)
        return data


_STORY_SAFE = SafeZone(top=0.14, bottom=0.20)
_REELS_SAFE = SafeZone(top=0.14, bottom=0.35, right=0.10)
_TIKTOK_SAFE = SafeZone(top=0.08, bottom=0.25, right=0.12, left=0.06)
_YT_BANNER_SAFE = SafeZone(top=0.3643, bottom=0.3643, left=0.2857, right=0.2857)
_GDN_KB = 150

_PRESETS: tuple[SizePreset, ...] = (
    # --- Google Display Network / IAB standard ad units -----------------------
    SizePreset("iab-medium-rectangle", "Medium Rectangle", 300, 250, "iab", _GDN_KB),
    SizePreset("iab-large-rectangle", "Large Rectangle", 336, 280, "iab", _GDN_KB),
    SizePreset("iab-leaderboard", "Leaderboard", 728, 90, "iab", _GDN_KB),
    SizePreset("iab-large-leaderboard", "Large Leaderboard", 970, 90, "iab", _GDN_KB),
    SizePreset("iab-billboard", "Billboard", 970, 250, "iab", _GDN_KB),
    SizePreset("iab-half-page", "Half Page", 300, 600, "iab", _GDN_KB),
    SizePreset("iab-wide-skyscraper", "Wide Skyscraper", 160, 600, "iab", _GDN_KB),
    SizePreset("iab-skyscraper", "Skyscraper", 120, 600, "iab", _GDN_KB),
    SizePreset("iab-portrait", "Portrait", 300, 1050, "iab", _GDN_KB),
    SizePreset("iab-mobile-banner", "Mobile Banner", 320, 50, "iab", _GDN_KB),
    SizePreset("iab-large-mobile-banner", "Large Mobile Banner", 320, 100, "iab", _GDN_KB),
    SizePreset("iab-banner", "Banner", 468, 60, "iab", _GDN_KB),
    SizePreset("iab-square", "Square", 250, 250, "iab", _GDN_KB),
    SizePreset("iab-small-square", "Small Square", 200, 200, "iab", _GDN_KB),
    SizePreset("iab-mobile-interstitial", "Mobile Interstitial", 320, 480, "iab", _GDN_KB),
    # --- Google Ads responsive display assets --------------------------------
    SizePreset("google-rda-landscape", "Responsive Landscape", 1200, 628, "google", 5120),
    SizePreset("google-rda-square", "Responsive Square", 1200, 1200, "google", 5120),
    SizePreset("google-rda-portrait", "Responsive Portrait", 960, 1200, "google", 5120),
    # --- Meta (Facebook / Instagram) -------------------------------------------
    SizePreset("meta-feed-square", "Feed Square", 1080, 1080, "meta", 30720),
    SizePreset("meta-feed-portrait", "Feed Portrait 4:5", 1080, 1350, "meta", 30720),
    SizePreset("meta-feed-landscape", "Feed Landscape", 1200, 628, "meta", 30720),
    SizePreset("meta-story", "Story / Reels", 1080, 1920, "meta", 30720, _STORY_SAFE),
    SizePreset("meta-reels-cover", "Reels (full UI)", 1080, 1920, "meta", 30720, _REELS_SAFE),
    SizePreset("facebook-cover", "Facebook Page Cover", 1640, 624, "meta"),
    SizePreset("facebook-event", "Facebook Event Cover", 1920, 1005, "meta"),
    # --- TikTok -------------------------------------------------------------------
    SizePreset("tiktok-infeed", "TikTok In-Feed", 1080, 1920, "tiktok", None, _TIKTOK_SAFE),
    # --- LinkedIn ----------------------------------------------------------------
    SizePreset("linkedin-single-image", "LinkedIn Single Image", 1200, 627, "linkedin", 5120),
    SizePreset("linkedin-square", "LinkedIn Square", 1200, 1200, "linkedin", 5120),
    SizePreset("linkedin-vertical", "LinkedIn Vertical", 628, 1200, "linkedin", 5120),
    SizePreset("linkedin-company-cover", "LinkedIn Company Cover", 1128, 191, "linkedin"),
    # --- X (Twitter) -------------------------------------------------------------
    SizePreset("x-post-landscape", "X Post 16:9", 1600, 900, "x", 5120),
    SizePreset("x-post-square", "X Post Square", 1080, 1080, "x", 5120),
    SizePreset("x-header", "X Header", 1500, 500, "x", 2048),
    # --- YouTube / Pinterest -----------------------------------------------------
    SizePreset("youtube-thumbnail", "YouTube Thumbnail", 1280, 720, "youtube", 2048),
    SizePreset("youtube-banner", "YouTube Channel Art", 2560, 1440, "youtube", 6144,
               _YT_BANNER_SAFE),
    SizePreset("pinterest-standard", "Pinterest Standard Pin", 1000, 1500, "pinterest", 20480),
    SizePreset("pinterest-square", "Pinterest Square Pin", 1000, 1000, "pinterest", 20480),
    # --- Web / email ---------------------------------------------------------------
    SizePreset("web-hero", "Website Hero 16:9", 1920, 1080, "web"),
    SizePreset("web-og-image", "Open Graph Image", 1200, 630, "web", 1024),
    SizePreset("email-header", "Email Header", 600, 200, "email", 200),
)

PRESETS: dict[str, SizePreset] = {p.id: p for p in _PRESETS}

PACKS: dict[str, tuple[str, ...]] = {
    "google-display": (
        "iab-medium-rectangle",
        "iab-large-rectangle",
        "iab-leaderboard",
        "iab-half-page",
        "iab-wide-skyscraper",
        "iab-mobile-banner",
        "iab-large-mobile-banner",
        "iab-billboard",
    ),
    "google-responsive": (
        "google-rda-landscape",
        "google-rda-square",
        "google-rda-portrait",
    ),
    "meta-ads": (
        "meta-feed-square",
        "meta-feed-portrait",
        "meta-feed-landscape",
        "meta-story",
    ),
    "social-organic": (
        "meta-feed-square",
        "meta-story",
        "linkedin-single-image",
        "x-post-landscape",
        "pinterest-standard",
        "youtube-thumbnail",
    ),
    "covers": (
        "facebook-cover",
        "linkedin-company-cover",
        "x-header",
        "youtube-banner",
    ),
    "starter": (
        "meta-feed-landscape",
        "meta-feed-square",
        "meta-story",
    ),
}

PACK_DESCRIPTIONS: dict[str, str] = {
    "google-display": "Top-performing IAB units for the Google Display Network (150 KB cap)",
    "google-responsive": "Image assets for Google Ads responsive display ads",
    "meta-ads": "Facebook & Instagram paid placements (feed + stories)",
    "social-organic": "One post size per major social network",
    "covers": "Profile/page cover images",
    "starter": "Landscape, square and vertical - the three sizes every campaign needs",
}

_CUSTOM_RE = re.compile(r"^\s*(\d{1,5})\s*[xX×*]\s*(\d{1,5})\s*$")


def get_preset(preset_id: str) -> SizePreset:
    """Return a preset by id or raise :class:`ValidationError`."""
    try:
        return PRESETS[preset_id]
    except KeyError as e:
        raise ValidationError(f"Unknown preset '{preset_id}'") from e


def list_presets(platform: str | None = None) -> list[SizePreset]:
    """List presets, optionally filtered by platform."""
    return [p for p in _PRESETS if platform is None or p.platform == platform]


def expand_pack(pack_id: str) -> list[SizePreset]:
    """Return the presets of a pack (``all`` expands to every preset)."""
    if pack_id == "all":
        return list(_PRESETS)
    if pack_id not in PACKS:
        raise ValidationError(f"Unknown pack '{pack_id}'. Available: {', '.join(sorted(PACKS))}")
    return [PRESETS[pid] for pid in PACKS[pack_id]]


def parse_size(spec: str) -> tuple[int, int]:
    """Parse ``"1200x628"`` style strings."""
    match = _CUSTOM_RE.match(spec or "")
    if not match:
        raise ValidationError(f"Invalid size '{spec}'. Expected WIDTHxHEIGHT, e.g. 1200x628")
    return int(match.group(1)), int(match.group(2))


def custom_preset(width: int, height: int, name: str | None = None) -> SizePreset:
    """Build an ad-hoc preset for a user supplied size."""
    label = name or f"Custom {width}x{height}"
    return SizePreset(f"custom-{width}x{height}", label, int(width), int(height), "custom")


def resolve_targets(
    *,
    presets: list[str] | None = None,
    packs: list[str] | None = None,
    sizes: list[str] | None = None,
) -> list[SizePreset]:
    """Resolve preset ids, pack ids and ``WxH`` strings into unique presets.

    Order is preserved and duplicates (same id) are dropped.
    """
    resolved: list[SizePreset] = []
    seen: set[str] = set()

    def _add(p: SizePreset) -> None:
        if p.id not in seen:
            seen.add(p.id)
            resolved.append(p)

    for pack in packs or []:
        for p in expand_pack(pack):
            _add(p)
    for pid in presets or []:
        _add(get_preset(pid))
    for spec in sizes or []:
        w, h = parse_size(spec)
        _add(custom_preset(w, h))
    return resolved
