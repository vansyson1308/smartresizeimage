"""Layout templates for different aspect ratios."""

from __future__ import annotations

from ..enums import ElementRole

# Layout templates for different aspect ratios
TEMPLATES = {
    "portrait_tall": {  # 9:16 (Instagram Story)
        "aspect_range": (0.5, 0.65),
        "zones": [
            {
                "id": "top_badge",
                "x": 0.1, "y": 0.02, "w": 0.8, "h": 0.08,
                "roles": [ElementRole.BADGE, ElementRole.LOGO],
            },
            {
                "id": "headline",
                "x": 0.05, "y": 0.12, "w": 0.9, "h": 0.25,
                "roles": [ElementRole.HEADLINE, ElementRole.SUBHEADLINE],
            },
            {
                "id": "cta",
                "x": 0.15, "y": 0.38, "w": 0.7, "h": 0.08,
                "roles": [ElementRole.CTA],
            },
            {
                "id": "hero",
                "x": 0.0, "y": 0.48, "w": 1.0, "h": 0.52,
                "roles": [ElementRole.HERO_IMAGE, ElementRole.ILLUSTRATION, ElementRole.PHOTO],
            },
        ],
    },
    "portrait": {  # 4:5 (Instagram Portrait)
        "aspect_range": (0.65, 0.85),
        "zones": [
            {
                "id": "top_badge",
                "x": 0.1, "y": 0.03, "w": 0.8, "h": 0.1,
                "roles": [ElementRole.BADGE, ElementRole.LOGO],
            },
            {
                "id": "headline",
                "x": 0.05, "y": 0.15, "w": 0.9, "h": 0.2,
                "roles": [ElementRole.HEADLINE, ElementRole.SUBHEADLINE],
            },
            {
                "id": "cta",
                "x": 0.2, "y": 0.36, "w": 0.6, "h": 0.08,
                "roles": [ElementRole.CTA],
            },
            {
                "id": "hero",
                "x": 0.05, "y": 0.46, "w": 0.9, "h": 0.5,
                "roles": [ElementRole.HERO_IMAGE, ElementRole.ILLUSTRATION, ElementRole.PHOTO],
            },
        ],
    },
    "square": {  # 1:1 (Instagram Square)
        "aspect_range": (0.85, 1.15),
        "zones": [
            {
                "id": "top_badge",
                "x": 0.05, "y": 0.03, "w": 0.4, "h": 0.12,
                "roles": [ElementRole.BADGE, ElementRole.LOGO],
            },
            {
                "id": "headline",
                "x": 0.05, "y": 0.18, "w": 0.55, "h": 0.25,
                "roles": [ElementRole.HEADLINE, ElementRole.SUBHEADLINE],
            },
            {
                "id": "cta",
                "x": 0.05, "y": 0.45, "w": 0.45, "h": 0.1,
                "roles": [ElementRole.CTA],
            },
            {
                "id": "hero",
                "x": 0.45, "y": 0.1, "w": 0.52, "h": 0.85,
                "roles": [ElementRole.HERO_IMAGE, ElementRole.ILLUSTRATION, ElementRole.PHOTO],
            },
        ],
    },
    "landscape": {  # 16:9 (YouTube)
        "aspect_range": (1.15, 2.0),
        "zones": [
            {
                "id": "left_content",
                "x": 0.03, "y": 0.05, "w": 0.5, "h": 0.9,
                "roles": [
                    ElementRole.BADGE, ElementRole.HEADLINE,
                    ElementRole.SUBHEADLINE, ElementRole.CTA,
                ],
            },
            {
                "id": "hero",
                "x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0,
                "roles": [
                    ElementRole.HERO_IMAGE, ElementRole.ILLUSTRATION,
                    ElementRole.PHOTO, ElementRole.LOGO,
                ],
            },
        ],
    },
    "landscape_wide": {  # 3:1 (Twitter Header)
        "aspect_range": (2.0, 4.0),
        "zones": [
            {
                "id": "left",
                "x": 0.02, "y": 0.1, "w": 0.35, "h": 0.8,
                "roles": [ElementRole.BADGE, ElementRole.HEADLINE],
            },
            {
                "id": "center",
                "x": 0.35, "y": 0.15, "w": 0.3, "h": 0.7,
                "roles": [ElementRole.SUBHEADLINE, ElementRole.CTA],
            },
            {
                "id": "hero",
                "x": 0.62, "y": 0.0, "w": 0.38, "h": 1.0,
                "roles": [
                    ElementRole.HERO_IMAGE, ElementRole.ILLUSTRATION,
                    ElementRole.LOGO,
                ],
            },
        ],
    },
}
