"""Shared constants for AutoBanner."""

from __future__ import annotations

from .enums import ElementRole

# Background-related roles (used for filtering in layout and composition)
BACKGROUND_ROLES = frozenset({
    ElementRole.BACKGROUND,
    ElementRole.BACKGROUND_PATTERN,
    ElementRole.OVERLAY,
})

# Supported file extensions
SUPPORTED_EXTENSIONS = (".psd", ".png", ".jpg", ".jpeg", ".webp")

# Role priority mapping (lower = higher priority)
ROLE_PRIORITIES = {
    ElementRole.HEADLINE: 1,
    ElementRole.LOGO: 2,
    ElementRole.CTA: 2,
    ElementRole.HERO_IMAGE: 3,
    ElementRole.SUBHEADLINE: 3,
    ElementRole.BADGE: 4,
    ElementRole.LABEL: 4,
    ElementRole.BODY_TEXT: 5,
    ElementRole.ILLUSTRATION: 5,
    ElementRole.PHOTO: 5,
    ElementRole.ICON: 6,
    ElementRole.GROUP: 7,
    ElementRole.UNKNOWN: 7,
    ElementRole.DECORATION: 8,
    ElementRole.BACKGROUND: 9,
    ElementRole.BACKGROUND_PATTERN: 9,
    ElementRole.OVERLAY: 9,
}
