"""Semantic roles for design elements."""

from __future__ import annotations

from enum import Enum


class ElementRole(Enum):
    """Semantic roles for design elements."""
    HEADLINE = "headline"
    SUBHEADLINE = "subheadline"
    BODY_TEXT = "body_text"
    CTA = "cta"
    BADGE = "badge"
    LABEL = "label"
    LOGO = "logo"
    HERO_IMAGE = "hero_image"
    ICON = "icon"
    PHOTO = "photo"
    ILLUSTRATION = "illustration"
    DECORATION = "decoration"
    BACKGROUND = "background"
    BACKGROUND_PATTERN = "background_pattern"
    OVERLAY = "overlay"
    GROUP = "group"
    UNKNOWN = "unknown"
