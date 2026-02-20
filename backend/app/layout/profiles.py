"""Layout profiles and profile selection for adaptive designer-like relayout."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..enums import ElementRole


@dataclass(frozen=True)
class LayoutProfile:
    """Profile parameters for aspect-ratio-aware layout constraints/scoring."""

    name: str
    margin_pct: float
    safe_area_pct: float
    grid_cols: int
    grid_rows: int
    baseline_spacing_pct: float
    min_sizes: dict[ElementRole, tuple[float, float]] = field(default_factory=dict)
    max_sizes: dict[ElementRole, tuple[float, float]] = field(default_factory=dict)
    role_priority: dict[ElementRole, int] = field(default_factory=dict)
    target_hero_ratio: float = 0.25
    text_block_max_width_pct: float = 0.7


_COMMON_ROLE_PRIORITY = {
    ElementRole.HEADLINE: 100,
    ElementRole.SUBHEADLINE: 90,
    ElementRole.CTA: 80,
    ElementRole.LOGO: 70,
    ElementRole.HERO_IMAGE: 95,
}


def pick_profile(target_w: int, target_h: int) -> LayoutProfile:
    """Pick LANDSCAPE/SQUARE/PORTRAIT profile by target aspect ratio."""
    aspect = target_w / max(1, target_h)

    if aspect > 1.2:
        return _landscape_profile()
    if aspect < 0.85:
        return _portrait_profile()
    return _square_profile()


def _portrait_profile() -> LayoutProfile:
    return LayoutProfile(
        name="PORTRAIT",
        margin_pct=0.06,
        safe_area_pct=0.08,
        grid_cols=4,
        grid_rows=12,
        baseline_spacing_pct=0.025,
        min_sizes={
            ElementRole.HEADLINE: (0.45, 0.08),
            ElementRole.SUBHEADLINE: (0.38, 0.05),
            ElementRole.CTA: (0.22, 0.04),
            ElementRole.LOGO: (0.12, 0.04),
            ElementRole.HERO_IMAGE: (0.45, 0.32),
        },
        max_sizes={
            ElementRole.HEADLINE: (0.92, 0.28),
            ElementRole.SUBHEADLINE: (0.90, 0.18),
            ElementRole.CTA: (0.50, 0.10),
            ElementRole.LOGO: (0.30, 0.15),
        },
        role_priority=_COMMON_ROLE_PRIORITY,
        target_hero_ratio=0.35,
        text_block_max_width_pct=0.88,
    )


def _square_profile() -> LayoutProfile:
    return LayoutProfile(
        name="SQUARE",
        margin_pct=0.05,
        safe_area_pct=0.07,
        grid_cols=6,
        grid_rows=6,
        baseline_spacing_pct=0.02,
        min_sizes={
            ElementRole.HEADLINE: (0.38, 0.08),
            ElementRole.SUBHEADLINE: (0.30, 0.05),
            ElementRole.CTA: (0.18, 0.04),
            ElementRole.LOGO: (0.10, 0.04),
            ElementRole.HERO_IMAGE: (0.36, 0.36),
        },
        max_sizes={
            ElementRole.HEADLINE: (0.75, 0.24),
            ElementRole.SUBHEADLINE: (0.65, 0.16),
            ElementRole.CTA: (0.42, 0.12),
            ElementRole.LOGO: (0.24, 0.14),
        },
        role_priority=_COMMON_ROLE_PRIORITY,
        target_hero_ratio=0.30,
        text_block_max_width_pct=0.70,
    )


def _landscape_profile() -> LayoutProfile:
    return LayoutProfile(
        name="LANDSCAPE",
        margin_pct=0.04,
        safe_area_pct=0.06,
        grid_cols=12,
        grid_rows=4,
        baseline_spacing_pct=0.02,
        min_sizes={
            ElementRole.HEADLINE: (0.25, 0.10),
            ElementRole.SUBHEADLINE: (0.20, 0.06),
            ElementRole.CTA: (0.14, 0.05),
            ElementRole.LOGO: (0.08, 0.05),
            ElementRole.HERO_IMAGE: (0.28, 0.45),
        },
        max_sizes={
            ElementRole.HEADLINE: (0.55, 0.32),
            ElementRole.SUBHEADLINE: (0.45, 0.20),
            ElementRole.CTA: (0.30, 0.12),
            ElementRole.LOGO: (0.20, 0.16),
        },
        role_priority=_COMMON_ROLE_PRIORITY,
        target_hero_ratio=0.36,
        text_block_max_width_pct=0.50,
    )
