"""Layout engine for intelligent element re-arrangement."""

from __future__ import annotations

import copy
import logging

from ..config import Config
from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import BoundingBox, DesignElement, LayoutResult
from .constraints import validate_layout
from .profiles import LayoutProfile, pick_profile
from .scoring import score_layout
from .solver import solve_layout
from .templates import TEMPLATES
from .typography import fit_text_block

logger = logging.getLogger("autobanner.layout")


_TEXT_ROLES = {
    ElementRole.HEADLINE,
    ElementRole.SUBHEADLINE,
    ElementRole.BODY_TEXT,
    ElementRole.CTA,
    ElementRole.LABEL,
}


class LayoutEngine:
    """Intelligent layout engine for re-arranging elements."""

    def __init__(self) -> None:
        self.last_layout_debug: dict[str, object] = {}

    def calculate_layout(
        self,
        elements: list[DesignElement],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> list[LayoutResult]:
        """Calculate new layout for all elements."""
        base_results = self._calculate_with_template(
            elements,
            source_size,
            target_size,
            self._select_template(target_size),
        )

        if not Config.LAYOUT_PROFILE_SCORING_ENABLED:
            return base_results

        profile = pick_profile(target_size[0], target_size[1])
        role_by_id = {e.id: e.role for e in elements}

        candidates = self._generate_candidates(elements, source_size, target_size)
        best_results = base_results
        best_score = float("-inf")
        best_violations: list[str] = []

        for candidate in candidates:
            solved, solver_meta = solve_layout(
                candidate,
                target_size=target_size,
                profile=profile,
                role_by_id=role_by_id,
                iterations=Config.LAYOUT_SOLVER_MAX_ITERS,
            )
            violations = validate_layout(solved, profile, target_size, role_by_id)
            score = score_layout(solved, profile, target_size, role_by_id)
            score -= solver_meta.get("overlap_area", 0.0) * 0.01
            if score > best_score:
                best_score = score
                best_results = solved
                best_violations = violations

        logger.info(
            "profile=%s, candidates=%d, best_score=%.2f, violations=%d",
            profile.name,
            len(candidates),
            best_score,
            len(best_violations),
        )

        self.last_layout_debug = {
            "profile": profile.name,
            "candidates": len(candidates),
            "best_score": float(best_score),
            "violations": list(best_violations),
            "results": [
                {
                    "element_id": r.element_id,
                    "x": r.new_bbox.x,
                    "y": r.new_bbox.y,
                    "width": r.new_bbox.width,
                    "height": r.new_bbox.height,
                    "visible": r.visible,
                }
                for r in best_results
            ],
        }

        if len(best_violations) >= 6:
            logger.warning("Adaptive scoring fallback to rigid template due to heavy violations")
            return base_results

        return best_results

    def _calculate_with_template(
        self,
        elements: list[DesignElement],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        template: dict,
    ) -> list[LayoutResult]:
        target_w, target_h = target_size
        profile = pick_profile(target_w, target_h)

        bg_elements = [e for e in elements if e.role in BACKGROUND_ROLES]
        content_elements = [e for e in elements if e.role not in BACKGROUND_ROLES]
        content_elements.sort(key=lambda e: (e.priority, -e.bbox.area))

        results: list[LayoutResult] = []

        for elem in bg_elements:
            bg_scale_w = target_w / elem.bbox.width if elem.bbox.width > 0 else 1.0
            bg_scale_h = target_h / elem.bbox.height if elem.bbox.height > 0 else 1.0
            results.append(
                LayoutResult(
                    element_id=elem.id,
                    new_bbox=BoundingBox(0, 0, target_w, target_h),
                    scale_factor=max(bg_scale_w, bg_scale_h),
                    visible=True,
                )
            )

        zone_assignments = self._assign_to_zones(content_elements, template)

        for elem_id, zone in zone_assignments.items():
            elem = next((e for e in content_elements if e.id == elem_id), None)
            if elem is None:
                continue

            zone_x = int(zone["x"] * target_w)
            zone_y = int(zone["y"] * target_h)
            zone_w = int(zone["w"] * target_w)
            zone_h = int(zone["h"] * target_h)

            if self._is_text_element(elem):
                result = self._layout_text_element(elem, profile, zone_x, zone_y, zone_w, zone_h)
            else:
                result = self._layout_visual_element(elem, zone_x, zone_y, zone_w, zone_h)

            results.append(result)

        assigned_ids = set(zone_assignments.keys())
        for elem in content_elements:
            if elem.id in assigned_ids:
                continue
            source_w, source_h = source_size
            scale = min(target_w / max(1, source_w), target_h / max(1, source_h))
            results.append(
                LayoutResult(
                    element_id=elem.id,
                    new_bbox=BoundingBox(
                        int(elem.bbox.x * scale),
                        int(elem.bbox.y * scale),
                        int(elem.bbox.width * scale),
                        int(elem.bbox.height * scale),
                    ),
                    scale_factor=scale,
                    visible=elem.priority <= 7,
                )
            )

        return results

    @staticmethod
    def _is_text_element(elem: DesignElement) -> bool:
        return elem.role in _TEXT_ROLES or bool(elem.text_content)

    def _layout_text_element(
        self,
        elem: DesignElement,
        profile: LayoutProfile,
        zone_x: int,
        zone_y: int,
        zone_w: int,
        zone_h: int,
    ) -> LayoutResult:
        text = elem.text_content or elem.name
        font_family = None
        if elem.font_info and isinstance(elem.font_info, dict):
            font_family = elem.font_info.get("family")

        min_font, max_font, max_lines = self._typography_bounds_for_role(elem.role)

        zone_w_pos = zone_w if zone_w > 0 else 1
        width_cap = int(min(zone_w, profile.text_block_max_width_pct * zone_w_pos))
        fit = fit_text_block(
            text=text,
            font_family=font_family,
            max_font=max_font,
            min_font=min_font,
            max_width=max(1, width_cap),
            max_lines=max_lines,
        )

        new_w = max(1, fit.bbox[0])
        new_h = max(1, fit.bbox[1])

        # If still overflowed at min font, expand block height (text-first behavior).
        if fit.overflow:
            new_h = min(int(zone_h * 1.5), max(new_h, zone_h))

        new_w = min(new_w, max(1, zone_w))
        new_h = min(new_h, max(1, int(zone_h * 1.5)))

        new_x = zone_x + (zone_w - new_w) // 2
        new_y = zone_y + (zone_h - min(new_h, zone_h)) // 2

        logger.info(
            "typography element=%s role=%s font=%d lines=%d overflow=%s",
            elem.id,
            elem.role.value,
            fit.font_size,
            len(fit.lines),
            fit.overflow,
        )

        scale = new_h / max(1, elem.bbox.height)
        return LayoutResult(
            element_id=elem.id,
            new_bbox=BoundingBox(new_x, new_y, new_w, new_h),
            scale_factor=scale,
            visible=True,
        )

    @staticmethod
    def _layout_visual_element(
        elem: DesignElement,
        zone_x: int,
        zone_y: int,
        zone_w: int,
        zone_h: int,
    ) -> LayoutResult:
        scale_x = zone_w / max(1, elem.bbox.width)
        scale_y = zone_h / max(1, elem.bbox.height)
        scale = min(scale_x, scale_y) if elem.maintain_aspect else (scale_x + scale_y) / 2
        scale = max(elem.min_scale, min(elem.max_scale, scale))

        new_w = int(elem.bbox.width * scale)
        new_h = int(elem.bbox.height * scale)
        new_x = zone_x + (zone_w - new_w) // 2
        new_y = zone_y + (zone_h - new_h) // 2

        return LayoutResult(
            element_id=elem.id,
            new_bbox=BoundingBox(new_x, new_y, new_w, new_h),
            scale_factor=scale,
            visible=True,
        )

    @staticmethod
    def _typography_bounds_for_role(role: ElementRole) -> tuple[int, int, int]:
        if role == ElementRole.HEADLINE:
            return 20, 64, 4
        if role == ElementRole.SUBHEADLINE:
            return 16, 42, 5
        if role == ElementRole.CTA:
            return 14, 30, 2
        return 12, 28, 5

    def _generate_candidates(
        self,
        elements: list[DesignElement],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> list[list[LayoutResult]]:
        base_template = self._select_template(target_size)
        candidates = [
            self._calculate_with_template(elements, source_size, target_size, base_template)
        ]

        profile = pick_profile(target_size[0], target_size[1])
        if profile.name != "PORTRAIT":
            return candidates

        # 4 additional portrait stacking variants (total 5).
        variants = [
            self._variant_shift(base_template, headline_dy=-0.03, hero_y=0.50),
            self._variant_shift(base_template, headline_dy=0.02, hero_y=0.46),
            self._variant_shift(base_template, cta_dy=0.03, hero_y=0.52),
            self._variant_shift(base_template, headline_dy=-0.01, cta_dy=0.02, hero_y=0.49),
        ]
        for template in variants:
            candidates.append(
                self._calculate_with_template(elements, source_size, target_size, template)
            )

        return candidates

    @staticmethod
    def _variant_shift(
        template: dict,
        headline_dy: float = 0.0,
        cta_dy: float = 0.0,
        hero_y: float | None = None,
    ) -> dict:
        t = copy.deepcopy(template)
        zones = t.get("zones", [])
        for zone in zones:
            zid = zone.get("id", "")
            if zid == "headline":
                zone["y"] = max(0.0, min(0.9, float(zone["y"]) + headline_dy))
            elif zid == "cta":
                zone["y"] = max(0.0, min(0.9, float(zone["y"]) + cta_dy))
            elif zid == "hero" and hero_y is not None:
                zone["y"] = max(0.0, min(0.95, hero_y))
                zone["h"] = max(0.05, 1.0 - zone["y"])
        return t

    def _select_template(self, target_size: tuple[int, int] | float) -> dict:
        if isinstance(target_size, tuple):
            target_w, target_h = target_size
            target_aspect = target_w / target_h if target_h > 0 else 1.0
        else:
            target_aspect = float(target_size)

        for template in TEMPLATES.values():
            min_aspect, max_aspect = template["aspect_range"]
            if min_aspect <= target_aspect < max_aspect:
                return template
        return TEMPLATES["landscape"]

    def _assign_to_zones(
        self,
        elements: list[DesignElement],
        template: dict,
    ) -> dict[str, dict]:
        assignments: dict[str, dict] = {}
        zone_occupancy: dict[str, int] = {z["id"]: 0 for z in template["zones"]}

        zones = template["zones"]
        max_per_zone = Config.MAX_ELEMENTS_PER_ZONE

        for elem in elements:
            best_zone = None
            for zone in zones:
                zone_id = zone["id"]
                if zone_occupancy[zone_id] >= max_per_zone:
                    continue
                if elem.role in zone["roles"]:
                    best_zone = zone
                    break

            if best_zone is not None:
                assignments[elem.id] = best_zone
                zone_occupancy[best_zone["id"]] += 1

        return assignments
