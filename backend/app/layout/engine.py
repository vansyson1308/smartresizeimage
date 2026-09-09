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
from .repair import apply_repair
from .scoring import score_layout
from .solver import solve_layout, total_overlap_area
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
        self._typography_debug: list[dict[str, object]] = []

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

        self._typography_debug = []

        profile = pick_profile(target_size[0], target_size[1])
        role_by_id = {e.id: e.role for e in elements}

        candidates = self._generate_candidates(elements, source_size, target_size)
        best_results = base_results
        best_score = float("-inf")
        best_violations: list[str] = []
        best_repair_steps: list[str] = []
        best_outside_margin_count = 0
        fallback_used = False
        fallback_reason = ""

        had_candidate_error = False
        elements_by_id = {e.id: e for e in elements}
        canvas_area = max(1, target_size[0] * target_size[1])

        base_repair = apply_repair(base_results, elements_by_id, profile, target_size)
        try:
            repaired_base, _ = solve_layout(
                base_repair.layout,
                target_size=target_size,
                profile=profile,
                role_by_id=role_by_id,
                iterations=max(8, Config.LAYOUT_SOLVER_MAX_ITERS // 2),
            )
        except Exception:
            repaired_base = base_repair.layout
        for candidate in candidates:
            try:
                solved, solver_meta = solve_layout(
                    candidate,
                    target_size=target_size,
                    profile=profile,
                    role_by_id=role_by_id,
                    iterations=Config.LAYOUT_SOLVER_MAX_ITERS,
                )
                repair = apply_repair(solved, elements_by_id, profile, target_size)
                repaired, solver_meta = solve_layout(
                    repair.layout,
                    target_size=target_size,
                    profile=profile,
                    role_by_id=role_by_id,
                    iterations=max(8, Config.LAYOUT_SOLVER_MAX_ITERS // 2),
                )

                violations = validate_layout(repaired, profile, target_size, role_by_id)
                outside_margin_count = len(
                    [v for v in violations if v.startswith("outside_margin:")]
                )
                if outside_margin_count > 0:
                    continue

                score = score_layout(repaired, profile, target_size, role_by_id)
                score -= solver_meta.get("overlap_area", 0.0) * 0.01
                score -= outside_margin_count * 1000.0
                if score > best_score:
                    best_score = score
                    best_results = repaired
                    best_violations = violations
                    best_repair_steps = repair.steps
                    best_outside_margin_count = outside_margin_count
            except Exception as e:
                had_candidate_error = True
                logger.warning(
                    "Adaptive solver candidate failed; using template fallback path: %s",
                    e,
                )

        if best_score == float("-inf"):
            logger.warning(
                "Adaptive solver failed for all candidates after repair; using rigid template"
            )
            fallback_used = True
            fallback_reason = "no_repairable_candidate"
            best_results = repaired_base
            best_violations = validate_layout(repaired_base, profile, target_size, role_by_id)
            best_repair_steps = list(base_repair.steps)
            best_outside_margin_count = len(
                [v for v in best_violations if v.startswith("outside_margin:")]
            )

        logger.info(
            "profile=%s, candidates=%d, best_score=%.2f, violations=%d",
            profile.name,
            len(candidates),
            best_score,
            len(best_violations),
        )

        self.last_layout_debug = {
            "profile": profile.name,
            "profile_name": profile.name,
            "candidates": len(candidates),
            "best_score": float(best_score),
            "violations": list(best_violations),
            "repair_applied": bool(best_repair_steps),
            "repair_steps": list(best_repair_steps),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "outside_margin_count": best_outside_margin_count,
            "typography": list(self._typography_debug),
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

        overlap_ratio = total_overlap_area(best_results, role_by_id) / canvas_area
        if best_outside_margin_count > 0:
            logger.warning("Adaptive scoring fallback: hard margin violations after repair")
            self.last_layout_debug["fallback_used"] = True
            self.last_layout_debug["fallback_reason"] = "outside_margin_after_repair"
            return repaired_base
        if overlap_ratio > 0.25:
            logger.warning("Adaptive scoring fallback: catastrophic overlap after repair")
            self.last_layout_debug["fallback_used"] = True
            self.last_layout_debug["fallback_reason"] = "catastrophic_overlap_after_repair"
            return repaired_base

        if had_candidate_error:
            logger.info("Adaptive scoring completed with fallback on failed candidates")

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

        # Group members per zone, preserving priority order, so that several
        # elements sharing a zone are stacked instead of centred on top of
        # each other.
        members_by_zone: dict[str, list[DesignElement]] = {}
        for elem in content_elements:
            zone = zone_assignments.get(elem.id)
            if zone is None:
                continue
            members_by_zone.setdefault(zone["id"], []).append(elem)

        zones_by_id = {z["id"]: z for z in template["zones"]}
        gap = max(4, int(profile.baseline_spacing_pct * target_h))
        for zone_id, members in members_by_zone.items():
            zone = zones_by_id[zone_id]
            zone_x = int(zone["x"] * target_w)
            zone_y = int(zone["y"] * target_h)
            zone_w = int(zone["w"] * target_w)
            zone_h = int(zone["h"] * target_h)

            n = len(members)
            slot_h = max(1, (zone_h - gap * (n - 1)) // n) if n > 1 else zone_h
            placed: list[LayoutResult] = []
            for elem in members:
                if self._is_text_element(elem):
                    result = self._layout_text_element(
                        elem, profile, zone_x, zone_y, zone_w, slot_h, target_w
                    )
                else:
                    result = self._layout_visual_element(elem, zone_x, zone_y, zone_w, slot_h)
                placed.append(result)

            if n > 1:
                total_h = sum(r.new_bbox.height for r in placed) + gap * (n - 1)
                y = zone_y + max(0, (zone_h - total_h) // 2)
                for r in placed:
                    b = r.new_bbox
                    b.x = zone_x + (zone_w - b.width) // 2
                    b.y = y
                    y += b.height + gap

            results.extend(placed)

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
        target_w: int | None = None,
    ) -> LayoutResult:
        text = elem.text_content or elem.name
        font_family = None
        if elem.font_info and isinstance(elem.font_info, dict):
            font_family = elem.font_info.get("family")

        min_font, max_font, max_lines = self._typography_bounds_for_role(elem.role)

        zone_w_pos = zone_w if zone_w > 0 else 1
        canvas_w = target_w if target_w else zone_w_pos
        width_cap = int(min(zone_w_pos, profile.text_block_max_width_pct * canvas_w))

        if elem.image is not None and elem.bbox.width > 0 and elem.bbox.height > 0:
            # Raster text cannot be re-wrapped: scale it uniformly so glyphs are
            # never distorted, then report the estimated glyph size honestly.
            return self._layout_raster_text_element(
                elem, zone_x, zone_y, zone_w, zone_h, width_cap, min_font, max_font
            )

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

        self._typography_debug.append(
            {
                "element_id": elem.id,
                "role": elem.role.value,
                "font_px": int(fit.font_size),
                "unit": "px",
                "lines": len(fit.lines),
                "overflow": bool(fit.overflow),
            }
        )

        scale = new_h / max(1, elem.bbox.height)
        return LayoutResult(
            element_id=elem.id,
            new_bbox=BoundingBox(new_x, new_y, new_w, new_h),
            scale_factor=scale,
            visible=True,
        )

    def _layout_raster_text_element(
        self,
        elem: DesignElement,
        zone_x: int,
        zone_y: int,
        zone_w: int,
        zone_h: int,
        width_cap: int,
        min_font: int,
        max_font: int,
    ) -> LayoutResult:
        src_w, src_h = elem.bbox.width, elem.bbox.height
        n_lines = max(1, (elem.text_content or "").count("\n") + 1)
        # Approximate glyph height of the source raster (cap height ~ 0.7 of line box).
        src_font_px = max(1.0, src_h / n_lines * 0.7)

        scale = min(width_cap / src_w, max(1, zone_h) / src_h)
        scale = min(scale, max_font / src_font_px)
        scale = max(scale, min(elem.min_scale, min_font / src_font_px))
        scale = min(scale, elem.max_scale)

        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        est_font = int(src_font_px * scale)
        overflow = new_w > zone_w or new_h > zone_h or est_font < min_font

        new_x = zone_x + (zone_w - new_w) // 2
        new_y = zone_y + (zone_h - min(new_h, zone_h)) // 2

        self._typography_debug.append(
            {
                "element_id": elem.id,
                "role": elem.role.value,
                "font_px": est_font,
                "unit": "px",
                "lines": n_lines,
                "overflow": bool(overflow),
                "raster": True,
                "estimated": True,
            }
        )
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

            if best_zone is None:
                # Every matching zone is full: overflow into the least occupied
                # matching zone. Members are stacked, so a third element is
                # still laid out deliberately instead of being dropped onto raw
                # source coordinates where it collides with everything.
                matching = [z for z in zones if elem.role in z["roles"]]
                if matching:
                    best_zone = min(matching, key=lambda z: zone_occupancy[z["id"]])

            if best_zone is not None:
                assignments[elem.id] = best_zone
                zone_occupancy[best_zone["id"]] += 1

        return assignments
