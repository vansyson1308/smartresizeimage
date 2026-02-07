"""Layout engine for intelligent element re-arrangement."""

from __future__ import annotations

import logging

from ..config import Config
from ..constants import BACKGROUND_ROLES
from ..models import BoundingBox, DesignElement, LayoutResult
from .templates import TEMPLATES

logger = logging.getLogger("autobanner.layout")


class LayoutEngine:
    """Intelligent layout engine for re-arranging elements.

    Features:
    - Aspect ratio-aware layouts
    - Priority-based placement
    - Constraint satisfaction
    - Multiple layout templates
    """

    def calculate_layout(
        self,
        elements: list[DesignElement],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> list[LayoutResult]:
        """Calculate new layout for all elements.

        Args:
            elements: List of classified design elements.
            source_size: Original canvas size (width, height).
            target_size: Target canvas size (width, height).

        Returns:
            List of LayoutResult with new positions and scales.
        """
        target_w, target_h = target_size
        target_aspect = target_w / target_h if target_h > 0 else 1.0

        # Select appropriate template
        template = self._select_template(target_aspect)

        # Separate background elements from content
        bg_elements = [e for e in elements if e.role in BACKGROUND_ROLES]
        content_elements = [e for e in elements if e.role not in BACKGROUND_ROLES]

        # Sort content by priority
        content_elements.sort(key=lambda e: (e.priority, -e.bbox.area))

        results: list[LayoutResult] = []

        # Handle background elements
        for elem in bg_elements:
            bg_scale_w = target_w / elem.bbox.width if elem.bbox.width > 0 else 1.0
            bg_scale_h = target_h / elem.bbox.height if elem.bbox.height > 0 else 1.0
            result = LayoutResult(
                element_id=elem.id,
                new_bbox=BoundingBox(0, 0, target_w, target_h),
                scale_factor=max(bg_scale_w, bg_scale_h),
                visible=True,
            )
            results.append(result)

        # Assign content elements to zones
        zone_assignments = self._assign_to_zones(content_elements, template, target_size)

        # Calculate positions within zones
        for elem_id, zone_info in zone_assignments.items():
            elem = next((e for e in content_elements if e.id == elem_id), None)
            if elem is None:
                logger.warning("Element %s not found in content_elements, skipping", elem_id)
                continue

            zone = zone_info["zone"]

            # Calculate zone bounds in pixels
            zone_x = int(zone["x"] * target_w)
            zone_y = int(zone["y"] * target_h)
            zone_w = int(zone["w"] * target_w)
            zone_h = int(zone["h"] * target_h)

            # Calculate scale to fit in zone
            scale_x = zone_w / max(1, elem.bbox.width)
            scale_y = zone_h / max(1, elem.bbox.height)

            scale = min(scale_x, scale_y) if elem.maintain_aspect else (scale_x + scale_y) / 2

            # Clamp scale
            scale = max(elem.min_scale, min(elem.max_scale, scale))

            # Calculate new size
            new_w = int(elem.bbox.width * scale)
            new_h = int(elem.bbox.height * scale)

            # Calculate position (centered in zone)
            new_x = zone_x + (zone_w - new_w) // 2
            new_y = zone_y + (zone_h - new_h) // 2

            result = LayoutResult(
                element_id=elem_id,
                new_bbox=BoundingBox(new_x, new_y, new_w, new_h),
                scale_factor=scale,
                visible=True,
            )
            results.append(result)

        # Handle unassigned elements
        assigned_ids = set(zone_assignments.keys())
        for elem in content_elements:
            if elem.id not in assigned_ids:
                source_w, source_h = source_size
                scale = min(
                    target_w / max(1, source_w),
                    target_h / max(1, source_h),
                )
                result = LayoutResult(
                    element_id=elem.id,
                    new_bbox=BoundingBox(
                        int(elem.bbox.x * scale),
                        int(elem.bbox.y * scale),
                        int(elem.bbox.width * scale),
                        int(elem.bbox.height * scale),
                    ),
                    scale_factor=scale,
                    visible=elem.priority <= 7,  # Hide low priority elements
                )
                results.append(result)

        return results

    def _select_template(self, target_aspect: float) -> dict:
        """Select the best template for the target aspect ratio."""
        for _name, template in TEMPLATES.items():
            min_aspect, max_aspect = template["aspect_range"]
            if min_aspect <= target_aspect < max_aspect:
                return template

        # Default to landscape if no match
        return TEMPLATES["landscape"]

    def _assign_to_zones(
        self,
        elements: list[DesignElement],
        template: dict,
        target_size: tuple[int, int],
    ) -> dict[str, dict]:
        """Assign elements to layout zones based on roles."""
        assignments: dict[str, dict] = {}
        zone_occupancy: dict[str, int] = {
            z["id"]: 0 for z in template["zones"]
        }

        zones = template["zones"]
        max_per_zone = Config.MAX_ELEMENTS_PER_ZONE

        for elem in elements:
            best_zone = None

            # Find a zone that accepts this element's role
            for zone in zones:
                zone_id = zone["id"]
                if zone_occupancy[zone_id] >= max_per_zone:
                    continue

                if elem.role in zone["roles"]:
                    best_zone = zone
                    break

            if best_zone:
                assignments[elem.id] = {
                    "zone": best_zone,
                    "element": elem,
                }
                zone_occupancy[best_zone["id"]] += 1

        return assignments
