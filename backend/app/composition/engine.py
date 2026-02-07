"""Composition engine for assembling final images."""

from __future__ import annotations

import logging

from PIL import Image

from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import CompositionResult, DesignElement, LayoutResult
from .background import BackgroundExtender
from .resize import high_quality_resize

logger = logging.getLogger("autobanner.composition")


class CompositionEngine:
    """Compose final image from elements and layout.

    Features:
    - High-quality scaling with gamma correction
    - Background extension/generation
    - Layer blending with effects
    - AI-powered inpainting (optional)
    """

    def __init__(self, use_ai_inpainting: bool = True) -> None:
        self.bg_extender = BackgroundExtender(use_ai_inpainting=use_ai_inpainting)

    def compose(
        self,
        elements: list[DesignElement],
        layout_results: list[LayoutResult],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> CompositionResult:
        """Compose final image.

        Args:
            elements: List of design elements with images.
            layout_results: Layout results with new positions.
            source_size: Original canvas size.
            target_size: Target canvas size.

        Returns:
            CompositionResult with final image.
        """
        target_w, target_h = target_size
        warnings: list[str] = []

        # Create result mapping
        layout_map: dict[str, LayoutResult] = {r.element_id: r for r in layout_results}

        # Separate background and content
        bg_elements = [
            (e, layout_map.get(e.id))
            for e in elements
            if e.role in BACKGROUND_ROLES and e.id in layout_map
        ]

        content_elements = [
            (e, layout_map.get(e.id))
            for e in elements
            if e.role not in BACKGROUND_ROLES and e.id in layout_map
        ]

        # Create canvas
        canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))

        # Compose background
        canvas = self._compose_background(canvas, bg_elements, source_size, target_size)

        # Sort content by z_index
        content_elements.sort(key=lambda x: x[0].z_index if x[0] else 0)

        # Compose content elements
        for elem, layout in content_elements:
            if not layout or not layout.visible or not elem.image:
                continue

            try:
                canvas = self._composite_element(canvas, elem, layout)
            except Exception as e:
                warnings.append(f"Could not composite '{elem.name}': {e}")

        return CompositionResult(
            image=canvas.convert("RGB"),
            layout_results=layout_results,
            warnings=warnings,
        )

    def _compose_background(
        self,
        canvas: Image.Image,
        bg_elements: list[tuple[DesignElement, LayoutResult]],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> Image.Image:
        """Compose background layer(s)."""
        target_w, target_h = target_size

        if not bg_elements:
            return canvas

        bg_elem, bg_layout = bg_elements[0]

        if not bg_elem.image:
            return canvas

        bg_image = bg_elem.image.convert("RGBA")
        bg_w, bg_h = bg_image.size

        need_extend_w = target_w > bg_w
        need_extend_h = target_h > bg_h

        if need_extend_w or need_extend_h:
            bg_image = self.bg_extender.extend(bg_image, target_size)
        else:
            # Scale to cover
            if bg_w > 0 and bg_h > 0:
                scale = max(target_w / bg_w, target_h / bg_h)
                new_w = int(bg_w * scale)
                new_h = int(bg_h * scale)
                bg_image = high_quality_resize(bg_image, (new_w, new_h))

                # Center crop
                x = (new_w - target_w) // 2
                y = (new_h - target_h) // 2
                bg_image = bg_image.crop((x, y, x + target_w, y + target_h))

        canvas.paste(bg_image, (0, 0))

        # Add overlays
        for elem, _layout in bg_elements[1:]:
            if elem.role == ElementRole.OVERLAY and elem.image:
                overlay = elem.image.convert("RGBA")
                overlay = high_quality_resize(overlay, target_size)
                canvas = Image.alpha_composite(canvas, overlay)

        return canvas

    def _composite_element(
        self,
        canvas: Image.Image,
        element: DesignElement,
        layout: LayoutResult,
    ) -> Image.Image:
        """Composite a single element onto the canvas."""
        if not element.image:
            return canvas

        elem_image = element.image.convert("RGBA")

        # Resize to new dimensions
        new_size = (layout.new_bbox.width, layout.new_bbox.height)
        if new_size[0] <= 0 or new_size[1] <= 0:
            return canvas

        resized = high_quality_resize(elem_image, new_size)

        # Apply opacity
        if element.opacity < 1.0:
            alpha = resized.split()[3]
            alpha = alpha.point(lambda p: int(p * element.opacity))
            resized.putalpha(alpha)

        # Paste with alpha
        canvas.paste(resized, (layout.new_bbox.x, layout.new_bbox.y), resized)

        return canvas
