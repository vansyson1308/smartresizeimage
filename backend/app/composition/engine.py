"""Composition engine for assembling final images."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from PIL import Image

from ..config import Config
from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..generative.text_plate import TextPlateConfig, apply_text_safe_plates
from ..models import CompositionResult, DesignElement, LayoutResult
from .background import BackgroundExtender
from .color import BlendMode, composite_pil_over, parse_blend_mode
from .content_aware_fit import ContentAwareFitStrategy, FitMode
from .effects import parse_drop_shadow_effect, render_drop_shadow
from .resize import high_quality_resize

logger = logging.getLogger("autobanner.composition")


class CompositionEngine:
    """Compose final image from elements and layout.

    Features:
    - High-quality scaling with gamma correction
    - Background extension/generation
    - Layer blending with effects
    - AI-powered inpainting (optional)
    - Content-aware fit for flat (single-layer) images
    """

    def __init__(self, use_ai_inpainting: bool = True) -> None:
        self.bg_extender = BackgroundExtender(use_ai_inpainting=use_ai_inpainting)
        self.content_aware_fit = ContentAwareFitStrategy(extender=self.bg_extender)

    def compose(
        self,
        elements: list[DesignElement],
        layout_results: list[LayoutResult],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        bg_outpaint_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> CompositionResult:
        """Compose final image.

        For flat images (single PNG/JPG from ImageParser), uses the
        content-aware fit strategy for dramatically better quality.
        For multi-layer sources (PSD), uses the existing zone-based
        layout and composition pipeline.

        Args:
            elements: List of design elements with images.
            layout_results: Layout results with new positions.
            source_size: Original canvas size.
            target_size: Target canvas size.

        Returns:
            CompositionResult with final image.
        """
        warnings: list[str] = []

        # Check if this is a flat image source — use content-aware fit
        if self._is_flat_image_source(elements):
            result_image = self._compose_flat_image(
                elements[0], source_size, target_size
            )
            return CompositionResult(
                image=result_image.convert("RGB"),
                layout_results=layout_results,
                warnings=warnings,
            )

        # Standard multi-element composition (PSD, etc.)
        return self._compose_multi_element(
            elements, layout_results, source_size, target_size, bg_outpaint_fn
        )

    @staticmethod
    def _is_flat_image_source(elements: list[DesignElement]) -> bool:
        """Check if elements come from a flat image (single-layer PNG/JPG).

        Detection is based on the ``_source_type`` metadata set by ImageParser.
        Falls back safely to False if metadata is absent.
        """
        if len(elements) != 1:
            return False
        return elements[0].effects.get("_source_type") == "flat_image"

    def _compose_flat_image(
        self,
        element: DesignElement,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> Image.Image:
        """Compose a flat image using content-aware fit strategy.

        This produces dramatically better results than the old blur-extend
        approach by intelligently scaling and extending edges.
        """
        if not element.image:
            return Image.new("RGBA", target_size, (255, 255, 255, 255))

        logger.info(
            "Using content-aware fit: %dx%d -> %dx%d",
            source_size[0], source_size[1],
            target_size[0], target_size[1],
        )

        focus_bbox = self._extract_hero_focus_bbox(element)
        return self.content_aware_fit.fit(
            element.image,
            target_size,
            mode=FitMode.SMART,
            focus_bbox=focus_bbox,
        )

    @staticmethod
    def _extract_hero_focus_bbox(
        element: DesignElement,
    ) -> tuple[int, int, int, int] | None:
        """Extract optional focus window (hero backbox) from element metadata."""
        effects = element.effects or {}
        candidates = (
            effects.get("hero_backbox"),
            effects.get("hero_bbox"),
            effects.get("focus_bbox"),
        )

        for candidate in candidates:
            if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
                continue
            try:
                x, y, w, h = [int(v) for v in candidate]
            except Exception:
                continue
            if w > 0 and h > 0:
                return (x, y, w, h)
        return None

    def _compose_multi_element(
        self,
        elements: list[DesignElement],
        layout_results: list[LayoutResult],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        bg_outpaint_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> CompositionResult:
        """Standard composition for multi-element sources (PSD files)."""
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

        if bg_outpaint_fn is not None:
            try:
                canvas = bg_outpaint_fn(canvas)
            except Exception as e:
                warnings.append(f"Background outpaint failed: {e}")

        canvas, text_plate_meta = self._apply_text_safe_plate(canvas, content_elements, target_size)

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
            metadata={"text_plate": text_plate_meta},
        )

    def _apply_text_safe_plate(
        self,
        canvas: Image.Image,
        content_elements: list[tuple[DesignElement, LayoutResult | None]],
        target_size: tuple[int, int],
    ) -> tuple[Image.Image, dict[str, object]]:
        """Apply optional readability plate behind text roles on busy backgrounds."""
        if not Config.TEXT_SAFE_PLATE_ENABLED:
            return canvas, {"applied": False, "reason": "disabled"}

        text_roles = {
            ElementRole.HEADLINE,
            ElementRole.SUBHEADLINE,
            ElementRole.BODY_TEXT,
            ElementRole.CTA,
            ElementRole.LABEL,
        }
        avoid_roles = {ElementRole.LOGO, ElementRole.HERO_IMAGE}

        text_boxes: list[tuple[int, int, int, int]] = []
        avoid_mask = np.zeros((target_size[1], target_size[0]), dtype=bool)

        for elem, layout in content_elements:
            if layout is None or not layout.visible:
                continue
            box = layout.new_bbox
            x1 = max(0, box.x)
            y1 = max(0, box.y)
            x2 = min(target_size[0], box.x2)
            y2 = min(target_size[1], box.y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if elem.role in text_roles:
                text_boxes.append((x1, y1, x2 - x1, y2 - y1))
            if elem.role in avoid_roles:
                avoid_mask[y1:y2, x1:x2] = True

        if not text_boxes:
            return canvas, {"applied": False, "reason": "no_text_boxes"}

        plate_cfg = TextPlateConfig(
            enabled=Config.TEXT_SAFE_PLATE_ENABLED,
            style=Config.TEXT_SAFE_PLATE_STYLE,
            busy_threshold=Config.TEXT_SAFE_BUSY_THRESHOLD,
            padding=Config.TEXT_SAFE_PLATE_PADDING,
            feather_radius=Config.TEXT_SAFE_PLATE_FEATHER,
            opacity=Config.TEXT_SAFE_PLATE_OPACITY,
            corner_radius=Config.TEXT_SAFE_PLATE_RADIUS,
        )

        plated, meta = apply_text_safe_plates(
            background=canvas,
            text_boxes=text_boxes,
            avoid_mask=avoid_mask,
            config=plate_cfg,
        )
        return plated, dict(meta)

    def _compose_background(
        self,
        canvas: Image.Image,
        bg_elements: list[tuple[DesignElement, LayoutResult | None]],
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

        canvas = composite_pil_over(
            canvas,
            bg_image,
            (0, 0),
            use_linear=Config.USE_LINEAR_COMPOSITING,
            blend_mode=BlendMode.NORMAL,
        )

        # Add overlays
        for elem, _layout in bg_elements[1:]:
            if elem.role == ElementRole.OVERLAY and elem.image:
                overlay = elem.image.convert("RGBA")
                overlay = high_quality_resize(overlay, target_size)
                canvas = composite_pil_over(
                    canvas,
                    overlay,
                    (0, 0),
                    use_linear=Config.USE_LINEAR_COMPOSITING,
                    blend_mode=BlendMode.NORMAL,
                )

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

        # Optional drop-shadow (rendered below element)
        drop_shadow = parse_drop_shadow_effect(element.effects)
        if drop_shadow is not None:
            scaled_shadow = drop_shadow.scaled(layout.scale_factor)
            shadow_img, shadow_off = render_drop_shadow(resized, scaled_shadow)
            canvas = composite_pil_over(
                canvas,
                shadow_img,
                (layout.new_bbox.x + shadow_off[0], layout.new_bbox.y + shadow_off[1]),
                use_linear=Config.USE_LINEAR_COMPOSITING,
                blend_mode=BlendMode.NORMAL,
            )

        # Blend main element
        blend_mode, supported = parse_blend_mode(element.blend_mode)
        if not supported:
            logger.warning(
                "Unsupported blend mode '%s' on element '%s'; falling back to normal",
                element.blend_mode,
                element.name,
            )

        canvas = composite_pil_over(
            canvas,
            resized,
            (layout.new_bbox.x, layout.new_bbox.y),
            use_linear=Config.USE_LINEAR_COMPOSITING,
            blend_mode=blend_mode,
        )

        return canvas
