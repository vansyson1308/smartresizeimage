"""Main ReLayout orchestrator engine."""

from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image, ImageDraw

from .classifier import SemanticClassifier
from .composition import CompositionEngine
from .enums import ElementRole
from .layout import LayoutEngine
from .models import CompositionResult, DesignElement
from .parser import get_parser
from .validators import validate_dimensions, validate_file_path

logger = logging.getLogger("autobanner.relayout")


class ReLayoutEngine:
    """Main engine that orchestrates the entire re-layout process."""

    def __init__(self, use_ai: bool = True) -> None:
        self.classifier = SemanticClassifier(use_ai=use_ai)
        self.layout_engine = LayoutEngine()
        self.compositor = CompositionEngine(use_ai_inpainting=use_ai)

        self.elements: list[DesignElement] = []
        self.source_size: tuple[int, int] = (0, 0)
        self.file_path: str | None = None

    def load_file(self, file_path: str) -> dict[str, Any]:
        """Load and analyze a design file (PSD, PNG, JPG, WEBP).

        Args:
            file_path: Path to the design file.

        Returns:
            Dict with analysis results for UI display.

        Raises:
            ValidationError: If file path is invalid.
            ParseError: If parsing fails.
        """
        validate_file_path(file_path)

        self.file_path = file_path

        # Get appropriate parser
        parser = get_parser(file_path)

        # Parse file
        self.elements, self.source_size = parser.parse(file_path)

        # Classify elements
        self.elements = self.classifier.classify_all(self.elements, self.source_size)

        # Prepare analysis for UI
        analysis: dict[str, Any] = {
            "file": os.path.basename(file_path),
            "size": self.source_size,
            "total_layers": len(self.elements),
            "elements": [],
        }

        for elem in self.elements:
            analysis["elements"].append(
                {
                    "id": elem.id,
                    "name": elem.name,
                    "type": elem.layer_type,
                    "role": elem.role.value,
                    "priority": elem.priority,
                    "bbox": {
                        "x": elem.bbox.x,
                        "y": elem.bbox.y,
                        "width": elem.bbox.width,
                        "height": elem.bbox.height,
                    },
                    "has_image": elem.image is not None,
                    "text": (
                        elem.text_content[:50] + "..."
                        if elem.text_content and len(elem.text_content) > 50
                        else elem.text_content
                    ),
                }
            )

        return analysis

    def update_element_role(self, element_id: str, new_role: str) -> bool:
        """Update an element's role (for user correction).

        Args:
            element_id: ID of the element to update.
            new_role: New role value string.

        Returns:
            True if updated successfully.
        """
        for elem in self.elements:
            if elem.id == element_id:
                try:
                    elem.role = ElementRole(new_role)
                    return True
                except ValueError as e:
                    logger.warning("Invalid role '%s': %s", new_role, e)
                    return False
        return False

    def update_element_priority(self, element_id: str, new_priority: int) -> bool:
        """Update an element's priority.

        Args:
            element_id: ID of the element to update.
            new_priority: New priority (1-9).

        Returns:
            True if updated successfully.
        """
        for elem in self.elements:
            if elem.id == element_id:
                elem.priority = max(1, min(9, new_priority))
                return True
        return False

    def relayout(self, target_size: tuple[int, int]) -> CompositionResult:
        """Re-layout elements to target size.

        Args:
            target_size: Target canvas size (width, height).

        Returns:
            CompositionResult with final image.

        Raises:
            ValueError: If no file has been loaded.
            ValidationError: If dimensions are invalid.
        """
        if not self.elements:
            raise ValueError("No file loaded. Call load_file() first.")

        validate_dimensions(target_size[0], target_size[1])

        # Calculate layout
        layout_results = self.layout_engine.calculate_layout(
            self.elements, self.source_size, target_size
        )

        # Compose final image
        result = self.compositor.compose(
            self.elements, layout_results, self.source_size, target_size
        )

        return result

    def batch_relayout(
        self, target_sizes: list[tuple[int, int, str]]
    ) -> dict[str, CompositionResult]:
        """Re-layout to multiple sizes.

        Args:
            target_sizes: List of (width, height, name) tuples.

        Returns:
            Dict mapping name to CompositionResult.
        """
        results: dict[str, CompositionResult] = {}

        for width, height, name in target_sizes:
            try:
                result = self.relayout((width, height))
                results[name] = result
            except Exception as e:
                logger.error("Error processing %s: %s", name, e, exc_info=True)

        return results

    def get_preview_image(self) -> Image.Image | None:
        """Get a preview of the loaded file with element bounding boxes."""
        if not self.elements:
            return None

        # Find background or create white canvas
        bg = None
        for elem in self.elements:
            if elem.role == ElementRole.BACKGROUND and elem.image:
                bg = elem.image.copy()
                break

        if bg is None:
            bg = Image.new("RGBA", self.source_size, (240, 240, 240, 255))
        else:
            bg = bg.convert("RGBA")
            if bg.size != self.source_size:
                bg = bg.resize(self.source_size, Image.Resampling.LANCZOS)

        # Draw bounding boxes
        draw = ImageDraw.Draw(bg)

        # Color coding by role
        role_colors = {
            ElementRole.HEADLINE: (255, 100, 100, 200),
            ElementRole.SUBHEADLINE: (255, 150, 100, 200),
            ElementRole.CTA: (100, 255, 100, 200),
            ElementRole.BADGE: (255, 255, 100, 200),
            ElementRole.LOGO: (100, 100, 255, 200),
            ElementRole.HERO_IMAGE: (255, 100, 255, 200),
            ElementRole.BACKGROUND: (150, 150, 150, 100),
            ElementRole.DECORATION: (200, 200, 200, 100),
        }
        default_color = (180, 180, 180, 150)

        for elem in self.elements:
            if elem.role == ElementRole.BACKGROUND:
                continue

            color = role_colors.get(elem.role, default_color)
            bbox = elem.bbox.to_tuple()

            draw.rectangle(bbox, outline=color[:3], width=2)
            label = f"{elem.role.value[:8]}: {elem.name[:15]}"
            draw.text((bbox[0] + 2, bbox[1] + 2), label, fill=color[:3])

        return bg.convert("RGB")
