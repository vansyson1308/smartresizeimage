"""PSD file parser for AutoBanner."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from ..config import Config
from ..exceptions import ParseError
from ..models import BoundingBox, DesignElement
from .base_parser import BaseParser

logger = logging.getLogger("autobanner.parser.psd")

# Optional import
try:
    from psd_tools import PSDImage

    HAS_PSD_TOOLS = True
except ImportError:
    HAS_PSD_TOOLS = False
    logger.info("psd-tools not installed. Run: pip install psd-tools")


class PSDParser(BaseParser):
    """Parse PSD files and extract all layers as DesignElements.

    Supports:
    - Pixel layers (images, photos)
    - Type layers (text with font info)
    - Shape layers
    - Groups (with hierarchy)
    - Layer effects (shadow, glow, etc.)
    - Blend modes and opacity
    """

    SUPPORTED_EXTENSIONS = {".psd"}

    def __init__(self) -> None:
        self.elements: list[DesignElement] = []
        self.psd_size: tuple[int, int] = (0, 0)
        self._z_counter = 0

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, psd_path: str) -> tuple[list[DesignElement], tuple[int, int]]:
        """Parse PSD file and return list of DesignElements.

        Args:
            psd_path: Path to PSD file.

        Returns:
            Tuple of (elements list, (width, height)).

        Raises:
            ParseError: If psd-tools is not installed or parsing fails.
        """
        if not HAS_PSD_TOOLS:
            raise ParseError("psd-tools is required. Run: pip install psd-tools")

        try:
            psd = PSDImage.open(psd_path)
        except Exception as e:
            raise ParseError(f"Failed to open PSD file '{psd_path}': {e}") from e

        self.psd_size = (psd.width, psd.height)
        self.elements = []
        self._z_counter = 0

        # Parse all layers recursively
        self._parse_layers(psd, parent_id=None)

        # Reverse z_index so top layers have higher values
        max_z = self._z_counter
        for elem in self.elements:
            elem.z_index = max_z - elem.z_index

        return self.elements, self.psd_size

    def _parse_layers(self, container: object, parent_id: str | None = None) -> None:
        """Recursively parse layers."""
        for layer in container:
            if not layer.visible:
                continue

            element = self._layer_to_element(layer, parent_id)
            if element:
                self.elements.append(element)

                # Parse children if it's a group
                if hasattr(layer, "__iter__") and layer.kind == "group":
                    self._parse_layers(layer, parent_id=element.id)

    def _layer_to_element(
        self, layer: object, parent_id: str | None
    ) -> DesignElement | None:
        """Convert a PSD layer to DesignElement."""
        try:
            # Get bounding box
            bbox = layer.bbox
            if not bbox or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return None

            element_bbox = BoundingBox(
                x=bbox[0],
                y=bbox[1],
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
            )

            # Skip tiny elements
            if (
                element_bbox.width < Config.MIN_ELEMENT_SIZE
                or element_bbox.height < Config.MIN_ELEMENT_SIZE
            ):
                return None

            # Generate unique ID using z_counter to avoid collisions
            unique_seed = f"{layer.name}_{self._z_counter}"
            elem_id = f"{layer.name}_{hashlib.md5(unique_seed.encode()).hexdigest()[:8]}"

            # Get layer type
            layer_type = layer.kind if hasattr(layer, "kind") else "unknown"

            # Create element
            element = DesignElement(
                id=elem_id,
                name=layer.name,
                layer_type=layer_type,
                bbox=element_bbox,
                visible=layer.visible,
                opacity=layer.opacity / 255.0 if hasattr(layer, "opacity") else 1.0,
                blend_mode=str(layer.blend_mode) if hasattr(layer, "blend_mode") else "normal",
                parent_id=parent_id,
                z_index=self._z_counter,
            )
            self._z_counter += 1

            # Extract text content for type layers
            if layer_type == "type" and hasattr(layer, "text"):
                element.text_content = layer.text
                element.font_info = self._extract_font_info(layer)

            # Render layer image (with transparency)
            if layer_type != "group":
                try:
                    element.image = layer.composite()
                except Exception as e:
                    logger.warning("Could not render layer '%s': %s", layer.name, e)

            # Extract effects
            element.effects = self._extract_effects(layer)

            return element

        except Exception as e:
            logger.warning("Error parsing layer '%s': %s", getattr(layer, "name", "unknown"), e)
            return None

    def _extract_font_info(self, layer: object) -> dict:
        """Extract font information from type layer."""
        try:
            info = {
                "text": layer.text if hasattr(layer, "text") else "",
            }

            # Try to get font details from engine_dict
            if hasattr(layer, "engine_dict") and layer.engine_dict:
                ed = layer.engine_dict
                if "StyleRun" in ed:
                    style_run = ed["StyleRun"]
                    if "RunArray" in style_run and style_run["RunArray"]:
                        first_run = style_run["RunArray"][0]
                        if "StyleSheet" in first_run:
                            ss = first_run["StyleSheet"]["StyleSheetData"]
                            info["font_size"] = ss.get("FontSize", 12)
                            info["font_name"] = ss.get("Font", "Unknown")
                            info["color"] = ss.get("FillColor", {}).get(
                                "Values", [1, 0, 0, 0]
                            )

            return info
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            logger.debug("Could not extract font info: %s", e)
            return {"text": getattr(layer, "text", "")}

    def _extract_effects(self, layer: object) -> dict:
        """Extract layer effects (shadow, glow, etc.)."""
        effects: dict = {}
        try:
            if hasattr(layer, "effects") and layer.effects:
                for effect in layer.effects:
                    effects[effect.name] = {
                        "enabled": effect.enabled,
                        "opacity": getattr(effect, "opacity", 100),
                    }
        except (AttributeError, TypeError) as e:
            logger.debug("Could not extract effects: %s", e)
        return effects
