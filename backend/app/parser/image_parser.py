"""Image file parser (PNG, JPG, WEBP) for AutoBanner."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ..enums import ElementRole
from ..exceptions import ParseError
from ..models import BoundingBox, DesignElement
from .base_parser import BaseParser

logger = logging.getLogger("autobanner.parser.image")


class ImageParser(BaseParser):
    """Parse raster image files (PNG, JPG, WEBP) as single-element designs.

    For non-PSD formats, the parser treats the entire image as a single
    BACKGROUND element, allowing the re-layout pipeline to process it
    with background extension and composition.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: str) -> tuple[list[DesignElement], tuple[int, int]]:
        """Parse image file into a single DesignElement.

        Args:
            file_path: Path to the image file.

        Returns:
            Tuple of ([single element], (width, height)).

        Raises:
            ParseError: If the image cannot be opened.
        """
        try:
            img = Image.open(file_path)
            img = img.convert("RGBA")
        except Exception as e:
            raise ParseError(f"Failed to open image '{file_path}': {e}") from e

        w, h = img.size
        logger.info("Parsed image %s: %dx%d", Path(file_path).name, w, h)

        element = DesignElement(
            id="source_image_0",
            name="Source Image",
            layer_type="pixel",
            bbox=BoundingBox(x=0, y=0, width=w, height=h),
            image=img,
            role=ElementRole.BACKGROUND,
            priority=9,
            visible=True,
            opacity=1.0,
            z_index=0,
        )

        return [element], (w, h)
