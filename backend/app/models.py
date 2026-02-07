"""Data structures for AutoBanner."""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image as PILImage

from .enums import ElementRole


@dataclass
class BoundingBox:
    """Bounding box for elements."""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def scale(self, factor: float) -> BoundingBox:
        return BoundingBox(
            x=int(self.x * factor),
            y=int(self.y * factor),
            width=int(self.width * factor),
            height=int(self.height * factor),
        )

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)


@dataclass
class DesignElement:
    """Represents a single design element extracted from a source file."""
    id: str
    name: str
    layer_type: str  # pixel, type, shape, group
    bbox: BoundingBox
    image: PILImage.Image | None = None
    text_content: str | None = None
    font_info: dict | None = None
    role: ElementRole = ElementRole.UNKNOWN
    priority: int = 5  # 1 = highest, 9 = lowest
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"
    effects: dict = field(default_factory=dict)
    children: list[DesignElement] = field(default_factory=list)
    parent_id: str | None = None
    z_index: int = 0

    # Layout hints
    anchor_horizontal: str = "center"  # left, center, right
    anchor_vertical: str = "center"  # top, center, bottom
    scalable: bool = True
    min_scale: float = 0.3
    max_scale: float = 2.0
    maintain_aspect: bool = True


@dataclass
class LayoutZone:
    """A zone in the layout grid."""
    id: str
    bbox: BoundingBox
    allowed_roles: list[ElementRole]
    priority: int = 1


@dataclass
class LayoutResult:
    """Result of layout calculation for one element."""
    element_id: str
    new_bbox: BoundingBox
    scale_factor: float
    visible: bool = True


@dataclass
class CompositionResult:
    """Final composition result."""
    image: PILImage.Image
    layout_results: list[LayoutResult]
    warnings: list[str] = field(default_factory=list)
