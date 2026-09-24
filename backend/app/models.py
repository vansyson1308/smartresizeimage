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

    def fit_aspect(self, src_w: int, src_h: int, tolerance: float = 0.01) -> BoundingBox:
        """Largest box with the ``src_w:src_h`` aspect ratio centred inside this one.

        Raster layers are never re-rendered, so they must be scaled uniformly:
        stretching a logo or a line of type to a reflowed box distorts the brand.
        Returns ``self`` when the aspect ratios already match within ``tolerance``.
        """
        if src_w <= 0 or src_h <= 0 or self.width <= 0 or self.height <= 0:
            return self
        box_ratio = self.width / self.height
        src_ratio = src_w / src_h
        if abs(box_ratio / src_ratio - 1.0) <= tolerance:
            return self
        scale = min(self.width / src_w, self.height / src_h)
        w = max(1, int(round(src_w * scale)))
        h = max(1, int(round(src_h * scale)))
        return BoundingBox(self.x + (self.width - w) // 2, self.y + (self.height - h) // 2, w, h)


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
    metadata: dict = field(default_factory=dict)
    gates_passed: bool = True
    fail_reasons: list[str] = field(default_factory=list)
    used_fallback: bool = False
