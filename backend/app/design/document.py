"""Typed design document (schema 1.0)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

SCHEMA_VERSION = "1.0"

ELEMENT_KINDS = ("text", "image", "shape", "group")
ORIGINS = ("psd_layer", "flat_image", "user", "generated", "recovered", "fixture", "import")

CONSTRAINT_TYPES = {
    # Elements move/scale together and keep their relative arrangement.
    "keep_group": {"elements": ">=2", "params": []},
    # elements[0] must be placed below elements[1] (reading order).
    "order_below": {"elements": "==2", "params": []},
    # Clear space around elements[0] scales with its size (ratio of its height).
    "clear_space": {"elements": "==1", "params": ["ratio"]},
    # Element must remain fully visible in every variant (e.g. legal copy).
    "keep_visible": {"elements": ">=1", "params": []},
    # Minimum rendered glyph size in px for a text element.
    "min_text_size": {"elements": "==1", "params": ["px"]},
    # Overlap between the two elements is intentional and must not be flagged.
    "allowed_overlap": {"elements": "==2", "params": []},
    # Keep element anchored to a canvas edge: params.edge in top/bottom/left/right.
    "anchor_edge": {"elements": "==1", "params": ["edge"]},
    # Element scale relative to canvas height must stay within [min, max].
    "scale_range": {"elements": "==1", "params": ["min", "max"]},
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class Provenance:
    """Where a value came from and how much to trust it."""

    origin: str = "import"
    source_ref: str | None = None
    confidence: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown provenance origin '{self.origin}'")
        self.confidence = float(max(0.0, min(1.0, self.confidence)))


@dataclass
class Geometry:
    """Axis-aligned box on the master canvas, in pixels (floats allowed)."""

    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass
class TextStyle:
    font_family: str = "DejaVu Sans"
    font_size: float = 24.0
    weight: str = "regular"  # regular | bold | light | medium | ...
    italic: bool = False
    color: str = "#000000"  # #RRGGBB or #RRGGBBAA
    letter_spacing: float = 0.0
    line_height: float = 1.2
    align: str = "left"  # left | center | right
    uppercase: bool = False


@dataclass
class TextRun:
    text: str
    style: TextStyle = field(default_factory=TextStyle)


@dataclass
class TextContent:
    """Native text: runs with styles, plus locale and wrapping policy."""

    runs: list[TextRun] = field(default_factory=list)
    locale: str = "en"
    max_lines: int | None = None
    # When True the string is a commercial claim (price, legal) and must be rendered verbatim.
    protected: bool = False

    @property
    def plain(self) -> str:
        return "".join(r.text for r in self.runs)

    @property
    def primary_style(self) -> TextStyle:
        return self.runs[0].style if self.runs else TextStyle()

    def replace_text(self, text: str) -> None:
        """Replace content while keeping the first run's style."""
        style = self.primary_style
        self.runs = [TextRun(text=text, style=style)]


@dataclass
class AssetRef:
    asset_id: str
    content_hash: str
    mime: str
    width: int
    height: int
    path: str  # relative path inside the project
    original_name: str = ""


@dataclass
class AllowedTransforms:
    move: bool = True
    scale_uniform: bool = True
    scale_free: bool = False  # distortion; off for logos, product shots, text
    crop: bool = False
    reflow: bool = False  # text may re-wrap / change font size within policy
    hide: bool = False  # element may be dropped in small variants


@dataclass
class FontRef:
    family: str
    weight: str = "regular"
    italic: bool = False
    path: str | None = None
    status: str = "unresolved"  # available | missing | substituted | unresolved
    substitute: str | None = None


@dataclass
class Element:
    id: str
    kind: str
    name: str
    role: str
    geometry: Geometry
    z_index: int = 0
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"
    text: TextContent | None = None
    asset: AssetRef | None = None
    shape: dict | None = None
    parent_id: str | None = None
    locked: bool = False
    priority: int = 5
    allowed: AllowedTransforms = field(default_factory=AllowedTransforms)
    provenance: Provenance = field(default_factory=Provenance)
    role_confidence: float = 1.0
    effects: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ELEMENT_KINDS:
            raise ValueError(f"unknown element kind '{self.kind}'")
        if self.kind == "text" and self.text is None:
            self.text = TextContent()

    @property
    def is_background(self) -> bool:
        return self.role in ("background", "background_pattern", "overlay")


@dataclass
class Constraint:
    id: str
    type: str
    elements: list[str]
    params: dict = field(default_factory=dict)
    hard: bool = True
    provenance: Provenance = field(default_factory=Provenance)
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.type not in CONSTRAINT_TYPES:
            raise ValueError(f"unknown constraint type '{self.type}'")


@dataclass
class DesignDocument:
    id: str
    name: str
    canvas_width: int
    canvas_height: int
    elements: list[Element] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    fonts: list[FontRef] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    # ---- lookups -----------------------------------------------------------------
    def element(self, element_id: str) -> Element:
        for e in self.elements:
            if e.id == element_id:
                return e
        raise KeyError(element_id)

    def has_element(self, element_id: str) -> bool:
        return any(e.id == element_id for e in self.elements)

    def content_elements(self) -> list[Element]:
        return [e for e in self.elements if not e.is_background and e.kind != "group"]

    def background_elements(self) -> list[Element]:
        return [e for e in self.elements if e.is_background]

    def constraints_for(self, element_id: str) -> list[Constraint]:
        return [c for c in self.constraints if element_id in c.elements and c.enabled]

    def allowed_overlaps(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for c in self.constraints:
            if c.enabled and c.type == "allowed_overlap" and len(c.elements) == 2:
                pairs.add((c.elements[0], c.elements[1]))
        return pairs

    # ---- mutation helpers --------------------------------------------------------
    def touch(self) -> None:
        self.updated_at = utc_now()

    def add_constraint(self, constraint: Constraint) -> None:
        for eid in constraint.elements:
            if not self.has_element(eid):
                raise KeyError(eid)
        self.constraints.append(constraint)
        self.touch()

    def remove_constraint(self, constraint_id: str) -> bool:
        before = len(self.constraints)
        self.constraints = [c for c in self.constraints if c.id != constraint_id]
        changed = len(self.constraints) != before
        if changed:
            self.touch()
        return changed

    def remove_element(self, element_id: str) -> bool:
        before = len(self.elements)
        self.elements = [e for e in self.elements if e.id != element_id]
        self.constraints = [c for c in self.constraints if element_id not in c.elements]
        changed = len(self.elements) != before
        if changed:
            self.touch()
        return changed

    def reorder(self, element_id: str, new_z: int) -> None:
        elem = self.element(element_id)
        elem.z_index = int(new_z)
        self.normalize_z()
        self.touch()

    def normalize_z(self) -> None:
        for i, e in enumerate(sorted(self.elements, key=lambda el: (el.z_index, el.id))):
            e.z_index = i

    def validate(self) -> list[str]:
        """Return schema-level problems (empty list means valid)."""
        problems: list[str] = []
        ids = [e.id for e in self.elements]
        if len(ids) != len(set(ids)):
            problems.append("duplicate element ids")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            problems.append("canvas must be positive")
        for e in self.elements:
            if e.geometry.width <= 0 or e.geometry.height <= 0:
                problems.append(f"element {e.id} has non-positive size")
            if e.kind == "text" and (e.text is None or not e.text.plain.strip()):
                problems.append(f"text element {e.id} has no content")
            if e.kind == "image" and e.asset is None:
                problems.append(f"image element {e.id} has no asset")
            if e.parent_id and e.parent_id not in ids:
                problems.append(f"element {e.id} references missing parent {e.parent_id}")
        for c in self.constraints:
            spec = CONSTRAINT_TYPES[c.type]
            for eid in c.elements:
                if eid not in ids:
                    problems.append(f"constraint {c.id} references missing element {eid}")
            rule = spec["elements"]
            n = len(c.elements)
            if rule == "==1" and n != 1 or rule == "==2" and n != 2:
                problems.append(f"constraint {c.id} needs {rule} elements, has {n}")
            if rule == ">=2" and n < 2 or rule == ">=1" and n < 1:
                problems.append(f"constraint {c.id} needs {rule} elements, has {n}")
            for p in spec["params"]:
                if p not in c.params:
                    problems.append(f"constraint {c.id} missing param '{p}'")
        return problems
