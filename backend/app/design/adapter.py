"""Bridges between parser/engine ``DesignElement`` lists and ``DesignDocument``.

The existing parser, classifier, layout and composition engines operate on
``DesignElement``. The document is the source of truth; these adapters convert
in both directions so the engines can be reused while the representation
evolves.
"""

from __future__ import annotations

import logging

from PIL import Image

from ..enums import ElementRole
from ..models import BoundingBox, DesignElement
from .assets import AssetStore
from .document import (
    AllowedTransforms,
    Constraint,
    DesignDocument,
    Element,
    FontRef,
    Geometry,
    Provenance,
    TextContent,
    TextRun,
    TextStyle,
    new_id,
)
from .fonts import FontRegistry, default_registry

logger = logging.getLogger("autobanner.design.adapter")

_ROLE_SOURCE_CONFIDENCE = {
    "rule": 0.9,
    "ai": 0.7,
    "heuristic": 0.5,
    "group": 1.0,
    "fixture": 1.0,
    "user": 1.0,
    "none": 0.2,
}

_TEXT_ROLES = {"headline", "subheadline", "body_text", "cta", "label", "badge"}
_IDENTITY_ROLES = {"logo"}
_SUBJECT_ROLES = {"hero_image", "photo", "illustration", "icon"}


def _color_from_font_info(info: dict | None) -> str:
    if not info:
        return "#000000"
    values = info.get("color")
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        vals = [float(v) for v in values[:4]]
        if len(vals) == 4 and all(0.0 <= v <= 1.0 for v in vals):
            # psd-tools FillColor "Values" is [A, R, G, B] in 0..1
            a, r, g, b = vals
            return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        if all(0.0 <= v <= 1.0 for v in vals[:3]):
            r, g, b = vals[:3]
            return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        if all(0 <= v <= 255 for v in vals[:3]):
            r, g, b = vals[:3]
            return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    return "#000000"


def _allowed_for_role(role: str, kind: str) -> AllowedTransforms:
    if kind == "text":
        return AllowedTransforms(move=True, scale_uniform=True, scale_free=False, reflow=True)
    if role in _IDENTITY_ROLES:
        return AllowedTransforms(move=True, scale_uniform=True, scale_free=False, crop=False)
    if role in _SUBJECT_ROLES:
        return AllowedTransforms(move=True, scale_uniform=True, scale_free=False, crop=False)
    if role in ("background", "background_pattern", "overlay"):
        return AllowedTransforms(move=False, scale_uniform=True, scale_free=True, crop=True)
    return AllowedTransforms()


def _origin_for(elem: DesignElement, default: str) -> str:
    if elem.effects.get("_source_type") == "flat_image":
        return "flat_image"
    return default


def document_from_elements(
    elements: list[DesignElement],
    source_size: tuple[int, int],
    assets: AssetStore,
    *,
    name: str = "Untitled",
    source_ref: str | None = None,
    origin: str = "psd_layer",
    registry: FontRegistry | None = None,
) -> DesignDocument:
    """Build a document from parsed+classified elements, storing pixel assets."""
    reg = registry or default_registry()
    doc = DesignDocument(
        id=new_id("doc"),
        name=name,
        canvas_width=int(source_size[0]),
        canvas_height=int(source_size[1]),
    )
    fonts: dict[tuple[str, str, bool], FontRef] = {}

    for elem in elements:
        role = elem.role.value if isinstance(elem.role, ElementRole) else str(elem.role)
        is_text = elem.layer_type == "type" and bool(elem.text_content)
        if elem.layer_type == "group":
            kind = "group"
        elif is_text:
            kind = "text"
        else:
            kind = "image" if elem.image is not None else "shape"

        source = str(elem.effects.get("_role_source", "none"))
        confidence = _ROLE_SOURCE_CONFIDENCE.get(source, 0.5)
        prov = Provenance(
            origin=_origin_for(elem, origin),
            source_ref=f"{source_ref or ''}#{elem.name}" if source_ref else elem.name,
            confidence=confidence,
            notes=f"role from {source}",
        )
        geometry = Geometry(
            x=float(elem.bbox.x),
            y=float(elem.bbox.y),
            width=float(max(1, elem.bbox.width)),
            height=float(max(1, elem.bbox.height)),
        )

        text = None
        asset = None
        shape = None
        if kind == "text":
            info = elem.font_info or {}
            family = str(info.get("font_name") or info.get("family") or "DejaVu Sans")
            size = float(info.get("font_size") or max(8.0, elem.bbox.height * 0.7))
            style = TextStyle(
                font_family=family,
                font_size=size,
                color=_color_from_font_info(info),
            )
            text = TextContent(runs=[TextRun(text=str(elem.text_content), style=style)])
            key = (family, style.weight, style.italic)
            if key not in fonts:
                resolved = reg.resolve(family, style.weight, style.italic)
                fonts[key] = FontRef(
                    family=family,
                    weight=style.weight,
                    italic=style.italic,
                    path=resolved.path,
                    status=resolved.status,
                    substitute=resolved.family if resolved.substituted else None,
                )
            # Keep the original raster as a reference asset for fidelity checks.
            if elem.image is not None:
                asset = assets.put(elem.image, elem.name)
        elif kind == "image":
            asset = assets.put(elem.image, elem.name)  # type: ignore[arg-type]
        elif kind == "shape":
            shape = {"type": "rect", "fill": "#00000000"}

        # Clean internal metadata copied into effects
        effects = {k: v for k, v in (elem.effects or {}).items() if not k.startswith("_")}

        doc.elements.append(
            Element(
                id=elem.id,
                kind=kind,
                name=elem.name,
                role=role,
                geometry=geometry,
                z_index=int(elem.z_index),
                visible=bool(elem.visible),
                opacity=float(elem.opacity),
                blend_mode=str(elem.blend_mode or "normal"),
                text=text,
                asset=asset,
                shape=shape,
                parent_id=elem.parent_id,
                priority=int(elem.priority),
                allowed=_allowed_for_role(role, kind),
                provenance=prov,
                role_confidence=confidence,
                effects=effects,
            )
        )

    doc.fonts = list(fonts.values())
    doc.normalize_z()
    doc.constraints.extend(propose_constraints(doc))
    return doc


def propose_constraints(doc: DesignDocument) -> list[Constraint]:
    """Propose reviewable default constraints from roles (provenance: generated)."""
    proposals: list[Constraint] = []
    prov = Provenance(origin="generated", confidence=0.6, notes="proposed from roles")
    by_role: dict[str, list[Element]] = {}
    for e in doc.content_elements():
        by_role.setdefault(e.role, []).append(e)

    for logo in by_role.get("logo", []):
        proposals.append(
            Constraint(
                id=new_id("c"),
                type="clear_space",
                elements=[logo.id],
                params={"ratio": 0.5},
                hard=False,
                provenance=prov,
            )
        )
    for role in ("headline", "cta", "logo"):
        for e in by_role.get(role, []):
            proposals.append(
                Constraint(
                    id=new_id("c"), type="keep_visible", elements=[e.id], hard=True, provenance=prov
                )
            )
    headlines = by_role.get("headline", [])
    ctas = by_role.get("cta", [])
    if headlines and ctas:
        proposals.append(
            Constraint(
                id=new_id("c"),
                type="order_below",
                elements=[ctas[0].id, headlines[0].id],
                hard=False,
                provenance=prov,
            )
        )
    subs = by_role.get("subheadline", [])
    if headlines and subs:
        proposals.append(
            Constraint(
                id=new_id("c"),
                type="keep_group",
                elements=[headlines[0].id, subs[0].id],
                hard=False,
                provenance=prov,
            )
        )
    for legal in by_role.get("label", []) + by_role.get("body_text", []):
        proposals.append(
            Constraint(
                id=new_id("c"),
                type="min_text_size",
                elements=[legal.id],
                params={"px": 10},
                hard=True,
                provenance=prov,
            )
        )
    return proposals


def elements_from_document(
    doc: DesignDocument,
    assets: AssetStore,
    *,
    registry: FontRegistry | None = None,
    text_overrides: dict[str, str] | None = None,
) -> list[DesignElement]:
    """Materialize engine elements. Text elements are native (no raster image)."""
    reg = registry or default_registry()
    overrides = text_overrides or {}
    out: list[DesignElement] = []
    for e in sorted(doc.elements, key=lambda el: el.z_index):
        if not e.visible or e.kind == "group":
            continue
        valid_roles = {r.value for r in ElementRole}
        role = ElementRole(e.role) if e.role in valid_roles else ElementRole.UNKNOWN
        bbox = BoundingBox(
            int(round(e.geometry.x)),
            int(round(e.geometry.y)),
            max(1, int(round(e.geometry.width))),
            max(1, int(round(e.geometry.height))),
        )
        image: Image.Image | None = None
        text_content = None
        font_info = None
        layer_type = "pixel"
        if e.kind == "text" and e.text is not None:
            layer_type = "type"
            text_content = overrides.get(e.id, e.text.plain)
            style = e.text.primary_style
            resolved = reg.resolve(style.font_family, style.weight, style.italic)
            font_info = {
                "family": resolved.path,
                "font_name": style.font_family,
                "font_size": style.font_size,
                "font_status": resolved.status,
                "color": style.color,
                "align": style.align,
            }
        elif e.kind == "image" and e.asset is not None:
            image = assets.get(e.asset)
        elif e.kind == "shape":
            layer_type = "shape"
            fill = (e.shape or {}).get("fill", "#00000000")
            image = Image.new("RGBA", (bbox.width, bbox.height), _hex_rgba(fill))

        de = DesignElement(
            id=e.id,
            name=e.name,
            layer_type=layer_type,
            bbox=bbox,
            image=image,
            text_content=text_content,
            font_info=font_info,
            role=role,
            priority=int(e.priority),
            visible=True,
            opacity=float(e.opacity),
            blend_mode=e.blend_mode,
            effects=dict(e.effects),
            parent_id=e.parent_id,
            z_index=int(e.z_index),
            maintain_aspect=not e.allowed.scale_free,
        )
        if e.provenance.origin == "flat_image" and e.is_background and len(doc.elements) == 1:
            de.effects["_source_type"] = "flat_image"
        out.append(de)
    return out


def _hex_rgba(color: str) -> tuple[int, int, int, int]:
    c = color.lstrip("#")
    if len(c) == 8:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(c[6:8], 16)
    if len(c) == 6:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255
    return 0, 0, 0, 0
