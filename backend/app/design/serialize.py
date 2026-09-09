"""JSON serialization and migration for ``DesignDocument``."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .document import (
    SCHEMA_VERSION,
    AllowedTransforms,
    AssetRef,
    Constraint,
    DesignDocument,
    Element,
    FontRef,
    Geometry,
    Provenance,
    TextContent,
    TextRun,
    TextStyle,
)


def document_to_dict(doc: DesignDocument) -> dict[str, Any]:
    payload = asdict(doc)
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def _provenance(d: dict | None) -> Provenance:
    d = d or {}
    return Provenance(
        origin=d.get("origin", "import"),
        source_ref=d.get("source_ref"),
        confidence=float(d.get("confidence", 1.0)),
        notes=d.get("notes", ""),
    )


def _style(d: dict | None) -> TextStyle:
    d = d or {}
    base = TextStyle()
    return TextStyle(
        font_family=d.get("font_family", base.font_family),
        font_size=float(d.get("font_size", base.font_size)),
        weight=d.get("weight", base.weight),
        italic=bool(d.get("italic", base.italic)),
        color=d.get("color", base.color),
        letter_spacing=float(d.get("letter_spacing", base.letter_spacing)),
        line_height=float(d.get("line_height", base.line_height)),
        align=d.get("align", base.align),
        uppercase=bool(d.get("uppercase", base.uppercase)),
    )


def _text(d: dict | None) -> TextContent | None:
    if d is None:
        return None
    runs = [
        TextRun(text=r.get("text", ""), style=_style(r.get("style"))) for r in d.get("runs", [])
    ]
    return TextContent(
        runs=runs,
        locale=d.get("locale", "en"),
        max_lines=d.get("max_lines"),
        protected=bool(d.get("protected", False)),
    )


def _asset(d: dict | None) -> AssetRef | None:
    if d is None:
        return None
    return AssetRef(
        asset_id=d["asset_id"],
        content_hash=d["content_hash"],
        mime=d.get("mime", "image/png"),
        width=int(d.get("width", 0)),
        height=int(d.get("height", 0)),
        path=d["path"],
        original_name=d.get("original_name", ""),
    )


def _allowed(d: dict | None) -> AllowedTransforms:
    d = d or {}
    base = AllowedTransforms()
    return AllowedTransforms(
        move=bool(d.get("move", base.move)),
        scale_uniform=bool(d.get("scale_uniform", base.scale_uniform)),
        scale_free=bool(d.get("scale_free", base.scale_free)),
        crop=bool(d.get("crop", base.crop)),
        reflow=bool(d.get("reflow", base.reflow)),
        hide=bool(d.get("hide", base.hide)),
    )


def _element(d: dict) -> Element:
    g = d["geometry"]
    return Element(
        id=d["id"],
        kind=d["kind"],
        name=d.get("name", d["id"]),
        role=d.get("role", "unknown"),
        geometry=Geometry(
            x=float(g["x"]),
            y=float(g["y"]),
            width=float(g["width"]),
            height=float(g["height"]),
            rotation=float(g.get("rotation", 0.0)),
        ),
        z_index=int(d.get("z_index", 0)),
        visible=bool(d.get("visible", True)),
        opacity=float(d.get("opacity", 1.0)),
        blend_mode=d.get("blend_mode", "normal"),
        text=_text(d.get("text")),
        asset=_asset(d.get("asset")),
        shape=d.get("shape"),
        parent_id=d.get("parent_id"),
        locked=bool(d.get("locked", False)),
        priority=int(d.get("priority", 5)),
        allowed=_allowed(d.get("allowed")),
        provenance=_provenance(d.get("provenance")),
        role_confidence=float(d.get("role_confidence", 1.0)),
        effects=dict(d.get("effects", {})),
    )


def _constraint(d: dict) -> Constraint:
    return Constraint(
        id=d["id"],
        type=d["type"],
        elements=list(d.get("elements", [])),
        params=dict(d.get("params", {})),
        hard=bool(d.get("hard", True)),
        provenance=_provenance(d.get("provenance")),
        enabled=bool(d.get("enabled", True)),
    )


def _font(d: dict) -> FontRef:
    return FontRef(
        family=d["family"],
        weight=d.get("weight", "regular"),
        italic=bool(d.get("italic", False)),
        path=d.get("path"),
        status=d.get("status", "unresolved"),
        substitute=d.get("substitute"),
    )


def migrate_document_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an on-disk document dict to the current schema.

    Every released schema version needs an explicit step here; unknown or
    newer versions are rejected rather than guessed.
    """
    version = str(payload.get("schema_version", "0.0"))
    if version == SCHEMA_VERSION:
        return payload
    if version == "0.0":
        # Pre-release payloads had no version field. Treat as 1.0-compatible.
        migrated = dict(payload)
        migrated["schema_version"] = "1.0"
        return migrated
    raise ValueError(f"unsupported design schema version '{version}' (current {SCHEMA_VERSION})")


def document_from_dict(payload: dict[str, Any]) -> DesignDocument:
    data = migrate_document_dict(payload)
    doc = DesignDocument(
        id=data["id"],
        name=data.get("name", data["id"]),
        canvas_width=int(data["canvas_width"]),
        canvas_height=int(data["canvas_height"]),
        elements=[_element(e) for e in data.get("elements", [])],
        constraints=[_constraint(c) for c in data.get("constraints", [])],
        fonts=[_font(f) for f in data.get("fonts", [])],
        metadata=dict(data.get("metadata", {})),
        schema_version=SCHEMA_VERSION,
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )
    problems = doc.validate()
    if problems:
        raise ValueError("invalid design document: " + "; ".join(problems))
    return doc
