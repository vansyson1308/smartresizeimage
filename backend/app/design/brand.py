"""Brand-level rules: carry confirmed rules across an owner's projects of one brand (H5).

A rule confirmed in one project (a user-added or correction-derived constraint on a
logo, CTA, …) is evidence about the brand, not just about that design. This module
unions such rules across projects by (constraint type, element role) and proposes
them into another project of the same brand as reviewable, soft constraints. Only
enabled constraints with provenance ``user`` or ``recovered`` are carried; unconfirmed
``generated`` proposals never propagate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..quality.contract import CheckResult, CheckStatus, Severity
from .document import Constraint, DesignDocument, Provenance, new_id

CARRIED_TYPES = ("scale_range", "min_text_size", "clear_space")
CARRIED_ORIGINS = ("user", "recovered")


@dataclass
class BrandRule:
    type: str
    role: str
    params: dict
    sources: list[str] = field(default_factory=list)  # project names/ids
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "role": self.role,
            "params": dict(self.params),
            "sources": list(self.sources),
            "notes": list(self.notes),
        }


def _merge(rule: BrandRule, params: dict) -> None:
    """Keep the stricter of two parameter sets for the same (type, role)."""
    if rule.type == "scale_range":
        lo = max(float(rule.params.get("min", 0.0)), float(params.get("min", 0.0)))
        hi = min(float(rule.params.get("max", 10.0)), float(params.get("max", 10.0)))
        if hi < lo:
            hi = 1.0
        rule.params = {"min": round(lo, 3), "max": round(hi, 3)}
    elif rule.type == "min_text_size":
        # compare on a common footing: px per 1000 px of measured canvas height
        def norm(p: dict) -> float:
            px = float(p.get("px", 0))
            measured = p.get("measured_on")
            if measured and len(measured) == 2 and int(measured[1]) > 0:
                return px * 1000.0 / int(measured[1])
            return px

        if norm(params) > norm(rule.params):
            rule.params = dict(params)
    elif rule.type == "clear_space":
        rule.params = {
            "ratio": max(float(rule.params.get("ratio", 0.5)), float(params.get("ratio", 0.5)))
        }


def collect_brand_rules(projects: list[tuple[str, DesignDocument]]) -> list[BrandRule]:
    """Union carried constraints across ``(project label, document)`` pairs, keyed by role."""
    rules: dict[tuple[str, str], BrandRule] = {}
    for label, doc in projects:
        roles = {e.id: e.role for e in doc.elements}
        for c in doc.constraints:
            if not c.enabled or c.type not in CARRIED_TYPES:
                continue
            if c.provenance.origin not in CARRIED_ORIGINS:
                continue
            for eid in c.elements:
                role = roles.get(eid)
                if not role or role in ("background", "unknown"):
                    continue
                key = (c.type, role)
                if key not in rules:
                    rules[key] = BrandRule(c.type, role, dict(c.params))
                else:
                    _merge(rules[key], c.params)
                rule = rules[key]
                if label not in rule.sources:
                    rule.sources.append(label)
                note = (c.provenance.notes or "").strip()
                if note and note not in rule.notes:
                    rule.notes.append(note)
    return sorted(rules.values(), key=lambda r: (r.role, r.type))


def propose_brand_rules(
    doc: DesignDocument, rules: list[BrandRule], *, brand: str
) -> list[Constraint]:
    """Add each brand rule to every matching element that has no rule of that type yet."""
    added: list[Constraint] = []
    for rule in rules:
        for e in doc.elements:
            if e.role != rule.role:
                continue
            if any(
                c.type == rule.type and e.id in c.elements and c.enabled for c in doc.constraints
            ):
                continue
            sources = ", ".join(rule.sources[:3])
            constraint = Constraint(
                id=new_id("c"),
                type=rule.type,
                elements=[e.id],
                params=dict(rule.params),
                hard=False,
                provenance=Provenance(
                    origin="generated",
                    confidence=0.5,
                    notes=f"brand rule ({brand}) from {sources}; confirm or remove",
                ),
            )
            doc.add_constraint(constraint)
            added.append(constraint)
    return added


# ---- brand profiles ---------------------------------------------------------------------
#
# A profile is what the brand book says (colours, fonts, logo rules, tone). It is applied
# in two ways: as reviewable rule proposals in every project of the brand, and as checks
# on every rendered variant (off-palette colour or off-brand font -> needs_review, never a
# silent pass). Nothing in the profile is used to train anything.

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
NEUTRALS = ("#ffffff", "#000000")
PALETTE_TOLERANCE = 24  # per-channel 0..255 distance still counted as the same colour


def _norm_hex(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if len(text) == 4 and text.startswith("#"):
        text = "#" + "".join(ch * 2 for ch in text[1:])
    if not _HEX.match(text):
        raise ValueError(f"{field_name}: use a colour like #1f3b63")
    return text.lower()


def _norm_font(raw: object, field_name: str) -> dict | None:
    if raw in (None, "", {}):
        return None
    if isinstance(raw, str):
        raw = {"family": raw}
    if not isinstance(raw, dict) or not str(raw.get("family") or "").strip():
        raise ValueError(f"{field_name}: needs a font family")
    return {
        "family": str(raw["family"]).strip()[:80],
        "weight": str(raw.get("weight") or "regular").strip().lower()[:24],
    }


def _norm_ratio(raw: object, field_name: str, lo: float, hi: float, default: float | None):
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: must be a number") from exc
    if not lo <= value <= hi:
        raise ValueError(f"{field_name}: must be between {lo} and {hi}")
    return round(value, 3)


def normalize_profile(raw: dict, *, brand: str) -> dict:
    """Validate a brand profile payload; raises ValueError with a field-level message."""
    if not isinstance(raw, dict):
        raise ValueError("profile must be an object")
    colors_in = raw.get("colors") or {}
    if not isinstance(colors_in, dict):
        raise ValueError("colors: must be an object")
    colors: dict = {}
    for key in ("primary", "secondary", "accent"):
        if colors_in.get(key):
            colors[key] = _norm_hex(colors_in[key], f"colors.{key}")
    palette = colors_in.get("palette") or []
    if isinstance(palette, str):
        palette = [c for c in re.split(r"[,\s]+", palette) if c]
    if not isinstance(palette, list):
        raise ValueError("colors.palette: must be a list of colours")
    colors["palette"] = sorted({_norm_hex(c, "colors.palette") for c in palette[:64]})
    colors["strict"] = bool(colors_in.get("strict", False))
    fonts_in = raw.get("fonts") or {}
    if not isinstance(fonts_in, dict):
        raise ValueError("fonts: must be an object")
    fonts = {
        "headline": _norm_font(fonts_in.get("headline"), "fonts.headline"),
        "body": _norm_font(fonts_in.get("body"), "fonts.body"),
    }
    logo_in = raw.get("logo") or {}
    if not isinstance(logo_in, dict):
        raise ValueError("logo: must be an object")
    logo = {
        "clear_space_ratio": _norm_ratio(logo_in.get("clear_space_ratio"),
                                         "logo.clear_space_ratio", 0.0, 3.0, None),
        "min_height_ratio": _norm_ratio(logo_in.get("min_height_ratio"),
                                        "logo.min_height_ratio", 0.0, 0.5, None),
        "always_visible": bool(logo_in.get("always_visible", True)),
    }
    text_in = raw.get("text") or {}
    if not isinstance(text_in, dict):
        raise ValueError("text: must be an object")
    text = {"min_px": int(_norm_ratio(text_in.get("min_px"), "text.min_px", 0, 200, 0) or 0)}
    do_not = raw.get("do_not") or []
    if isinstance(do_not, str):
        do_not = [ln.strip() for ln in do_not.splitlines() if ln.strip()]
    if not isinstance(do_not, list):
        raise ValueError("do_not: must be a list of sentences")
    return {
        "brand": brand,
        "colors": colors,
        "fonts": fonts,
        "logo": logo,
        "text": text,
        "tone": str(raw.get("tone") or "").strip()[:2000],
        "do_not": [str(x).strip()[:200] for x in do_not[:50] if str(x).strip()],
    }


def profile_palette(profile: dict) -> list[str]:
    colors = profile.get("colors") or {}
    out = [colors[k] for k in ("primary", "secondary", "accent") if colors.get(k)]
    out.extend(colors.get("palette") or [])
    if not colors.get("strict"):
        out.extend(NEUTRALS)
    return sorted({c.lower() for c in out})


def _rgb(hex_color: str) -> tuple[int, int, int]:
    c = hex_color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def color_in_palette(color: str, palette: list[str]) -> bool:
    try:
        r, g, b = _rgb(_norm_hex(color[:7], "color"))
    except ValueError:
        return False
    for p in palette:
        pr, pg, pb = _rgb(p)
        if max(abs(r - pr), abs(g - pg), abs(b - pb)) <= PALETTE_TOLERANCE:
            return True
    return False


def _font_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def font_matches(requested: str, brand_font: dict | None) -> bool | None:
    """True/False when a brand font is set for the slot, None when there is nothing to check."""
    if not brand_font:
        return None
    want = _font_key(brand_font["family"])
    have = _font_key(requested)
    return bool(want) and (have.startswith(want) or want.startswith(have))


def _slot_for_role(role: str) -> str | None:
    if role == "headline":
        return "headline"
    if role in ("subheadline", "body_text", "label", "badge"):
        return "body"
    return None  # cta may use either


def propose_profile_rules(doc: DesignDocument, profile: dict, *, brand: str) -> list[Constraint]:
    """Turn the profile's logo and text rules into reviewable soft constraints."""
    added: list[Constraint] = []
    note = f"brand profile ({brand}); confirm or remove"

    def has(e, ctype: str) -> bool:
        return any(c.type == ctype and e.id in c.elements and c.enabled for c in doc.constraints)

    def add(e, ctype: str, params: dict) -> None:
        constraint = Constraint(
            id=new_id("c"), type=ctype, elements=[e.id], params=params, hard=False,
            provenance=Provenance(origin="generated", confidence=0.6, notes=note),
        )
        doc.add_constraint(constraint)
        added.append(constraint)

    logo = profile.get("logo") or {}
    text = profile.get("text") or {}
    for e in doc.elements:
        if e.role == "logo":
            ratio = logo.get("clear_space_ratio")
            if ratio and not has(e, "clear_space"):
                add(e, "clear_space", {"ratio": float(ratio)})
            if logo.get("always_visible", True) and not has(e, "keep_visible"):
                add(e, "keep_visible", {})
            min_h = logo.get("min_height_ratio")
            if min_h and not has(e, "scale_range"):
                # the logo's master height relative to the canvas sets the floor
                master_ratio = e.geometry.height / max(1.0, float(doc.canvas_height))
                if master_ratio > 0:
                    add(e, "scale_range", {"min": round(min(1.0, float(min_h) / master_ratio), 3),
                                           "max": 3.0})
        elif e.kind == "text" and e.text is not None and int(text.get("min_px") or 0) > 0:
            if not has(e, "min_text_size"):
                add(e, "min_text_size", {"px": int(text["min_px"]),
                                         "measured_on": [doc.canvas_width, doc.canvas_height]})
    return added


def brand_profile_checks(
    doc: DesignDocument, layout: list, typography: dict[str, dict], profile: dict
) -> list[CheckResult]:
    """Palette and font checks on the rendered variant's visible text."""
    out: list[CheckResult] = []
    placed = {r.element_id for r in layout if r.visible}
    palette = profile_palette(profile)
    fonts = profile.get("fonts") or {}
    brand = profile.get("brand") or "brand"
    for e in doc.elements:
        if e.id not in placed or e.kind != "text" or e.text is None:
            continue
        if palette:
            off = sorted({r.style.color for r in e.text.runs
                          if not color_in_palette(r.style.color, palette)})
            if off:
                out.append(CheckResult(
                    "brand_palette", CheckStatus.NEEDS_REVIEW, Severity.MINOR,
                    f"{e.name}: colour {', '.join(off)} is not in the {brand} palette",
                    subject_id=e.id, details={"colors": off, "palette": palette},
                ))
            else:
                out.append(CheckResult(
                    "brand_palette", CheckStatus.PASS, Severity.MINOR,
                    f"{e.name}: colours are in the {brand} palette", subject_id=e.id,
                ))
        slot = _slot_for_role(e.role)
        candidates = [fonts.get(slot)] if slot else [fonts.get("headline"), fonts.get("body")]
        candidates = [f for f in candidates if f]
        if candidates:
            requested = (typography.get(e.id) or {}).get("font_family_requested") or \
                e.text.primary_style.font_family
            ok = any(font_matches(requested, f) for f in candidates)
            wanted = " or ".join(f["family"] for f in candidates)
            out.append(CheckResult(
                "brand_font",
                CheckStatus.PASS if ok else CheckStatus.NEEDS_REVIEW, Severity.MINOR,
                (f"{e.name}: font {requested} matches the {brand} font" if ok else
                 f"{e.name}: font {requested} is not the {brand} font ({wanted})"),
                subject_id=e.id, details={"requested": requested, "brand_fonts": wanted},
            ))
    return out
