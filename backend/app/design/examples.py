"""Learn adaptation rules from approved example variants (H1).

Given a master document and one or more approved variants (documents with
their own canvas sizes and placed elements), this module matches elements
across documents and infers, per aspect class, a layout ``Family`` (text
column, subject slot, logo slot, alignment, stacking order) plus reviewable
constraint proposals. Everything inferred carries a confidence that grows
with agreement across examples; a single example is explicitly
under-determined (confidence 0.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .document import Constraint, DesignDocument, Element, Geometry, Provenance, new_id
from .planner import Family, Region, aspect_class
from .serialize import document_from_dict, document_to_dict

SUBJECT_ROLES = {"hero_image", "photo", "illustration", "icon"}
TEXT_ROLES = {"headline", "subheadline", "body_text", "cta", "label", "badge"}


@dataclass
class Match:
    master_id: str
    example_id: str
    method: str  # asset | text | role
    confidence: float


@dataclass
class InferredFamily:
    aspect: str
    family: Family
    confidence: float
    examples: int
    hierarchy_scale: float | None
    constraints: list[Constraint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        f = self.family
        return {
            "aspect": self.aspect,
            "family": f.name,
            "confidence": round(self.confidence, 3),
            "examples": self.examples,
            "hierarchy_scale": self.hierarchy_scale,
            "regions": {
                "text": [f.text.x, f.text.y, f.text.w, f.text.h],
                "subject": [f.subject.x, f.subject.y, f.subject.w, f.subject.h],
                "logo": [f.logo.x, f.logo.y, f.logo.w, f.logo.h],
            },
            "text_align": f.text_align,
            "subject_first": f.subject_first,
            "constraints": [c.type for c in self.constraints],
            "notes": list(self.notes),
        }


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def match_elements(master: DesignDocument, example: DesignDocument) -> list[Match]:
    """Match master elements to example elements by asset hash, text, then role."""
    matches: list[Match] = []
    used: set[str] = set()
    ex_by_hash: dict[str, list[Element]] = {}
    ex_by_text: dict[str, list[Element]] = {}
    for e in example.elements:
        if e.asset is not None:
            ex_by_hash.setdefault(e.asset.content_hash, []).append(e)
        if e.kind == "text" and e.text is not None:
            ex_by_text.setdefault(_norm_text(e.text.plain), []).append(e)
            for t in e.text.translations.values():
                ex_by_text.setdefault(_norm_text(t), []).append(e)

    for m in master.elements:
        if m.is_background:
            continue
        candidate: Element | None = None
        method = ""
        conf = 0.0
        if m.asset is not None:
            for c in ex_by_hash.get(m.asset.content_hash, []):
                if c.id not in used:
                    candidate, method, conf = c, "asset", 0.98
                    break
        if candidate is None and m.kind == "text" and m.text is not None:
            keys = [_norm_text(m.text.plain)] + [
                _norm_text(t) for t in m.text.translations.values()
            ]
            for key in keys:
                for c in ex_by_text.get(key, []):
                    if c.id not in used:
                        candidate, method, conf = c, "text", 0.9
                        break
                if candidate is not None:
                    break
        if candidate is None:
            same_role = [
                c
                for c in example.elements
                if c.role == m.role and c.id not in used and not c.is_background
            ]
            if len(same_role) == 1:
                candidate, method, conf = same_role[0], "role", 0.6
        if candidate is not None:
            used.add(candidate.id)
            matches.append(Match(m.id, candidate.id, method, conf))
    return matches


def _union(elements: list[Element], cw: int, ch: int) -> Region | None:
    if not elements:
        return None
    x1 = min(e.geometry.x for e in elements)
    y1 = min(e.geometry.y for e in elements)
    x2 = max(e.geometry.x2 for e in elements)
    y2 = max(e.geometry.y2 for e in elements)
    return Region(x1 / cw, y1 / ch, max(0.01, (x2 - x1) / cw), max(0.01, (y2 - y1) / ch))


def _iou(a: Region, b: Region) -> float:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _average(regions: list[Region]) -> Region:
    n = len(regions)
    return Region(
        sum(r.x for r in regions) / n,
        sum(r.y for r in regions) / n,
        sum(r.w for r in regions) / n,
        sum(r.h for r in regions) / n,
    )


def infer_families(
    master: DesignDocument,
    examples: list[DesignDocument],
) -> dict[str, InferredFamily]:
    """Infer one family per aspect class from approved example documents."""
    per_class: dict[str, list[dict]] = {}
    for ex in examples:
        cw, ch = ex.canvas_width, ex.canvas_height
        cls = aspect_class(cw / max(1, ch))
        matches = match_elements(master, ex)
        ex_by_id = {e.id: e for e in ex.elements}
        m_by_id = {e.id: e for e in master.elements}
        texts, subjects, logos = [], [], []
        scales: list[float] = []
        for mt in matches:
            me, ee = m_by_id[mt.master_id], ex_by_id[mt.example_id]
            if me.role in TEXT_ROLES or me.kind == "text":
                texts.append(ee)
                if me.text is not None and ee.text is not None and me.text.primary_style.font_size:
                    canvas_scale = min(cw / master.canvas_width, ch / master.canvas_height)
                    if canvas_scale > 0:
                        scales.append(
                            ee.text.primary_style.font_size
                            / me.text.primary_style.font_size
                            / canvas_scale
                        )
            elif me.role in SUBJECT_ROLES:
                subjects.append(ee)
            elif me.role == "logo":
                logos.append(ee)
        text_r = _union(texts, cw, ch)
        subj_r = _union(subjects, cw, ch)
        logo_r = _union(logos, cw, ch)
        if text_r is None and subj_r is None:
            continue
        align = "left"
        if texts:
            xs = [e.geometry.x / cw for e in texts]
            centers = [(e.geometry.x + e.geometry.width / 2) / cw for e in texts]
            centered_geometry = max(xs) - min(xs) > 0.03 and max(centers) - min(centers) <= 0.03
            centered_style = any(
                e.text is not None and e.text.primary_style.align == "center" for e in texts
            )
            if centered_geometry or centered_style:
                align = "center"
        subject_first = bool(subj_r and text_r and subj_r.y + subj_r.h * 0.5 < text_r.y)
        per_class.setdefault(cls, []).append(
            {
                "text": text_r,
                "subject": subj_r,
                "logo": logo_r,
                "align": align,
                "subject_first": subject_first,
                "scale": sorted(scales)[len(scales) // 2] if scales else None,
                "matches": len(matches),
                "logos": logos,
                "canvas": (cw, ch),
            }
        )

    out: dict[str, InferredFamily] = {}
    for cls, obs in per_class.items():
        n = len(obs)
        text_regions = [o["text"] for o in obs if o["text"] is not None]
        subj_regions = [o["subject"] for o in obs if o["subject"] is not None]
        logo_regions = [o["logo"] for o in obs if o["logo"] is not None]
        # Default fallbacks when a role is absent from the examples.
        text = _average(text_regions) if text_regions else Region(0.06, 0.12, 0.5, 0.7)
        subject = _average(subj_regions) if subj_regions else Region(0.58, 0.1, 0.38, 0.8)
        logo = _average(logo_regions) if logo_regions else Region(0.8, 0.04, 0.16, 0.12)
        # Agreement across examples: mean IoU of each region with the average.
        agreement = 1.0
        for regions, avg in ((text_regions, text), (subj_regions, subject)):
            if len(regions) > 1:
                agreement = min(agreement, sum(_iou(r, avg) for r in regions) / len(regions))
        confidence = 0.5 if n == 1 else min(0.9, 0.5 + 0.2 * (n - 1) * agreement)
        align_votes = [o["align"] for o in obs]
        align = max(set(align_votes), key=align_votes.count)
        subject_first = sum(1 for o in obs if o["subject_first"]) * 2 > n
        scales = [o["scale"] for o in obs if o["scale"]]
        hierarchy = round(sorted(scales)[len(scales) // 2], 3) if scales else None
        family = Family(
            f"learned_{cls}", text, subject, logo, text_align=align, subject_first=subject_first
        )
        notes = [f"inferred from {n} approved example(s); region agreement {agreement:.2f}"]
        if n == 1:
            notes.append("single example: under-determined, confirm before relying on it")
        constraints = _propose_constraints(master, obs, cls)
        out[cls] = InferredFamily(cls, family, confidence, n, hierarchy, constraints, notes)
    return out


def _propose_constraints(master: DesignDocument, obs: list[dict], cls: str) -> list[Constraint]:
    """Constraints supported by every example: logo edge anchoring, subject scale range."""
    proposals: list[Constraint] = []
    prov = Provenance(
        origin="recovered",
        confidence=0.5 if len(obs) == 1 else 0.7,
        notes=f"learned from approved {cls} example(s)",
    )
    logos = [e for e in master.elements if e.role == "logo"]
    if logos:
        edges: list[str] = []
        for o in obs:
            lr = o["logo"]
            if lr is None:
                edges = []
                break
            if lr.y <= 0.06:
                edges.append("top")
            elif lr.y + lr.h >= 0.94:
                edges.append("bottom")
            else:
                edges.append("none")
        if edges and len(set(edges)) == 1 and edges[0] != "none":
            proposals.append(
                Constraint(
                    id=new_id("c"),
                    type="anchor_edge",
                    elements=[logos[0].id],
                    params={"edge": edges[0]},
                    hard=False,
                    provenance=prov,
                )
            )
    subjects = [e for e in master.elements if e.role in SUBJECT_ROLES]
    heights = [o["subject"].h for o in obs if o["subject"] is not None]
    if subjects and heights:
        lo, hi = min(heights), max(heights)
        proposals.append(
            Constraint(
                id=new_id("c"),
                type="scale_range",
                elements=[subjects[0].id],
                params={
                    "min": round(max(0.05, lo * 0.85), 3),
                    "max": round(min(1.0, hi * 1.15), 3),
                },
                hard=False,
                provenance=prov,
            )
        )
    return proposals


def learned_families(master: DesignDocument, examples: list[DesignDocument]) -> dict[str, Family]:
    return {cls: inf.family for cls, inf in infer_families(master, examples).items()}


def example_from_plan(
    master: DesignDocument,
    target: tuple[int, int],
    placements: list[dict],
    typography: dict[str, dict] | None = None,
    *,
    example_id: str = "example",
) -> DesignDocument:
    """Turn a rendered variant's plan (as stored in its report) into an example document.

    Hidden elements are dropped, backgrounds fill the canvas, and text elements
    take the rendered font size so hierarchy can be measured.
    """
    ex = document_from_dict(document_to_dict(master))
    ex.id = example_id
    ex.canvas_width, ex.canvas_height = target
    by_id = {p["element_id"]: p for p in placements}
    typography = typography or {}
    kept: list[Element] = []
    for e in ex.elements:
        p = by_id.get(e.id)
        if e.is_background:
            e.geometry = Geometry(0, 0, target[0], target[1])
            kept.append(e)
            continue
        if p is None or not p.get("visible", True):
            continue
        e.geometry = Geometry(int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"]))
        if e.kind == "text" and e.text is not None and e.id in typography:
            e.text.primary_style.font_size = float(typography[e.id]["font_px"])
        kept.append(e)
    ex.elements = kept
    return ex
