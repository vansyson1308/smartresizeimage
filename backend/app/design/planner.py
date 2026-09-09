"""Constraint-aware planner: place document elements for a target size.

The planner treats the master as intent, not coordinates. It derives the
reading order and hierarchy from the master, chooses a layout family for the
target aspect ratio, sizes text from the master hierarchy, keeps subjects and
logos uniform, honours document constraints (keep_visible, keep_group,
order_below, clear_space, anchor_edge, scale_range) and reports conflicts it
cannot resolve instead of hiding them.

It is the "constraints" planner selected by ``Config.DESIGN_PLANNER``; the
legacy zone-template engine remains available as "zones".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..enums import ElementRole
from ..models import BoundingBox, DesignElement, LayoutResult
from .document import Constraint, DesignDocument, Element
from .fonts import FontRegistry, default_registry
from .text_render import fit_text

logger = logging.getLogger("autobanner.design.planner")

TEXT_ROLES = {"headline", "subheadline", "body_text", "cta", "label", "badge"}
SUBJECT_ROLES = {"hero_image", "photo", "illustration", "icon"}
IDENTITY_ROLES = {"logo"}
BACKGROUND_ROLES = {"background", "background_pattern", "overlay"}

# Font size bounds per role in px at the target (min, max)
FONT_BOUNDS = {
    "headline": (20, 140),
    "subheadline": (16, 80),
    "body_text": (12, 40),
    "cta": (14, 56),
    "label": (10, 32),
    "badge": (12, 48),
}


@dataclass
class Region:
    x: float
    y: float
    w: float
    h: float

    def box(self, tw: int, th: int) -> BoundingBox:
        return BoundingBox(int(self.x * tw), int(self.y * th), int(self.w * tw), int(self.h * th))


@dataclass
class Family:
    """A layout family: named regions as canvas fractions."""

    name: str
    text: Region
    subject: Region
    logo: Region
    text_align: str = "left"
    subject_first: bool = False  # reading order: subject before text (stacked families)


def families_for(aspect: float) -> list[Family]:
    if aspect >= 1.25:  # landscape
        return [
            Family("landscape_text_left", Region(0.05, 0.10, 0.50, 0.80),
                   Region(0.58, 0.06, 0.38, 0.88), Region(0.80, 0.04, 0.16, 0.14)),
            Family("landscape_text_right", Region(0.45, 0.10, 0.50, 0.80),
                   Region(0.04, 0.06, 0.38, 0.88), Region(0.04, 0.04, 0.16, 0.14),
                   text_align="left"),
            Family("landscape_wide_text", Region(0.05, 0.14, 0.62, 0.72),
                   Region(0.70, 0.10, 0.27, 0.80), Region(0.82, 0.04, 0.14, 0.12)),
        ]
    if aspect <= 0.8:  # portrait
        return [
            Family("portrait_text_top", Region(0.07, 0.16, 0.86, 0.36),
                   Region(0.10, 0.54, 0.80, 0.42), Region(0.07, 0.04, 0.30, 0.09)),
            Family("portrait_subject_top", Region(0.07, 0.56, 0.86, 0.38),
                   Region(0.10, 0.12, 0.80, 0.42), Region(0.07, 0.03, 0.30, 0.08),
                   subject_first=True),
            Family("portrait_centered", Region(0.08, 0.12, 0.84, 0.30),
                   Region(0.15, 0.46, 0.70, 0.40), Region(0.35, 0.03, 0.30, 0.07),
                   text_align="center"),
        ]
    return [  # square-ish
        Family("square_text_left", Region(0.06, 0.12, 0.46, 0.76),
               Region(0.54, 0.16, 0.42, 0.72), Region(0.06, 0.03, 0.22, 0.08)),
        Family("square_text_top", Region(0.07, 0.15, 0.86, 0.34),
               Region(0.20, 0.52, 0.60, 0.44), Region(0.07, 0.04, 0.26, 0.09)),
        Family("square_subject_top", Region(0.07, 0.58, 0.86, 0.36),
               Region(0.20, 0.10, 0.60, 0.44), Region(0.70, 0.03, 0.26, 0.08),
               subject_first=True),
    ]


@dataclass
class Plan:
    family: str
    layout: list[LayoutResult]
    score: float
    conflicts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    text_px: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "score": round(self.score, 3),
            "conflicts": list(self.conflicts),
            "decisions": list(self.decisions),
            "text_px": dict(self.text_px),
        }


def plan_layout(
    doc: DesignDocument,
    elements: list[DesignElement],
    target: tuple[int, int],
    *,
    registry: FontRegistry | None = None,
    families: list[Family] | None = None,
) -> Plan:
    """Return the best-scoring plan across layout families for ``target``."""
    reg = registry or default_registry()
    tw, th = target
    aspect = tw / max(1, th)
    candidates = families or families_for(aspect)
    best: Plan | None = None
    for fam in candidates:
        plan = _plan_family(doc, elements, target, fam, reg)
        logger.debug("family=%s score=%.2f conflicts=%s", fam.name, plan.score, plan.conflicts)
        if best is None or plan.score > best.score:
            best = plan
    assert best is not None
    return best


# --------------------------------------------------------------------------------------
# planning one family
# --------------------------------------------------------------------------------------


def _plan_family(
    doc: DesignDocument,
    elements: list[DesignElement],
    target: tuple[int, int],
    fam: Family,
    reg: FontRegistry,
) -> Plan:
    tw, th = target
    cw, ch = doc.canvas_width, doc.canvas_height
    scale = min(tw / max(1, cw), th / max(1, ch))
    doc_by_id = {e.id: e for e in doc.elements}
    by_id = {e.id: e for e in elements}
    decisions: list[str] = []
    conflicts: list[str] = []
    layout: list[LayoutResult] = []
    text_px: dict[str, int] = {}

    # background(s): always the full canvas
    for e in elements:
        if e.role in {ElementRole.BACKGROUND, ElementRole.BACKGROUND_PATTERN, ElementRole.OVERLAY}:
            layout.append(LayoutResult(e.id, BoundingBox(0, 0, tw, th), max(tw / cw, th / ch)))

    content = [e for e in elements if e.role.value not in BACKGROUND_ROLES]
    texts = sorted(
        [e for e in content if e.role.value in TEXT_ROLES or e.layer_type == "type"],
        key=lambda e: (e.bbox.y, e.bbox.x),
    )
    subjects = [e for e in content if e.role.value in SUBJECT_ROLES]
    logos = [e for e in content if e.role.value in IDENTITY_ROLES]
    others = [e for e in content if e not in texts and e not in subjects and e not in logos]

    # --- logo(s): uniform scale into the logo slot, respecting scale_range
    logo_box = fam.logo.box(tw, th)
    for i, e in enumerate(logos):
        slot = _split_horizontal(logo_box, len(logos), i)
        box = _fit_uniform(e.bbox, slot, prefer_scale=scale)
        box = _apply_scale_range(doc_by_id.get(e.id), doc, box, e.bbox, th)
        layout.append(LayoutResult(e.id, box, box.width / max(1, e.bbox.width)))

    # --- subject(s): uniform scale into the subject region, keep master prominence hint
    subj_box = fam.subject.box(tw, th)
    # respect logo clear space: the subject slot must not enter the logo's clear zone
    for lg in logos:
        lr = next((r for r in layout if r.element_id == lg.id), None)
        if lr is None:
            continue
        ratio = 0.5
        for c in doc.constraints_for(lg.id):
            if c.type == "clear_space":
                ratio = float(c.params.get("ratio", ratio))
        subj_box = _avoid_zone(subj_box, _clear_zone(lr.new_bbox, ratio), th)
    master_area = cw * ch
    for i, e in enumerate(subjects):
        slot = _split_vertical(subj_box, len(subjects), i) if len(subjects) > 1 else subj_box
        # target area share: keep the master's share, bounded by the region
        share = e.bbox.area / max(1, master_area)
        box = _fit_uniform(e.bbox, slot, prefer_scale=None)
        wanted = (share * tw * th) ** 0.5
        if box.width * box.height > 0 and (box.width * box.height) ** 0.5 > wanted * 1.35:
            # region is much larger than the master share: don't blow the subject up
            k = wanted * 1.35 / max(1.0, (box.width * box.height) ** 0.5)
            box = _scale_box_about_center(box, k)
        box = _apply_scale_range(doc_by_id.get(e.id), doc, box, e.bbox, th)
        layout.append(LayoutResult(e.id, box, box.width / max(1, e.bbox.width)))
        decisions.append(f"subject:{e.id}:{fam.name}")

    # --- text stack: reading order from the master, sizes from master hierarchy
    text_box = fam.text.box(tw, th)
    # keep the text column clear of the logo slot when they overlap vertically
    logo_ids = {lg.id for lg in logos}
    text_box = _avoid(text_box, [r.new_bbox for r in layout if r.element_id in logo_ids], th)
    stack = _plan_text_stack(texts, doc_by_id, text_box, scale, reg, th, fam.text_align)

    # Content pressure: when the copy cannot fit its column even after shrinking,
    # give the text more room and let the subject give way (never the other way
    # round: copy is approved content, the subject may scale down to a floor).
    needed = _stack_height(stack)
    if stack and needed > text_box.height and _stacked_family(fam):
        min_subject_h = int(0.18 * th)
        extra = min(needed - text_box.height, max(0, subj_box.height - min_subject_h))
        if extra > 0:
            if fam.subject_first:
                # subject above text: shrink subject height, move text up
                new_subj = BoundingBox(subj_box.x, subj_box.y, subj_box.width,
                                       subj_box.height - extra)
                new_text = BoundingBox(text_box.x, text_box.y - extra, text_box.width,
                                       text_box.height + extra)
            else:
                new_subj = BoundingBox(subj_box.x, subj_box.y + extra, subj_box.width,
                                       subj_box.height - extra)
                new_text = BoundingBox(text_box.x, text_box.y, text_box.width,
                                       text_box.height + extra)
            layout = [r for r in layout if r.element_id not in {s.id for s in subjects}]
            for i, e in enumerate(subjects):
                slot = new_subj
                if len(subjects) > 1:
                    slot = _split_vertical(new_subj, len(subjects), i)
                box = _fit_uniform(e.bbox, slot, prefer_scale=None)
                box = _apply_scale_range(doc_by_id.get(e.id), doc, box, e.bbox, th)
                layout.append(LayoutResult(e.id, box, box.width / max(1, e.bbox.width)))
            stack = _plan_text_stack(texts, doc_by_id, new_text, scale, reg, th, fam.text_align)
            decisions.append(f"content_pressure:text+{extra}px")
    for e, box, px, overflow in stack:
        layout.append(LayoutResult(e.id, box, px / max(1.0, _master_px(doc_by_id.get(e.id), e))))
        text_px[e.id] = px
        if overflow:
            conflicts.append(f"text_overflow:{e.id}")

    # --- others (badges, decorations, unknown): scale uniformly, keep relative master position
    for e in others:
        box = BoundingBox(
            int(e.bbox.x / cw * tw), int(e.bbox.y / ch * th),
            max(1, int(e.bbox.width * scale)), max(1, int(e.bbox.height * scale)),
        )
        layout.append(LayoutResult(e.id, box, scale, visible=e.priority <= 7))

    # --- constraints: keep_group ordering / order_below / anchor_edge
    _apply_constraint_adjustments(doc, layout, target, conflicts, decisions)

    score = _score(doc, layout, target, by_id, text_px, conflicts, fam)
    return Plan(fam.name, layout, score, conflicts, decisions, text_px)


def _plan_text_stack(
    texts: list[DesignElement],
    doc_by_id: dict[str, Element],
    column: BoundingBox,
    scale: float,
    reg: FontRegistry,
    th: int,
    align: str,
) -> list[tuple[DesignElement, BoundingBox, int, bool]]:
    """Stack text elements top-down in the column; shrink proportionally to fit height."""
    if not texts:
        return []
    gap = max(6, int(0.018 * th))
    # Try the preferred sizes first; if the stack is taller than the column, scale all
    # sizes by a common factor so hierarchy is preserved.
    factor = 1.0
    for _ in range(8):
        placed: list[tuple[DesignElement, BoundingBox, int, bool]] = []
        y = column.y
        total = 0
        for e in texts:
            de = doc_by_id.get(e.id)
            content = _content(e, de)
            lo, hi = FONT_BOUNDS.get(e.role.value, (12, 60))
            master_px = _master_px(de, e)
            preferred = max(lo, min(hi, int(round(master_px * scale * factor))))
            max_lines = de.text.max_lines if de is not None and de.text is not None else None
            fitted = fit_text(content, column.width, None, min_px=lo, max_px=preferred,
                              registry=reg, max_lines=max_lines)
            h = max(1, fitted.height)
            w = min(column.width, max(1, fitted.width))
            x = column.x
            if align == "center":
                x = column.x + (column.width - w) // 2
            placed.append((e, BoundingBox(x, y, w, h), fitted.font_px, fitted.overflow))
            y += h + gap
            total += h + gap
        total -= gap
        if total <= column.height or factor <= 0.45:
            # vertically centre the stack inside the column
            offset = max(0, (column.height - total) // 2)
            if offset:
                placed = [(e, BoundingBox(b.x, b.y + offset, b.width, b.height), px, ov)
                          for e, b, px, ov in placed]
            return placed
        factor *= max(0.5, column.height / max(1, total)) * 0.98
    return placed


def _stack_height(stack: list[tuple[DesignElement, BoundingBox, int, bool]]) -> int:
    if not stack:
        return 0
    top = min(b.y for _, b, _, _ in stack)
    bottom = max(b.y2 for _, b, _, _ in stack)
    return bottom - top


def _stacked_family(fam: Family) -> bool:
    """True when text and subject share the vertical axis (portrait/square-top)."""
    t, s = fam.text, fam.subject
    horizontal_overlap = t.x < s.x + s.w and s.x < t.x + t.w
    vertical_separated = t.y + t.h <= s.y + 0.02 or s.y + s.h <= t.y + 0.02
    return horizontal_overlap and vertical_separated


def _content(e: DesignElement, de: Element | None):
    from .document import TextContent

    if de is not None and de.text is not None:
        content = TextContent(
            runs=[type(r)(text=r.text, style=r.style) for r in de.text.runs],
            locale=de.text.locale,
            max_lines=de.text.max_lines,
            protected=de.text.protected,
        )
        if e.text_content and e.text_content != de.text.plain:
            content.replace_text(e.text_content)
        return content
    content = TextContent()
    content.replace_text(e.text_content or e.name)
    return content


def _master_px(de: Element | None, e: DesignElement) -> float:
    if de is not None and de.text is not None:
        return float(de.text.primary_style.font_size)
    return max(8.0, e.bbox.height * 0.6)


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def _fit_uniform(src: BoundingBox, slot: BoundingBox, prefer_scale: float | None) -> BoundingBox:
    sw, sh = max(1, src.width), max(1, src.height)
    k = min(slot.width / sw, slot.height / sh)
    if prefer_scale is not None:
        k = min(k, max(prefer_scale, 0.3))
    w, h = max(1, int(sw * k)), max(1, int(sh * k))
    return BoundingBox(slot.x + (slot.width - w) // 2, slot.y + (slot.height - h) // 2, w, h)


def _scale_box_about_center(b: BoundingBox, k: float) -> BoundingBox:
    cx, cy = b.center
    w, h = max(1, int(b.width * k)), max(1, int(b.height * k))
    return BoundingBox(cx - w // 2, cy - h // 2, w, h)


def _split_horizontal(b: BoundingBox, n: int, i: int) -> BoundingBox:
    if n <= 1:
        return b
    w = b.width // n
    return BoundingBox(b.x + i * w, b.y, w, b.height)


def _split_vertical(b: BoundingBox, n: int, i: int) -> BoundingBox:
    if n <= 1:
        return b
    h = b.height // n
    return BoundingBox(b.x, b.y + i * h, b.width, h)


def _clear_zone(b: BoundingBox, ratio: float) -> BoundingBox:
    pad = int(b.height * ratio)
    return BoundingBox(b.x - pad, b.y - pad, b.width + 2 * pad, b.height + 2 * pad)


def _avoid_zone(slot: BoundingBox, zone: BoundingBox, th: int) -> BoundingBox:
    """Cut ``slot`` so it no longer intersects ``zone`` (from the nearest side)."""
    if _inter(slot, zone) == 0:
        return slot
    gap = max(4, int(0.01 * th))
    # candidate cuts: below the zone, above the zone, right of it, left of it
    candidates = []
    if zone.y2 + gap < slot.y2:
        candidates.append(BoundingBox(slot.x, zone.y2 + gap, slot.width, slot.y2 - zone.y2 - gap))
    if zone.y - gap > slot.y:
        candidates.append(BoundingBox(slot.x, slot.y, slot.width, zone.y - gap - slot.y))
    if zone.x2 + gap < slot.x2:
        candidates.append(BoundingBox(zone.x2 + gap, slot.y, slot.x2 - zone.x2 - gap, slot.height))
    if zone.x - gap > slot.x:
        candidates.append(BoundingBox(slot.x, slot.y, zone.x - gap - slot.x, slot.height))
    if not candidates:
        return slot
    return max(candidates, key=lambda b: b.area)


def _avoid(column: BoundingBox, blockers: list[BoundingBox], th: int) -> BoundingBox:
    """Shrink the column from the top if a blocker (logo) intrudes into it."""
    for b in blockers:
        if b.x < column.x2 and column.x < b.x2 and b.y < column.y2 and column.y < b.y2:
            new_top = b.y2 + max(6, int(0.015 * th))
            if new_top < column.y2 - 20:
                column = BoundingBox(column.x, new_top, column.width, column.y2 - new_top)
    return column


def _apply_scale_range(
    de: Element | None, doc: DesignDocument, box: BoundingBox, src: BoundingBox, th: int
) -> BoundingBox:
    if de is None:
        return box
    for c in doc.constraints_for(de.id):
        if c.type != "scale_range":
            continue
        lo, hi = float(c.params.get("min", 0.0)), float(c.params.get("max", 10.0))
        rel = box.height / max(1, th)
        if rel < lo:
            box = _scale_box_about_center(box, lo * th / max(1, box.height))
        elif rel > hi:
            box = _scale_box_about_center(box, hi * th / max(1, box.height))
    return box


def _apply_constraint_adjustments(
    doc: DesignDocument,
    layout: list[LayoutResult],
    target: tuple[int, int],
    conflicts: list[str],
    decisions: list[str],
) -> None:
    tw, th = target
    layout_map = {r.element_id: r for r in layout}
    for c in doc.constraints:
        if not c.enabled:
            continue
        if c.type == "anchor_edge" and c.elements:
            lr = layout_map.get(c.elements[0])
            if lr is None:
                continue
            edge = str(c.params.get("edge", "top"))
            m = int(0.04 * min(tw, th))
            b = lr.new_bbox
            if edge == "top":
                lr.new_bbox = BoundingBox(b.x, m, b.width, b.height)
            elif edge == "bottom":
                lr.new_bbox = BoundingBox(b.x, th - m - b.height, b.width, b.height)
            elif edge == "left":
                lr.new_bbox = BoundingBox(m, b.y, b.width, b.height)
            elif edge == "right":
                lr.new_bbox = BoundingBox(tw - m - b.width, b.y, b.width, b.height)
            decisions.append(f"anchor:{c.elements[0]}:{edge}")
        elif c.type == "order_below" and len(c.elements) == 2:
            a, b = layout_map.get(c.elements[0]), layout_map.get(c.elements[1])
            if a is None or b is None:
                continue
            if a.new_bbox.y < b.new_bbox.y:
                # swap vertical positions to honour the reading order
                ay, by = a.new_bbox.y, b.new_bbox.y
                a.new_bbox = BoundingBox(a.new_bbox.x, by, a.new_bbox.width, a.new_bbox.height)
                b.new_bbox = BoundingBox(b.new_bbox.x, ay, b.new_bbox.width, b.new_bbox.height)
                decisions.append(f"reorder:{c.elements[0]}_below_{c.elements[1]}")
        elif c.type == "keep_visible":
            for eid in c.elements:
                lr = layout_map.get(eid)
                if lr is None or not lr.visible:
                    conflicts.append(f"keep_visible:{eid}")


def _score(
    doc: DesignDocument,
    layout: list[LayoutResult],
    target: tuple[int, int],
    by_id: dict[str, DesignElement],
    text_px: dict[str, int],
    conflicts: list[str],
    fam: Family,
) -> float:
    tw, th = target
    score = 100.0
    content = [r for r in layout if by_id.get(r.element_id) and
               by_id[r.element_id].role.value not in BACKGROUND_ROLES and r.visible]
    allowed = doc.allowed_overlaps()
    # overlaps between content boxes
    for i in range(len(content)):
        for j in range(i + 1, len(content)):
            a, b = content[i], content[j]
            if (a.element_id, b.element_id) in allowed or (b.element_id, a.element_id) in allowed:
                continue
            inter = _inter(a.new_bbox, b.new_bbox)
            if inter > 0:
                smaller = min(a.new_bbox.area, b.new_bbox.area)
                score -= 40.0 * inter / max(1, smaller)
    # outside canvas
    for r in content:
        b = r.new_bbox
        if b.x < 0 or b.y < 0 or b.x2 > tw or b.y2 > th:
            score -= 25.0
    # text size: reward larger relative to the target height (hierarchy preserved by construction)
    for px in text_px.values():
        score += min(6.0, 60.0 * px / max(1, th))
    # subject prominence: prefer a share similar to the master
    cw, ch = doc.canvas_width, doc.canvas_height
    for r in content:
        e = by_id[r.element_id]
        if e.role.value in SUBJECT_ROLES:
            master_share = e.bbox.area / max(1, cw * ch)
            share = r.new_bbox.area / max(1, tw * th)
            score -= 30.0 * abs(share - master_share)
    # clear space around logos
    for c in doc.constraints:
        if not c.enabled or c.type != "clear_space" or not c.elements:
            continue
        lr = next((r for r in content if r.element_id == c.elements[0]), None)
        if lr is None:
            continue
        pad = int(lr.new_bbox.height * float(c.params.get("ratio", 0.5)))
        zone = BoundingBox(lr.new_bbox.x - pad, lr.new_bbox.y - pad,
                           lr.new_bbox.width + 2 * pad, lr.new_bbox.height + 2 * pad)
        for other in content:
            if other.element_id != lr.element_id and _inter(zone, other.new_bbox) > 0:
                score -= 4.0
    score -= 15.0 * len(conflicts)
    return score


def _inter(a: BoundingBox, b: BoundingBox) -> int:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    return (ix2 - ix1) * (iy2 - iy1) if ix1 < ix2 and iy1 < iy2 else 0


def constraint_ids(doc: DesignDocument) -> list[str]:
    return [c.id for c in doc.constraints if isinstance(c, Constraint)]
