"""Rules from correction history (H5).

A reviewer's rejection ("logo too small", "CTA hard to read") followed by an
approved variant of the same size is evidence of what the brand wants. This
module pairs the rejection snapshot (the rejected plan and the reason) with the
next approved plan and derives a reviewable constraint proposal from the
difference. Nothing is trained; every proposal names its evidence and carries
confidence 0.5 until a person confirms it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .document import Constraint, DesignDocument, Provenance, new_id

ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "logo": ("logo", "brand mark", "wordmark"),
    "cta": ("cta", "button", "call to action", "call-to-action"),
    "headline": ("headline", "title", "heading"),
    "subheadline": ("subheadline", "subtitle", "sub-headline", "subhead"),
    "hero_image": ("hero", "product", "photo", "image", "picture", "packshot"),
}
TEXT_WORDS = ("text", "copy", "read", "legib", "font", "type", "wording")
SIZE_UP_WORDS = (
    "too small",
    "small",
    "tiny",
    "bigger",
    "larger",
    "unreadable",
    "hard to read",
    "illegible",
    "can't read",
    "cannot read",
)
SIZE_DOWN_WORDS = ("too big", "too large", "huge", "smaller", "dominant", "overpower")
OVERLAP_WORDS = ("overlap", "cover", "hidden", "behind", "occlud", "obscur", "on top of")
TEXT_ROLES = {"headline", "subheadline", "body_text", "cta", "label", "badge"}


@dataclass
class RejectionSnapshot:
    """What was rejected: the plan at rejection time and the reason."""

    variant_id: str
    name: str
    width: int
    height: int
    reason: str
    ts: str
    placements: list[dict] = field(default_factory=list)
    typography: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "reason": self.reason,
            "ts": self.ts,
            "placements": list(self.placements),
            "typography": dict(self.typography),
        }

    @staticmethod
    def from_dict(d: dict) -> RejectionSnapshot:
        return RejectionSnapshot(
            variant_id=str(d.get("variant_id", "")),
            name=str(d.get("name", "")),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            reason=str(d.get("reason", "")),
            ts=str(d.get("ts", "")),
            placements=list(d.get("placements") or []),
            typography=dict(d.get("typography") or {}),
        )


@dataclass
class CorrectionProposal:
    kind: str  # scale_range | min_text_size | clear_space
    element_id: str
    params: dict
    reason: str
    rejected_variant: str
    approved_variant: str
    size: tuple[int, int]
    evidence: str
    confidence: float = 0.5

    def to_constraint(self) -> Constraint:
        return Constraint(
            id=new_id("c"),
            type=self.kind,
            elements=[self.element_id],
            params=dict(self.params),
            hard=False,
            provenance=Provenance(
                origin="recovered",
                confidence=self.confidence,
                notes=f"from rejection '{self.reason[:60]}' ({self.size[0]}x{self.size[1]})",
            ),
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "element_id": self.element_id,
            "params": dict(self.params),
            "reason": self.reason,
            "rejected_variant": self.rejected_variant,
            "approved_variant": self.approved_variant,
            "size": [self.size[0], self.size[1]],
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class UnresolvedCorrection:
    reason: str
    rejected_variant: str
    approved_variant: str | None
    note: str

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "rejected_variant": self.rejected_variant,
            "approved_variant": self.approved_variant,
            "note": self.note,
        }


def _mentions(reason: str, words: tuple[str, ...]) -> bool:
    return any(w in reason for w in words)


def _roles_in(reason: str) -> list[str]:
    """Roles named in the reason, ordered by where they are first mentioned."""
    found: list[tuple[int, str]] = []
    for role, words in ROLE_KEYWORDS.items():
        positions = [reason.find(w) for w in words if w in reason]
        if positions:
            found.append((min(positions), role))
    return [role for _, role in sorted(found)]


def _boxes(placements: list[dict]) -> dict[str, dict]:
    return {p["element_id"]: p for p in placements if p.get("visible", True)}


def _overlaps(a: dict, b: dict) -> bool:
    return (
        a["x"] < b["x"] + b["width"]
        and b["x"] < a["x"] + a["width"]
        and a["y"] < b["y"] + b["height"]
        and b["y"] < a["y"] + a["height"]
    )


def _already_covered(doc: DesignDocument, proposal: CorrectionProposal) -> bool:
    for c in doc.constraints:
        if c.type != proposal.kind or proposal.element_id not in c.elements or not c.enabled:
            continue
        if proposal.kind == "scale_range":
            lo_have = float(c.params.get("min", 0.0))
            lo_want = float(proposal.params.get("min", 0.0))
            hi_have = float(c.params.get("max", 10.0))
            hi_want = float(proposal.params.get("max", 10.0))
            if lo_have >= lo_want and hi_have <= hi_want:
                return True
        elif proposal.kind == "min_text_size":
            if int(c.params.get("px", 0)) >= int(proposal.params.get("px", 0)):
                return True
        else:
            return True
    return False


def derive_corrections(
    doc: DesignDocument,
    rejections: list[RejectionSnapshot],
    approved: list[tuple[dict, dict]],
) -> tuple[list[CorrectionProposal], list[UnresolvedCorrection]]:
    """Pair each rejection with the next approved plan of the same size and derive rules.

    ``approved`` holds ``(variant_record_dict, plan_dict)`` for approved, finished
    variants. A rejection pairs with the earliest approved variant of the same size
    whose ``updated_at`` is not earlier than the rejection (timestamps have second
    precision, and a variant that is approved is by definition no longer rejected,
    so an equal stamp still means "approved after").
    """
    proposals: dict[tuple[str, str], CorrectionProposal] = {}
    unresolved: list[UnresolvedCorrection] = []
    roles_by_id = {e.id: e.role for e in doc.elements}
    for rej in sorted(rejections, key=lambda r: r.ts):
        reason = re.sub(r"\s+", " ", rej.reason.strip().lower())
        if not reason:
            continue
        candidates = [
            (rec, plan)
            for rec, plan in approved
            if int(rec.get("width", 0)) == rej.width
            and int(rec.get("height", 0)) == rej.height
            and str(rec.get("updated_at", "")) >= rej.ts
            and plan.get("placements")
        ]
        if not candidates:
            unresolved.append(
                UnresolvedCorrection(
                    rej.reason, rej.variant_id, None, "no approved variant of this size yet"
                )
            )
            continue
        rec, plan = sorted(candidates, key=lambda rp: str(rp[0].get("updated_at", "")))[0]
        r_boxes, a_boxes = _boxes(rej.placements), _boxes(plan.get("placements", []))
        a_typo = plan.get("typography", {}) or {}
        roles = _roles_in(reason)
        if roles and _mentions(reason, OVERLAP_WORDS):
            # "headline hidden behind the hero": the first-named element is the victim
            roles = roles[:1]
        targets = [eid for eid, role in roles_by_id.items() if role in roles]
        if not targets and _mentions(reason, TEXT_WORDS):
            targets = [eid for eid, role in roles_by_id.items() if role in TEXT_ROLES]
        derived: list[CorrectionProposal] = []
        size = (rej.width, rej.height)
        for eid in targets:
            rb, ab = r_boxes.get(eid), a_boxes.get(eid)
            if rb is None or ab is None:
                continue
            role = roles_by_id[eid]
            is_text = role in TEXT_ROLES
            if _mentions(reason, OVERLAP_WORDS):
                overlapped_before = any(
                    _overlaps(rb, o) for k, o in r_boxes.items() if k != eid and k in a_boxes
                )
                overlapped_after = any(_overlaps(ab, o) for k, o in a_boxes.items() if k != eid)
                if overlapped_before and not overlapped_after:
                    derived.append(
                        CorrectionProposal(
                            "clear_space",
                            eid,
                            {"ratio": 0.5},
                            rej.reason,
                            rej.variant_id,
                            str(rec.get("id")),
                            size,
                            "overlapped another element when rejected; clear when approved",
                        )
                    )
                    continue
            if is_text:
                px_r = int((rej.typography.get(eid) or {}).get("font_px", 0))
                px_a = int((a_typo.get(eid) or {}).get("font_px", 0))
                if _mentions(reason, SIZE_UP_WORDS) and px_a > px_r > 0:
                    derived.append(
                        CorrectionProposal(
                            "min_text_size",
                            eid,
                            {"px": px_a, "measured_on": [rej.width, rej.height]},
                            rej.reason,
                            rej.variant_id,
                            str(rec.get("id")),
                            size,
                            f"font {px_r}px when rejected, {px_a}px when approved",
                        )
                    )
                continue
            h_r = rb["height"] / max(1, rej.height)
            h_a = ab["height"] / max(1, rej.height)
            if _mentions(reason, SIZE_UP_WORDS) and h_a > h_r * 1.05:
                derived.append(
                    CorrectionProposal(
                        "scale_range",
                        eid,
                        {"min": round(h_a, 3), "max": 1.0},
                        rej.reason,
                        rej.variant_id,
                        str(rec.get("id")),
                        size,
                        f"height {h_r:.2f} of canvas when rejected, {h_a:.2f} when approved",
                    )
                )
            elif _mentions(reason, SIZE_DOWN_WORDS) and h_a < h_r * 0.95:
                derived.append(
                    CorrectionProposal(
                        "scale_range",
                        eid,
                        {"min": 0.05, "max": round(h_a, 3)},
                        rej.reason,
                        rej.variant_id,
                        str(rec.get("id")),
                        size,
                        f"height {h_r:.2f} of canvas when rejected, {h_a:.2f} when approved",
                    )
                )
        if not derived:
            unresolved.append(
                UnresolvedCorrection(
                    rej.reason,
                    rej.variant_id,
                    str(rec.get("id")),
                    "no rule derived: the reason names no known element/direction or the "
                    "approved plan does not differ in that way",
                )
            )
            continue
        for prop in derived:
            if _already_covered(doc, prop):
                continue
            proposals[(prop.kind, prop.element_id)] = prop  # latest wins
    return list(proposals.values()), unresolved
