"""Brand-level rules: carry confirmed rules across an owner's projects of one brand (H5).

A rule confirmed in one project (a user-added or correction-derived constraint on a
logo, CTA, …) is evidence about the brand, not just about that design. This module
unions such rules across projects by (constraint type, element role) and proposes
them into another project of the same brand as reviewable, soft constraints. Only
enabled constraints with provenance ``user`` or ``recovered`` are carried; unconfirmed
``generated`` proposals never propagate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
