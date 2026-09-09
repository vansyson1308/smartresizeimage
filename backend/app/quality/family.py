"""Cross-variant consistency checks for a family of variants from one job.

These run after every variant of a job has rendered. They compare plans, not
pixels: reading order of text, typographic hierarchy ratios, asset identity
(nothing dropped in some sizes only) and the layout family used per
orientation. Results are attached to each variant's quality report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract import CheckResult, CheckStatus, Severity


@dataclass
class VariantSnapshot:
    variant_id: str
    target: tuple[int, int]
    family: str | None
    placements: list[dict]  # {element_id, x, y, width, height, visible}
    typography: dict[str, dict]  # element_id -> {font_px, ...}
    roles: dict[str, str]  # element_id -> role
    hideable: set[str] = field(default_factory=set)
    master_px: dict[str, float] = field(default_factory=dict)  # element_id -> master font size


def _aspect_class(w: int, h: int) -> str:
    a = w / max(1, h)
    return "landscape" if a >= 1.25 else ("portrait" if a <= 0.8 else "square")


def _reading_order(snap: VariantSnapshot) -> list[str]:
    text_ids = [eid for eid in snap.typography]
    boxes = {p["element_id"]: p for p in snap.placements if p.get("visible", True)}
    ordered = sorted(
        (eid for eid in text_ids if eid in boxes),
        key=lambda eid: (boxes[eid]["y"], boxes[eid]["x"]),
    )
    return ordered


def family_consistency_checks(
    snapshots: list[VariantSnapshot],
    *,
    hierarchy_tolerance: float = 0.30,
) -> dict[str, list[CheckResult]]:
    """Return checks per variant id. A single variant gets no family checks."""
    out: dict[str, list[CheckResult]] = {s.variant_id: [] for s in snapshots}
    if len(snapshots) < 2:
        return out

    # 1) asset identity: every element visible somewhere must be visible everywhere
    visible_sets = {
        s.variant_id: {p["element_id"] for p in s.placements if p.get("visible", True)}
        for s in snapshots
    }
    union = set().union(*visible_sets.values())
    for s in snapshots:
        missing = sorted(
            eid for eid in union - visible_sets[s.variant_id] if eid not in s.hideable
        )
        out[s.variant_id].append(
            CheckResult(
                "family_identity",
                CheckStatus.FAIL if missing else CheckStatus.PASS,
                Severity.MAJOR,
                (
                    "Same elements as the rest of the family"
                    if not missing
                    else f"Missing elements present in other sizes: {', '.join(missing[:4])}"
                ),
                details={"missing": missing},
            )
        )

    # 2) reading order of text (only elements shared by all variants)
    shared_text = set.intersection(*[set(s.typography) for s in snapshots]) if snapshots else set()
    orders = {
        s.variant_id: [eid for eid in _reading_order(s) if eid in shared_text] for s in snapshots
    }
    reference = orders[snapshots[0].variant_id]
    for s in snapshots:
        same = orders[s.variant_id] == reference
        out[s.variant_id].append(
            CheckResult(
                "family_reading_order",
                CheckStatus.PASS if same else CheckStatus.NEEDS_REVIEW,
                Severity.MAJOR,
                "Text reads in the same order as the other sizes"
                if same
                else "Text order differs from the other sizes",
                details={"order": orders[s.variant_id], "reference": reference},
            )
        )

    # 3) hierarchy: pairwise font ratios stay within tolerance of the master's ratio
    #    (the design intent); when the master size is unknown, of the family median.
    pairs = sorted(shared_text)
    ratios: dict[tuple[str, str], list[tuple[str, float]]] = {}
    master = snapshots[0].master_px
    for s in snapshots:
        for i, a in enumerate(pairs):
            for b in pairs[i + 1 :]:
                pa = float(s.typography[a].get("font_px", 0) or 0)
                pb = float(s.typography[b].get("font_px", 0) or 0)
                if pa > 0 and pb > 0:
                    ratios.setdefault((a, b), []).append((s.variant_id, pa / pb))
    drift: dict[str, list[str]] = {s.variant_id: [] for s in snapshots}
    for (a, b), values in ratios.items():
        ma, mb = float(master.get(a, 0) or 0), float(master.get(b, 0) or 0)
        if ma > 0 and mb > 0:
            reference = ma / mb
        else:
            sorted_vals = sorted(v for _, v in values)
            reference = sorted_vals[len(sorted_vals) // 2]
        for vid, v in values:
            if reference > 0 and abs(v - reference) / reference > hierarchy_tolerance:
                drift[vid].append(f"{a}/{b}")
    for s in snapshots:
        bad = drift[s.variant_id]
        out[s.variant_id].append(
            CheckResult(
                "family_hierarchy",
                CheckStatus.PASS if not bad else CheckStatus.NEEDS_REVIEW,
                Severity.MAJOR,
                "Type hierarchy matches the other sizes"
                if not bad
                else f"Type hierarchy drifts from the family for: {', '.join(bad[:3])}",
                details={"drift_pairs": bad, "tolerance": hierarchy_tolerance},
            )
        )

    # 4) one family per orientation
    by_class: dict[str, set[str]] = {}
    for s in snapshots:
        by_class.setdefault(_aspect_class(*s.target), set()).add(s.family or "?")
    for s in snapshots:
        fams = by_class[_aspect_class(*s.target)]
        consistent = len(fams) == 1
        out[s.variant_id].append(
            CheckResult(
                "family_layout",
                CheckStatus.PASS if consistent else CheckStatus.NEEDS_REVIEW,
                Severity.MINOR,
                "One composition per orientation across the family"
                if consistent
                else f"Different compositions used for the same orientation: {sorted(fams)}",
                details={"families": sorted(fams)},
            )
        )
    return out
