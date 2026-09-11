"""Layout grammar: layout families composed from a small vocabulary.

The hand-written families in ``planner.families_for`` are three fixed compositions
per orientation. The grammar spans the same space compositionally so the planner
can search it:

    arrangement   side (text column beside the subject) | stack (text above/below)
    text_side     left | right            (side)
    text_position top | bottom            (stack)
    text_share    how much of the usable width (side) or height (stack) the copy gets
    logo_corner   tl | tr
    text_align    left | center

Every generated family records these *traits*, which is what creative directions
("copy first", "text left", "subject on top", "centered") filter on. Landscape
formats only take side arrangements, portrait only stacks, square takes both.
Nothing here is learned: the grammar is a deterministic enumeration that the
document's constraints, the scoring function and the quality contract judge.
"""

from __future__ import annotations

from .planner import Family, Region, aspect_class

SIDE_SHARES = (0.45, 0.55, 0.65)
STACK_SHARES = (0.32, 0.40, 0.48)
MARGIN_X = 0.05
GAP = 0.04


def _side_family(cls: str, text_side: str, share: float, corner: str) -> Family:
    usable_w = 1.0 - 2 * MARGIN_X
    text_w = round(usable_w * share - GAP / 2, 3)
    subj_w = round(usable_w * (1.0 - share) - GAP / 2, 3)
    if text_side == "left":
        text = Region(MARGIN_X, 0.12, text_w, 0.78)
        subject = Region(round(MARGIN_X + text_w + GAP, 3), 0.06, subj_w, 0.88)
    else:
        subject = Region(MARGIN_X, 0.06, subj_w, 0.88)
        text = Region(round(MARGIN_X + subj_w + GAP, 3), 0.12, text_w, 0.78)
    logo_w, logo_h = (0.16, 0.12) if cls == "landscape" else (0.22, 0.08)
    logo = Region(MARGIN_X if corner == "tl" else round(1.0 - MARGIN_X - logo_w, 3),
                  0.04, logo_w, logo_h)
    name = f"g_{cls}_side_{text_side}{int(round(share * 100))}_{corner}"
    return Family(
        name, text, subject, logo, text_align="left",
        traits={"grammar": True, "arrangement": "side", "text_side": text_side,
                "text_share": share, "logo_corner": corner, "text_align": "left"},
    )


def _stack_family(cls: str, text_position: str, share: float, corner: str, align: str) -> Family:
    top, bottom = 0.12, 0.96
    usable_h = bottom - top
    text_h = round(usable_h * share - GAP / 2, 3)
    subj_h = round(usable_h * (1.0 - share) - GAP / 2, 3)
    if text_position == "top":
        text = Region(0.07, top + 0.01, 0.86, text_h)
        subject = Region(0.10, round(top + 0.01 + text_h + GAP, 3), 0.80, subj_h)
        subject_first = False
    else:
        subject = Region(0.10, top, 0.80, subj_h)
        text = Region(0.07, round(top + subj_h + GAP, 3), 0.86, text_h)
        subject_first = True
    logo_w, logo_h = 0.30, 0.08
    logo = Region(0.07 if corner == "tl" else round(1.0 - 0.07 - logo_w, 3), 0.03, logo_w, logo_h)
    name = f"g_{cls}_stack_{text_position}{int(round(share * 100))}_{corner}_{align[0]}"
    return Family(
        name, text, subject, logo, text_align=align, subject_first=subject_first,
        traits={"grammar": True, "arrangement": "stack", "text_position": text_position,
                "text_share": share, "logo_corner": corner, "text_align": align},
    )


def grammar_families(aspect: float) -> list[Family]:
    """All grammar families that suit ``aspect`` (12 landscape, 24 portrait, 36 square)."""
    cls = aspect_class(aspect)
    out: list[Family] = []
    if cls in ("landscape", "square"):
        for text_side in ("left", "right"):
            for share in SIDE_SHARES:
                for corner in ("tl", "tr"):
                    out.append(_side_family(cls, text_side, share, corner))
    if cls in ("portrait", "square"):
        for text_position in ("top", "bottom"):
            for share in STACK_SHARES:
                for corner in ("tl", "tr"):
                    for align in ("left", "center"):
                        out.append(_stack_family(cls, text_position, share, corner, align))
    return out


# ---- creative directions ---------------------------------------------------------------

DIRECTION_KEYS = ("family", "arrangement", "text_side", "text_position", "text_align",
                  "emphasis", "mood")
_TOKENS = {
    "copy": ("emphasis", "copy"), "copy-first": ("emphasis", "copy"), "text": ("emphasis", "copy"),
    "subject": ("emphasis", "subject"), "subject-first": ("emphasis", "subject"),
    "hero": ("emphasis", "subject"), "balanced": ("emphasis", "balanced"),
    "text-left": ("text_side", "left"), "left": ("text_side", "left"),
    "text-right": ("text_side", "right"), "right": ("text_side", "right"),
    "text-top": ("text_position", "top"), "top": ("text_position", "top"),
    "subject-top": ("text_position", "bottom"), "text-bottom": ("text_position", "bottom"),
    "bottom": ("text_position", "bottom"),
    "center": ("text_align", "center"), "centered": ("text_align", "center"),
    "side": ("arrangement", "side"), "stack": ("arrangement", "stack"),
    "stacked": ("arrangement", "stack"),
}
_VALUES = {
    "arrangement": ("side", "stack"), "text_side": ("left", "right"),
    "text_position": ("top", "bottom"), "text_align": ("left", "center", "right"),
    "emphasis": ("copy", "subject", "balanced"),
}


def parse_direction(raw: object) -> dict | None:
    """Normalise a direction given as a dict or as tokens (``"copy text-left"``).

    Returns None for an empty direction; raises ValueError for unknown tokens/values.
    """
    if raw in (None, "", {}, []):
        return None
    out: dict = {}
    if isinstance(raw, str):
        for token in raw.replace(",", " ").replace("+", " ").split():
            low = token.strip().lower()
            if low.startswith("family:"):
                out["family"] = token.split(":", 1)[1].strip()
            elif low in _TOKENS:
                key, value = _TOKENS[low]
                out[key] = value
            else:
                raise ValueError(f"unknown direction token {token!r}")
        return out or None
    if not isinstance(raw, dict):
        raise ValueError("direction must be an object or a token string")
    for key, value in raw.items():
        if key not in DIRECTION_KEYS:
            raise ValueError(f"unknown direction key {key!r}")
        if value in (None, ""):
            continue
        if key == "mood":
            out["mood"] = str(value).strip()[:200]
        elif key == "family":
            out["family"] = str(value).strip()[:80]
        else:
            v = str(value).strip().lower()
            if v not in _VALUES[key]:
                raise ValueError(f"direction {key}: use one of {', '.join(_VALUES[key])}")
            out[key] = v
    return out or None


def family_traits(fam: Family) -> dict:
    """Traits of any family; hand-written and learned ones are inferred from their regions."""
    if fam.traits:
        return fam.traits
    t, s = fam.text, fam.subject
    horizontal_overlap = t.x < s.x + s.w and s.x < t.x + t.w
    vertical_separated = t.y + t.h <= s.y + 0.02 or s.y + s.h <= t.y + 0.02
    stacked = horizontal_overlap and vertical_separated
    traits: dict = {"grammar": False, "arrangement": "stack" if stacked else "side",
                    "text_align": fam.text_align,
                    "logo_corner": "tl" if fam.logo.x + fam.logo.w / 2 < 0.5 else "tr"}
    if stacked:
        traits["text_position"] = "bottom" if fam.subject_first else "top"
        traits["text_share"] = round(t.h / max(1e-6, t.h + s.h), 3)
    else:
        traits["text_side"] = "left" if t.x < s.x else "right"
        traits["text_share"] = round(t.w / max(1e-6, t.w + s.w), 3)
    return traits


def apply_direction(
    candidates: list[Family], direction: dict | None
) -> tuple[list[Family], list[str]]:
    """Keep the candidates that satisfy ``direction``; report what could not be met.

    Filters are applied one by one; a filter that would leave nothing is skipped and
    reported (``direction_unmet:<key>``) so a portrait format asked for "text left"
    still gets a plan. ``emphasis`` keeps the half of the remaining candidates with the
    larger (copy) or smaller (subject) text share, or the middle half (balanced).
    """
    if not direction:
        return list(candidates), []
    kept = list(candidates)
    notes: list[str] = []
    if direction.get("family"):
        pinned = [f for f in kept if f.name == direction["family"]]
        if pinned:
            return pinned, []
        notes.append(f"direction_unmet:family:{direction['family']}")
    for key in ("arrangement", "text_side", "text_position", "text_align"):
        want = direction.get(key)
        if not want:
            continue
        narrowed = [f for f in kept if family_traits(f).get(key) == want]
        if narrowed:
            kept = narrowed
        else:
            notes.append(f"direction_unmet:{key}:{want}")
    emphasis = direction.get("emphasis")
    if emphasis and len(kept) > 1:
        ranked = sorted(kept, key=lambda f: family_traits(f).get("text_share", 0.5))
        n = len(ranked)
        half = max(1, n // 2)
        if emphasis == "copy":
            kept = ranked[n - half:]
        elif emphasis == "subject":
            kept = ranked[:half]
        else:
            lo = max(0, (n - half) // 2)
            kept = ranked[lo:lo + half]
    return kept, notes


def direction_key(direction: dict | None) -> str:
    """Stable key for grouping briefs that share a direction (mood is descriptive only)."""
    if not direction:
        return ""
    items = sorted((k, v) for k, v in direction.items() if k != "mood" and v)
    return "|".join(f"{k}={v}" for k, v in items)
