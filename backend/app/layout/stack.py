"""Role-aware stack layout: overlap-free relayout for layered designs.

Every ad size falls into one of three archetypes, and designers lay banners
out the same way each time:

* **strip** (728x90, 320x50, 970x250 ...): one row - logo | copy | hero | CTA
* **landscape** (1200x628, 300x250 ...): copy column beside the hero
* **vertical** (square, 4:5, stories, skyscrapers): logo, copy, hero, CTA stacked

Elements are placed along an axis, so they cannot overlap. A single scale
factor is solved (by bisection) for each group, so the source hierarchy
(headline bigger than sub-copy, and so on) is preserved. Raster layers are
only ever scaled uniformly, never stretched. When a size is too small for
everything to stay legible, the least important elements are dropped one by
one (body copy, then sub-copy, secondary visuals, logo, hero). The headline
and CTA are never dropped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import BoundingBox, DesignElement, LayoutResult
from .profiles import LayoutProfile, pick_profile

_COPY_ROLES = (
    ElementRole.HEADLINE,
    ElementRole.SUBHEADLINE,
    ElementRole.BODY_TEXT,
    ElementRole.LABEL,
    ElementRole.BADGE,
)
_VISUAL_ROLES = (ElementRole.HERO_IMAGE, ElementRole.PHOTO, ElementRole.ILLUSTRATION)

# Minimum rendered height (px) for an element to count as legible. Keep the
# small-copy value in sync with the QA check (service._MIN_TEXT_HEIGHT_PX).
MIN_SMALL_TEXT_PX = 9
_MIN_HEIGHT = {
    ElementRole.HEADLINE: 12,
    ElementRole.SUBHEADLINE: MIN_SMALL_TEXT_PX,
    ElementRole.BODY_TEXT: MIN_SMALL_TEXT_PX,
    ElementRole.LABEL: MIN_SMALL_TEXT_PX,
    ElementRole.BADGE: 10,
    ElementRole.CTA: 12,
    ElementRole.LOGO: 10,
}
# Lower number = dropped first when space runs out.
_DROP_ORDER = {
    ElementRole.BODY_TEXT: 0,
    ElementRole.LABEL: 0,
    ElementRole.BADGE: 1,
    ElementRole.SUBHEADLINE: 2,
    ElementRole.LOGO: 4,
}
_MAX_UPSCALE = 2.5  # beyond this raster layers look soft


@dataclass
class _Item:
    elem: DesignElement
    w0: float
    h0: float
    cap_w: float = 1.0  # max width as a fraction of the region width
    cap_h: float = 1.0  # max height as a fraction of the region height
    children: list[_Item] = field(default_factory=list)  # composite copy block (strips)
    child_gap: float = 0.0

    @property
    def role(self) -> ElementRole:
        return self.elem.role

    @property
    def aspect(self) -> float:
        return self.w0 / max(1e-6, self.h0)


@dataclass
class StackLayoutDebug:
    mode: str
    dropped: list[str]
    scale: float


def _natural_size(elem: DesignElement) -> tuple[float, float]:
    if elem.image is not None:
        return float(elem.image.width), float(elem.image.height)
    return float(max(1, elem.bbox.width)), float(max(1, elem.bbox.height))


def _solve_scale(total_at: Callable[[float], float], budget: float, s_max: float) -> float:
    """Largest s in [0, s_max] with total_at(s) <= budget (total_at is monotone)."""
    if total_at(s_max) <= budget:
        return s_max
    lo, hi = 0.0, s_max
    for _ in range(40):
        mid = (lo + hi) / 2
        if total_at(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def _v_height(item: _Item, s: float, rw: float, rh: float) -> float:
    return min(s * item.h0, item.cap_w * rw / item.aspect, item.cap_h * rh)


def _h_width(item: _Item, s: float, rw: float, rh: float) -> float:
    h = min(s * item.h0, item.cap_h * rh, item.cap_w * rw / item.aspect)
    return h * item.aspect


class StackLayoutEngine:
    """Deterministic role-aware layout for layered designs."""

    def __init__(self) -> None:
        self.last_debug: StackLayoutDebug | None = None
        self._profile: LayoutProfile = pick_profile(1, 1)

    # ------------------------------------------------------------------ public
    def calculate(
        self,
        elements: list[DesignElement],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> list[LayoutResult] | None:
        """Return layout results, or ``None`` when there is nothing to arrange."""
        tw, th = target_size
        background, foreground, hidden = [], [], []
        for e in elements:
            if e.role in BACKGROUND_ROLES:
                background.append(e)
            elif e.image is None or e.role == ElementRole.GROUP or not e.visible:
                hidden.append(e)
            else:
                foreground.append(e)
        if not foreground:
            return None

        roles = self._assign(foreground, source_size)
        self._profile = pick_profile(tw, th)
        ratio = tw / th
        mode = "strip" if ratio >= 2.8 else "landscape" if ratio >= 1.3 else "vertical"

        dropped: list[str] = []
        boxes: dict[str, BoundingBox] = {}
        for _ in range(len(foreground) + 1):
            if mode == "strip":
                boxes, too_small = self._strip(roles, tw, th)
            elif mode == "landscape" and roles["hero"] is not None:
                boxes, too_small = self._landscape(roles, source_size, tw, th)
            else:
                boxes, too_small = self._vertical(roles, tw, th)
            victim = self._pick_victim(roles, too_small)
            if victim is None:
                break
            dropped.append(victim.id)
            self._remove(roles, victim)

        self._place_secondary(roles, boxes, source_size, target_size, dropped)

        results: list[LayoutResult] = []
        for e in elements:
            if e.role in BACKGROUND_ROLES:
                results.append(LayoutResult(e.id, BoundingBox(0, 0, tw, th), 1.0, True))
            elif e.id in boxes:
                b = boxes[e.id]
                w0, h0 = _natural_size(e)
                results.append(LayoutResult(e.id, b, b.height / max(1.0, h0), True))
            else:
                results.append(LayoutResult(e.id, BoundingBox(0, 0, 1, 1), 0.0, False))
        scale = max((r.scale_factor for r in results if r.visible and r.scale_factor), default=1)
        self.last_debug = StackLayoutDebug(mode=mode, dropped=dropped, scale=round(scale, 4))
        return results

    # --------------------------------------------------------------- grouping
    @staticmethod
    def _assign(
        foreground: list[DesignElement], source_size: tuple[int, int]
    ) -> dict[str, object]:
        canvas = max(1, source_size[0] * source_size[1])
        by_y = sorted(foreground, key=lambda e: (e.bbox.y, e.bbox.x))
        logo = next((e for e in by_y if e.role == ElementRole.LOGO), None)
        cta = next((e for e in by_y if e.role == ElementRole.CTA), None)
        copy = [e for e in by_y if e.role in _COPY_ROLES or (
            e.role in (ElementRole.CTA, ElementRole.LOGO) and e not in (cta, logo))]

        def is_visual(e: DesignElement) -> bool:
            if e.role in _VISUAL_ROLES:
                return True
            return e.role in (ElementRole.UNKNOWN, ElementRole.ICON) and (
                e.bbox.width * e.bbox.height > 0.08 * canvas
            )

        visuals = sorted(
            (e for e in foreground if is_visual(e)),
            key=lambda e: e.bbox.width * e.bbox.height,
            reverse=True,
        )
        hero = visuals[0] if visuals else None
        placed = {id(x) for x in [logo, cta, hero, *copy] if x is not None}
        secondary = [e for e in foreground if id(e) not in placed]
        return {"logo": logo, "cta": cta, "copy": copy, "hero": hero, "secondary": secondary}

    @staticmethod
    def _remove(roles: dict[str, object], victim: DesignElement) -> None:
        for key in ("logo", "cta", "hero"):
            if roles[key] is victim:
                roles[key] = None
        roles["copy"] = [e for e in roles["copy"] if e is not victim]  # type: ignore[union-attr]

    @staticmethod
    def _pick_victim(
        roles: dict[str, object], too_small: list[DesignElement]
    ) -> DesignElement | None:
        if not too_small:
            return None
        candidates = [e for e in roles["copy"]]  # type: ignore[union-attr]
        if roles["logo"] is not None:
            candidates.append(roles["logo"])  # type: ignore[arg-type]
        candidates = [
            e for e in candidates if e.role in _DROP_ORDER and e.role != ElementRole.HEADLINE
        ]
        if not candidates:
            return None  # only headline/CTA/hero left: keep them, even if small
        # An illegible droppable item goes first; otherwise free room by
        # dropping the least important element (lowest in the stack first).
        small = [e for e in candidates if e in too_small]
        pool = small or candidates
        return min(pool, key=lambda e: (_DROP_ORDER[e.role], -e.bbox.y))

    # ----------------------------------------------------------------- modes
    def _margins(self, tw: int, th: int, strip: bool) -> tuple[int, int]:
        # The layout profile's margins are per axis (x% of width, x% of height).
        pct = self._profile.margin_pct
        mx, my = math.ceil(pct * tw) + 1, math.ceil(pct * th) + 1
        if strip:
            return max(2, mx, round(0.025 * tw)), max(2, my, round(0.12 * th))
        return max(4, mx), max(4, my)

    def _stack(
        self,
        items: list[_Item],
        x: float,
        y: float,
        rw: float,
        rh: float,
        gap: float,
        align: str,
        s_max: float = _MAX_UPSCALE,
    ) -> tuple[dict[str, BoundingBox], float, float]:
        """Vertical stack of items in a region; returns boxes, scale and used height."""
        if not items:
            return {}, 0.0, 0.0
        budget = rh - gap * (len(items) - 1)

        def total(s: float) -> float:
            return sum(_v_height(it, s, rw, rh) for it in items)

        s = _solve_scale(total, max(1.0, budget), s_max)
        heights = [_v_height(it, s, rw, rh) for it in items]
        used = sum(heights) + gap * (len(items) - 1)
        cy = y + (rh - used) / 2
        boxes: dict[str, BoundingBox] = {}
        for it, h in zip(items, heights, strict=True):
            w = h * it.aspect
            if align == "left":
                bx = x
            elif align == "right":
                bx = x + rw - w
            else:
                bx = x + (rw - w) / 2
            boxes[it.elem.id] = BoundingBox(
                round(bx), round(cy), max(1, round(w)), max(1, round(h))
            )
            cy += h + gap
        return boxes, s, used

    def _item(
        self,
        e: DesignElement,
        cap_w: float = 1.0,
        cap_h: float = 1.0,
        region: tuple[float, float, int, int] | None = None,
    ) -> _Item:
        """Build an item; ``region`` = (rw, rh, tw, th) applies the profile's max sizes."""
        w0, h0 = _natural_size(e)
        limit = self._profile.max_sizes.get(e.role)
        if limit is not None and region is not None:
            rw, rh, tw, th = region
            cap_w = min(cap_w, limit[0] * tw / max(1.0, rw))
            cap_h = min(cap_h, limit[1] * th / max(1.0, rh))
        return _Item(e, w0, h0, cap_w=cap_w, cap_h=cap_h)

    def _vertical(
        self, roles: dict[str, object], tw: int, th: int
    ) -> tuple[dict[str, BoundingBox], list[DesignElement]]:
        mx, my = self._margins(tw, th, strip=False)
        rw, rh = tw - 2 * mx, th - 2 * my
        gap = max(3, round(0.03 * min(tw, th)))
        narrow = tw / th < 0.45

        region = (rw, rh, tw, th)
        top: list[_Item] = []
        if roles["logo"] is not None:
            top.append(
                self._item(roles["logo"], cap_w=0.5 if narrow else 0.3, cap_h=0.1, region=region)
            )
        for e in roles["copy"]:  # type: ignore[union-attr]
            cap = 0.92 if e.role == ElementRole.HEADLINE else 0.85
            top.append(self._item(e, cap_w=cap, region=region))
        bottom: list[_Item] = []
        if roles["cta"] is not None:
            bottom.append(self._item(
                roles["cta"], cap_w=0.9 if narrow else 0.6, cap_h=0.12, region=region
            ))

        hero = roles["hero"]
        copy_items = top + bottom
        if hero is None:
            boxes, _, _ = self._stack(copy_items, mx, my, rw, rh, gap, "center")
            return boxes, self._too_small(copy_items, boxes, rw, rh)

        # The hero should reach the profile's prominence target; copy gets the
        # rest of the height (between 30% and 60%).
        hero_elem: DesignElement = hero  # type: ignore[assignment]
        hw0, hh0 = _natural_size(hero_elem)
        want_h = math.sqrt(self._profile.target_hero_ratio * tw * th * hh0 / max(1.0, hw0))
        want_h = min(want_h, 0.92 * rw * hh0 / max(1.0, hw0))
        copy_share = min(0.6, max(0.3, (rh - want_h - 2 * gap) / rh))
        n_gaps = len(copy_items)  # gaps between copy items + around the hero
        budget = copy_share * rh - gap * max(0, len(copy_items) - 1)

        def total(s: float) -> float:
            return sum(_v_height(it, s, rw, rh) for it in copy_items)

        s = _solve_scale(total, max(1.0, budget), _MAX_UPSCALE)
        heights = {it.elem.id: _v_height(it, s, rw, rh) for it in copy_items}
        copy_h = sum(heights.values()) + gap * max(0, len(copy_items) - 1)

        hero_item = self._item(hero, cap_w=0.92)  # type: ignore[arg-type]
        remaining = rh - copy_h - gap * min(n_gaps, 2)
        hero_h = min(
            remaining, _MAX_UPSCALE * hero_item.h0, hero_item.cap_w * rw / hero_item.aspect
        )
        hero_h = max(1.0, hero_h)

        order = [*top, hero_item, *bottom]
        used = copy_h + hero_h + gap * (2 if bottom else 1)
        # Breathing room: spread leftover height between the groups
        # (logo | copy | hero | CTA) instead of clustering everything mid-canvas.
        logo_item = top[0] if top and top[0].elem is roles["logo"] else None
        breaks = {id(hero_item)} | {id(it) for it in bottom}
        if logo_item is not None and len(top) > 1:
            breaks.add(id(top[1]))
        extra = min(max(0.0, rh - used) / (len(breaks) + 2), 2.5 * gap)
        used += extra * len(breaks)
        cy = my + (rh - used) / 2
        boxes: dict[str, BoundingBox] = {}
        for it in order:
            if id(it) in breaks:
                cy += extra
            h = hero_h if it is hero_item else heights[it.elem.id]
            w = h * it.aspect
            boxes[it.elem.id] = BoundingBox(
                round(mx + (rw - w) / 2), round(cy), max(1, round(w)), max(1, round(h))
            )
            cy += h + gap
        return boxes, self._too_small(copy_items, boxes, rw, rh)

    def _landscape(
        self,
        roles: dict[str, object],
        source_size: tuple[int, int],
        tw: int,
        th: int,
    ) -> tuple[dict[str, BoundingBox], list[DesignElement]]:
        mx, my = self._margins(tw, th, strip=False)
        rw, rh = tw - 2 * mx, th - 2 * my
        col_gap = max(4, round(0.04 * tw))
        gap = max(3, round(0.035 * th))
        hero: DesignElement = roles["hero"]  # type: ignore[assignment]

        copy_elems = list(roles["copy"])  # type: ignore[arg-type]
        copy_cx = (
            sum(e.bbox.x + e.bbox.width / 2 for e in copy_elems) / len(copy_elems)
            if copy_elems else source_size[0] / 2
        )
        hero_right = hero.bbox.x + hero.bbox.width / 2 >= copy_cx

        hero_item = self._item(hero)
        hero_h = min(rh, _MAX_UPSCALE * hero_item.h0)
        hero_w = hero_h * hero_item.aspect
        if hero_w > 0.45 * rw:
            hero_w = 0.45 * rw
            hero_h = hero_w / hero_item.aspect
        copy_w = rw - hero_w - col_gap
        hx = mx + rw - hero_w if hero_right else mx
        cx = mx if hero_right else mx + hero_w + col_gap
        boxes = {
            hero.id: BoundingBox(round(hx), round(my + (rh - hero_h) / 2),
                                 max(1, round(hero_w)), max(1, round(hero_h)))
        }

        region = (copy_w, rh, tw, th)
        items: list[_Item] = []
        if roles["logo"] is not None:
            items.append(self._item(roles["logo"], cap_w=0.4, cap_h=0.16, region=region))
        items += [self._item(e, region=region) for e in copy_elems]
        if roles["cta"] is not None:
            items.append(self._item(roles["cta"], cap_w=0.7, cap_h=0.2, region=region))
        copy_boxes, _, _ = self._stack(items, cx, my, copy_w, rh, gap, "left")
        boxes.update(copy_boxes)
        return boxes, self._too_small(items, boxes, copy_w, rh)

    def _strip(
        self, roles: dict[str, object], tw: int, th: int
    ) -> tuple[dict[str, BoundingBox], list[DesignElement]]:
        mx, my = self._margins(tw, th, strip=True)
        rw, rh = tw - 2 * mx, th - 2 * my
        gap = max(4, round(0.02 * tw))

        copy_elems = list(roles["copy"])  # type: ignore[arg-type]
        block: _Item | None = None
        if copy_elems:
            kids = [self._item(e) for e in copy_elems]
            inner_gap = 0.12
            w0 = max(k.w0 for k in kids)
            h0 = sum(k.h0 for k in kids) * (1 + inner_gap * (len(kids) - 1) / max(1, len(kids)))
            block = _Item(copy_elems[0], w0, h0, cap_h=1.0, children=kids, child_gap=inner_gap)

        row: list[_Item] = []
        if roles["logo"] is not None:
            row.append(self._item(roles["logo"], cap_h=0.75))
        if block is not None:
            row.append(block)
        if roles["hero"] is not None:
            row.append(self._item(roles["hero"]))
        if roles["cta"] is not None:
            row.append(self._item(roles["cta"], cap_h=0.7))
        if not row:
            return {}, []

        budget = rw - gap * (len(row) - 1)

        def total(s: float) -> float:
            return sum(_h_width(it, s, rw, rh) for it in row)

        s = _solve_scale(total, max(1.0, budget), _MAX_UPSCALE)
        widths = [_h_width(it, s, rw, rh) for it in row]

        boxes: dict[str, BoundingBox] = {}
        # Left group (logo, copy) flush left; right group (hero, CTA) flush right.
        left = [i for i, it in enumerate(row) if it is block or it.elem is roles["logo"]]
        right = [i for i in range(len(row)) if i not in left]
        x = float(mx)
        for i in left:
            x = self._place_strip_item(row[i], widths[i], x, my, rh, boxes) + gap
        x = float(mx + rw - sum(widths[i] for i in right) - gap * max(0, len(right) - 1))
        for i in right:
            x = self._place_strip_item(row[i], widths[i], x, my, rh, boxes) + gap

        check = [c for it in row for c in (it.children or [it])]
        return boxes, self._too_small(check, boxes, rw, rh)

    def _place_strip_item(
        self,
        it: _Item,
        width: float,
        x: float,
        my: float,
        rh: float,
        boxes: dict[str, BoundingBox],
    ) -> float:
        height = width / it.aspect
        top = my + (rh - height) / 2
        if not it.children:
            boxes[it.elem.id] = BoundingBox(round(x), round(top), max(1, round(width)),
                                            max(1, round(height)))
            return x + width
        # Copy block: children stacked, left-aligned, sharing the block's scale.
        scale = height / it.h0
        gap = it.child_gap * scale * (sum(k.h0 for k in it.children) / len(it.children))
        y = top
        for k in it.children:
            kh = k.h0 * scale
            kw = kh * k.aspect
            boxes[k.elem.id] = BoundingBox(round(x), round(y), max(1, round(kw)),
                                           max(1, round(kh)))
            y += kh + gap
        return x + width

    @staticmethod
    def _too_small(
        items: list[_Item], boxes: dict[str, BoundingBox], rw: float, rh: float
    ) -> list[DesignElement]:
        """Illegible items whose size is limited by the shared scale.

        Items pinned by their own width/height cap (e.g. a long headline in a
        160px skyscraper) cannot grow by dropping other elements, so they are
        not reported - dropping content would not help them.
        """
        out = []
        for it in items:
            b = boxes.get(it.elem.id)
            need = _MIN_HEIGHT.get(it.role)
            if b is None or need is None or b.height >= need:
                continue
            cap = min(it.cap_w * rw / it.aspect, it.cap_h * rh)
            if b.height < cap - 1:
                out.append(it.elem)
        return out

    # ------------------------------------------------------- secondary visuals
    @staticmethod
    def _place_secondary(
        roles: dict[str, object],
        boxes: dict[str, BoundingBox],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        dropped: list[str],
    ) -> None:
        """Keep small visuals/decor at their relative position when they fit freely."""
        sw, sh = source_size
        tw, th = target_size
        scale = min(tw / max(1, sw), th / max(1, sh))
        placed = list(boxes.values())
        for e in roles["secondary"]:  # type: ignore[union-attr]
            w0, h0 = _natural_size(e)
            w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
            cx = (e.bbox.x + e.bbox.width / 2) / max(1, sw) * tw
            cy = (e.bbox.y + e.bbox.height / 2) / max(1, sh) * th
            x = int(min(max(cx - w / 2, 0), tw - w))
            y = int(min(max(cy - h / 2, 0), th - h))
            box = BoundingBox(x, y, w, h)
            if any(_overlap(box, p) > 0.05 * box.area for p in placed):
                dropped.append(e.id)
                continue
            boxes[e.id] = box
            placed.append(box)


def _overlap(a: BoundingBox, b: BoundingBox) -> int:
    ix = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    iy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return ix * iy
