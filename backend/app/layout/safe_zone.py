"""Keep key content out of platform UI overlay zones.

Stories, Reels and TikTok overlay the top and bottom of the canvas with
profile headers, captions and reply bars. :func:`fit_group_into_safe_rect`
moves (and if necessary uniformly shrinks) the whole foreground group so every
key element lands inside the safe rectangle. Treating the foreground as one
rigid group preserves the designer's relative arrangement, so no new overlaps
are introduced between elements.
"""

from __future__ import annotations

from ..models import BoundingBox

Rect = tuple[int, int, int, int]  # x1, y1, x2, y2


def _inside(box: BoundingBox, safe: Rect) -> bool:
    return box.x >= safe[0] and box.y >= safe[1] and box.x2 <= safe[2] and box.y2 <= safe[3]


def fit_group_into_safe_rect(
    boxes: dict[str, BoundingBox],
    critical_ids: set[str],
    safe: Rect,
) -> tuple[dict[str, BoundingBox], float]:
    """Return adjusted boxes and the uniform scale applied to the group.

    Boxes are returned unchanged (scale ``1.0``) when every critical box is
    already inside ``safe`` or when there is nothing critical to protect.
    """
    critical = {k: b for k, b in boxes.items() if k in critical_ids}
    if not critical or all(_inside(b, safe) for b in critical.values()):
        return dict(boxes), 1.0

    gx1 = min(b.x for b in boxes.values())
    gy1 = min(b.y for b in boxes.values())
    gx2 = max(b.x2 for b in boxes.values())
    gy2 = max(b.y2 for b in boxes.values())
    group_w = max(1, gx2 - gx1)
    group_h = max(1, gy2 - gy1)
    safe_w = max(1, safe[2] - safe[0])
    safe_h = max(1, safe[3] - safe[1])

    scale = min(1.0, safe_w / group_w, safe_h / group_h)
    new_w = group_w * scale
    new_h = group_h * scale

    # Keep the group's centre where it was as far as the safe rect allows.
    cx = (gx1 + gx2) / 2.0
    cy = (gy1 + gy2) / 2.0
    nx1 = min(max(cx - new_w / 2.0, safe[0]), safe[2] - new_w)
    ny1 = min(max(cy - new_h / 2.0, safe[1]), safe[3] - new_h)

    adjusted: dict[str, BoundingBox] = {}
    for key, b in boxes.items():
        x = nx1 + (b.x - gx1) * scale
        y = ny1 + (b.y - gy1) * scale
        w = max(1, int(round(b.width * scale)))
        h = max(1, int(round(b.height * scale)))
        adjusted[key] = BoundingBox(int(round(x)), int(round(y)), w, h)
    return adjusted, scale
