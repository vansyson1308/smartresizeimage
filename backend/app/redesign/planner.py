"""Target-first planning helpers for Phase 3 redesign."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .anchors import Anchor


@dataclass(frozen=True)
class RedesignPlan:
    fill_mask: np.ndarray
    decor_mask: np.ndarray
    ground_mask: np.ndarray
    horizon_hint: tuple[int, int]
    skyline_band: tuple[int, int]
    decor_zones: list[tuple[int, int, int, int]]

    def summary(self) -> dict[str, object]:
        return {
            "horizon_hint": {"y1": self.horizon_hint[0], "y2": self.horizon_hint[1]},
            "skyline_band": {"y1": self.skyline_band[0], "y2": self.skyline_band[1]},
            "decor_zones": [
                {"x": x, "y": y, "width": w, "height": h} for x, y, w, h in self.decor_zones
            ],
        }


def infer_horizon_hint(image: Image.Image) -> tuple[int, int]:
    """Infer horizon/ground transition from row-wise color/edge changes."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h = arr.shape[0]
    if h < 8:
        return (max(0, h // 2 - 2), min(h, h // 2 + 2))

    row_mean = arr.mean(axis=(1, 2))
    row_diff = np.abs(np.diff(row_mean, prepend=row_mean[:1]))
    gx = np.abs(np.diff(arr, axis=1)).mean(axis=(1, 2)) if arr.shape[1] > 1 else np.zeros(h)
    score = row_diff * 0.65 + gx * 0.35

    y_start = int(h * 0.45)
    y_end = max(y_start + 1, int(h * 0.92))
    idx = y_start + int(np.argmax(score[y_start:y_end]))
    band = max(4, int(h * 0.03))
    return max(0, idx - band), min(h, idx + band)


def build_target_first_plan(
    anchors: list[Anchor],
    protected_mask: np.ndarray,
    target_size: tuple[int, int],
    source_background: Image.Image,
) -> RedesignPlan:
    """Build deterministic background/decor/ground fill zones with flat-illustration hints."""
    w, h = target_size
    fill_mask = ~protected_mask

    horizon_hint = infer_horizon_hint(source_background.resize(target_size).convert("RGB"))
    skyline_band = (
        max(0, horizon_hint[0] - int(0.04 * h)),
        min(h, horizon_hint[1] + int(0.04 * h)),
    )

    decor_mask = np.zeros((h, w), dtype=bool)
    top_h = max(1, int(0.24 * h))
    side_w = max(1, int(0.26 * w))
    zones = [
        (0, 0, side_w, top_h),
        (w - side_w, 0, side_w, top_h),
        (0, int(0.18 * h), int(0.12 * w), int(0.28 * h)),
        (w - int(0.12 * w), int(0.18 * h), int(0.12 * w), int(0.28 * h)),
    ]
    for x, y, zw, zh in zones:
        x2 = min(w, x + zw)
        y2 = min(h, y + zh)
        decor_mask[y:y2, x:x2] = True
    decor_mask &= fill_mask

    ground_mask = np.zeros((h, w), dtype=bool)
    ground_mask[horizon_hint[1] :, :] = True
    ground_mask &= fill_mask

    return RedesignPlan(
        fill_mask=fill_mask,
        decor_mask=decor_mask,
        ground_mask=ground_mask,
        horizon_hint=horizon_hint,
        skyline_band=skyline_band,
        decor_zones=zones,
    )
