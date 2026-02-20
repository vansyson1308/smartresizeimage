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
    """Infer horizon/ground transition from row-wise color + edge changes with smoothing."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h = arr.shape[0]
    if h < 8:
        return (max(0, h // 2 - 2), min(h, h // 2 + 2))

    luma = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)
    row_mean = luma.mean(axis=1)
    row_std = luma.std(axis=1)
    row_diff = np.abs(np.diff(row_mean, prepend=row_mean[:1]))
    row_std_diff = np.abs(np.diff(row_std, prepend=row_std[:1]))
    gx = np.abs(np.diff(luma, axis=1)).mean(axis=1) if arr.shape[1] > 1 else np.zeros(h)

    score = row_diff * 0.45 + gx * 0.35 + row_std_diff * 0.20
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel = kernel / kernel.sum()
    score = np.convolve(score, kernel, mode="same")

    y_start = int(h * 0.38)
    y_end = max(y_start + 1, int(h * 0.94))
    idx = y_start + int(np.argmax(score[y_start:y_end]))
    band = max(5, int(h * 0.035))
    return max(0, idx - band), min(h, idx + band)


def build_target_first_plan(
    anchors: list[Anchor],
    protected_mask: np.ndarray,
    target_size: tuple[int, int],
    source_background: Image.Image,
) -> RedesignPlan:
    """Build deterministic background/decor/ground fill zones with flat-illustration hints."""
    _ = anchors
    w, h = target_size
    fill_mask = ~protected_mask

    horizon_hint = infer_horizon_hint(source_background.resize(target_size).convert("RGB"))
    skyline_band = (
        max(0, horizon_hint[0] - int(0.05 * h)),
        min(h, horizon_hint[1] + int(0.05 * h)),
    )

    decor_mask = np.zeros((h, w), dtype=bool)
    top_h = max(1, int(0.22 * h))
    side_w = max(1, int(0.23 * w))
    zones = [
        (int(0.03 * w), int(0.02 * h), side_w, top_h),
        (w - side_w - int(0.03 * w), int(0.02 * h), side_w, top_h),
        (int(0.02 * w), int(0.20 * h), int(0.10 * w), int(0.28 * h)),
        (w - int(0.12 * w), int(0.20 * h), int(0.10 * w), int(0.28 * h)),
    ]
    for x, y, zw, zh in zones:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x1 + max(1, zw))
        y2 = min(h, y1 + max(1, zh))
        decor_mask[y1:y2, x1:x2] = True
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
