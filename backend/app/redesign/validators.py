"""Candidate validators for Phase 3 redesign."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .anchors import Anchor


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    hard_fail: bool
    score: float
    reason: str = ""


def _ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1 = 6.5025
    c2 = 58.5225
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(num / den) if den > 0 else 0.0


class AnchorIntegrityValidator:
    def __init__(self, min_ssim: float = 0.995) -> None:
        self.min_ssim = min_ssim

    def validate(self, image: Image.Image, anchors: list[Anchor]) -> ValidationResult:
        img = image.convert("RGBA")
        per_anchor: list[float] = []
        for a in anchors:
            if not a.protected:
                continue
            b = a.target_bbox
            patch = img.crop((b.x, b.y, b.x2, b.y2)).resize(a.image.size)
            ssim = _ssim_like(np.array(patch), np.array(a.image))
            per_anchor.append(ssim)
            if ssim < self.min_ssim:
                return ValidationResult(
                    False,
                    True,
                    0.0,
                    f"anchor_integrity:{a.element_id}:{ssim:.4f}",
                )

        avg = float(sum(per_anchor) / len(per_anchor)) if per_anchor else 1.0
        return ValidationResult(True, False, avg * 100.0, "")


class OCRTextValidator:
    def validate(self, _image: Image.Image, _anchors: list[Anchor]) -> ValidationResult:
        return ValidationResult(True, False, 100.0, "ocr_skipped")


class SeamArtifactHeuristic:
    def validate(self, image: Image.Image, fill_mask: np.ndarray) -> ValidationResult:
        arr = np.array(image.convert("L"), dtype=np.float32)
        gx = np.abs(np.diff(arr, axis=1))
        gy = np.abs(np.diff(arr, axis=0))
        grad = np.pad(gx, ((0, 0), (0, 1))) + np.pad(gy, ((0, 1), (0, 0)))
        if fill_mask.shape != grad.shape or not fill_mask.any():
            return ValidationResult(True, False, 90.0, "seam_skip")

        boundary = np.zeros_like(fill_mask, dtype=bool)
        boundary[:, 1:] |= fill_mask[:, 1:] != fill_mask[:, :-1]
        boundary[1:, :] |= fill_mask[1:, :] != fill_mask[:-1, :]
        boundary_grad = float(np.mean(grad[boundary])) if boundary.any() else 0.0

        # repeated-edge detector: very similar adjacent columns in fill area
        rep = 0.0
        col_sim_count = 0
        for x in range(1, arr.shape[1]):
            mask_col = fill_mask[:, x] & fill_mask[:, x - 1]
            if not mask_col.any():
                continue
            d = np.abs(arr[:, x][mask_col] - arr[:, x - 1][mask_col]).mean()
            rep += max(0.0, 3.0 - d)
            col_sim_count += 1
        rep_score = rep / max(1, col_sim_count)

        # periodic banding detector (lag-2 similarity catches striped repeats)
        col_profile = arr.mean(axis=0)
        if col_profile.shape[0] > 3:
            periodic = float(np.mean(np.abs(col_profile[2:] - col_profile[:-2]) < 2.5))
        else:
            periodic = 0.0

        penalty = min(95.0, boundary_grad * 0.15 + rep_score * 12.0 + periodic * 55.0)
        return ValidationResult(True, False, max(0.0, 100.0 - penalty), "")


class PaletteDistanceValidator:
    def validate(
        self,
        image: Image.Image,
        source_bg: Image.Image,
        fill_mask: np.ndarray,
    ) -> ValidationResult:
        a = np.array(image.convert("RGB"), dtype=np.float32)
        b = np.array(source_bg.resize(image.size).convert("RGB"), dtype=np.float32)
        if fill_mask.shape != a[:, :, 0].shape or not fill_mask.any():
            return ValidationResult(True, False, 100.0, "")
        diff = np.linalg.norm(a[fill_mask] - b[fill_mask], axis=1)
        mean = float(diff.mean()) if diff.size else 0.0
        score = max(0.0, 100.0 - min(100.0, mean * 0.9))
        return ValidationResult(True, False, score, "")


class DecorCutoffValidator:
    def validate(self, decor_stats: dict[str, int | float]) -> ValidationResult:
        total = int(decor_stats.get("particles", 0))
        cut = int(decor_stats.get("edge_cutoff", 0))
        if total <= 0:
            return ValidationResult(True, False, 70.0, "no_particles")
        ratio = cut / max(1, total)
        score = max(0.0, 100.0 - min(100.0, ratio * 300.0))
        return ValidationResult(True, False, score, "")


class HorizonContinuityValidator:
    def validate(self, image: Image.Image, horizon_hint: tuple[int, int]) -> ValidationResult:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        y1, y2 = horizon_hint
        y = max(1, min(arr.shape[0] - 2, (y1 + y2) // 2))
        band = arr[y - 1 : y + 2]
        # continuity proxy: avoid large step changes along x
        d = np.abs(np.diff(band.mean(axis=0), axis=0)).mean()
        score = max(0.0, 100.0 - min(100.0, d * 1.6))
        return ValidationResult(True, False, score, "")
