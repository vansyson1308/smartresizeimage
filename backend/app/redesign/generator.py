"""Background generation adapters for Phase 3 redesign."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from ..composition.background import BackgroundExtender
from .planner import RedesignPlan


class BackgroundGenerator:
    def generate(
        self,
        source_background: Image.Image,
        target_size: tuple[int, int],
        fill_mask: np.ndarray,
        decor_mask: np.ndarray,
        seed: int,
        variant: int,
        plan: RedesignPlan,
        recipe: str,
    ) -> tuple[Image.Image, dict[str, object]]:
        raise NotImplementedError


@dataclass
class DeterministicFlatGenerator(BackgroundGenerator):
    """Headless-safe deterministic flat-illustration generator."""

    def __post_init__(self) -> None:
        self._ext = BackgroundExtender(use_ai_inpainting=False)

    def generate(
        self,
        source_background: Image.Image,
        target_size: tuple[int, int],
        fill_mask: np.ndarray,
        decor_mask: np.ndarray,
        seed: int,
        variant: int,
        plan: RedesignPlan,
        recipe: str,
    ) -> tuple[Image.Image, dict[str, object]]:
        rng = random.Random(seed + variant * 101 + hash(recipe) % 997)
        src = self._ext.extend(source_background.convert("RGBA"), target_size)
        w, h = target_size

        palette = _extract_palette(src)
        canvas = _render_base_gradient((w, h), palette, plan.horizon_hint, rng)
        src_arr = np.array(src.convert("RGBA"), dtype=np.uint8)
        canvas = _stitch_skyline(canvas, src_arr, plan, rng)
        canvas = _boundary_polish(canvas, src_arr, fill_mask, rng)
        canvas = _micro_noise_unify(canvas, fill_mask, rng)
        canvas = _palette_lock(canvas, palette)

        decor_stats = {"particles": 0, "edge_cutoff": 0}
        if recipe != "background_only":
            strength = 0.45 if recipe == "light_decor" else 1.0
            img = Image.fromarray(canvas, mode="RGBA")
            img, decor_stats = _draw_decor(img, decor_mask, fill_mask, palette, rng, strength)
            canvas = np.array(img, dtype=np.uint8)

        # hard protected lock
        canvas[~fill_mask] = src_arr[~fill_mask]

        # retry once with strict clamp if drift is high
        drift = _palette_drift(canvas, src_arr, fill_mask)
        if drift > 42.0:
            canvas = _palette_lock(canvas, palette[:3])
            canvas[~fill_mask] = src_arr[~fill_mask]

        repetition = _repetition_penalty(canvas[:, :, :3], fill_mask)
        out = Image.fromarray(canvas, mode="RGBA")
        sy1, sy2 = plan.skyline_band
        y1, y2 = plan.horizon_hint
        return out, {
            "palette_stats": {
                "dominant": [list(c) for c in palette[:4]],
                "count": len(palette),
                "drift": round(float(_palette_drift(canvas, src_arr, fill_mask)), 3),
            },
            "horizon_hint": {"y1": y1, "y2": y2},
            "skyline_band": {"y1": sy1, "y2": sy2},
            "decor_stats": decor_stats,
            "recipe": recipe,
            "repetition_penalty": round(float(repetition), 3),
        }


@dataclass
class GenerativeFillAdapter(BackgroundGenerator):
    fallback: DeterministicFlatGenerator

    def generate(
        self,
        source_background: Image.Image,
        target_size: tuple[int, int],
        fill_mask: np.ndarray,
        decor_mask: np.ndarray,
        seed: int,
        variant: int,
        plan: RedesignPlan,
        recipe: str,
    ) -> tuple[Image.Image, dict[str, object]]:
        img, meta = self.fallback.generate(
            source_background,
            target_size,
            fill_mask,
            decor_mask,
            seed,
            variant,
            plan,
            recipe,
        )
        meta["prompt_style"] = "flat illustration background extension, consistent palette, masked"
        return img, meta


def _render_base_gradient(
    size: tuple[int, int],
    palette: list[tuple[int, int, int]],
    horizon_hint: tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    w, h = size
    sky_top = np.array(palette[0], dtype=np.float32)
    sky_mid = np.array(palette[min(1, len(palette) - 1)], dtype=np.float32)
    ground = np.array(palette[min(2, len(palette) - 1)], dtype=np.float32)
    y1, y2 = horizon_hint
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        if y < y1:
            t = y / max(1, y1)
            col = sky_top * (1 - t) + sky_mid * t
        elif y < y2:
            t = (y - y1) / max(1, y2 - y1)
            col = sky_mid * (1 - t) + ground * t
        else:
            col = ground
        noise = (rng.random() - 0.5) * 1.8
        out[y, :, :3] = np.clip(col + noise, 0, 255)
        out[y, :, 3] = 255
    return out


def _stitch_skyline(
    canvas: np.ndarray,
    src_arr: np.ndarray,
    plan: RedesignPlan,
    rng: random.Random,
) -> np.ndarray:
    h, w = canvas.shape[:2]
    sy1, sy2 = plan.skyline_band
    band_h = max(1, sy2 - sy1)
    strip_w = max(12, w // 14)
    x = 0
    while x < w:
        sx = rng.randint(0, max(0, w - strip_w))
        patch = src_arr[sy1:sy2, sx : sx + strip_w].copy()
        if patch.size == 0:
            break
        if (x // strip_w) % 2 == 1:
            patch = patch[:, ::-1]
        jitter = rng.randint(-2, 2)
        yy1 = max(0, min(h - band_h, sy1 + jitter))
        yy2 = yy1 + band_h
        xx2 = min(w, x + strip_w)
        pw = xx2 - x
        patch = patch[:band_h, :pw]
        if pw <= 0:
            break
        # y-band blend to avoid steps
        alpha = np.linspace(0.35, 0.95, patch.shape[0], dtype=np.float32)[:, None, None]
        dst = canvas[yy1:yy2, x:xx2].astype(np.float32)
        src = patch.astype(np.float32)
        canvas[yy1:yy2, x:xx2] = np.clip(src * alpha + dst * (1 - alpha), 0, 255).astype(np.uint8)
        x += strip_w
    return canvas


def _boundary_polish(
    canvas: np.ndarray,
    src_arr: np.ndarray,
    fill_mask: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    boundary = np.zeros_like(fill_mask, dtype=bool)
    boundary[:, 1:] |= fill_mask[:, 1:] != fill_mask[:, :-1]
    boundary[1:, :] |= fill_mask[1:, :] != fill_mask[:-1, :]
    if not boundary.any():
        return canvas
    band = boundary.copy()
    for _ in range(2):
        dil = band.copy()
        dil[:, 1:] |= band[:, :-1]
        dil[:, :-1] |= band[:, 1:]
        dil[1:, :] |= band[:-1, :]
        dil[:-1, :] |= band[1:, :]
        band = dil
    mix = 0.30 + rng.random() * 0.15
    dst = canvas[band].astype(np.float32)
    src = src_arr[band].astype(np.float32)
    canvas[band] = np.clip(dst * (1 - mix) + src * mix, 0, 255).astype(np.uint8)
    return canvas


def _micro_noise_unify(canvas: np.ndarray, fill_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    noise = np.zeros_like(canvas[:, :, :3], dtype=np.int16)
    noise[:, :, 0] = int((rng.random() - 0.5) * 2)
    noise[:, :, 1] = int((rng.random() - 0.5) * 2)
    noise[:, :, 2] = int((rng.random() - 0.5) * 2)
    arr = canvas[:, :, :3].astype(np.int16)
    arr[fill_mask] = np.clip(arr[fill_mask] + noise[fill_mask], 0, 255)
    canvas[:, :, :3] = arr.astype(np.uint8)
    return canvas


def _draw_decor(
    img: Image.Image,
    decor_mask: np.ndarray,
    fill_mask: np.ndarray,
    palette: list[tuple[int, int, int]],
    rng: random.Random,
    strength: float,
) -> tuple[Image.Image, dict[str, int]]:
    draw = ImageDraw.Draw(img)
    h, w = decor_mask.shape
    particles = int(40 + 70 * strength)
    stats = {"particles": 0, "edge_cutoff": 0}
    ys, xs = np.where(decor_mask)
    if len(xs) == 0:
        return img, stats
    points = list(zip(xs.tolist(), ys.tolist(), strict=False))
    min_spacing = 8
    placed: list[tuple[int, int]] = []
    for _ in range(particles * 3):
        if stats["particles"] >= particles:
            break
        px, py = points[rng.randint(0, len(points) - 1)]
        r = rng.randint(2, 6)
        if px - r < 4 or px + r >= w - 4 or py - r < 4 or py + r >= h - 4:
            stats["edge_cutoff"] += 1
            continue
        if not fill_mask[py, px]:
            continue
        if any((px - qx) ** 2 + (py - qy) ** 2 < min_spacing * min_spacing for qx, qy in placed):
            continue
        placed.append((px, py))
        col = palette[rng.randint(0, min(3, len(palette) - 1))]
        alpha = rng.randint(145, 225)
        if rng.random() < 0.70:
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(*col, alpha))
        else:
            draw.line((px - r, py, px + r, py), fill=(*col, alpha), width=1)
            draw.line((px, py - r, px, py + r), fill=(*col, alpha), width=1)
        stats["particles"] += 1
    return img, stats


def _palette_lock(canvas: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    flat = canvas[:, :, :3].reshape(-1, 3).astype(np.int16)
    pal = np.array(palette, dtype=np.int16)
    d = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d, axis=1)
    q = pal[idx].astype(np.uint8).reshape(canvas.shape[0], canvas.shape[1], 3)
    out = canvas.copy()
    out[:, :, :3] = q
    return out


def _palette_drift(canvas: np.ndarray, src_arr: np.ndarray, fill_mask: np.ndarray) -> float:
    if not fill_mask.any():
        return 0.0
    lhs = canvas[:, :, :3][fill_mask].astype(np.float32)
    rhs = src_arr[:, :, :3][fill_mask].astype(np.float32)
    diff = np.linalg.norm(lhs - rhs, axis=1)
    return float(diff.mean()) if diff.size else 0.0


def _repetition_penalty(arr: np.ndarray, fill_mask: np.ndarray) -> float:
    if arr.shape[1] < 4:
        return 0.0
    gray = arr.mean(axis=2)
    prof = gray.mean(axis=0)
    periodic = float(np.mean(np.abs(prof[2:] - prof[:-2]) < 2.0)) if prof.shape[0] > 3 else 0.0
    rep = 0.0
    c = 0
    for x in range(1, gray.shape[1]):
        m = fill_mask[:, x] & fill_mask[:, x - 1]
        if not m.any():
            continue
        rep += max(0.0, 3.0 - float(np.abs(gray[:, x][m] - gray[:, x - 1][m]).mean()))
        c += 1
    return periodic * 50.0 + (rep / max(1, c)) * 8.0


def _extract_palette(image: Image.Image, n: int = 5) -> list[tuple[int, int, int]]:
    q = image.convert("RGB").resize((160, 160)).quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = sorted(q.getcolors() or [], reverse=True)
    out: list[tuple[int, int, int]] = []
    for _cnt, idx in counts[:n]:
        base = idx * 3
        if base + 2 < len(pal):
            out.append((pal[base], pal[base + 1], pal[base + 2]))
    if not out:
        out = [(140, 190, 240), (120, 160, 220), (80, 90, 130)]
    return out


def make_generator() -> tuple[BackgroundGenerator, str]:
    det = DeterministicFlatGenerator()
    flag = os.environ.get("AUTOBANNER_ENABLE_GENERATIVE_REDESIGN", "false").lower() == "true"
    if flag:
        return GenerativeFillAdapter(fallback=det), "generative_adapter"
    return det, "deterministic_flat"
