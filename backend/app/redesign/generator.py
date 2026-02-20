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
    ) -> tuple[Image.Image, dict[str, object]]:
        rng = random.Random(seed + variant * 101)
        src = self._ext.extend(source_background.convert("RGBA"), target_size)
        w, h = target_size

        palette = _extract_palette(src)
        sky_top = np.array(palette[0], dtype=np.float32)
        sky_mid = np.array(palette[min(1, len(palette) - 1)], dtype=np.float32)
        ground = np.array(palette[min(2, len(palette) - 1)], dtype=np.float32)

        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        y1, y2 = plan.horizon_hint

        for y in range(h):
            if y < y1:
                t = y / max(1, y1)
                col = sky_top * (1 - t) + sky_mid * t
            elif y < y2:
                t = (y - y1) / max(1, y2 - y1)
                col = sky_mid * (1 - t) + ground * t
            else:
                col = ground
            noise = (rng.random() - 0.5) * 2.0
            canvas[y, :, :3] = np.clip(col + noise, 0, 255)
            canvas[y, :, 3] = 255

        # Continue skyline structure with patch strips.
        sy1, sy2 = plan.skyline_band
        src_arr = np.array(src.convert("RGBA"), dtype=np.uint8)
        skyline_h = max(1, sy2 - sy1)
        strip_w = max(8, w // 12)
        x = 0
        while x < w:
            sx = (x + rng.randint(0, strip_w)) % max(1, w - strip_w)
            patch = src_arr[sy1:sy2, sx : sx + strip_w].copy()
            if (x // strip_w) % 2 == 1:
                patch = patch[:, ::-1]
            jitter = rng.randint(-3, 3)
            yy1 = max(0, min(h - skyline_h, sy1 + jitter))
            yy2 = yy1 + skyline_h
            xx2 = min(w, x + strip_w)
            pw = xx2 - x
            canvas[yy1:yy2, x:xx2] = patch[:skyline_h, :pw]
            x += strip_w

        img = Image.fromarray(canvas, mode="RGBA")

        # Procedural decor in decor zones.
        draw = ImageDraw.Draw(img)
        decor_stats = {"particles": 0, "edge_cutoff": 0}
        for _ in range(70 + variant * 10):
            px = rng.randint(8, w - 9)
            py = rng.randint(8, max(8, int(h * 0.35)))
            if not decor_mask[py, px]:
                continue
            r = rng.randint(2, 7)
            col = palette[rng.randint(0, min(3, len(palette) - 1))]
            alpha = rng.randint(140, 235)
            if px - r < 0 or px + r >= w or py - r < 0 or py + r >= h:
                decor_stats["edge_cutoff"] += 1
                continue
            if rng.random() < 0.65:
                draw.ellipse((px - r, py - r, px + r, py + r), fill=(*col, alpha))
            else:
                draw.line((px - r, py, px + r, py), fill=(*col, alpha), width=1)
                draw.line((px, py - r, px, py + r), fill=(*col, alpha), width=1)
            decor_stats["particles"] += 1

        out_arr = np.array(img, dtype=np.uint8)
        src_rgba = np.array(src.convert("RGBA"), dtype=np.uint8)
        out_arr[~fill_mask] = src_rgba[~fill_mask]
        out = Image.fromarray(out_arr, mode="RGBA")

        return out, {
            "palette_stats": {
                "dominant": [list(c) for c in palette[:4]],
                "count": len(palette),
            },
            "horizon_hint": {"y1": y1, "y2": y2},
            "skyline_band": {"y1": sy1, "y2": sy2},
            "decor_stats": decor_stats,
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
    ) -> tuple[Image.Image, dict[str, object]]:
        # Placeholder adapter: use deterministic output while maintaining interface.
        img, meta = self.fallback.generate(
            source_background,
            target_size,
            fill_mask,
            decor_mask,
            seed,
            variant,
            plan,
        )
        meta["prompt_style"] = "flat illustration background extension, masked"
        return img, meta


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
