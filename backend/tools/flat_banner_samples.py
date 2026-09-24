"""Deterministic, realistic *flattened* banners with ground-truth element boxes.

Used to measure auto-layering (flat image -> pseudo layers) accuracy. Unlike the
Phase 2.1 bench fixtures, the text, CTA and logo are actually rendered into the
pixels, so a detector only sees what a real exported PNG/JPG would contain.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADLINES = [
    "SUMMER SALE",
    "NEW COLLECTION",
    "BLACK FRIDAY DEALS",
    "FRESH & HEALTHY",
    "GRAND OPENING",
    "BACK TO SCHOOL",
]
SUBS = [
    "Up to 50% off selected items",
    "Free shipping on every order",
    "Limited time only - don't miss out",
    "Discover the season's best picks",
]
CTAS = ["SHOP NOW", "LEARN MORE", "GET STARTED", "ORDER TODAY"]

# fmt: off
PALETTES = [
    # background top, background bottom, text, cta fill, cta text, hero
    ((255, 214, 102), (255, 170, 60), (40, 30, 90), (40, 30, 90), (255, 255, 255), (230, 70, 90)),
    ((24, 40, 90), (60, 20, 110), (255, 255, 255), (255, 200, 40), (30, 30, 30), (80, 200, 220)),
    ((230, 245, 235), (190, 230, 205), (20, 80, 50), (20, 120, 70), (255, 255, 255), (250, 150, 60)),  # noqa: E501
    ((250, 240, 250), (225, 205, 240), (90, 30, 110), (230, 60, 120), (255, 255, 255), (120, 90, 220)),  # noqa: E501
]
# fmt: on


@dataclass(frozen=True)
class Sample:
    name: str
    image: Image.Image
    boxes: dict[str, dict[str, int]]  # role -> bbox (x, y, width, height)
    layers: dict[str, Image.Image]  # role -> full-canvas RGBA layer ("background" included)

    def design_elements(self) -> list:
        """The sample as layered DesignElements, like a parsed PSD."""
        # Import from the same package root the caller uses (``backend.app`` in
        # tests, ``app`` when run from backend/), so enum identities match.
        root = "backend.app" if __name__.startswith("backend.") else "app"
        enums = importlib.import_module(f"{root}.enums")
        models = importlib.import_module(f"{root}.models")
        ElementRole = enums.ElementRole  # noqa: N806
        BoundingBox, DesignElement = models.BoundingBox, models.DesignElement  # noqa: N806

        out = []
        for z, (role, layer) in enumerate(self.layers.items()):
            box = self.boxes.get(role, {"x": 0, "y": 0, "width": layer.width,
                                        "height": layer.height})
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            out.append(DesignElement(
                id=role, name=role, layer_type="pixel",
                bbox=BoundingBox(x, y, w, h),
                image=layer.crop((x, y, x + w, y + h)),
                role=ElementRole(role), z_index=z,
            ))
        return out


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _gradient(size: tuple[int, int], top: tuple[int, ...], bottom: tuple[int, ...]) -> Image.Image:
    w, h = size
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    a = np.array(top, dtype=np.float32)[None, None, :]
    b = np.array(bottom, dtype=np.float32)[None, None, :]
    arr = np.broadcast_to(a * (1 - t) + b * t, (h, w, 3)).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill) -> tuple:
    draw.text(xy, text, font=font, fill=fill)
    return draw.textbbox(xy, text, font=font)


def _to_box(b: tuple[int, int, int, int]) -> dict[str, int]:
    return {"x": int(b[0]), "y": int(b[1]), "width": int(b[2] - b[0]), "height": int(b[3] - b[1])}


def make_sample(index: int, size: tuple[int, int] = (1200, 628), seed: int = 7) -> Sample:
    rng = random.Random(seed * 1000 + index)
    w, h = size
    top, bottom, text_c, cta_c, cta_text_c, hero_c = PALETTES[index % len(PALETTES)]
    bg = _gradient(size, top, bottom).convert("RGBA")
    layers: dict[str, Image.Image] = {"background": bg}

    def layer(role: str) -> ImageDraw.ImageDraw:
        layers[role] = Image.new("RGBA", size, (0, 0, 0, 0))
        return ImageDraw.Draw(layers[role])

    d = ImageDraw.Draw(bg)
    layout = index % 3  # 0: text left/hero right, 1: hero left/text right, 2: centered text

    # subtle decoration dots (should be treated as background/decoration)
    for _ in range(18):
        cx, cy, r = rng.randint(0, w), rng.randint(0, h), rng.randint(2, 5)
        shade = tuple(min(255, c + 25) for c in top)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=shade)

    boxes: dict[str, dict[str, int]] = {}
    margin = int(0.06 * w)
    text_w = int(0.5 * w)
    if layout == 0:
        tx, hero_cx = margin, int(0.75 * w)
    elif layout == 1:
        tx, hero_cx = int(0.46 * w), int(0.23 * w)
    else:
        tx, hero_cx = int(0.25 * w), None

    headline = HEADLINES[index % len(HEADLINES)]
    hsize = int(h * 0.13)
    font_h = _font(hsize)
    while d.textlength(headline, font=font_h) > text_w and hsize > 20:
        hsize -= 4
        font_h = _font(hsize)
    y = int(0.2 * h)
    d = layer("headline")
    hb = _text_box(d, (tx, y), headline, font_h, text_c)
    boxes["headline"] = _to_box(hb)

    sub = SUBS[index % len(SUBS)]
    font_s = _font(max(14, int(hsize * 0.38)))
    d = layer("subheadline")
    sb = _text_box(d, (tx, hb[3] + int(0.05 * h)), sub, font_s, text_c)
    boxes["subheadline"] = _to_box(sb)

    cta = CTAS[index % len(CTAS)]
    font_c = _font(max(14, int(hsize * 0.34)))
    tw = d.textlength(cta, font=font_c)
    pad_x, pad_y = int(0.025 * w), int(0.03 * h)
    cy0 = sb[3] + int(0.08 * h)
    cb = (tx, cy0, int(tx + tw + 2 * pad_x), int(cy0 + font_c.size + 2 * pad_y))
    d = layer("cta")
    d.rounded_rectangle(cb, radius=int(0.02 * h), fill=cta_c)
    d.text((tx + pad_x, cy0 + pad_y), cta, font=font_c, fill=cta_text_c)
    boxes["cta"] = _to_box(cb)

    # logo: badge in a top corner
    lw, lh = int(0.12 * w), int(0.09 * h)
    lx = w - lw - int(0.03 * w) if layout != 1 else int(0.03 * w)
    ly = int(0.04 * h)
    d = layer("logo")
    d.rounded_rectangle((lx, ly, lx + lw, ly + lh), radius=6, fill=(255, 255, 255))
    d.text((lx + int(0.02 * w), ly + int(0.025 * h)), "BRAND", font=_font(int(lh * 0.4)),
           fill=text_c if layout != 1 else (30, 30, 30))
    boxes["logo"] = _to_box((lx, ly, lx + lw, ly + lh))

    if hero_cx is not None:
        r = int(0.3 * h)
        hcy = int(0.58 * h)
        d = layer("hero_image")
        d.ellipse((hero_cx - r, hcy - r, hero_cx + r, hcy + r), fill=hero_c)
        # "face" details so the hero is not a flat disc
        e1, e2 = r // 3, r // 6
        white, ink = (255, 255, 255), (40, 40, 40)
        d.ellipse((hero_cx - e1, hcy - e1, hero_cx - e2, hcy - e2), fill=white)
        d.ellipse((hero_cx + e2, hcy - e1, hero_cx + e1, hcy - e2), fill=white)
        d.arc((hero_cx - r // 2, hcy, hero_cx + r // 2, hcy + r // 2), 20, 160, fill=ink, width=6)
        boxes["hero_image"] = _to_box((hero_cx - r, hcy - r, hero_cx + r, hcy + r))

    img = Image.new("RGBA", size)
    for lyr in layers.values():
        img.alpha_composite(lyr)
    return Sample(f"flat_{index:02d}", img.convert("RGB"), boxes, layers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="backend/tests/fixtures/outputs/flat_samples")
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        s = make_sample(i)
        s.image.save(out / f"{s.name}.png")
        (out / f"{s.name}.json").write_text(json.dumps(s.boxes, indent=2))
    print(f"Wrote {args.count} samples to {out}")


if __name__ == "__main__":
    main()
