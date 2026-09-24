"""Render the README demo images with the real engine.

Every picture in ``docs/demo/`` is produced by this script: it draws three
master banners as separate layers (like a PSD), runs them through
``RenderService`` exactly as the API/CLI would, and lays the real outputs out
on showcase boards labelled with each file's real size and weight.

    python backend/tools/make_demo.py            # writes docs/demo/*.jpg

Fonts: uses Liberation/DejaVu Sans Bold when installed, otherwise Pillow's
built-in font (outputs then differ slightly in typography only).
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from backend.app.config import Config
from backend.app.enums import ElementRole
from backend.app.export import ExportOptions
from backend.app.models import BoundingBox, DesignElement
from backend.app.presets import custom_preset, get_preset
from backend.app.relayout import ReLayoutEngine
from backend.app.service import RenderRequest, RenderService

W, H = 1200, 628
_FONT_CANDIDATES = {
    True: [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
    False: [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
}


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES[bold]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# --------------------------------------------------------------------- masters
@dataclass
class Master:
    name: str
    title: str
    layers: list[tuple[str, ElementRole, Image.Image]]  # (id, role, full-canvas RGBA)

    def flat(self) -> Image.Image:
        canvas = Image.new("RGBA", (W, H))
        for _, _, layer in self.layers:
            canvas.alpha_composite(layer)
        return canvas.convert("RGB")

    def elements(self) -> list[DesignElement]:
        out = []
        for z, (eid, role, layer) in enumerate(self.layers):
            box = layer.getbbox() or (0, 0, W, H)
            if role == ElementRole.BACKGROUND:
                box = (0, 0, W, H)
            x1, y1, x2, y2 = box
            out.append(DesignElement(
                id=eid, name=eid, layer_type="pixel",
                bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                image=layer.crop(box), role=role, z_index=z,
            ))
        return out


def _layer() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _gradient(top: tuple[int, int, int], bottom: tuple[int, int, int],
              diagonal: bool = False) -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (yy / H * 0.7 + xx / W * 0.3) if diagonal else yy / H
    a, b = np.array(top, np.float32), np.array(bottom, np.float32)
    arr = a * (1 - t[..., None]) + b * t[..., None]
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


def _text_layer(xy: tuple[int, int], text: str, size: int, fill, bold: bool = True):
    img, d = _layer()
    d.text(xy, text, font=font(size, bold), fill=fill)
    return img


def _pill(xy: tuple[int, int], text: str, size: int, fill, text_fill, pad=(34, 18)):
    img, d = _layer()
    f = font(size)
    tw = d.textlength(text, font=f)
    x, y = xy
    h = size + 2 * pad[1]
    d.rounded_rectangle((x, y, x + tw + 2 * pad[0], y + h), radius=h // 2, fill=fill)
    d.text((x + pad[0], y + pad[1] - size * 0.08), text, font=f, fill=text_fill)
    return img


def _soft_dots(bg: Image.Image, color, seed: int, n: int = 22) -> None:
    rng = np.random.default_rng(seed)
    over = Image.new("RGBA", bg.size)
    d = ImageDraw.Draw(over)
    for _ in range(n):
        x, y = rng.integers(0, W), rng.integers(0, H)
        r = int(rng.integers(6, 28))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*color, int(rng.integers(40, 90))))
    bg.alpha_composite(over.filter(ImageFilter.GaussianBlur(1.5)))


def coffee_master() -> Master:
    bg = _gradient((252, 241, 225), (236, 205, 170))
    _soft_dots(bg, (255, 255, 255), 1)
    hero, d = _layer()
    cx, cy = 880, 360
    d.ellipse((cx - 210, cy + 150, cx + 210, cy + 210), fill=(255, 250, 244))  # saucer
    d.ellipse((cx - 170, cy + 140, cx + 170, cy + 190), fill=(230, 214, 196))
    d.polygon([(cx - 150, cy - 70), (cx + 150, cy - 70), (cx + 115, cy + 160),
               (cx - 115, cy + 160)], fill=(196, 84, 52))  # cup body
    d.ellipse((cx - 115, cy + 130, cx + 115, cy + 185), fill=(196, 84, 52))
    d.arc((cx + 100, cy - 30, cx + 220, cy + 90), -80, 80, fill=(196, 84, 52), width=24)
    d.ellipse((cx - 150, cy - 105, cx + 150, cy - 35), fill=(160, 64, 38))  # rim
    d.ellipse((cx - 132, cy - 96, cx + 132, cy - 44), fill=(92, 52, 32))  # coffee
    d.ellipse((cx - 60, cy - 84, cx + 40, cy - 58), fill=(160, 110, 70))  # crema
    for i, sx in enumerate((-60, 0, 60)):  # steam
        pts = [(cx + sx + 18 * math.sin(k / 4 + i), cy - 120 - k * 6) for k in range(22)]
        d.line(pts, fill=(255, 255, 255, 200), width=10, joint="curve")
    return Master(
        "coffee", "Coffee shop",
        [
            ("background", ElementRole.BACKGROUND, bg),
            ("logo", ElementRole.LOGO, _pill((70, 52), "BREW & CO", 26, (74, 44, 30),
                                             (255, 244, 230), pad=(26, 14))),
            ("headline", ElementRole.HEADLINE, _text_layer((70, 170), "Morning Brew", 92,
                                                           (74, 44, 30))),
            ("subheadline", ElementRole.SUBHEADLINE,
             _text_layer((74, 290), "Fresh roasted · delivered by 8am", 36, (122, 82, 58),
                         bold=False)),
            ("cta", ElementRole.CTA, _pill((74, 380), "ORDER NOW", 32, (226, 108, 44),
                                           (255, 255, 255))),
            ("hero", ElementRole.HERO_IMAGE, hero),
        ],
    )


def tech_master() -> Master:
    bg = _gradient((18, 24, 64), (72, 28, 110), diagonal=True)
    _soft_dots(bg, (120, 140, 255), 2, n=30)
    hero, d = _layer()
    glow = Image.new("RGBA", (W, H))
    ImageDraw.Draw(glow).ellipse((700, 70, 1140, 510), fill=(90, 180, 255, 110))
    hero.alpha_composite(glow.filter(ImageFilter.GaussianBlur(40)))
    d = ImageDraw.Draw(hero)
    d.rounded_rectangle((820, 80, 1030, 540), radius=34, fill=(20, 22, 30))  # phone
    screen = _gradient((64, 200, 255), (160, 70, 255)).crop((0, 0, 186, 420))
    mask = Image.new("L", screen.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 185, 419), radius=24, fill=255)
    hero.paste(screen, (832, 100), mask)
    d.ellipse((917, 108, 933, 124), fill=(20, 22, 30))
    for i in range(3):
        d.rounded_rectangle((852, 330 + i * 50, 998, 360 + i * 50), radius=8,
                            fill=(255, 255, 255, 90))
    badge, bd = _layer()
    bd.ellipse((1015, 70, 1160, 215), fill=(255, 64, 96))
    bd.text((1040, 112), "-50%", font=font(44), fill=(255, 255, 255))
    return Master(
        "tech", "Electronics",
        [
            ("background", ElementRole.BACKGROUND, bg),
            ("logo", ElementRole.LOGO, _text_layer((72, 60), "VOLT", 40, (160, 220, 255))),
            ("headline", ElementRole.HEADLINE, _text_layer((70, 160), "Flash Sale", 104,
                                                           (255, 255, 255))),
            ("subheadline", ElementRole.SUBHEADLINE,
             _text_layer((74, 290), "Flagship phones at half price", 38, (200, 210, 255),
                         bold=False)),
            ("cta", ElementRole.CTA, _pill((74, 380), "SHOP NOW", 32, (255, 214, 64),
                                           (24, 24, 40))),
            ("hero", ElementRole.HERO_IMAGE, hero),
            ("badge", ElementRole.BADGE, badge),
        ],
    )


def travel_master() -> Master:
    bg = _gradient((120, 214, 230), (255, 220, 170))
    hero, d = _layer()
    d.ellipse((870, 90, 1050, 270), fill=(255, 196, 70))  # sun
    d.polygon([(620, 560), (820, 250), (1020, 560)], fill=(38, 120, 120))  # mountains
    d.polygon([(860, 560), (1030, 300), (1190, 560)], fill=(24, 92, 100))
    d.polygon([(790, 300), (820, 250), (850, 300)], fill=(255, 255, 255))
    d.rectangle((600, 540, 1190, 572), fill=(24, 92, 100))
    for x in (690, 730):  # palms
        d.line([(x, 560), (x + 10, 430)], fill=(80, 60, 40), width=10)
        for ang in (-150, -110, -70, -30):
            r = math.radians(ang)
            d.line([(x + 10, 430), (x + 10 + 70 * math.cos(r), 430 + 40 * math.sin(r))],
                   fill=(30, 140, 80), width=12)
    return Master(
        "travel", "Travel agency",
        [
            ("background", ElementRole.BACKGROUND, bg),
            ("logo", ElementRole.LOGO, _pill((70, 50), "SkyTrip", 28, (255, 255, 255),
                                             (16, 96, 120), pad=(24, 12))),
            ("headline", ElementRole.HEADLINE, _text_layer((70, 160), "Summer in Bali", 84,
                                                           (12, 60, 80))),
            ("subheadline", ElementRole.SUBHEADLINE,
             _text_layer((74, 272), "Return flights from $299", 38, (30, 90, 110),
                         bold=False)),
            ("cta", ElementRole.CTA, _pill((74, 366), "BOOK NOW", 32, (12, 60, 80),
                                           (255, 255, 255))),
            ("hero", ElementRole.HERO_IMAGE, hero),
        ],
    )


# ----------------------------------------------------------------------- boards
BOARD_BG = (244, 245, 247)
INK = (28, 31, 36)
MUTED = (104, 112, 124)


@dataclass
class Tile:
    image: Image.Image
    label: str
    sub: str = ""


def _tile_card(tile: Tile, scale: float) -> Image.Image:
    img = tile.image.convert("RGB")
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.Resampling.LANCZOS)
    pad, label_h = 0, 44
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(probe.textlength(tile.label, font=font(15)),
                 probe.textlength(tile.sub, font=font(13, bold=False)))
    card = Image.new("RGB", (max(img.width, int(text_w) + 4), img.height + label_h), BOARD_BG)
    shadow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle((2, 3, img.width + 2, img.height + 3), fill=(0, 0, 0, 40))
    card.paste(shadow.filter(ImageFilter.GaussianBlur(3)), (0, 0),
               shadow.filter(ImageFilter.GaussianBlur(3)))
    card.paste(img, (pad, 0))
    d = ImageDraw.Draw(card)
    d.text((0, img.height + 8), tile.label, font=font(15), fill=INK)
    if tile.sub:
        d.text((0, img.height + 26), tile.sub, font=font(13, bold=False), fill=MUTED)
    return card


def board(title: str, subtitle: str, rows: list[tuple[list[Tile], float]],
          width: int = 1600) -> Image.Image:
    """Shelf-pack rows of tiles (each row has its own display scale)."""
    margin, gap = 40, 28
    cards_rows = []
    for tiles, scale in rows:
        cards = [_tile_card(t, scale) for t in tiles]
        lines, line, lw = [], [], 0
        for c in cards:
            if line and lw + gap + c.width > width - 2 * margin:
                lines.append(line)
                line, lw = [], 0
            lw += (gap if line else 0) + c.width
            line.append(c)
        if line:
            lines.append(line)
        cards_rows.append((lines, scale))
    height = margin + 70
    for lines, _ in cards_rows:
        for line in lines:
            height += max(c.height for c in line) + gap
        height += 10
    out = Image.new("RGB", (width, height + margin - gap), BOARD_BG)
    d = ImageDraw.Draw(out)
    d.text((margin, margin - 6), title, font=font(30), fill=INK)
    d.text((margin, margin + 34), subtitle, font=font(16, bold=False), fill=MUTED)
    y = margin + 70
    for lines, _ in cards_rows:
        for line in lines:
            x = margin
            for c in line:
                out.paste(c, (x, y))
                x += c.width + gap
            y += max(c.height for c in line) + gap
        y += 10
    return out


# ---------------------------------------------------------------------- render
IAB_ROW = ["iab-leaderboard", "iab-mobile-banner", "iab-medium-rectangle",
           "iab-half-page", "iab-wide-skyscraper", "iab-large-mobile-banner"]
SOCIAL_ROW = ["meta-feed-square", "meta-feed-portrait", "meta-story", "linkedin-single-image",
              "x-header", "pinterest-standard"]


def _layered_engine(master: Master) -> ReLayoutEngine:
    engine = ReLayoutEngine(use_ai=False)
    engine.elements = master.elements()
    engine.source_size = (W, H)
    return engine


def _render(service: RenderService, src: Path, engine: ReLayoutEngine | None,
            preset_ids: list[str], auto_layers: bool = False, mode: str = "phase21"):
    presets = [get_preset(p) if not p.startswith("custom:") else
               custom_preset(*map(int, p[7:].split("x"))) for p in preset_ids]
    report = service.render(
        src, RenderRequest(targets=presets, export=ExportOptions(format="webp"),
                           auto_layers=auto_layers, mode=mode),
        engine=engine,
    )
    tiles = []
    for a in report.assets:
        assert a.ok and a.encoded is not None, a.error
        img = Image.open(io.BytesIO(a.encoded.data))
        cap_kb = a.encoded.max_kb
        cap = f" / {cap_kb} KB cap" if cap_kb and cap_kb < 1000 else ""
        n = len(a.warnings)
        warn = f" · {n} QA note{'s' if n > 1 else ''}" if n else ""
        if not a.preset.safe_zone.is_empty:
            img = _show_safe_zone(img, a.preset.safe_zone.to_pixels(*img.size))
            warn += " · UI zones shaded"
        tiles.append(Tile(img, f"{a.preset.name} {a.preset.width}×{a.preset.height}",
                          f"{a.encoded.size_kb:.0f} KB WebP{cap}{warn}"))
    return tiles, report


def _show_safe_zone(img: Image.Image, safe: tuple[int, int, int, int]) -> Image.Image:
    """Shade the platform UI overlay bands (display only; not part of the output)."""
    out = img.convert("RGBA")
    over = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    w, h = out.size
    x1, y1, x2, y2 = safe
    for band in ((0, 0, w, y1), (0, y2, w, h), (0, 0, x1, h), (x2, 0, w, h)):
        if band[2] > band[0] and band[3] > band[1]:
            d.rectangle(band, fill=(220, 40, 60, 60))
    d.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(220, 40, 60, 200), width=max(2, w // 300))
    out.alpha_composite(over)
    return out.convert("RGB")


def make_showcase(service: RenderService, master: Master, tmp: Path) -> Image.Image:
    src = tmp / f"{master.name}.png"
    master.flat().save(src)
    iab, _ = _render(service, src, _layered_engine(master), IAB_ROW)
    social, _ = _render(service, src, _layered_engine(master), SOCIAL_ROW)
    return board(
        f"{master.title}: 1 master → {len(iab) + len(social)} sizes",
        "Input: layered 1200×628 design (top). Every tile is the engine's real output with its "
        "real WebP file size. Display scale: IAB rows 1:1, social rows 1:3.",
        [([Tile(master.flat(), "MASTER 1200×628", "input")], 0.5), (iab, 1.0), (social, 1 / 3)],
    )


def make_auto_layers(service: RenderService, master: Master, tmp: Path) -> Image.Image:
    src = tmp / f"{master.name}_flat.png"
    master.flat().save(src)
    small = ["iab-leaderboard", "iab-medium-rectangle"]
    plain_s, _ = _render(service, src, None, small, auto_layers=False)
    auto_s, rep = _render(service, src, None, small, auto_layers=True)
    plain_story, _ = _render(service, src, None, ["meta-story"], auto_layers=False)
    auto_story, _ = _render(service, src, None, ["meta-story"], auto_layers=True)
    for t in plain_s + plain_story:
        t.label = "Plain resize · " + t.label
    for t in auto_s + auto_story:
        t.label = "Auto-layers · " + t.label
    detected = rep.source["layers"] - 1
    return board(
        "Flat PNG in, real relayout out (auto-layers)",
        f"The same flattened PNG, no layers. Plain resize can only shrink the whole picture; "
        f"auto-layers found {detected} elements and rearranged them. IAB 1:1, stories 1:3.",
        [([Tile(master.flat(), "FLAT PNG 1200×628", "input, no layers")], 0.4),
         (plain_s, 1.0), (auto_s, 1.0), (plain_story + auto_story, 1 / 3)],
    )


def make_engine_comparison(service: RenderService, masters: list[Master], tmp: Path):
    sizes = ["iab-leaderboard", "iab-medium-rectangle", "meta-feed-square", "meta-story"]
    rows = []
    for master in masters:
        src = tmp / f"{master.name}.png"
        master.flat().save(src)
        for engine_name in ("legacy", "stack"):
            Config.LAYOUT_ENGINE = engine_name
            try:
                tiles, _ = _render(service, src, _layered_engine(master), sizes)
            finally:
                Config.LAYOUT_ENGINE = "stack"
            label = "v1 engine" if engine_name == "legacy" else "v2 stack layout"
            for t in tiles:
                t.label = f"{label} · {t.label}"
            rows.append((tiles, 0.36))
    return board(
        "Same layered input, old engine vs new engine",
        "Rows alternate v1 (zone templates) and v2 (role-aware stack layout). "
        "v1 overlaps and crops elements; v2 never overlaps and keeps the hierarchy. Scale 0.36.",
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    service = RenderService()
    masters = [coffee_master(), tech_master(), travel_master()]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for m in masters:
            make_showcase(service, m, tmp).save(out / f"showcase_{m.name}.jpg", quality=86,
                                                 optimize=True, progressive=True)
            print("wrote", out / f"showcase_{m.name}.jpg")
        make_auto_layers(service, masters[0], tmp).save(out / "auto_layers.jpg", quality=86,
                                                       optimize=True, progressive=True)
        make_engine_comparison(service, masters[:2], tmp).save(
            out / "engine_comparison.jpg", quality=86, optimize=True, progressive=True)
    print("done")


if __name__ == "__main__":
    main()
