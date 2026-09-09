"""Native text layout and rasterization.

Text is measured and drawn with real fonts (FreeType via Pillow; complex
scripts and ligatures use libraqm when Pillow was built with it). Line breaks
respect explicit newlines and wrap at word boundaries; a word longer than the
box is broken by characters rather than dropped. Overflow is reported, never
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont, features

from .document import TextContent, TextStyle
from .fonts import FontRegistry, ResolvedFont, default_registry

HAS_RAQM = bool(features.check("raqm"))


@dataclass
class TextLayout:
    """Result of fitting a text block into a box."""

    font_px: int
    lines: list[str]
    line_height_px: int
    width: int
    height: int
    overflow: bool
    resolved_font: ResolvedFont
    measured_widths: list[int] = field(default_factory=list)

    @property
    def substituted_font(self) -> bool:
        return self.resolved_font.substituted


def _hex_to_rgba(color: str) -> tuple[int, int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 6:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255
    if len(c) == 8:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(c[6:8], 16)
    return 0, 0, 0, 255


def _measure(font: ImageFont.ImageFont, text: str, letter_spacing: float = 0.0) -> int:
    if not text:
        return 0
    try:
        width = font.getlength(text)
    except AttributeError:  # very old Pillow fallback
        width = font.getbbox(text)[2]
    if letter_spacing:
        width += letter_spacing * max(0, len(text) - 1)
    return int(round(width))


def wrap_lines(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    letter_spacing: float = 0.0,
) -> list[str]:
    """Wrap ``text`` into lines no wider than ``max_width`` without dropping characters."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if _measure(font, candidate, letter_spacing) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            # word alone is too wide: break by characters
            token = word
            while token:
                cut = len(token)
                while cut > 1 and _measure(font, token[:cut], letter_spacing) > max_width:
                    cut -= 1
                if cut == len(token):
                    current = token
                    token = ""
                else:
                    lines.append(token[:cut])
                    token = token[cut:]
        lines.append(current)
    return lines if lines else [""]


def fit_text(
    content: TextContent,
    max_width: int,
    max_height: int | None,
    *,
    min_px: int,
    max_px: int,
    registry: FontRegistry | None = None,
    max_lines: int | None = None,
) -> TextLayout:
    """Find the largest font size in [min_px, max_px] whose wrapped block fits.

    If even ``min_px`` overflows, the layout at ``min_px`` is returned with
    ``overflow=True`` so the caller can surface the conflict.
    """
    reg = registry or default_registry()
    style = content.primary_style
    text = content.plain.upper() if style.uppercase else content.plain
    resolved = reg.resolve_for_text(style.font_family, text, style.weight, style.italic)
    max_lines = max_lines or content.max_lines
    max_width = max(1, int(max_width))
    lo, hi = max(1, int(min_px)), max(int(min_px), int(max_px))

    best: TextLayout | None = None
    for px in range(hi, lo - 1, -1):
        font = reg.load(resolved, px)
        lines = wrap_lines(text, font, max_width, style.letter_spacing)
        line_h = int(round(px * style.line_height))
        height = line_h * len(lines)
        widths = [_measure(font, ln, style.letter_spacing) for ln in lines]
        width = max(widths) if widths else 0
        fits_lines = max_lines is None or len(lines) <= max_lines
        fits_height = max_height is None or height <= max_height
        fits_width = width <= max_width
        layout = TextLayout(px, lines, line_h, width, height, False, resolved, widths)
        if fits_lines and fits_height and fits_width:
            return layout
        best = layout
    assert best is not None
    best.overflow = True
    return best


def render_text(
    content: TextContent,
    layout: TextLayout,
    box_width: int,
    box_height: int | None = None,
    *,
    registry: FontRegistry | None = None,
) -> Image.Image:
    """Rasterize a fitted layout into an RGBA image of ``box_width`` x height."""
    reg = registry or default_registry()
    style: TextStyle = content.primary_style
    font = reg.load(layout.resolved_font, layout.font_px)
    height = box_height if box_height else max(1, layout.height)
    img = Image.new("RGBA", (max(1, int(box_width)), max(1, int(height))), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = _hex_to_rgba(style.color)
    y = 0
    # Vertical centring inside a taller box keeps the same bbox as the plan.
    if box_height and layout.height < box_height:
        y = (box_height - layout.height) // 2
    for line, width in zip(layout.lines, layout.measured_widths, strict=False):
        if style.align == "center":
            x = (img.width - width) // 2
        elif style.align == "right":
            x = img.width - width
        else:
            x = 0
        if style.letter_spacing:
            cx = float(x)
            for ch in line:
                draw.text((cx, y), ch, font=font, fill=color)
                cx += _measure(font, ch) + style.letter_spacing
        else:
            draw.text((x, y), line, font=font, fill=color)
        y += layout.line_height_px
    return img


def measure_text_block(content: TextContent, font_px: int, max_width: int) -> tuple[int, int]:
    """Convenience: (width, height) of the block at a fixed size."""
    layout = fit_text(content, max_width, None, min_px=font_px, max_px=font_px)
    return layout.width, layout.height
