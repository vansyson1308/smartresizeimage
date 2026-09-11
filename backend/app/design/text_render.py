"""Native text layout and rasterization.

Text is measured and drawn with real fonts (FreeType via Pillow; complex
scripts and ligatures use libraqm when Pillow was built with it). Line breaks
respect explicit newlines and wrap at word boundaries; a word longer than the
box is broken by characters rather than dropped. Overflow is reported, never
hidden.

A text block is a sequence of *runs*, each with its own style (family, weight,
italic, colour, relative size, letter spacing, case). Fitting scales the whole
block: the primary (first) run's size is the block size and every other run keeps
its size ratio to it, so a "50% **OFF**" with a bigger bold "OFF" stays that way
at every target size. Runs are shaped to a shared baseline per line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont, features

from .document import TextContent, TextStyle
from .fonts import FontRegistry, ResolvedFont, default_registry

HAS_RAQM = bool(features.check("raqm"))
_TOKEN = re.compile(r"\n| +|[^ \n]+")


@dataclass
class Segment:
    """A piece of one line drawn with a single run's style."""

    text: str
    run: int
    width: int


@dataclass
class TextLayout:
    """Result of fitting a text block into a box."""

    font_px: int
    lines: list[str]
    line_height_px: int
    width: int
    height: int
    overflow: bool
    resolved_font: ResolvedFont  # the primary run's face
    measured_widths: list[int] = field(default_factory=list)
    run_fonts: list[ResolvedFont] = field(default_factory=list)  # one per run
    run_px: list[int] = field(default_factory=list)  # rendered size per run
    segments: list[list[Segment]] = field(default_factory=list)  # per line

    @property
    def substituted_font(self) -> bool:
        return any(f.substituted for f in self.run_fonts) or self.resolved_font.substituted


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
    """Wrap single-style ``text`` into lines no wider than ``max_width``."""
    content = TextContent()
    content.replace_text(text)
    shaped = _shape(content, [font], [letter_spacing], [False], max_width)
    return [ln.text for ln in shaped]


@dataclass
class _Line:
    segments: list[Segment]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    @property
    def width(self) -> int:
        return sum(s.width for s in self.segments)


def _shape(
    content: TextContent,
    fonts: list[ImageFont.ImageFont],
    spacings: list[float],
    uppercase: list[bool],
    max_width: int,
) -> list[_Line]:
    """Break the runs of ``content`` into lines no wider than ``max_width``.

    Words never straddle a run boundary (a run boundary inside a word yields two
    adjacent segments that are kept together); a word wider than the box is broken
    by characters inside its own run. Nothing is dropped.
    """
    lines: list[_Line] = []
    current: list[Segment] = []
    width = 0

    def trimmed(segs: list[Segment]) -> list[Segment]:
        # trailing spaces end no line: they neither show nor count towards the width
        segs = list(segs)
        while segs and not segs[-1].text.strip():
            segs.pop()
        if segs and segs[-1].text.endswith(" "):
            last = segs[-1]
            text = last.text.rstrip(" ")
            segs[-1] = Segment(text, last.run, _measure(fonts[last.run], text, spacings[last.run]))
        return segs

    def flush() -> None:
        nonlocal current, width
        lines.append(_Line(trimmed(current)))
        current, width = [], 0

    def push(text: str, run: int) -> None:
        nonlocal width
        w = _measure(fonts[run], text, spacings[run])
        if current and current[-1].run == run:
            current[-1] = Segment(current[-1].text + text, run, current[-1].width + w)
        else:
            current.append(Segment(text, run, w))
        width += w

    # Tokenise every run; a word that continues into the next run is one unit.
    units: list[list[tuple[str, int]]] = []  # each unit: [(text, run), ...]
    pending_word: list[tuple[str, int]] = []
    for idx, run in enumerate(content.runs):
        text = run.text.upper() if uppercase[idx] else run.text
        for tok in _TOKEN.findall(text):
            if tok == "\n" or tok.startswith(" "):
                if pending_word:
                    units.append(pending_word)
                    pending_word = []
                units.append([(tok, idx)])
            else:
                pending_word.append((tok, idx))
    if pending_word:
        units.append(pending_word)

    for unit in units:
        tok, run = unit[0]
        if tok == "\n":
            flush()
            continue
        if tok.startswith(" "):
            if current:  # leading spaces on a wrapped line are dropped, never mid-word
                push(tok, run)
            continue
        unit_width = sum(_measure(fonts[r], t, spacings[r]) for t, r in unit)
        if current and width + unit_width > max_width:
            flush()
        if unit_width <= max_width or not unit:
            for t, r in unit:
                push(t, r)
            continue
        # a word wider than the box: break it by characters, piece by piece
        for t, r in unit:
            token = t
            while token:
                cut = len(token)
                while cut > 1 and width + _measure(fonts[r], token[:cut], spacings[r]) > max_width:
                    cut -= 1
                if cut < len(token) and current and width > 0 and _measure(
                    fonts[r], token[:cut], spacings[r]
                ) + width > max_width:
                    flush()
                    continue
                push(token[:cut], r)
                token = token[cut:]
                if token:
                    flush()
    lines.append(_Line(trimmed(current)))
    return lines


def _run_styles(content: TextContent) -> list[TextStyle]:
    return [r.style for r in content.runs] if content.runs else [TextStyle()]


def _run_texts(content: TextContent) -> list[str]:
    return [r.text for r in content.runs] if content.runs else [""]


def _resolve_runs(content: TextContent, reg: FontRegistry) -> list[ResolvedFont]:
    out = []
    for text, style in zip(_run_texts(content), _run_styles(content), strict=True):
        sample = text.upper() if style.uppercase else text
        out.append(reg.resolve_for_text(style.font_family, sample, style.weight, style.italic))
    return out


def _run_sizes(content: TextContent, px: int) -> list[int]:
    styles = _run_styles(content)
    base = max(1.0, float(styles[0].font_size))
    return [max(1, int(round(px * float(st.font_size) / base))) for st in styles]


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
    """Find the largest primary size in [min_px, max_px] whose wrapped block fits.

    If even ``min_px`` overflows, the layout at ``min_px`` is returned with
    ``overflow=True`` so the caller can surface the conflict.
    """
    reg = registry or default_registry()
    styles = _run_styles(content)
    primary = styles[0]
    resolved = _resolve_runs(content, reg)
    max_lines = max_lines or content.max_lines
    max_width = max(1, int(max_width))
    lo, hi = max(1, int(min_px)), max(int(min_px), int(max_px))
    line_factor = max(st.line_height for st in styles)
    spacings = [st.letter_spacing for st in styles]
    upper = [st.uppercase for st in styles]

    best: TextLayout | None = None
    for px in range(hi, lo - 1, -1):
        sizes = _run_sizes(content, px)
        fonts = [reg.load(rf, size) for rf, size in zip(resolved, sizes, strict=True)]
        shaped = _shape(content, fonts, spacings, upper, max_width)
        line_h = int(round(px * line_factor))
        height = line_h * len(shaped)
        widths = [ln.width for ln in shaped]
        width = max(widths) if widths else 0
        fits_lines = max_lines is None or len(shaped) <= max_lines
        fits_height = max_height is None or height <= max_height
        fits_width = width <= max_width
        layout = TextLayout(
            px, [ln.text for ln in shaped], line_h, width, height, False, resolved[0], widths,
            run_fonts=resolved, run_px=sizes, segments=[ln.segments for ln in shaped],
        )
        if fits_lines and fits_height and fits_width:
            return layout
        best = layout
    assert best is not None
    best.overflow = True
    assert primary is not None
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
    styles = _run_styles(content)
    primary = styles[0]
    run_fonts = layout.run_fonts or [layout.resolved_font]
    run_px = layout.run_px or [layout.font_px]
    fonts = [reg.load(rf, px) for rf, px in zip(run_fonts, run_px, strict=True)]
    ascents = [font.getmetrics()[0] if hasattr(font, "getmetrics") else px
               for font, px in zip(fonts, run_px, strict=True)]
    height = box_height if box_height else max(1, layout.height)
    img = Image.new("RGBA", (max(1, int(box_width)), max(1, int(height))), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 0
    # Vertical centring inside a taller box keeps the same bbox as the plan.
    if box_height and layout.height < box_height:
        y = (box_height - layout.height) // 2
    segments = layout.segments or [[Segment(ln, 0, w)] for ln, w in
                                   zip(layout.lines, layout.measured_widths, strict=False)]
    for segs, width in zip(segments, layout.measured_widths, strict=False):
        if primary.align == "center":
            x = (img.width - width) // 2
        elif primary.align == "right":
            x = img.width - width
        else:
            x = 0
        # Every run of the line sits on one baseline: the tallest ascender decides it.
        baseline = y + max((ascents[s.run] for s in segs), default=ascents[0])
        cx = float(x)
        for seg in segs:
            style = styles[seg.run] if seg.run < len(styles) else primary
            font = fonts[seg.run] if seg.run < len(fonts) else fonts[0]
            color = _hex_to_rgba(style.color)
            if style.letter_spacing:
                for ch in seg.text:
                    draw.text((cx, baseline), ch, font=font, fill=color, anchor="ls")
                    cx += _measure(font, ch) + style.letter_spacing
            else:
                draw.text((cx, baseline), seg.text, font=font, fill=color, anchor="ls")
                cx += _measure(font, seg.text)
        y += layout.line_height_px
    return img


def measure_text_block(content: TextContent, font_px: int, max_width: int) -> tuple[int, int]:
    """Convenience: (width, height) of the block at a fixed size."""
    layout = fit_text(content, max_width, None, min_px=font_px, max_px=font_px)
    return layout.width, layout.height
