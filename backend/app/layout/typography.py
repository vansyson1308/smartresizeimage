"""Typography reflow helpers for adaptive layout."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class TextFitResult:
    """Result of fitting text in a constrained block."""

    font_size: int
    lines: list[str]
    bbox: tuple[int, int]
    overflow: bool = False


def load_font(font_family: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load font with safe fallback."""
    if font_family:
        try:
            return ImageFont.truetype(font_family, size=size)
        except Exception:  # noqa: BLE001
            pass

    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:  # noqa: BLE001
            continue

    return ImageFont.load_default()


def measure_text(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    """Measure single/multi-line text using PIL metrics."""
    img = Image.new("L", (4, 4), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    return max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])


def wrap_text_to_width(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap text to width without dropping characters."""
    if text == "":
        return [""]

    words = text.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if current == "" else f"{current} {word}"
        w, _ = measure_text(font, candidate)

        if w <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        # handle very long token by character split
        token = word
        while token:
            best = ""
            for i in range(1, len(token) + 1):
                part = token[:i]
                pw, _ = measure_text(font, part)
                if pw <= max_width:
                    best = part
                else:
                    break

            if best == "":
                best = token[0]

            if len(best) == len(token):
                current = best
                token = ""
            else:
                lines.append(best)
                token = token[len(best) :]

    if current:
        lines.append(current)

    return lines if lines else [""]


def fit_text_block(
    text: str,
    font_family: str | None,
    max_font: int,
    min_font: int,
    max_width: int,
    max_lines: int,
) -> TextFitResult:
    """Fit text by reducing font and wrapping lines."""
    min_font = max(1, min_font)
    max_font = max(min_font, max_font)
    max_width = max(1, max_width)
    max_lines = max(1, max_lines)

    best: TextFitResult | None = None

    for font_size in range(max_font, min_font - 1, -1):
        font = load_font(font_family, font_size)
        lines = wrap_text_to_width(text, font, max_width)

        if len(lines) <= max_lines:
            block_w = 0
            block_h = 0
            for line in lines:
                lw, lh = measure_text(font, line)
                block_w = max(block_w, lw)
                block_h += lh + 2
            block_h = max(1, block_h - 2)
            return TextFitResult(
                font_size=font_size,
                lines=lines,
                bbox=(block_w, block_h),
                overflow=False,
            )

        # keep track of smallest-overflow candidate
        truncated_lines = lines[:max_lines]
        if len(lines) > max_lines:
            # Append remainder to last line without dropping text
            remainder = " ".join(lines[max_lines - 1 :])
            truncated_lines[-1] = remainder

        block_w = 0
        block_h = 0
        for line in truncated_lines:
            lw, lh = measure_text(font, line)
            block_w = max(block_w, min(lw, max_width))
            block_h += lh + 2
        block_h = max(1, block_h - 2)
        best = TextFitResult(
            font_size=font_size,
            lines=truncated_lines,
            bbox=(min(block_w, max_width), block_h),
            overflow=True,
        )

    assert best is not None
    return best
