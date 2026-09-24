"""Image export with format selection and file-weight budgets.

Ad networks reject creatives above a hard file-size cap (150 KB on the Google
Display Network, for example). :func:`encode_image` searches the encoder
quality space so the output fits the budget with the highest quality possible,
and reports honestly when the budget cannot be met.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from .exceptions import ValidationError

FORMATS: dict[str, tuple[str, str, str]] = {
    # key: (PIL format, file extension, MIME type)
    "png": ("PNG", "png", "image/png"),
    "jpeg": ("JPEG", "jpg", "image/jpeg"),
    "jpg": ("JPEG", "jpg", "image/jpeg"),
    "webp": ("WEBP", "webp", "image/webp"),
}

_MIN_QUALITY = 35
_MAX_QUALITY = 95
_PNG_PALETTE_STEPS = (256, 128, 64, 32)


@dataclass(frozen=True)
class ExportOptions:
    """How a rendered image should be encoded."""

    format: str = "png"
    quality: int = 90
    max_kb: int | None = None

    def __post_init__(self) -> None:
        fmt = self.format.lower()
        if fmt not in FORMATS:
            raise ValidationError(
                f"Unsupported export format '{self.format}'. Use one of: png, jpeg, webp"
            )
        object.__setattr__(self, "format", "jpeg" if fmt == "jpg" else fmt)
        if not 1 <= int(self.quality) <= 100:
            raise ValidationError("quality must be between 1 and 100")
        if self.max_kb is not None and int(self.max_kb) <= 0:
            raise ValidationError("max_kb must be a positive integer")

    @property
    def extension(self) -> str:
        return FORMATS[self.format][1]

    @property
    def mime_type(self) -> str:
        return FORMATS[self.format][2]


@dataclass(frozen=True)
class EncodedImage:
    """Encoded bytes plus what it took to produce them."""

    data: bytes
    format: str
    extension: str
    mime_type: str
    quality: int | None
    within_budget: bool
    max_kb: int | None
    palette_colors: int | None = None

    @property
    def size_kb(self) -> float:
        return round(len(self.data) / 1024.0, 1)

    def describe(self) -> dict[str, object]:
        return {
            "format": self.format,
            "bytes": len(self.data),
            "size_kb": self.size_kb,
            "quality": self.quality,
            "max_kb": self.max_kb,
            "within_budget": self.within_budget,
            "palette_colors": self.palette_colors,
        }


def _encode(image: Image.Image, fmt: str, quality: int | None) -> bytes:
    buf = io.BytesIO()
    pil_format = FORMATS[fmt][0]
    if pil_format == "JPEG":
        image.convert("RGB").save(
            buf, "JPEG", quality=quality, optimize=True, progressive=True, subsampling="4:2:0"
        )
    elif pil_format == "WEBP":
        image.save(buf, "WEBP", quality=quality, method=6)
    else:
        image.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _fits(data: bytes, max_kb: int | None) -> bool:
    return max_kb is None or len(data) <= max_kb * 1024


def encode_image(image: Image.Image, options: ExportOptions | None = None) -> EncodedImage:
    """Encode ``image`` according to ``options``.

    * Lossy formats: if a budget is set, binary-search the highest quality
      (between 35 and the requested quality) that fits.
    * PNG: try lossless first; if over budget, fall back to adaptive palette
      quantization with progressively fewer colors.

    The smallest attempt is returned with ``within_budget=False`` when no
    setting satisfies the budget, so callers can surface a warning instead of
    silently shipping an asset the ad network will reject.
    """
    opts = options or ExportOptions()
    fmt = opts.format
    ext = opts.extension
    mime = opts.mime_type
    src = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image

    if fmt == "png":
        data = _encode(src, "png", None)
        if _fits(data, opts.max_kb):
            return EncodedImage(data, fmt, ext, mime, None, True, opts.max_kb)
        best = data
        for colors in _PNG_PALETTE_STEPS:
            quantized = src.convert("RGB").quantize(
                colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG
            )
            candidate = _encode(quantized, "png", None)
            if len(candidate) < len(best):
                best = candidate
            if _fits(candidate, opts.max_kb):
                return EncodedImage(candidate, fmt, ext, mime, None, True, opts.max_kb, colors)
        return EncodedImage(best, fmt, ext, mime, None, False, opts.max_kb)

    quality = int(opts.quality)
    data = _encode(src, fmt, quality)
    if _fits(data, opts.max_kb):
        return EncodedImage(data, fmt, ext, mime, quality, True, opts.max_kb)

    lo, hi = _MIN_QUALITY, min(quality - 1, _MAX_QUALITY)
    best_fit: tuple[bytes, int] | None = None
    smallest = (data, quality)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _encode(src, fmt, mid)
        if len(candidate) < len(smallest[0]):
            smallest = (candidate, mid)
        if _fits(candidate, opts.max_kb):
            best_fit = (candidate, mid)
            lo = mid + 1
        else:
            hi = mid - 1

    if best_fit is not None:
        return EncodedImage(best_fit[0], fmt, ext, mime, best_fit[1], True, opts.max_kb)
    return EncodedImage(smallest[0], fmt, ext, mime, smallest[1], False, opts.max_kb)
