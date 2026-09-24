"""Input validation for AutoBanner."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .config import Config
from .constants import SUPPORTED_EXTENSIONS
from .enums import ElementRole
from .exceptions import ValidationError


def validate_dimensions(width: int, height: int) -> None:
    """Validate target dimensions.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.

    Raises:
        ValidationError: If dimensions are invalid.
    """
    if not isinstance(width, int | float) or not isinstance(height, int | float):
        raise ValidationError(
            f"Dimensions must be numbers, got {type(width).__name__} and {type(height).__name__}"
        )

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValidationError(f"Dimensions must be positive, got {width}x{height}")
    if width > Config.MAX_IMAGE_SIZE or height > Config.MAX_IMAGE_SIZE:
        raise ValidationError(
            f"Dimensions exceed maximum {Config.MAX_IMAGE_SIZE}, got {width}x{height}"
        )
    if width < Config.MIN_ELEMENT_SIZE or height < Config.MIN_ELEMENT_SIZE:
        raise ValidationError(
            f"Dimensions below minimum {Config.MIN_ELEMENT_SIZE}, got {width}x{height}"
        )


def validate_file_path(path: str) -> None:
    """Validate input file exists and has supported extension.

    Args:
        path: Path to the input file.

    Raises:
        ValidationError: If file doesn't exist or format is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"File not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported format '{p.suffix}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


# Magic-byte signatures for accepted formats. Extension checks alone let a
# renamed payload (e.g. an SVG or TIFF bomb) reach the decoders.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".psd": (b"8BPS",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".webp": (b"RIFF",),
}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sniff_extension(header: bytes) -> str | None:
    """Return the canonical extension for a file header, or ``None``."""
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"
    for ext, sigs in _SIGNATURES.items():
        if ext == ".webp":
            continue
        if any(header.startswith(sig) for sig in sigs):
            return ".jpg" if ext == ".jpeg" else ext
    return None


def validate_upload(path: str, max_bytes: int | None = None) -> None:
    """Validate an untrusted uploaded file before any decoder touches it.

    Checks existence, extension, byte size, that the file content matches its
    extension, and (for raster formats) the declared pixel dimensions without
    decoding pixel data.

    Raises:
        ValidationError: If any check fails.
    """
    validate_file_path(path)
    p = Path(path)
    limit = Config.MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    size = p.stat().st_size
    if size == 0:
        raise ValidationError("Uploaded file is empty")
    if size > limit:
        raise ValidationError(f"File is {size / 1048576:.1f} MB; limit is {limit / 1048576:.0f} MB")

    with p.open("rb") as fh:
        header = fh.read(32)
    detected = sniff_extension(header)
    declared = ".jpg" if p.suffix.lower() == ".jpeg" else p.suffix.lower()
    if detected is None or detected != declared:
        raise ValidationError(
            f"File content does not match its '{p.suffix}' extension"
            + (f" (looks like {detected})" if detected else "")
        )

    if declared == ".psd":
        # PSD header: signature(4) version(2) reserved(6) channels(2) height(4) width(4)
        height = int.from_bytes(header[14:18], "big")
        width = int.from_bytes(header[18:22], "big")
    else:
        try:
            with Image.open(p) as img:
                width, height = img.size
        except Exception as e:  # PIL raises many types for corrupt input
            raise ValidationError(f"Could not read image header: {e}") from e
    validate_source_dimensions(width, height)


def validate_source_dimensions(width: int, height: int) -> None:
    """Reject sources that are empty or large enough to exhaust memory."""
    if width <= 0 or height <= 0:
        raise ValidationError(f"Source has invalid dimensions {width}x{height}")
    if max(width, height) > Config.MAX_SOURCE_DIMENSION:
        raise ValidationError(
            f"Source {width}x{height} exceeds the {Config.MAX_SOURCE_DIMENSION}px limit"
        )
    if width * height > Config.MAX_SOURCE_PIXELS:
        raise ValidationError(
            f"Source has {width * height / 1e6:.0f} MP; limit is "
            f"{Config.MAX_SOURCE_PIXELS / 1e6:.0f} MP"
        )


def safe_filename(name: str, default: str = "output", max_length: int = 80) -> str:
    """Make ``name`` safe to use as a single path component (no traversal)."""
    cleaned = _SAFE_NAME_RE.sub("_", name or "").strip("._-")
    cleaned = cleaned[:max_length].rstrip("._-")
    return cleaned or default


_ANCHOR_ROLES = frozenset(role.value for role in ElementRole)


def validate_manual_anchors(
    anchors: object,
    source_size: tuple[int, int],
    max_anchors: int = 32,
) -> list[dict[str, int | str]]:
    """Validate and normalise user supplied anchor boxes.

    Boxes are clipped to the source canvas; boxes that end up empty are
    rejected so the redesign pipeline never receives degenerate geometry.

    Returns:
        A new list of normalised anchor dicts.

    Raises:
        ValidationError: On malformed input.
    """
    if not isinstance(anchors, list):
        raise ValidationError("anchors must be a JSON list")
    if len(anchors) > max_anchors:
        raise ValidationError(f"At most {max_anchors} anchors are allowed")
    sw, sh = source_size
    normalised: list[dict[str, int | str]] = []
    used_ids: set[str] = set()
    for idx, raw in enumerate(anchors):
        if not isinstance(raw, dict):
            raise ValidationError(f"anchor #{idx} must be an object")
        try:
            x = int(raw["x"])
            y = int(raw["y"])
            w = int(raw["width"])
            h = int(raw["height"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValidationError(
                f"anchor #{idx} needs integer x, y, width and height"
            ) from e
        role = str(raw.get("role", "hero_image"))
        if role not in _ANCHOR_ROLES:
            raise ValidationError(f"anchor #{idx} has unknown role '{role}'")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(sw, x + w), min(sh, y + h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValidationError(f"anchor #{idx} lies outside the {sw}x{sh} source canvas")
        anchor_id = safe_filename(str(raw.get("id", f"anchor_{idx}")), f"anchor_{idx}", 40)
        base_id, n = anchor_id, 2
        while anchor_id in used_ids:  # ids key layout lookups; keep them unique
            anchor_id, n = f"{base_id}_{n}", n + 1
        used_ids.add(anchor_id)
        normalised.append(
            {
                "id": anchor_id,
                "role": role,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )
    return normalised
