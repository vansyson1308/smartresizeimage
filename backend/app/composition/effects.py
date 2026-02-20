"""Drop-shadow effect helpers (PSD-like MVP subset)."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageFilter


@dataclass(frozen=True)
class DropShadowParams:
    """Normalized drop-shadow parameters."""

    offset_x: float
    offset_y: float
    blur_radius: float
    spread: float
    color: tuple[int, int, int, int]
    opacity: float

    def scaled(self, factor: float) -> DropShadowParams:
        return DropShadowParams(
            offset_x=self.offset_x * factor,
            offset_y=self.offset_y * factor,
            blur_radius=max(0.0, self.blur_radius * factor),
            spread=self.spread,
            color=self.color,
            opacity=self.opacity,
        )


def parse_drop_shadow_effect(effects: dict) -> DropShadowParams | None:
    """Extract drop-shadow effect parameters from generic effect metadata.

    Supports synthetic tests and a PSD-like subset keys.
    Returns None when effect is absent/disabled/invalid.
    """
    if not effects:
        return None

    candidate = None
    for key in ("drop_shadow", "drop shadow", "Drop Shadow", "dropshadow"):
        if key in effects:
            candidate = effects.get(key)
            break
    if not isinstance(candidate, dict):
        return None

    if candidate.get("enabled") is False:
        return None

    opacity = _norm_opacity(candidate.get("opacity", 1.0))
    blur = float(
        candidate.get("blur_radius", candidate.get("blur", candidate.get("radius", 4.0)))
    )
    spread = float(candidate.get("spread", 0.0))
    offset_x = float(
        candidate.get("offset_x", candidate.get("distance_x", candidate.get("x", 0.0)))
    )
    offset_y = float(
        candidate.get("offset_y", candidate.get("distance_y", candidate.get("y", 2.0)))
    )
    color = _norm_color(candidate.get("color", (0, 0, 0, 255)))

    return DropShadowParams(
        offset_x=offset_x,
        offset_y=offset_y,
        blur_radius=max(0.0, blur),
        spread=min(1.0, max(0.0, spread)),
        color=color,
        opacity=min(1.0, max(0.0, opacity)),
    )


def render_drop_shadow(
    element_image: Image.Image,
    params: DropShadowParams,
) -> tuple[Image.Image, tuple[int, int]]:
    """Render shadow image from element alpha.

    Returns shadow image and (dx, dy) offset relative to element top-left.
    """
    src = element_image.convert("RGBA")
    alpha = src.split()[3]

    if params.spread > 0.0:
        spread_px = max(1, int(params.spread * 8))
        alpha = alpha.filter(ImageFilter.MaxFilter(size=spread_px * 2 + 1))

    if params.blur_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))

    base_pad = int(params.blur_radius * 2 + 2)
    left = base_pad + max(0, -int(round(params.offset_x)))
    top = base_pad + max(0, -int(round(params.offset_y)))
    right = base_pad + max(0, int(round(params.offset_x)))
    bottom = base_pad + max(0, int(round(params.offset_y)))

    w, h = src.size
    canvas_w = w + left + right
    canvas_h = h + top + bottom

    shadow_alpha = Image.new("L", (canvas_w, canvas_h), 0)
    paste_x = left + int(round(params.offset_x))
    paste_y = top + int(round(params.offset_y))
    shadow_alpha.paste(alpha, (paste_x, paste_y))

    color_a = int(params.color[3] * params.opacity)
    color_img = Image.new(
        "RGBA",
        (canvas_w, canvas_h),
        (params.color[0], params.color[1], params.color[2], color_a),
    )

    color_img.putalpha(
        shadow_alpha.point(lambda p: int(p * (color_a / 255.0)))
    )

    # Offset to place rendered shadow relative to element position.
    return color_img, (-left, -top)


def _norm_opacity(v: object) -> float:
    try:
        value = float(v)
    except (TypeError, ValueError):
        return 1.0
    if value > 1.0:
        value = value / 100.0
    return min(1.0, max(0.0, value))


def _norm_color(v: object) -> tuple[int, int, int, int]:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        r = int(v[0])
        g = int(v[1])
        b = int(v[2])
        a = int(v[3]) if len(v) >= 4 else 255
        return (_clamp255(r), _clamp255(g), _clamp255(b), _clamp255(a))
    if isinstance(v, dict):
        # TODO: map full psd-tools color model for layer effects if needed.
        return (
            _clamp255(int(v.get("r", 0))),
            _clamp255(int(v.get("g", 0))),
            _clamp255(int(v.get("b", 0))),
            _clamp255(int(v.get("a", 255))),
        )
    return (0, 0, 0, 255)


def _clamp255(x: int) -> int:
    return min(255, max(0, x))
