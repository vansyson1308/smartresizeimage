"""Render a document at its master canvas size, exactly as authored.

Unlike variant generation, no re-layout happens: every element is drawn at its
own geometry, text at its own font size (wrapped to its box). This is the
preview shown in the interpretation/correction surface and the reference for
"did the import understand the design".
"""

from __future__ import annotations

from PIL import Image

from ..composition.color import BlendMode, composite_pil_over, parse_blend_mode
from ..composition.resize import high_quality_resize
from ..config import Config
from .assets import AssetStore
from .document import DesignDocument, Element
from .fonts import FontRegistry, default_registry
from .text_render import fit_text, render_text


def _hex_rgba(color: str) -> tuple[int, int, int, int]:
    c = (color or "").lstrip("#")
    if len(c) == 8:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(c[6:8], 16)
    if len(c) == 6:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255
    return 0, 0, 0, 0


def render_element(
    element: Element,
    assets: AssetStore,
    size: tuple[int, int],
    registry: FontRegistry | None = None,
    text_override: str | None = None,
) -> Image.Image | None:
    """Rasterize one element into an RGBA image of ``size``."""
    reg = registry or default_registry()
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    if element.kind == "image" and element.asset is not None:
        img = assets.get(element.asset)
        if img.size != (w, h):
            img = high_quality_resize(img, (w, h))
        return img
    if element.kind == "shape":
        fill = (element.shape or {}).get("fill", "#00000000")
        return Image.new("RGBA", (w, h), _hex_rgba(fill))
    if element.kind == "text" and element.text is not None:
        content = element.text
        if text_override is not None:
            from .document import TextContent

            content = TextContent(
                runs=[type(r)(text=r.text, style=r.style) for r in element.text.runs],
                locale=element.text.locale,
                max_lines=element.text.max_lines,
                protected=element.text.protected,
            )
            content.replace_text(text_override)
        px = max(4, int(round(content.primary_style.font_size)))
        layout = fit_text(content, w, None, min_px=px, max_px=px, registry=reg)
        return render_text(content, layout, w, max(h, layout.height), registry=reg)
    return None


def render_master(
    doc: DesignDocument,
    assets: AssetStore,
    registry: FontRegistry | None = None,
    *,
    scale: float = 1.0,
) -> Image.Image:
    """Composite the document at (canvas * scale)."""
    reg = registry or default_registry()
    cw = max(1, int(round(doc.canvas_width * scale)))
    ch = max(1, int(round(doc.canvas_height * scale)))
    canvas = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
    for element in sorted(doc.elements, key=lambda e: e.z_index):
        if not element.visible or element.kind == "group":
            continue
        g = element.geometry
        x = int(round(g.x * scale))
        y = int(round(g.y * scale))
        w = max(1, int(round(g.width * scale)))
        h = max(1, int(round(g.height * scale)))
        if element.kind == "text" and element.text is not None and scale != 1.0:
            # scale the font with the canvas for previews
            style = element.text.primary_style
            original = style.font_size
            style.font_size = max(1.0, original * scale)
            try:
                layer = render_element(element, assets, (w, h), reg)
            finally:
                style.font_size = original
        else:
            layer = render_element(element, assets, (w, h), reg)
        if layer is None:
            continue
        if element.opacity < 1.0:
            alpha = layer.split()[3].point(lambda p, o=element.opacity: int(p * o))
            layer.putalpha(alpha)
        mode, _supported = parse_blend_mode(element.blend_mode)
        canvas = composite_pil_over(
            canvas,
            layer,
            (x, y),
            use_linear=Config.USE_LINEAR_COMPOSITING,
            blend_mode=mode if isinstance(mode, BlendMode) else BlendMode.NORMAL,
        )
    return canvas.convert("RGB")
