"""Auto-layering: split a flattened banner (PNG/JPG/WEBP) into pseudo-layers.

A flat export has no layer information, so extreme aspect changes (e.g. a
1200x628 banner to a 728x90 leaderboard) can only shrink the whole picture.
This module estimates the background, detects the foreground blocks (headline,
sub-copy, CTA button, logo, hero), classifies them by geometry and returns them
as :class:`DesignElement` objects so the layered layout engine can rearrange
them - the same way it handles a PSD.

The detector is deliberately conservative: when the background cannot be
modelled or the result looks implausible it returns ``None`` and the caller
falls back to the content-aware fit of the whole image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from ..enums import ElementRole
from ..models import BoundingBox, DesignElement

logger = logging.getLogger("autobanner.parser.auto_layers")

_WORK_MAX_SIDE = 640
_BORDER_FRAC = 0.04
_RESIDUAL_THRESHOLD = 28.0
_MIN_COMPONENT_FRAC = 0.0012
_MAX_FOREGROUND_FRAC = 0.62
_MAX_ELEMENTS = 10
_MIN_FILL = 0.08
_MAX_COMPONENTS = 300  # bound merge work on text-heavy / noisy inputs


@dataclass
class _Block:
    x1: int
    y1: int
    x2: int
    y2: int
    fill: float  # foreground pixels / bbox area (on the un-merged mask)
    area: int  # foreground pixel count
    labels: frozenset[int] = frozenset()  # connected-component ids it is made of

    def merged(self, other: _Block) -> _Block:
        total = self.area + other.area
        return _Block(
            min(self.x1, other.x1), min(self.y1, other.y1),
            max(self.x2, other.x2), max(self.y2, other.y2),
            fill=(self.fill * self.area + other.fill * other.area) / max(1, total),
            area=total,
            labels=self.labels | other.labels,
        )

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def aspect(self) -> float:
        return self.w / max(1, self.h)


def _design_matrix(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    # Quadratic surface: captures flat fills, linear and radial-ish gradients.
    return np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys], axis=1)


def _fit_background(rgb: np.ndarray, sample_mask: np.ndarray) -> np.ndarray | None:
    h, w, _ = rgb.shape
    ys, xs = np.nonzero(sample_mask)
    if xs.size < 50:
        return None
    step = max(1, xs.size // 20000)
    xs, ys = xs[::step], ys[::step]
    a = _design_matrix(xs / w, ys / h)
    coef, *_ = np.linalg.lstsq(a, rgb[ys, xs].astype(np.float64), rcond=None)
    gy, gx = np.mgrid[0:h, 0:w]
    full = _design_matrix(gx.ravel() / w, gy.ravel() / h) @ coef
    return full.reshape(h, w, 3)


def _foreground_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Robustly fit the background and threshold the colour residual."""
    h, w, _ = rgb.shape
    border = np.zeros((h, w), dtype=bool)
    bw, bh = max(2, int(w * _BORDER_FRAC)), max(2, int(h * _BORDER_FRAC))
    border[:bh, :] = border[-bh:, :] = True
    border[:, :bw] = border[:, -bw:] = True

    sample = border
    residual = None
    for _ in range(3):
        bg = _fit_background(rgb, sample)
        if bg is None:
            return None
        residual = np.linalg.norm(rgb.astype(np.float64) - bg, axis=2)
        sample = residual < _RESIDUAL_THRESHOLD * 0.6
    assert residual is not None

    # The border itself must be explained by the model, or this is not a
    # "design on a background" image (e.g. a full-bleed photo).
    if float(np.mean(residual[border] < _RESIDUAL_THRESHOLD)) < 0.8:
        return None

    # No morphological opening here: at working resolution small type has
    # 1px strokes. Speckle and thin pattern lines are rejected later by
    # component area and fill ratio instead.
    return residual > _RESIDUAL_THRESHOLD


def _components(
    mask: np.ndarray, close_w: int, close_h: int
) -> tuple[list[_Block], np.ndarray]:
    closed = ndimage.binary_closing(mask, structure=np.ones((close_h, close_w)))
    closed = ndimage.binary_fill_holes(closed)
    labels, count = ndimage.label(closed)
    blocks: list[_Block] = []
    total = mask.size
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        region = labels[sl] == idx
        area = int(region.sum())
        if area < _MIN_COMPONENT_FRAC * total:
            continue
        raw = mask[sl] & region
        bbox_area = max(1, region.shape[0] * region.shape[1])
        fill = float(raw.sum()) / bbox_area
        if fill < _MIN_FILL and bbox_area > 0.02 * total:
            continue  # large but sparse: background line work / texture, not content
        blocks.append(
            _Block(
                sl[1].start, sl[0].start, sl[1].stop, sl[0].stop,
                fill=fill,
                area=area,
                labels=frozenset({idx}),
            )
        )
    blocks.sort(key=lambda b: b.area, reverse=True)
    return blocks[:_MAX_COMPONENTS], labels


def _merge_words(blocks: list[_Block]) -> list[_Block]:
    """Join blocks on the same text line (large type has word gaps wider than
    the closing kernel).

    Single left-to-right sweep per pass: each block is only compared with the
    open line fragments it could extend, instead of rescanning all pairs.
    """
    def joinable(a: _Block, b: _Block) -> bool:
        if a.fill > 0.8 or b.fill > 0.8:
            return False  # solid plates (CTA/logo) are never words
        similar_height = 0.7 <= b.h / max(1, a.h) <= 1.43
        overlap_y = min(a.y2, b.y2) - max(a.y1, b.y1)
        gap = b.x1 - a.x2
        return similar_height and overlap_y > 0.6 * min(a.h, b.h) and -2 <= gap < 0.9 * a.h

    current = list(blocks)
    for _ in range(4):  # merged fragments can enable one more join; converges fast
        current.sort(key=lambda b: (b.x1, b.y1))
        out: list[_Block] = []
        for b in current:
            for i in range(len(out) - 1, -1, -1):
                if joinable(out[i], b):
                    out[i] = out[i].merged(b)
                    break
            else:
                out.append(b)
        if len(out) == len(current):
            return out
        current = out
    return current


def _merge_text_lines(blocks: list[_Block]) -> list[_Block]:
    """Merge vertically adjacent text lines that belong to one paragraph."""
    blocks = sorted(blocks, key=lambda b: (b.y1, b.x1))
    merged: list[_Block] = []
    for b in blocks:
        if merged:
            p = merged[-1]
            similar_height = 0.75 <= b.h / max(1, p.h) <= 1.33
            gap = b.y1 - p.y2
            overlap_x = min(p.x2, b.x2) - max(p.x1, b.x1)
            if (
                similar_height and 0 <= gap < 0.5 * p.h
                and overlap_x > 0.3 * min(p.w, b.w)
                and p.aspect > 2.0 and b.aspect > 2.0
            ):
                merged[-1] = p.merged(b)
                continue
        merged.append(b)
    return merged


def _classify(blocks: list[_Block], w: int, h: int) -> list[tuple[_Block, ElementRole]]:
    """Assign roles from geometry: solid pills are CTA/logo, wide sparse blocks are text."""
    canvas = w * h
    roles: list[tuple[_Block, ElementRole]] = []
    texts: list[_Block] = []
    for b in blocks:
        area_frac = b.w * b.h / canvas
        solid = b.fill > 0.8
        near_top = b.y1 < 0.2 * h
        near_side = b.x1 < 0.25 * w or b.x2 > 0.75 * w
        if area_frac > 0.06 and b.aspect < 2.2:
            roles.append((b, ElementRole.HERO_IMAGE))
        elif solid and near_top and near_side and area_frac < 0.05:
            roles.append((b, ElementRole.LOGO))
        elif solid and 1.6 <= b.aspect <= 10 and area_frac < 0.08:
            roles.append((b, ElementRole.CTA))
        elif b.aspect >= 1.8:
            texts.append(b)
        elif area_frac > 0.02:
            roles.append((b, ElementRole.ILLUSTRATION))
        else:
            roles.append((b, ElementRole.DECORATION))

    # Tallest text block is the headline, the rest are sub/body copy.
    texts.sort(key=lambda b: b.h, reverse=True)
    for i, b in enumerate(texts):
        role = (
            ElementRole.HEADLINE if i == 0
            else ElementRole.SUBHEADLINE if i == 1
            else ElementRole.BODY_TEXT
        )
        roles.append((b, role))

    # Wordmark logos on a near-invisible plate show up as a small text block
    # tucked into a top corner; promote the first such block if no logo exists.
    if not any(r == ElementRole.LOGO for _, r in roles):
        for i, (b, r) in enumerate(roles):
            if (
                r in (ElementRole.SUBHEADLINE, ElementRole.BODY_TEXT)
                and b.y2 < 0.22 * h
                and (b.x1 > 0.7 * w or b.x2 < 0.3 * w)
                and b.w * b.h / canvas < 0.02
            ):
                roles[i] = (b, ElementRole.LOGO)
                break

    # A banner has at most one CTA and one logo: demote extras.
    for unique in (ElementRole.CTA, ElementRole.LOGO):
        found = [i for i, (_, r) in enumerate(roles) if r == unique]
        for i in found[1:]:
            roles[i] = (roles[i][0], ElementRole.BADGE)
    return roles


def decompose_flat_image(image: Image.Image) -> list[DesignElement] | None:
    """Split a flat banner into background + foreground pseudo-layers.

    Returns ``None`` when the image does not look like a design on a
    modelable background, or when nothing plausible is found.
    """
    src = image.convert("RGBA")
    sw, sh = src.size
    scale = min(1.0, _WORK_MAX_SIDE / max(sw, sh))
    ww, wh = max(8, int(round(sw * scale))), max(8, int(round(sh * scale)))
    work = np.asarray(src.convert("RGB").resize((ww, wh), Image.Resampling.BOX))

    mask = _foreground_mask(work)
    if mask is None:
        logger.info("auto-layers: background not modelable, keeping flat image")
        return None
    fg_frac = float(mask.mean())
    if fg_frac < 0.005 or fg_frac > _MAX_FOREGROUND_FRAC:
        logger.info("auto-layers: implausible foreground ratio %.3f", fg_frac)
        return None

    close_w = max(3, int(ww * 0.022))
    close_h = max(2, int(wh * 0.012))
    components, labels = _components(mask, close_w, close_h)
    blocks = _merge_text_lines(_merge_words(components))
    blocks = sorted(blocks, key=lambda b: b.w * b.h, reverse=True)[:_MAX_ELEMENTS]
    if not blocks:
        return None

    roles = _classify(blocks, ww, wh)
    inv = 1.0 / scale

    # Full-resolution foreground mask for alpha and background cleanup.
    mask_full = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (sw, sh), Image.Resampling.NEAREST
    )
    mask_full = mask_full.filter(ImageFilter.MaxFilter(5))
    # Component ids at full resolution, so each layer only takes its own pixels
    # (a badge inside the hero's rectangle must not be copied into the hero).
    labels_full = np.asarray(
        Image.fromarray(labels.astype(np.int32), mode="I").resize(
            (sw, sh), Image.Resampling.NEAREST
        )
    )
    labels_full = ndimage.grey_dilation(labels_full, size=(5, 5))

    elements: list[DesignElement] = []
    fg_union = np.zeros((sh, sw), dtype=bool)
    for idx, (b, role) in enumerate(roles):
        pad = max(2, int(3 * inv))
        x1 = max(0, int(b.x1 * inv) - pad)
        y1 = max(0, int(b.y1 * inv) - pad)
        x2 = min(sw, int(np.ceil(b.x2 * inv)) + pad)
        y2 = min(sh, int(np.ceil(b.y2 * inv)) + pad)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = src.crop((x1, y1, x2, y2))
        own = np.isin(labels_full[y1:y2, x1:x2], list(b.labels))
        region = (np.asarray(mask_full.crop((x1, y1, x2, y2))) > 0) & own
        if role in (ElementRole.CTA, ElementRole.LOGO, ElementRole.BADGE):
            # Solid plates: fill the holes (the label text) but keep the plate's
            # own silhouette so rounded corners do not carry background patches.
            region = ndimage.binary_fill_holes(region)
        alpha = Image.fromarray(region.astype(np.uint8) * 255).filter(
            ImageFilter.GaussianBlur(0.8)
        )
        crop.putalpha(alpha)
        fg_union[y1:y2, x1:x2] |= np.asarray(alpha) > 0
        elements.append(
            DesignElement(
                id=f"auto_{role.value}_{idx}",
                name=f"Auto {role.value.replace('_', ' ')}",
                layer_type="pixel",
                bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                image=crop,
                role=role,
                z_index=10 + idx,
                effects={"_auto_layer": True},
            )
        )

    if not elements:
        return None

    background = _clean_background(src, fg_union)
    elements.insert(
        0,
        DesignElement(
            id="auto_background",
            name="Auto background",
            layer_type="pixel",
            bbox=BoundingBox(0, 0, sw, sh),
            image=background,
            role=ElementRole.BACKGROUND,
            priority=9,
            z_index=0,
            effects={"_auto_layer": True},
        ),
    )
    logger.info(
        "auto-layers: %d elements (%s)",
        len(elements) - 1,
        ", ".join(e.role.value for e in elements[1:]),
    )
    return elements


def _clean_background(src: Image.Image, fg: np.ndarray) -> Image.Image:
    """Remove detected foreground from the background plate by inpainting."""
    rgb = np.asarray(src.convert("RGB")).copy()
    try:
        import cv2  # type: ignore

        from ..composition.content_aware_fit import _fast_inpaint

        mask = ndimage.binary_dilation(fg, iterations=3).astype(np.uint8) * 255
        filled = _fast_inpaint(cv2, rgb, mask, 5)
        return Image.fromarray(filled, "RGB").convert("RGBA")
    except Exception as e:  # headless environments without cv2
        logger.info("auto-layers: inpaint unavailable (%s), using blurred fill", e)
        blurred = np.asarray(
            Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(25)), dtype=np.uint8
        )
        rgb[fg] = blurred[fg]
        return Image.fromarray(rgb, "RGB").convert("RGBA")
