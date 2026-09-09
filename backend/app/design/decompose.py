"""Honest decomposition of a flat image into reviewable elements.

A flat PNG/JPG carries no structure. This module recovers what it can and
labels every inference with provenance ``recovered`` and a confidence:

- text lines via OCR (tesseract) with per-line alpha masks and the recognised
  string kept as *unverified* metadata (the element stays raster text until a
  user converts it to native text);
- one salient subject via edge density + GrabCut refinement (OpenCV) or a
  rectangular fallback;
- a clean background obtained by inpainting the recovered regions.

Nothing here claims pixel-exact reconstruction: hidden content behind a
subject is inferred (inpainted) and marked as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("autobanner.design.decompose")

OCR_MAX_SIDE = 1600


@dataclass
class RecoveredElement:
    kind: str  # "text" | "subject"
    bbox: tuple[int, int, int, int]  # x, y, w, h in source pixels
    image: Image.Image  # RGBA crop with recovered alpha
    confidence: float
    role_guess: str
    text: str | None = None
    text_color: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DecompositionResult:
    background: Image.Image
    elements: list[RecoveredElement]
    method: dict
    notes: list[str] = field(default_factory=list)


def _cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------
# text recovery
# --------------------------------------------------------------------------------------


def _ocr_lines(image: Image.Image) -> list[dict]:
    """OCR the image and return line boxes: {x, y, w, h, text, conf}."""
    try:
        import os

        import pytesseract  # type: ignore

        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    except Exception as exc:  # noqa: BLE001
        logger.info("OCR unavailable: %s", exc)
        return []
    rgb = image.convert("RGB")
    scale = 1.0
    longest = max(rgb.size)
    if longest > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / longest
        rgb = rgb.resize(
            (int(rgb.width * scale), int(rgb.height * scale)), Image.Resampling.LANCZOS
        )
    try:
        data = pytesseract.image_to_data(
            rgb, output_type=pytesseract.Output.DICT, config="--psm 11", timeout=30.0
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("OCR failed: %s", exc)
        return []
    lines: dict[tuple[int, int, int], dict] = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 35:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        x, y = int(data["left"][i] / scale), int(data["top"][i] / scale)
        w, h = int(data["width"][i] / scale), int(data["height"][i] / scale)
        if w < 4 or h < 6:
            continue
        line = lines.setdefault(
            key, {"x1": x, "y1": y, "x2": x + w, "y2": y + h, "words": [], "confs": []}
        )
        line["x1"] = min(line["x1"], x)
        line["y1"] = min(line["y1"], y)
        line["x2"] = max(line["x2"], x + w)
        line["y2"] = max(line["y2"], y + h)
        line["words"].append(text)
        line["confs"].append(conf)
    out = []
    for line in lines.values():
        out.append(
            {
                "x": line["x1"],
                "y": line["y1"],
                "w": line["x2"] - line["x1"],
                "h": line["y2"] - line["y1"],
                "text": " ".join(line["words"]),
                "conf": float(sum(line["confs"]) / len(line["confs"])) / 100.0,
            }
        )
    return sorted(out, key=lambda b: (b["y"], b["x"]))


def _merge_lines_into_blocks(lines: list[dict], gap_factor: float = 0.7) -> list[dict]:
    """Merge vertically adjacent lines of similar height/left edge into blocks."""
    blocks: list[dict] = []
    for line in lines:
        merged = False
        for blk in blocks:
            close_y = (
                0 <= line["y"] - (blk["y"] + blk["h"]) <= gap_factor * max(blk["line_h"], line["h"])
            )
            similar_h = abs(line["h"] - blk["line_h"]) <= 0.5 * max(blk["line_h"], line["h"])
            overlap_x = line["x"] < blk["x"] + blk["w"] and blk["x"] < line["x"] + line["w"]
            if close_y and similar_h and overlap_x:
                x1 = min(blk["x"], line["x"])
                y1 = min(blk["y"], line["y"])
                x2 = max(blk["x"] + blk["w"], line["x"] + line["w"])
                y2 = max(blk["y"] + blk["h"], line["y"] + line["h"])
                blk.update({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
                blk["text"] += "\n" + line["text"]
                blk["confs"].append(line["conf"])
                blk["lines"] += 1
                merged = True
                break
        if not merged:
            blocks.append({**line, "line_h": line["h"], "confs": [line["conf"]], "lines": 1})
    for blk in blocks:
        blk["conf"] = float(sum(blk["confs"]) / len(blk["confs"]))
    return blocks


def _text_alpha_crop(image: Image.Image, box: tuple[int, int, int, int], pad: int = 4):
    """Cut the text out of its background using colour distance to the border colour."""
    x, y, w, h = box
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(image.width, x + w + pad), min(image.height, y + h + pad)
    crop = np.asarray(image.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.float32)
    if crop.size == 0:
        return None, None
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(crop - bg, axis=2)
    # soft alpha: 0 at <25, 1 at >90 colour distance
    alpha = np.clip((dist - 25.0) / 65.0, 0.0, 1.0)
    if alpha.mean() < 0.01:
        return None, None
    text_pixels = crop[alpha > 0.6]
    color = np.median(text_pixels, axis=0) if len(text_pixels) else (255.0 - bg)
    rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(crop, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    hex_color = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in color])
    return Image.fromarray(rgba, mode="RGBA"), hex_color


LOW_OCR_CONFIDENCE = 0.65


def _guess_text_role(block: dict, all_blocks: list[dict]) -> str:
    """Role from typographic cues. Unreadable small blocks are treated as logo/badge marks."""
    text = block["text"].strip()
    upper = text.upper()
    words = upper.split()
    confident = [b for b in all_blocks if b["conf"] >= LOW_OCR_CONFIDENCE] or all_blocks
    tallest = max(b["line_h"] for b in confident) if confident else block["line_h"]
    if block["conf"] < LOW_OCR_CONFIDENCE and len(words) <= 3:
        return "logo"
    if any(ch in text for ch in "$€£%") and len(words) <= 4:
        return "label"
    if block["line_h"] >= 0.85 * tallest and len(words) <= 8:
        return "headline"
    if len(words) <= 3 and text == upper and block["lines"] == 1:
        return "cta"
    if block["line_h"] >= 0.5 * tallest:
        return "subheadline"
    return "body_text"


# --------------------------------------------------------------------------------------
# subject recovery
# --------------------------------------------------------------------------------------


def _find_subject(
    image: Image.Image, exclude: np.ndarray
) -> tuple[tuple[int, int, int, int], float] | None:
    """Locate the most salient blob outside ``exclude`` (True = text) -> (bbox, confidence)."""
    gray = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = gx + gy
    edges[exclude] = 0.0
    # coarse grid energy so texture-free blobs (flat illustrations) still register via outline
    h, w = edges.shape
    cell = max(8, min(w, h) // 48)
    gh, gw = h // cell, w // cell
    if gh < 2 or gw < 2:
        return None
    grid = edges[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    # also colour deviation from the global median (subjects differ from the background)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    med = np.median(rgb.reshape(-1, 3), axis=0)
    dev = np.linalg.norm(rgb - med, axis=2)
    dev[exclude] = 0.0
    dev_grid = dev[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    energy = grid / (grid.max() + 1e-6) + dev_grid / (dev_grid.max() + 1e-6)
    thresh = np.percentile(energy, 80)
    mask = energy >= max(thresh, 0.15)
    cv2 = _cv2()
    if cv2 is not None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if n <= 1:
            return None
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, area = stats[best]
        comp = labels == best
    else:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x, y = xs.min(), ys.min()
        bw, bh = xs.max() - x + 1, ys.max() - y + 1
        area = len(xs)
        comp = mask
    box = (int(x * cell), int(y * cell), int(bw * cell), int(bh * cell))
    fill = float(area / max(1, bw * bh))
    size_share = (box[2] * box[3]) / max(1, w * h)
    if size_share < 0.02 or size_share > 0.85:
        return None
    confidence = 0.25 + 0.5 * fill + (0.15 if cv2 is not None else 0.0)
    _ = comp
    return box, float(min(0.85, confidence))


def _subject_alpha_crop(image: Image.Image, box: tuple[int, int, int, int]):
    """Cut the subject with GrabCut when OpenCV is present; rectangular otherwise."""
    x, y, w, h = box
    # Expand the crop so the detected box sits inside it: GrabCut needs some
    # definite background around the rectangle it is initialised with.
    margin = max(6, int(0.06 * max(w, h)))
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(image.width, x + w + margin), min(image.height, y + h + margin)
    crop_rgb = np.asarray(image.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.uint8)
    cv2 = _cv2()
    alpha = np.full(crop_rgb.shape[:2], 255, dtype=np.uint8)
    method = "rect"
    if cv2 is not None and crop_rgb.shape[0] > 20 and crop_rgb.shape[1] > 20:
        try:
            mask = np.zeros(crop_rgb.shape[:2], np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            rect = (
                max(1, x - x1),
                max(1, y - y1),
                min(crop_rgb.shape[1] - 2, w),
                min(crop_rgb.shape[0] - 2, h),
            )
            bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            cv2.grabCut(bgr, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            if fg.mean() > 10:
                fg = cv2.GaussianBlur(fg, (5, 5), 0)
                alpha = fg
                method = "grabcut"
        except Exception as exc:  # noqa: BLE001
            logger.info("grabCut failed, rectangular subject: %s", exc)
    rgba = np.dstack([crop_rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA"), (x1, y1, x2 - x1, y2 - y1), method


# --------------------------------------------------------------------------------------
# background
# --------------------------------------------------------------------------------------


def _inpaint_background(image: Image.Image, holes: np.ndarray) -> tuple[Image.Image, str]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not holes.any():
        return Image.fromarray(rgb, mode="RGB").convert("RGBA"), "none"
    cv2 = _cv2()
    if cv2 is not None:
        try:
            mask = (holes.astype(np.uint8)) * 255
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), mode="RGB").convert(
                "RGBA"
            ), "telea"
        except Exception as exc:  # noqa: BLE001
            logger.info("inpaint failed, using blurred fill: %s", exc)
    # fallback: fill holes with a heavily blurred copy
    blurred = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(25)), dtype=np.uint8)
    out = rgb.copy()
    out[holes] = blurred[holes]
    return Image.fromarray(out, mode="RGB").convert("RGBA"), "blur_fill"


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def decompose_flat_image(image: Image.Image, *, run_ocr: bool = True) -> DecompositionResult:
    src = image.convert("RGBA")
    w, h = src.size
    notes: list[str] = []
    elements: list[RecoveredElement] = []
    holes = np.zeros((h, w), dtype=bool)

    lines = _ocr_lines(src) if run_ocr else []
    blocks = _merge_lines_into_blocks(lines)
    if run_ocr and not lines:
        notes.append("No text was recognised (OCR unavailable or nothing legible).")
    for blk in blocks:
        box = (blk["x"], blk["y"], blk["w"], blk["h"])
        crop, color = _text_alpha_crop(src, box)
        if crop is None:
            continue
        pad = 4
        x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
        holes[y1 : y1 + crop.height, x1 : x1 + crop.width] |= np.asarray(crop.split()[3]) > 40
        role = _guess_text_role(blk, blocks)
        if role == "logo":
            elements.append(
                RecoveredElement(
                    kind="mark",
                    bbox=(x1, y1, crop.width, crop.height),
                    image=crop,
                    confidence=0.4,
                    role_guess="logo",
                    text=None,
                    notes=[
                        "Small block OCR could not read; treated as a logo/badge mark. "
                        "Replace with the original asset when available."
                    ],
                )
            )
            continue
        elements.append(
            RecoveredElement(
                kind="text",
                bbox=(x1, y1, crop.width, crop.height),
                image=crop,
                confidence=float(min(0.9, blk["conf"])),
                role_guess=role,
                text=blk["text"],
                text_color=color,
                notes=[
                    "Recognised by OCR; string not verified. Convert to native text after review."
                ],
            )
        )

    subject = _find_subject(src, holes)
    if subject is not None:
        box, conf = subject
        crop, real_box, method = _subject_alpha_crop(src, box)
        # Pixels already claimed by recovered text/marks are not part of the subject.
        rx, ry, rw, rh = real_box
        claimed = holes[ry : ry + rh, rx : rx + rw]
        if claimed.any():
            arr = np.asarray(crop, dtype=np.uint8).copy()
            arr[:, :, 3][claimed[: arr.shape[0], : arr.shape[1]]] = 0
            crop = Image.fromarray(arr, mode="RGBA")
        elements.append(
            RecoveredElement(
                kind="subject",
                bbox=real_box,
                image=crop,
                confidence=conf,
                role_guess="hero_image",
                notes=[f"Salient region cut with {method}; edges are approximate."],
            )
        )
        # The whole detected region is treated as unknown background: remnants of
        # the subject around the cut would otherwise be inpainted back in.
        holes[real_box[1] : real_box[1] + real_box[3], real_box[0] : real_box[0] + real_box[2]] = (
            True
        )
    else:
        notes.append("No distinct subject found; the picture is kept as background.")

    background, fill_method = _inpaint_background(src, holes)
    if holes.any():
        notes.append(
            f"Background behind recovered elements was inferred ({fill_method}); "
            "it is not original pixels."
        )
    return DecompositionResult(
        background=background,
        elements=elements,
        method={
            "ocr": "tesseract" if lines else ("none" if run_ocr else "disabled"),
            "subject": "edge_energy+grabcut" if _cv2() is not None else "edge_energy+rect",
            "background_fill": fill_method,
        },
        notes=notes,
    )
