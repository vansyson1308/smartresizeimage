"""Rendered-output checks: what the customer actually sees.

These checks look at the final composited pixels rather than at layout
metadata. They are deliberately independent from the layout engine and the
scoring used to select candidates, so a bad output cannot mark itself good.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..composition.resize import high_quality_resize
from ..constants import BACKGROUND_ROLES
from ..enums import ElementRole
from ..models import DesignElement, LayoutResult
from .contract import CheckResult, CheckStatus, Severity

logger = logging.getLogger("autobanner.quality.rendered")

TEXT_ROLES = frozenset(
    {
        ElementRole.HEADLINE,
        ElementRole.SUBHEADLINE,
        ElementRole.BODY_TEXT,
        ElementRole.CTA,
        ElementRole.LABEL,
        ElementRole.BADGE,
    }
)
IDENTITY_ROLES = frozenset({ElementRole.LOGO})
SUBJECT_ROLES = frozenset(
    {ElementRole.HERO_IMAGE, ElementRole.PHOTO, ElementRole.ILLUSTRATION, ElementRole.ICON}
)


def severity_for_role(role: ElementRole) -> Severity:
    if role in TEXT_ROLES or role in IDENTITY_ROLES:
        return Severity.CRITICAL
    if role in SUBJECT_ROLES:
        return Severity.MAJOR
    return Severity.MINOR


@dataclass(frozen=True)
class VisibilityThresholds:
    pass_min: float = 0.97
    review_min: float = 0.85
    pixel_tolerance: int = 28  # mean abs RGB difference to count a pixel as "as expected"
    opaque_alpha: int = 250
    min_samples: int = 40


def element_visibility_checks(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    composite: Image.Image,
    thresholds: VisibilityThresholds | None = None,
) -> list[CheckResult]:
    """Verify each content element is actually visible in the composite.

    For every visible content element, the element image is resized to its
    placed bbox exactly like the compositor does; the opaque pixels are then
    compared with the final composite. Pixels that differ are occluded by a
    later element, clipped by the canvas or altered by post-processing.
    """
    th = thresholds or VisibilityThresholds()
    comp = np.asarray(composite.convert("RGB"), dtype=np.int16)
    canvas_h, canvas_w = comp.shape[:2]
    layout_map = {r.element_id: r for r in layout_results}
    results: list[CheckResult] = []

    for elem in elements:
        if elem.role in BACKGROUND_ROLES:
            continue
        layout = layout_map.get(elem.id)
        if layout is None or not layout.visible:
            continue
        severity = severity_for_role(elem.role)
        if elem.image is None:
            results.append(
                CheckResult(
                    check_id="element_visible",
                    status=CheckStatus.NOT_CHECKED,
                    severity=severity,
                    subject_id=elem.id,
                    message=f"{_label(elem)} has no pixel data to verify",
                    details={"reason": "no_image"},
                )
            )
            continue

        b = layout.new_bbox
        if b.width <= 0 or b.height <= 0:
            results.append(
                CheckResult(
                    check_id="element_visible",
                    status=CheckStatus.FAIL,
                    severity=severity,
                    subject_id=elem.id,
                    message=f"{_label(elem)} collapsed to zero size",
                    details={"bbox": _bbox_dict(b)},
                )
            )
            continue

        expected = np.asarray(
            high_quality_resize(elem.image.convert("RGBA"), (b.width, b.height)), dtype=np.int16
        )
        alpha = expected[:, :, 3]
        opaque = alpha >= th.opaque_alpha
        total_opaque = int(opaque.sum())

        # Canvas clipping: opaque pixels placed outside the canvas are lost.
        x1, y1 = max(0, b.x), max(0, b.y)
        x2, y2 = min(canvas_w, b.x2), min(canvas_h, b.y2)
        inside = np.zeros_like(opaque)
        if x2 > x1 and y2 > y1:
            inside[y1 - b.y : y2 - b.y, x1 - b.x : x2 - b.x] = True
        clipped = int((opaque & ~inside).sum())
        clipped_fraction = clipped / total_opaque if total_opaque else 0.0

        if total_opaque < th.min_samples:
            results.append(
                CheckResult(
                    check_id="element_visible",
                    status=CheckStatus.NEEDS_REVIEW,
                    severity=severity,
                    subject_id=elem.id,
                    message=f"{_label(elem)} is too soft/transparent to verify automatically",
                    details={"opaque_pixels": total_opaque, "min_samples": th.min_samples},
                )
            )
            continue

        sample = opaque & inside
        if not sample.any():
            visible_fraction = 0.0
            mean_diff = None
        else:
            region = comp[y1:y2, x1:x2]
            exp_rgb = expected[y1 - b.y : y2 - b.y, x1 - b.x : x2 - b.x, :3]
            diff = np.abs(region - exp_rgb).mean(axis=2)
            sub_sample = sample[y1 - b.y : y2 - b.y, x1 - b.x : x2 - b.x]
            matched = diff[sub_sample] <= th.pixel_tolerance
            # visible = matched inside pixels / all opaque pixels (clipped ones count as lost)
            visible_fraction = float(matched.sum()) / float(total_opaque)
            mean_diff = float(diff[sub_sample].mean())

        non_normal = elem.opacity < 0.999 or (elem.blend_mode or "normal").lower() not in (
            "normal",
            "blendmode.normal",
        )
        details = {
            "visible_fraction": round(visible_fraction, 4),
            "clipped_fraction": round(clipped_fraction, 4),
            "opaque_pixels": total_opaque,
            "mean_pixel_diff": None if mean_diff is None else round(mean_diff, 2),
            "bbox": _bbox_dict(b),
            "role": elem.role.value,
        }

        if visible_fraction >= th.pass_min:
            status = CheckStatus.PASS
            message = f"{_label(elem)} is fully visible"
        elif non_normal and visible_fraction >= th.review_min * 0.5:
            status = CheckStatus.NEEDS_REVIEW
            message = f"{_label(elem)} uses opacity/blend; visibility could not be verified exactly"
            details["reason"] = "non_normal_blend"
        elif visible_fraction >= th.review_min:
            status = CheckStatus.NEEDS_REVIEW
            message = (
                f"{_label(elem)} is partially hidden "
                f"({(1 - visible_fraction) * 100:.0f}% of its pixels are covered or clipped)"
            )
        else:
            status = CheckStatus.FAIL
            hidden = (1 - visible_fraction) * 100
            if clipped_fraction > 0.5:
                message = f"{_label(elem)} is clipped by the canvas edge ({hidden:.0f}% lost)"
            else:
                message = f"{_label(elem)} is hidden behind another element ({hidden:.0f}% covered)"

        results.append(
            CheckResult(
                check_id="element_visible",
                status=status,
                severity=severity,
                subject_id=elem.id,
                message=message,
                details=details,
            )
        )

    return results


# ---------------------------------------------------------------------------
# OCR-based legibility
# ---------------------------------------------------------------------------

OcrFn = Callable[[Image.Image], str]

_OCR_STATE: dict[str, object] = {
    "checked": False,
    "available": False,
    "version": None,
    "reason": "",
}


def ocr_engine_status() -> dict[str, object]:
    """Return availability of the tesseract OCR engine (cached)."""
    if _OCR_STATE["checked"]:
        return dict(_OCR_STATE)
    _OCR_STATE["checked"] = True
    try:
        import pytesseract  # type: ignore

        version = str(pytesseract.get_tesseract_version())
        _OCR_STATE.update({"available": True, "version": version, "reason": ""})
    except Exception as exc:  # noqa: BLE001
        _OCR_STATE.update({"available": False, "version": None, "reason": str(exc)[:200]})
    return dict(_OCR_STATE)


OCR_TIMEOUT_S = 20.0
OCR_MAX_SIDE = 1400


def default_ocr(image: Image.Image) -> str:
    """OCR with bounded cost: capped input size, single thread, hard timeout.

    Busy or noisy backgrounds can make tesseract crawl for minutes; a timeout
    turns that into ``NOT_CHECKED`` instead of stalling a whole batch.
    """
    import os

    import pytesseract  # type: ignore

    # Tesseract spawns OpenMP threads per call; one thread per process keeps
    # batch evaluation from oversubscribing the machine.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    rgb = image.convert("RGB")
    longest = max(rgb.size)
    if longest > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / longest
        rgb = rgb.resize(
            (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return str(pytesseract.image_to_string(rgb, config="--psm 6", timeout=OCR_TIMEOUT_S))


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z%$€£&+\- ]+", " ", text.upper())
    return re.sub(r"\s+", " ", cleaned).strip()


def text_similarity(expected: str, observed: str) -> tuple[float, float]:
    """Return (sequence ratio, token recall) between normalized strings."""
    e = normalize_text(expected)
    o = normalize_text(observed)
    if not e:
        return 1.0, 1.0
    ratio = difflib.SequenceMatcher(None, e, o).ratio()
    e_tokens = e.split()
    o_tokens = set(o.split())
    recall = sum(1 for t in e_tokens if t in o_tokens) / max(1, len(e_tokens))
    return float(ratio), float(recall)


@dataclass(frozen=True)
class LegibilityThresholds:
    pass_min: float = 0.80
    review_min: float = 0.50
    min_crop_height: int = 64
    max_crop_height: int = 320
    padding: int = 8


def text_legibility_checks(
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    composite: Image.Image,
    ocr: OcrFn | None = None,
    visibility_by_id: dict[str, CheckResult] | None = None,
    thresholds: LegibilityThresholds | None = None,
) -> list[CheckResult]:
    """OCR the region of every native text element and compare with its content.

    OCR is one tool, not an oracle: when the pixel-level visibility check
    passed but OCR disagrees, the result is NEEDS_REVIEW, not FAIL. When OCR is
    unavailable the check is reported as NOT_CHECKED.
    """
    th = thresholds or LegibilityThresholds()
    layout_map = {r.element_id: r for r in layout_results}
    results: list[CheckResult] = []

    engine = ocr
    engine_meta: dict[str, object] = {"engine": "custom"}
    if engine is None:
        status = ocr_engine_status()
        if status["available"]:
            engine = default_ocr
            engine_meta = {"engine": "tesseract", "version": status["version"]}
        else:
            engine_meta = {"engine": "tesseract", "reason": status["reason"]}

    for elem in elements:
        text = (elem.text_content or "").strip()
        if not text or elem.role in BACKGROUND_ROLES:
            continue
        layout = layout_map.get(elem.id)
        if layout is None or not layout.visible:
            continue
        severity = severity_for_role(elem.role) if elem.role in TEXT_ROLES else Severity.MAJOR

        if engine is None:
            results.append(
                CheckResult(
                    check_id="text_legible",
                    status=CheckStatus.NOT_CHECKED,
                    severity=severity,
                    subject_id=elem.id,
                    message=f"Could not OCR {_label(elem)}: OCR engine unavailable",
                    details={**engine_meta, "expected": text},
                )
            )
            continue

        b = layout.new_bbox
        x1 = max(0, b.x - th.padding)
        y1 = max(0, b.y - th.padding)
        x2 = min(composite.width, b.x2 + th.padding)
        y2 = min(composite.height, b.y2 + th.padding)
        if x2 <= x1 or y2 <= y1:
            results.append(
                CheckResult(
                    check_id="text_legible",
                    status=CheckStatus.FAIL,
                    severity=severity,
                    subject_id=elem.id,
                    message=f"{_label(elem)} lies entirely outside the canvas",
                    details={"expected": text, "bbox": _bbox_dict(b)},
                )
            )
            continue

        vis = (visibility_by_id or {}).get(elem.id)
        if vis is not None and vis.status == CheckStatus.FAIL:
            # The pixels are not there; OCR cannot add information.
            results.append(
                CheckResult(
                    check_id="text_legible",
                    status=CheckStatus.FAIL,
                    severity=severity,
                    subject_id=elem.id,
                    message=f"{_label(elem)} is not legible (element is hidden or clipped)",
                    details={**engine_meta, "expected": text, "reason": "not_visible"},
                )
            )
            continue

        crop = composite.convert("RGB").crop((x1, y1, x2, y2))
        if crop.height < th.min_crop_height:
            scale = th.min_crop_height / max(1, crop.height)
            crop = crop.resize(
                (max(1, int(crop.width * scale)), th.min_crop_height), Image.Resampling.LANCZOS
            )
        elif crop.height > th.max_crop_height:
            # Glyphs far larger than ~60px do not help tesseract; shrinking bounds cost.
            scale = th.max_crop_height / crop.height
            crop = crop.resize(
                (max(1, int(crop.width * scale)), th.max_crop_height), Image.Resampling.LANCZOS
            )
        try:
            observed = engine(crop)
        except Exception as exc:  # noqa: BLE001
            reason = "ocr_timeout" if "timeout" in str(exc).lower() else "ocr_error"
            results.append(
                CheckResult(
                    check_id="text_legible",
                    status=CheckStatus.NOT_CHECKED,
                    severity=severity,
                    subject_id=elem.id,
                    message=f"Could not OCR {_label(elem)} ({reason})",
                    details={
                        **engine_meta,
                        "expected": text,
                        "reason": reason,
                        "error": str(exc)[:160],
                    },
                )
            )
            continue

        ratio, recall = text_similarity(text, observed)
        score = max(ratio, recall)
        details = {
            **engine_meta,
            "expected": text,
            "observed": normalize_text(observed)[:200],
            "ratio": round(ratio, 3),
            "token_recall": round(recall, 3),
        }
        pixels_ok = vis is not None and vis.status == CheckStatus.PASS

        if score >= th.pass_min:
            status, message = CheckStatus.PASS, f"{_label(elem)} reads correctly"
        elif score >= th.review_min or pixels_ok:
            status = CheckStatus.NEEDS_REVIEW
            message = f"{_label(elem)} may be hard to read (OCR agreement {score * 100:.0f}%)"
        else:
            status = CheckStatus.FAIL
            message = f"{_label(elem)} is not legible (OCR agreement {score * 100:.0f}%)"

        results.append(
            CheckResult(
                check_id="text_legible",
                status=status,
                severity=severity,
                subject_id=elem.id,
                message=message,
                details=details,
            )
        )

    return results


def _label(elem: DesignElement) -> str:
    role = elem.role.value.replace("_", " ")
    if elem.text_content:
        snippet = elem.text_content.strip().replace("\n", " ")
        if len(snippet) > 24:
            snippet = snippet[:21] + "..."
        return f"{role.capitalize()} '{snippet}'"
    return f"{role.capitalize()} '{elem.name}'"


def _bbox_dict(b) -> dict[str, int]:
    return {"x": int(b.x), "y": int(b.y), "width": int(b.width), "height": int(b.height)}
