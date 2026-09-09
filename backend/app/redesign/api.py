"""Public Phase 3 target-first redesign API."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from ..composition.engine import CompositionEngine
from ..models import CompositionResult, DesignElement, LayoutResult
from .anchors import extract_anchors, extract_anchors_from_boxes
from .generator import make_generator
from .planner import build_target_first_plan
from .selector import select_best_candidate


@dataclass(frozen=True)
class RedesignDebug:
    mode: str
    generator_type: str
    selected_id: int
    selected_reason: str
    candidates: list[dict]
    anchors: list[dict]
    horizon_hint: dict[str, int]
    skyline_band: dict[str, int]
    palette_stats: dict[str, object]


def run_target_first_redesign(
    *,
    elements: list[DesignElement],
    layout_results: list[LayoutResult],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    manual_anchors: list[dict[str, int | str]] | None = None,
    n_candidates: int = 8,
) -> CompositionResult:
    """Execute Phase 3 target-first redesign using anchored brand-locked workflow."""
    compositor = CompositionEngine(use_ai_inpainting=False)
    base = compositor.compose(elements, layout_results, source_size, target_size)
    source_bg = base.image.convert("RGBA")
    # Phase 3 regenerates the background around the anchors; the base plate (if
    # any) is painted over. Report what actually happened instead of claiming a
    # plate was applied.
    base_text_plate_meta = dict(base.metadata.get("text_plate", {}))
    base_text_plate_meta.update(
        {
            "applied": False,
            "plates": 0,
            "reason": "phase3_regenerates_background",
            "base_plate_applied": bool(base.metadata.get("text_plate", {}).get("applied", False)),
            "source": "phase3_target_first",
        }
    )
    base_text_plate_meta.setdefault("busy_threshold", 0.2)

    if manual_anchors:
        src_ref = _pick_source_background(elements, source_size)
        anchors_bundle = extract_anchors_from_boxes(
            src_ref,
            manual_anchors,
            layout_results,
            target_size,
            mask_padding=8,
        )
    else:
        anchors_bundle = extract_anchors(elements, layout_results, target_size, mask_padding=8)

    plan = build_target_first_plan(
        anchors_bundle.anchors,
        anchors_bundle.protected_mask,
        target_size,
        source_bg,
    )
    generator, generator_name = make_generator()

    picked = select_best_candidate(
        generator=generator,
        source_background=source_bg,
        anchors=anchors_bundle.anchors,
        plan=plan,
        target_size=target_size,
        n=n_candidates,
    )

    debug = RedesignDebug(
        mode="phase3_target_first",
        generator_type=generator_name,
        selected_id=picked.selected_id,
        selected_reason=picked.selected_reason,
        candidates=[asdict(c) for c in picked.candidates],
        anchors=[
            {
                "element_id": a.element_id,
                "role": a.role.value,
                "protected": a.protected,
                "bbox": {
                    "x": a.target_bbox.x,
                    "y": a.target_bbox.y,
                    "width": a.target_bbox.width,
                    "height": a.target_bbox.height,
                },
                "scale": round(a.target_bbox.width / max(1, a.source_bbox.width), 4),
            }
            for a in anchors_bundle.anchors
        ],
        horizon_hint={"y1": plan.horizon_hint[0], "y2": plan.horizon_hint[1]},
        skyline_band={"y1": plan.skyline_band[0], "y2": plan.skyline_band[1]},
        palette_stats=dict(picked.selected_meta.get("palette_stats", {})),
    )

    warnings: list[str] = []
    fail_reasons: list[str] = []
    used_fallback = picked.status != "valid"
    if picked.status == "degraded":
        warnings.append("All Phase 3 candidates failed anchor integrity; best-effort output.")
        fail_reasons.append("phase3_all_candidates_rejected")
    elif picked.status == "last_resort":
        warnings.append("Phase 3 produced no usable candidate; source background stretched.")
        fail_reasons.append("phase3_last_resort")

    result = CompositionResult(
        image=picked.image.convert("RGB"),
        layout_results=layout_results,
        warnings=warnings,
        metadata={
            "redesign": {**asdict(debug), "selection_status": picked.status},
            "text_plate": base_text_plate_meta,
            "protected_ratio": float(np.mean(anchors_bundle.protected_mask))
            if anchors_bundle.protected_mask.size
            else 0.0,
        },
        gates_passed=None,
        fail_reasons=fail_reasons,
        used_fallback=used_fallback,
    )
    return result


def _pick_source_background(
    elements: list[DesignElement],
    source_size: tuple[int, int],
) -> Image.Image:
    for e in elements:
        if e.role.value == "background" and e.image is not None:
            return e.image.convert("RGBA")
    return Image.new("RGBA", source_size, (235, 235, 235, 255))
