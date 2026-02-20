"""Best-of-N candidate selection for Phase 3 redesign."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .anchors import Anchor
from .generator import BackgroundGenerator
from .planner import RedesignPlan
from .validators import (
    AnchorIntegrityValidator,
    DecorCutoffValidator,
    HorizonContinuityValidator,
    OCRTextValidator,
    PaletteDistanceValidator,
    SeamArtifactHeuristic,
)


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: int
    score: float
    rejected: bool
    rejection_reason: str
    penalties: dict[str, float]


@dataclass(frozen=True)
class SelectionResult:
    image: Image.Image
    selected_id: int
    selected_reason: str
    candidates: list[CandidateRecord]
    selected_meta: dict[str, object]


def _compose_anchors(base: Image.Image, anchors: list[Anchor]) -> Image.Image:
    out = base.convert("RGBA")
    for a in anchors:
        target = a.image.resize((a.target_bbox.width, a.target_bbox.height))
        out.alpha_composite(target, dest=(a.target_bbox.x, a.target_bbox.y))
    return out


def select_best_candidate(
    generator: BackgroundGenerator,
    source_background: Image.Image,
    anchors: list[Anchor],
    plan: RedesignPlan,
    target_size: tuple[int, int],
    n: int = 4,
    seed: int = 42,
) -> SelectionResult:
    integrity = AnchorIntegrityValidator()
    ocr = OCRTextValidator()
    seam = SeamArtifactHeuristic()
    palette = PaletteDistanceValidator()
    decor = DecorCutoffValidator()
    horizon = HorizonContinuityValidator()

    records: list[CandidateRecord] = []
    best_img: Image.Image | None = None
    best_score = -1e9
    best_id = 0
    best_reason = ""
    best_meta: dict[str, object] = {}

    for i in range(max(1, n)):
        bg, gen_meta = generator.generate(
            source_background=source_background,
            target_size=target_size,
            fill_mask=plan.fill_mask,
            decor_mask=plan.decor_mask,
            seed=seed,
            variant=i,
            plan=plan,
        )
        cand = _compose_anchors(bg, anchors)

        hard = integrity.validate(cand, anchors)
        if not hard.passed:
            records.append(CandidateRecord(i, 0.0, True, hard.reason, penalties={"hard": 100.0}))
            continue

        _ = ocr.validate(cand, anchors)
        seam_v = seam.validate(cand, plan.fill_mask)
        pal_v = palette.validate(cand, source_background, plan.fill_mask)
        decor_v = decor.validate(gen_meta.get("decor_stats", {}))
        hor_v = horizon.validate(cand, plan.horizon_hint)

        penalties = {
            "seam": max(0.0, 100.0 - seam_v.score),
            "palette": max(0.0, 100.0 - pal_v.score),
            "decor": max(0.0, 100.0 - decor_v.score),
            "horizon": max(0.0, 100.0 - hor_v.score),
        }

        score = 100.0
        score -= penalties["seam"] * 0.35
        score -= penalties["palette"] * 0.25
        score -= penalties["decor"] * 0.20
        score -= penalties["horizon"] * 0.20
        score = max(0.0, min(100.0, score))

        records.append(CandidateRecord(i, score, False, "", penalties=penalties))
        if score > best_score:
            best_score = score
            best_img = cand
            best_id = i
            best_reason = "highest_valid_score"
            best_meta = gen_meta

    if best_img is None:
        fallback = _compose_anchors(source_background.resize(target_size).convert("RGBA"), anchors)
        return SelectionResult(fallback, 0, "all_candidates_rejected", records, selected_meta={})

    return SelectionResult(best_img, best_id, best_reason, records, selected_meta=best_meta)
