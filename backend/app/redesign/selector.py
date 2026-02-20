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
    recipe: str
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
    n: int = 8,
    seed: int = 42,
) -> SelectionResult:
    integrity = AnchorIntegrityValidator()
    ocr = OCRTextValidator()
    seam = SeamArtifactHeuristic()
    palette = PaletteDistanceValidator()
    decor = DecorCutoffValidator()
    horizon = HorizonContinuityValidator()

    recipes = ["background_only", "light_decor", "strong_decor"]
    records: list[CandidateRecord] = []
    best_img: Image.Image | None = None
    best_score = -1e9
    best_id = 0
    best_reason = ""
    best_meta: dict[str, object] = {}
    best_any_img: Image.Image | None = None
    best_any_score = -1e9
    best_any_id = 0

    for i in range(max(1, n)):
        recipe = recipes[i % len(recipes)]
        bg, gen_meta = generator.generate(
            source_background=source_background,
            target_size=target_size,
            fill_mask=plan.fill_mask,
            decor_mask=plan.decor_mask,
            seed=seed,
            variant=i,
            plan=plan,
            recipe=recipe,
        )
        cand = _compose_anchors(bg, anchors)

        hard = integrity.validate(cand, anchors)
        if not hard.passed:
            records.append(
                CandidateRecord(i, recipe, 0.0, True, hard.reason, penalties={"hard": 100.0})
            )
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
            "repetition": float(gen_meta.get("repetition_penalty", 0.0)),
        }

        score = 100.0
        score -= penalties["seam"] * 0.34
        score -= penalties["palette"] * 0.24
        score -= penalties["decor"] * 0.16
        score -= penalties["horizon"] * 0.18
        score -= penalties["repetition"] * 0.08
        score = max(0.0, min(100.0, score))

        records.append(CandidateRecord(i, recipe, score, False, "", penalties=penalties))
        if score > best_any_score:
            best_any_score = score
            best_any_img = cand
            best_any_id = i

        # retry path: if candidate has severe seam/palette drift, regenerate stricter as same slot
        if penalties["seam"] > 22 or penalties["palette"] > 24:
            bg2, meta2 = generator.generate(
                source_background=source_background,
                target_size=target_size,
                fill_mask=plan.fill_mask,
                decor_mask=plan.decor_mask,
                seed=seed + 17,
                variant=i,
                plan=plan,
                recipe="background_only",
            )
            cand2 = _compose_anchors(bg2, anchors)
            hard2 = integrity.validate(cand2, anchors)
            if hard2.passed:
                seam2 = seam.validate(cand2, plan.fill_mask)
                pal2 = palette.validate(cand2, source_background, plan.fill_mask)
                decor2 = decor.validate(meta2.get("decor_stats", {}))
                hor2 = horizon.validate(cand2, plan.horizon_hint)
                score2 = (
                    100.0
                    - (100.0 - seam2.score) * 0.34
                    - (100.0 - pal2.score) * 0.24
                    - (100.0 - decor2.score) * 0.16
                    - (100.0 - hor2.score) * 0.18
                )
                score2 = max(0.0, min(100.0, score2))
                if score2 > score:
                    score = score2
                    cand = cand2
                    gen_meta = meta2

        if score > best_score:
            best_score = score
            best_img = cand
            best_id = i
            best_reason = "highest_valid_score"
            best_meta = gen_meta

    # Always keep phase3 output: never route to phase21/baseline.
    # Use best valid, else best generated.
    if best_img is None:
        if best_any_img is not None:
            return SelectionResult(
                best_any_img,
                best_any_id,
                "best_any_candidate",
                records,
                selected_meta={},
            )
        fallback = _compose_anchors(
            source_background.resize(target_size).convert("RGBA"), anchors
        )
        return SelectionResult(fallback, 0, "phase3_last_resort", records, selected_meta={})

    return SelectionResult(best_img, best_id, best_reason, records, selected_meta=best_meta)
