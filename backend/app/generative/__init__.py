"""Generative utilities (Phase 2 scaffolding)."""

from .decor import apply_optional_decor_synthesis, build_decor_zone_mask, run_ocr_negative_gate
from .engine import GenerativeBackend, GenerativeOutpaintEngine, OutpaintPolicy
from .gates import GateReport, evaluate_quality_gates
from .harmonize import apply_color_grading_safe, apply_grounding_shadow_safe, extract_mascot_masks
from .masks import Masks, build_masks
from .text_plate import TextPlateConfig, apply_text_safe_plates, compute_busy_score

__all__ = [
    "GenerativeBackend",
    "GenerativeOutpaintEngine",
    "OutpaintPolicy",
    "Masks",
    "build_masks",
    "apply_color_grading_safe",
    "apply_grounding_shadow_safe",
    "extract_mascot_masks",
    "apply_optional_decor_synthesis",
    "build_decor_zone_mask",
    "run_ocr_negative_gate",
    "GateReport",
    "evaluate_quality_gates",
    "TextPlateConfig",
    "apply_text_safe_plates",
    "compute_busy_score",
]
