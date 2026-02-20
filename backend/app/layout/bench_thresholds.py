"""Thresholds for Phase 2.1 benchmark pass/fail checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchThresholds:
    max_overlap_area_ratio: float = 0.10
    max_outside_margin_ratio: float = 0.05
    min_hero_prominence_ratio: float = 0.12
    min_total_score: float = 250.0


DEFAULT_THRESHOLDS = BenchThresholds()
