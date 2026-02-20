# BENCH DELTA — Phase 3 flat-illustration patch

## What changed
- Planner now infers `horizon_hint` and `skyline_band` for flat-banner continuity and creates decor zones (top corners + side sparkle lanes) while excluding protected regions.
- Deterministic Phase 3 generator now uses a flat-illustration pipeline:
  - palette-bound sky gradient extrapolation,
  - skyline strip continuation (patch/mirror + jitter),
  - procedural fireworks/confetti decor in non-protected zones.
- Validators/scoring were expanded with flat-style penalties:
  - seam/repetition,
  - palette drift,
  - decor edge cutoff,
  - horizon continuity.
- Debug payload now includes horizon/skyline/palette summaries and per-candidate penalty breakdowns used by best-of-N selection.

## Verification commands
```bash
ruff check .
pytest -q
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
python backend/tools/run_layout_bench.py --mode phase3 --seed 42
```

## Results (seed=42)
- Phase 2.1 benchmark (non-regression):
  - `27/36` pass (`75.0%`)
  - `outside_margin_ratio`: `0/36` failures
- Phase 3 benchmark (patched):
  - `25/36` pass (`69.4%`)
  - `outside_margin_ratio`: `0/36` failures
  - `anchor_integrity`: `0/36` failures (100% integrity pass)
  - top remaining failures: `text_plate` (6), `overlap_area_ratio` (5), `total_score` (5)

## Acceptance / blockers
- Phase 3 target `>=70%` is narrowly missed (`69.4%`) by one case.
- Repro: run `python backend/tools/run_layout_bench.py --mode phase3 --seed 42` and inspect failures in `backend/tests/fixtures/outputs/bench_phase21/report.md`.
- Next tuning priority: reduce plate false-positives on busy-but-readable regions and cut residual overlap in portrait-heavy crowded cases.
