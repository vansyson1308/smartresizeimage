# BENCH DELTA — PR-A (Hard-margin constraints + repair pass)

## What changed
- Added deterministic repair utilities to enforce margin-safe bounding boxes and cap oversized boxes:
  - `clamp_bbox_to_margins`
  - `rescale_to_fit`
  - `apply_repair`
- Integrated repair into adaptive candidate evaluation and fallback policy:
  - repair -> re-solve -> re-validate before fallback
  - fallback now records reason and returns repaired layout instead of raw rigid template
- Made margin enforcement hard inside solver iterations using clamp helper.
- Extended benchmark layout debug payload with:
  - `profile_name`
  - `repair_applied`
  - `repair_steps`
  - `fallback_used`
  - `fallback_reason`

## Benchmark commands
```bash
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
```

## Before vs After (Phase 2.1 only, 36 runs)
- Pass rate:
  - Before: `0/36`
  - After:  `0/36`
- `outside_margin_ratio` failures:
  - Before: `36/36`
  - After:  `0/36` ✅
- Top remaining failure counts (after):
  - `min_font_size`: `32/36`
  - `total_score`: `24/36`
  - `overlap_area_ratio`: `8/36`
  - `text_plate`: `4/36`

## Interpretation
PR-A achieved the primary constraint objective (eliminate outside-margin failures) and reduced fallback dominance from margin-break conditions, but additional work is still needed for typography sizing and overlap quality to move pass-rate above zero.
