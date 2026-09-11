# Preserved evidence: Phase 3 benchmark false positive (commit 9d913a0)

Reproduced on 2026-09-09 in this repository at commit `9d913a0720af338bf1dda2d4a0908cce7d35dd0a`
(Python 3.11.15, Pillow 10.4.0, NumPy 1.26.4, SciPy 1.17.1, opencv-python-headless 4.11.0.86,
psd-tools 1.19.0; `backend/requirements-ci.txt`).

Command:

```bash
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode phase3 --seed 42 \
  --cases case_01_hero_headline_cta_logo --sizes 1080x1920 --outdir <scratch>
```

Legacy result (`legacy_summary.json`): **PASS**, score 32.69, 13 recorded violations
(10 overlaps, 3 min-size), `fail_reasons: []`.

What the customer would have seen (`after_thumb.png`, downscaled from 1080x1920): the headline,
subheadline and CTA are entirely covered by the hero; the logo is half covered. All four Phase 3
candidates were rejected (`anchor_integrity:headline:0.0044`) and the "phase3_last_resort" image
was returned as a normal result with `text_plate.applied = true`.

Root causes (all fixed in the same change set that added this evidence):

1. `backend/app/layout/solver.py` included the full-canvas background in collision resolution and
   forced a single vertical stack on side-by-side elements, so every content element was pushed to
   the bottom and clamped ("catastrophic_overlap_after_repair").
2. `backend/app/layout/bench_metrics.py` only inspected layout metadata; overlap was measured
   relative to the whole canvas (0.085 < 0.10) and the score threshold (32) was met.
3. `backend/app/redesign/api.py` forced `text_plate.applied = True` and returned last-resort output
   without any failure marker.

Contract v2 (`backend/app/quality`) evaluates the rendered pixels and fails this output with:
`element_visible:headline` (99% covered), `element_visible:sub`, `element_visible:cta`,
`element_visible:logo` (48% covered) and `text_legible:*` (OCR agreement 0%).
Regression test: `backend/tests/test_quality_contract.py::test_contract_v2_fails_historical_false_positive_for_the_right_reason`.
