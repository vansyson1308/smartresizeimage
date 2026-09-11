# Design pipeline ablations: layout grammar (2026-09-11)

Contract v2.0.0. Same fixtures, seeds and sizes for every configuration; holdout cases were never used for tuning. Wilson 95% intervals on acceptance.

- environment: `{"numpy": "1.26.4", "ocr_engine": "tesseract", "ocr_version": "5.3.4", "pillow": "10.4.0", "python": "3.11.15"}`
- git commit: `fdae532eebdc38aec08a94840f0ef2b114642333`
- sizes: 1200x628, 1080x1080, 1080x1920, 300x250; cases: 15 (holdout from case 10)

| Config | Split | Runs | Accepted | Needs review | Failed | Acceptance (95% CI) | Mean s | Repair steps | Family-issue runs | Agreement w/ designer (held-out sizes) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| no_grammar | tuning | 36 | 32 | 2 | 2 | 0.89 (0.75–0.96) | 1.44 | 0.19 | 2 | 0.2574 |
| no_grammar | holdout | 24 | 22 | 1 | 1 | 0.92 (0.74–0.98) | 1.41 | 0.21 | 1 | 0.2591 |
| grammar | tuning | 36 | 32 | 2 | 2 | 0.89 (0.75–0.96) | 1.38 | 0.19 | 2 | 0.2579 |
| grammar | holdout | 24 | 22 | 1 | 1 | 0.92 (0.74–0.98) | 1.44 | 0.21 | 1 | 0.2595 |

## Configurations

- `no_grammar`: planner=constraints, repair=True, plates=True, ocr=True, joint=True, learned=False, grammar=False — joint planning over the three hand-written families per orientation only
- `grammar`: planner=constraints, repair=True, plates=True, ocr=True, joint=True, learned=False, grammar=True — joint planning over hand-written + grammar-composed families (12 landscape / 24 portrait / 36 square candidates)

## Notes

- `no_ocr` cannot reach `accepted` by contract (a critical check did not run); it measures how much OCR costs, not quality.
- Synthetic fixtures; not customer validation.
- H1 protocol: `designer_ref` plans every size with the second-best hand-written family per orientation (a stand-in for a composition the planner would not pick on its own). `learned` sees exactly one designer_ref variant per aspect class as the approved example (the largest size of that class) and must reproduce the designer's plan on the other sizes of that class. `Agreement` is the mean IoU of planned element boxes against the designer plan on those held-out sizes; it is reported for every constraint-planner config, so `joint`/`full` give the no-example counterfactual and `designer_ref` the ceiling (1.0 by construction). `Mean s` for `learned` includes the setup (render example, match, infer) on the example sizes.
## Reading (2026-09-11)

- Same acceptance with and without the grammar (tuning 32/36, holdout 22/24), the same six
  non-accepted runs (all 300×250: three honest "CTA does not fit at 14 px" failures on the
  long-CTA cases, three OCR-doubt reviews on busy backgrounds), the same family-consistency
  issues and the same time per variant (≈1.4 s; the extra candidates cost planning time
  that stays below the render + verify cost).
- Grammar families won 6 of 60 runs (`g_square_stack_top40_tl_l` on the square size of the
  large-logo cases); everywhere else a hand-written family kept the best score (ties go to
  the hand-written family, which comes first).
- Conclusion: on this synthetic corpus the grammar adds no measurable acceptance; its value
  is coverage for creative directions (text right, centred, subject on top, larger or
  smaller text share) that the three hand-written families cannot honour, and a search
  space that stays judged by the same scoring and contract. Claims beyond this corpus are
  not made.
- Earlier in the same session the subject-integrity oracle was miscalibrated for flat
  assets (edge pixels dominated a Pearson correlation), which showed as 15/60 non-accepted
  runs in a first run of this ablation; the oracle now uses solid pixels and a mean colour
  difference (plus structure for textured assets), and the numbers above are from the
  corrected run. The first run is kept only in the session log.
