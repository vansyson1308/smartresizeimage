# Design pipeline ablations

Contract v2.0.0. Same fixtures, seeds and sizes for every configuration; holdout cases were never used for tuning. Wilson 95% intervals on acceptance.

- environment: `{"numpy": "1.26.4", "ocr_engine": "tesseract", "ocr_version": "5.3.4", "pillow": "10.4.0", "python": "3.11.15"}`
- git commit: `23143383727cc5bd0e48a9179409260f978c4026`
- sizes: 1200x628, 1500x500, 300x250, 1080x1080, 600x600, 1080x1920, 1080x1350; cases: 15 (holdout from case 10)

| Config | Split | Runs | Accepted | Needs review | Failed | Acceptance (95% CI) | Mean s | Repair steps | Family-issue runs | Agreement w/ designer (held-out sizes) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| joint | tuning | 63 | 59 | 2 | 2 | 0.94 (0.85–0.97) | 1.07 | 0.22 | 4 | 0.2374 |
| joint | holdout | 42 | 40 | 1 | 1 | 0.95 (0.84–0.99) | 1.12 | 0.24 | 2 | 0.238 |
| designer_ref | tuning | 63 | 59 | 2 | 2 | 0.94 (0.85–0.97) | 1.11 | 0.22 | 0 | 0.9945 |
| designer_ref | holdout | 42 | 40 | 1 | 1 | 0.95 (0.84–0.99) | 1.11 | 0.24 | 0 | 0.9941 |
| learned | tuning | 63 | 62 | 1 | 0 | 0.98 (0.92–1.00) | 2.70 | 0.22 | 0 | 0.647 |
| learned | holdout | 42 | 42 | 0 | 0 | 1.00 (0.92–1.00) | 2.71 | 0.24 | 0 | 0.6584 |

## Configurations

- `joint`: planner=constraints, repair=True, plates=True, ocr=True, joint=True, learned=False — one family per orientation chosen jointly for the size set
- `designer_ref`: planner=constraints, repair=True, plates=True, ocr=True, joint=True, learned=False — reference: the designer's (second-best) family applied directly to all sizes
- `learned`: planner=constraints, repair=True, plates=True, ocr=True, joint=True, learned=True — H1: families inferred from one approved example per orientation (the designer's example uses the second-best hand-written family)

## Notes

- `no_ocr` cannot reach `accepted` by contract (a critical check did not run); it measures how much OCR costs, not quality.
- Synthetic fixtures; not customer validation.
- H1 protocol: `designer_ref` plans every size with the second-best hand-written family per orientation (a stand-in for a composition the planner would not pick on its own). `learned` sees exactly one designer_ref variant per aspect class as the approved example (the largest size of that class) and must reproduce the designer's plan on the other sizes of that class. `Agreement` is the mean IoU of planned element boxes against the designer plan on those held-out sizes; it is reported for every constraint-planner config, so `joint`/`full` give the no-example counterfactual and `designer_ref` the ceiling (1.0 by construction). `Mean s` for `learned` includes the setup (render example, match, infer) on the example sizes.