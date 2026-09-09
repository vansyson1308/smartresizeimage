# Design pipeline ablations

Contract v2.0.0. Same fixtures, seeds and sizes for every configuration; holdout cases were never used for tuning. Wilson 95% intervals on acceptance.

- environment: `{"numpy": "1.26.4", "ocr_engine": "tesseract", "ocr_version": "5.3.4", "pillow": "10.4.0", "python": "3.11.15"}`
- git commit: `62ee670cf202b574c297a1a3306b8ce884b38147`
- sizes: 1200x628, 1080x1080, 1080x1920, 300x250; cases: 15 (holdout from case 10)

| Config | Split | Runs | Accepted | Needs review | Failed | Acceptance (95% CI) | Mean s | Repair steps | Family-issue runs |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| full | tuning | 36 | 32 | 2 | 2 | 0.89 (0.75–0.96) | 1.24 | 0.19 | 6 |
| full | holdout | 24 | 22 | 1 | 1 | 0.92 (0.74–0.98) | 1.28 | 0.21 | 3 |
| joint | tuning | 36 | 32 | 2 | 2 | 0.89 (0.75–0.96) | 1.25 | 0.19 | 2 |
| joint | holdout | 24 | 22 | 1 | 1 | 0.92 (0.74–0.98) | 1.11 | 0.21 | 1 |

## Configurations

- `full`: planner=constraints, repair=True, plates=True, ocr=True, joint=False — production defaults
- `joint`: planner=constraints, repair=True, plates=True, ocr=True, joint=True — one family per orientation chosen jointly for the size set

## Notes

- `no_ocr` cannot reach `accepted` by contract (a critical check did not run); it measures how much OCR costs, not quality.
- Synthetic fixtures; not customer validation.