# Design pipeline ablations

Contract v2.0.0. Same fixtures, seeds and sizes for every configuration; holdout cases were never used for tuning. Wilson 95% intervals on acceptance.

- environment: `{"numpy": "1.26.4", "ocr_engine": "tesseract", "ocr_version": "5.3.4", "pillow": "10.4.0", "python": "3.11.15"}`
- git commit: `6dded998824f48a93291010a9d67a43ca761511d`
- sizes: 1200x628, 1080x1080, 1080x1920, 300x250; cases: 12 (holdout from case 10)

| Config | Split | Runs | Accepted | Needs review | Failed | Acceptance (95% CI) | Mean s | Repair steps |
|---|---|---:|---:|---:|---:|---|---:|---:|
| full | tuning | 36 | 32 | 2 | 2 | 0.89 (0.75–0.96) | 1.31 | 0.19 |
| full | holdout | 12 | 12 | 0 | 0 | 1.00 (0.76–1.00) | 1.27 | 0.17 |
| zones | tuning | 36 | 27 | 0 | 9 | 0.75 (0.59–0.86) | 2.56 | 0.86 |
| zones | holdout | 12 | 9 | 0 | 3 | 0.75 (0.47–0.91) | 2.75 | 0.75 |
| no_repair | tuning | 36 | 32 | 2 | 2 | 0.89 (0.75–0.96) | 1.29 | 0.19 |
| no_repair | holdout | 12 | 12 | 0 | 0 | 1.00 (0.76–1.00) | 1.18 | 0.17 |
| no_plates | tuning | 36 | 26 | 8 | 2 | 0.72 (0.56–0.84) | 1.31 | 0.19 |
| no_plates | holdout | 12 | 12 | 0 | 0 | 1.00 (0.76–1.00) | 1.31 | 0.17 |
| zones_no_repair | tuning | 36 | 23 | 0 | 13 | 0.64 (0.48–0.78) | 2.41 | 0.64 |
| zones_no_repair | holdout | 12 | 8 | 0 | 4 | 0.67 (0.39–0.86) | 2.53 | 0.58 |

## Configurations

- `full`: planner=constraints, repair=True, plates=True, ocr=True — production defaults
- `zones`: planner=zones, repair=True, plates=True, ocr=True — legacy zone-template planner
- `no_repair`: planner=constraints, repair=False, plates=True, ocr=True — verify only, no repair loop
- `no_plates`: planner=constraints, repair=True, plates=False, ocr=True — text-safe plates disabled
- `zones_no_repair`: planner=zones, repair=False, plates=True, ocr=True — zone planner without repair

## Notes

- `no_ocr` cannot reach `accepted` by contract (a critical check did not run); it measures how much OCR costs, not quality.
- Synthetic fixtures; not customer validation.