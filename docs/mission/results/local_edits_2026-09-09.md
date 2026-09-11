# Local edits (H3): out-of-scope change on campaign revisions

Contract v2.0.0. Same fixtures and sizes for both modes; `reference` keeps the base variant's plan, `replan` plans from scratch. Scope = the edited element's box before and after, padded by 36px.

- environment: `{"numpy": "1.26.4", "ocr_engine": "tesseract", "ocr_version": "5.3.4", "pillow": "10.4.0", "python": "3.11.15"}`
- git commit: `25c794cb95acac3be4e7508fec95fb698e0da941`
- sizes: 1200x628, 1080x1080, 1080x1920; cases: 15

| Mode | Runs | Zero out-of-scope diff | Mean out-of-scope diff | Runs with moved elements | Failed | Re-planned (copy no longer fit) | Mean s |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 180 | 174 | 0.0007 | 0 | 0 | 0 | 1.02 |
| replan | 180 | 156 | 0.0033 | 21 | 0 | 0 | 1.02 |

## By edit

| Mode | Edit | Runs | Zero out-of-scope diff | Runs with moved elements |
|---|---|---:|---:|---:|
| reference | cta_longer | 45 | 39 | 0 |
| reference | headline_copy | 45 | 45 | 0 |
| reference | logo_swap | 45 | 45 | 0 |
| reference | sub_colour | 45 | 45 | 0 |
| replan | cta_longer | 45 | 45 | 0 |
| replan | headline_copy | 45 | 21 | 21 |
| replan | logo_swap | 45 | 45 | 0 |
| replan | sub_colour | 45 | 45 | 0 |

## Notes

- Out-of-scope diff is a pixel measurement on synthetic fixtures; it says whether a revision touched anything but the edited element, not whether the result is good.
- `Re-planned` counts revisions where the new copy did not fit its previous box and the element fell back to a fresh plan (reported as `layout_change`).
- OCR is off in this measurement (pixels decide locality), so verdicts are `needs_review` by contract unless a structural check fails; only `Failed` is reported.
