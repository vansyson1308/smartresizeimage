# Rules from corrections (H5): repeat rejections with and without brand memory

Contract v2.0.0. Reviewer = rule (logo height < 0.09 of canvas height → 'logo too small'; CTA font < 0.032 of canvas height → 'CTA hard to read'). Campaigns = fixture cases in order, one synthetic brand; the designer's fix after a rejection is the matching constraint; derived rules are carried by role into later campaigns when memory is on.

- environment: `{"numpy": "1.26.4", "ocr_engine": "tesseract", "ocr_version": "5.3.4", "pillow": "10.4.0", "python": "3.11.15"}`
- git commit: `816677273a52a96ae39c20adbe7e956f359ec0b7`
- sizes: 1200x628, 1080x1080, 300x250; campaigns: 15

| Memory | Variants | Rejections | Repeat rejections | By reason | Mean s |
|---|---:|---:|---:|---|---:|
| memory_off | 45 | 30 | 28 | logo too small: 30 | 0.50 |
| memory_on | 45 | 3 | 0 | CTA hard to read: 1, logo too small: 2 | 0.51 |

## Per campaign

| Campaign | Rejections (memory off) | Rejections (memory on) | Rules carried |
|---|---:|---:|---:|
| case_01_hero_headline_cta_logo | 2 | 2 | 0 |
| case_02_long_text | 2 | 0 | 1 |
| case_03_large_logo_long_cta | 2 | 1 | 1 |
| case_04_busy_bg | 2 | 0 | 2 |
| case_05_offcenter_hero | 2 | 0 | 2 |
| case_06_hero_headline_cta_logo | 2 | 0 | 2 |
| case_07_long_text | 2 | 0 | 2 |
| case_08_large_logo_long_cta | 2 | 0 | 2 |
| case_09_busy_bg | 2 | 0 | 2 |
| case_10_offcenter_hero | 2 | 0 | 2 |
| case_11_hero_headline_cta_logo | 2 | 0 | 2 |
| case_12_long_text | 2 | 0 | 2 |
| case_13_large_logo_long_cta | 2 | 0 | 2 |
| case_14_busy_bg | 2 | 0 | 2 |
| case_15_offcenter_hero | 2 | 0 | 2 |

## Notes

- The reviewer is a deterministic rule, so 'repeat rejections' measures whether the mechanism closes the loop, not whether the rules match a real brand's taste.
- Rules are carried by role across campaigns here; the product stores corrections per project and applies a rule only after a person adds it.
- OCR is off; verdicts are not part of this measurement.