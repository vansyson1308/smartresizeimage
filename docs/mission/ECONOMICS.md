# Unit economics (measured inputs, explicit scenarios)

Everything below separates **measured** numbers (this environment, synthetic fixtures) from
**assumptions**. No price here is evidence of willingness to pay.

## Measured (2026-09-09, 4 vCPU Intel Xeon 2.1 GHz, no GPU, tesseract 5.3.4)

| Quantity | Value | Source |
|---|---|---|
| Compute per accepted variant (design pipeline, incl. OCR verification) | mean 1.5 s wall, ≈1.5 CPU-s (single-threaded) | `run_layout_bench.py --mode design`, 36 runs |
| Compute per variant, legacy phase21 path | mean 3.2 s | same benchmark |
| Compute per variant, Phase 3 procedural redesign (8 candidates) | ≈43 s under contention (≈15–20 s uncontended, not re-measured) | phase3 benchmark |
| Flat-image decomposition (1200×628, OCR + GrabCut + inpaint) | ≈1.3 s | `test_decompose` smoke |
| Storage per project after a 4-variant journey (synthetic) | ≈350 KB (variants 244 KB, history 94 KB, assets 6 KB) | Playwright journey data dir |
| Storage per variant PNG (synthetic flat art, 1080×1920) | ≈60 KB | same |
| Peak RSS of the API process during generation | not instrumented yet | — |

Real photography will dominate storage: a 3 MB source and 1–2 MB per variant PNG are typical,
so budget ≈10–20 MB per photographic project with 8 variants (assumption, not measured).

## Cost scenarios (assumptions, to be replaced by invoices)

Assume a 4-vCPU cloud VM at USD 0.15/hour (assumption) fully busy ≈ 2,400 variants/hour
at 1.5 CPU-s each on one core, ≈ 9,600/hour on four cores → compute ≈ USD 0.00002 per
variant; even at 10× overhead (retries, OCR timeouts, previews) compute stays below USD
0.001 per variant. Storage at USD 0.02/GB-month (assumption): 20 MB per project ≈ USD 0.0004
per project-month. Generation compute and storage are therefore not the binding cost;
support, review time, and any future model-provider calls are.

| Scenario | Customers | Price/month (hypothesis) | ARR (before churn, discounts, tax) |
|---|---:|---:|---:|
| Agency tier | 200 | USD 499 | USD 1,197,600 |
| Mixed | 300 team @ 199 + 100 agency @ 499 | — | USD 1,315,200 |
| Usage-based add-on | 150 @ 299 + variants over 2,000/month at USD 0.05 | — | ≥ USD 538,200 + usage |

These are arithmetic scenarios, not forecasts. Failed generations cost compute only (no
provider calls today); retries are bounded (3 repair rounds) and metered per owner.

## What would change the picture

- A generative provider for background expansion (per-call cost, latency, failure rate) —
  currently `BLOCKED_EXTERNAL`; the adapter records provider/cost fields once wired.
- Human review time: the pilot instruments (`/api/pilot/summary`) measure time-to-decision
  and first-pass acceptance; operator time per accepted variant is the number to publish.
- Support and onboarding effort for PSDs with unsupported effects.

## Proposed limits until costs are measured on real work

- Free/trial: 3 projects, 50 variants/day (`AUTOBANNER_QUOTA_*`).
- Team: 50 projects, 2,000 variants/month, 5 GB.
- Agency: 500 projects, 20,000 variants/month, 50 GB, API access.
