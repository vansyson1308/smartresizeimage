# Experiment ledger

Each entry: hypothesis, closest prior art, implementation, frozen evaluation, budget,
expected gain, decision rule, outcome. Research must not destabilize the product: everything
lands behind flags with a rollback path.

| ID | Hypothesis | Prior art | Status | Budget | Decision rule |
|---|---|---|---|---|---|
| H1 | A few approved variants yield reusable adaptation rules with less setup than manual responsive templates. | PosterO (example-conditioned layouts), CHILI/Celtra template rules | PLANNED (needs multi-variant fixtures + IR) | CPU only | Adopt if held-out sizes need >= 30% fewer corrections vs one-master, counting import/match/approve time. |
| H2 | Joint planning across a variant family improves consistency and reduces corrections vs independent resizing. | DesignAsCode retargeting, iPoster constraints | PLANNED (needs IR + planner) | CPU only | Adopt if family-consistency errors drop without lowering acceptance; equal compute/review budget. |
| H3 | A local-edit representation reduces unintended changes on campaign revision. | Layered editing (Qwen-Image-Layered) | PLANNED | CPU only | Adopt if pixel diff outside edited scope is zero on the regression set. |
| H4 | Calibrated verification + targeted repair improves throughput vs missed critical errors. | Verification-driven repair | IN PROGRESS: verification layer shipped (contract v2); repair loop pending | CPU only | Measure missed-error rate on held-out set before/after repair; report sample size. |
| H5 | Retrieval from approved correction history reduces recurring mistakes per brand without training. | RAG-style retrieval | PLANNED (needs correction store) | CPU only | Adopt if repeat-correction rate drops on the same brand's held-out campaigns. |

## Results log

- 2026-09-09 — H4 (verification): contract v2 rejects the preserved false positive for the
  right reason (`element_visible:headline` 99% covered, OCR agreement 0%). Calibration of the
  verdict against human judgement is NOT measured yet (no human reviewers available).
