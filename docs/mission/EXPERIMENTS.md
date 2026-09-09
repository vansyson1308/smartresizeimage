# Experiment ledger

Each entry: hypothesis, closest prior art, implementation, frozen evaluation, budget,
expected gain, decision rule, outcome. Research must not destabilize the product: everything
lands behind flags with a rollback path.

| ID | Hypothesis | Prior art | Status | Budget | Decision rule |
|---|---|---|---|---|---|
| H1 | A few approved variants yield reusable adaptation rules with less setup than manual responsive templates. | PosterO (example-conditioned layouts), CHILI/Celtra template rules | PLANNED (needs multi-variant fixtures + IR) | CPU only | Adopt if held-out sizes need >= 30% fewer corrections vs one-master, counting import/match/approve time. |
| H2 | Joint planning across a variant family improves consistency and reduces corrections vs independent resizing. | DesignAsCode retargeting, iPoster constraints | ABLATED, ADOPTED: joint family choice cuts family-consistency issues 9 → 3 runs at equal acceptance and compute (below); correction-time effect unmeasured (no humans) | CPU only | Adopt if family-consistency errors drop without lowering acceptance; equal compute/review budget. |
| H3 | A local-edit representation reduces unintended changes on campaign revision. | Layered editing (Qwen-Image-Layered) | PLANNED | CPU only | Adopt if pixel diff outside edited scope is zero on the regression set. |
| H4 | Calibrated verification + targeted repair improves throughput vs missed critical errors. | Verification-driven repair | ABLATED (below): repair helps a weak planner, adds nothing to the constraint planner on this corpus; calibration vs humans still unmeasured | CPU only | Measure missed-error rate on held-out set before/after repair; report sample size. |
| H5 | Retrieval from approved correction history reduces recurring mistakes per brand without training. | RAG-style retrieval | PLANNED (needs correction store) | CPU only | Adopt if repeat-correction rate drops on the same brand's held-out campaigns. |

## Results log

- 2026-09-09 — H4 (verification): contract v2 rejects the preserved false positive for the
  right reason (`element_visible:headline` 99% covered, OCR agreement 0%). Calibration of the
  verdict against human judgement is NOT measured yet (no human reviewers available).
- 2026-09-09 — Ablations (`backend/tools/run_ablations.py`, 12 synthetic cases × 4 sizes incl.
  300×250, seed 42, tuning = cases 1–9, frozen holdout = cases 10–12; full table in
  `results/ablations_2026-09-09.md`):

  | Config | Tuning accepted | Holdout accepted | Mean s |
  |---|---:|---:|---:|
  | full (constraints + repair + plates) | 32/36 (0.89, CI 0.75–0.96) | 12/12 (1.00, CI 0.76–1.00) | 1.31 |
  | zones planner | 27/36 (0.75) | 9/12 (0.75) | 2.56 |
  | no repair | 32/36 (0.89) | 12/12 | 1.29 |
  | no plates | 26/36 (0.72) | 12/12 | 1.31 |
  | zones, no repair | 23/36 (0.64) | 8/12 (0.67) | 2.41 |

  Reading: the constraint planner is the main driver (+14 pp tuning, +25 pp holdout over
  zones, at half the time); repair adds +11 pp to the zone planner but nothing to the
  constraint planner here; contrast-aware plates add +17 pp on tuning (busy backgrounds) and
  nothing on the holdout, which contains no busy-background case — a coverage gap to close
  when the corpus grows. The 4 non-accepted `full` runs are all 300×250: two honest
  "CTA does not fit at 14px" failures and two "hard to read" reviews. Intervals are wide
  (n=36/12); no claim beyond this synthetic corpus. `repair_steps` includes group compaction
  applied before the repair loop, so it is non-zero even with repair off.
- 2026-09-09 — H2 joint family planning (`results/ablations_joint_2026-09-09.md`; corpus
  grown to 15 cases so the holdout (cases 10–15, 24 runs) includes a busy background;
  4 sizes per case incl. 300×250):

  | Config | Tuning accepted | Holdout accepted | Runs with family issues | Orientations with >1 family |
  |---|---:|---:|---:|---:|
  | full (independent per size) | 32/36 | 22/24 | 9/60 | 3/45 |
  | joint (one family per orientation) | 32/36 | 22/24 | 3/60 | 0/45 |

  Joint choice removes all mixed-family orientations and cuts family-consistency issues by
  two thirds at identical acceptance and compute (1.2 s/run). The 3 remaining issues are
  `large_logo_long_cta` cases where the long CTA is clamped to its minimum size at 300×250,
  so the headline:CTA ratio drifts beyond 30% — an honest hierarchy warning, not a planner
  bug. Consistency is measured on plans (reading order, ratios, identity, family), not on
  human judgement; correction-time impact remains unmeasured. Adopted as the default for
  multi-size jobs (`ProjectService.request_variants`).
