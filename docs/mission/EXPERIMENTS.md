# Experiment ledger

Each entry: hypothesis, closest prior art, implementation, frozen evaluation, budget,
expected gain, decision rule, outcome. Research must not destabilize the product: everything
lands behind flags with a rollback path.

| ID | Hypothesis | Prior art | Status | Budget | Decision rule |
|---|---|---|---|---|---|
| H1 | A few approved variants yield reusable adaptation rules with less setup than manual responsive templates. | PosterO (example-conditioned layouts), CHILI/Celtra template rules | ABLATED, ADOPTED (proxy): one approved example per orientation lifts agreement with the designer's plan on held-out sizes 0.24 → 0.85 and cuts held-out sizes needing a re-layout 60/60 → 6/60 (−90%) at +3.7 s setup per case; human correction counts still unmeasured (below) | CPU only | Adopt if held-out sizes need >= 30% fewer corrections vs one-master, counting import/match/approve time. |
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
- 2026-09-09 — H1 learning from approved examples (`results/ablations_h1_2026-09-09.md`;
  15 cases × 7 sizes, tuning cases 1–9, frozen holdout 10–15; run from the working tree
  later committed as `7ffbd5d`). Protocol: the "designer" composition is the second-best
  hand-written family per orientation (one the planner would not pick on its own); the
  largest size of each orientation (1500×500, 1080×1080, 1080×1920) rendered with it is the
  single approved example; the other four sizes (1200×628, 300×250, 600×600, 1080×1350) are
  held out. `agreement` = mean IoU of planned element boxes against the designer plan on the
  held-out sizes; a held-out size with agreement < 0.5 counts as "needs a re-layout" (the
  correction proxy).

  | Config | Agreement, tuning | Agreement, holdout | Held-out sizes with agreement < 0.5 | Accepted (tuning / holdout) | Family-issue runs | Mean s on example sizes |
  |---|---:|---:|---:|---:|---:|---:|
  | joint (no example) | 0.24 | 0.24 | 60/60 | 59/63 / 40/42 | 6 | 1.52 |
  | learned (one example per orientation) | 0.65 | 0.66 | 18/60 (tuning 11/36, holdout 7/24) | 62/63 / 42/42 | 0 | 5.29 |
  | designer_ref (ceiling) | 0.99 | 0.99 | 0/60 | 59/63 / 40/42 | 0 | 1.57 |

  Per orientation (learned vs joint): landscape 0.85 vs 0.17, square 0.58 vs 0.25, portrait
  0.59 vs 0.27. Reading: one example is enough to make the planner follow the designer's
  side-by-side composition almost exactly; stacked compositions (square/portrait) transfer
  only partially because the example's union regions do not carry content pressure to a
  smaller canvas. Setup cost is ≈3.8 s per case for three orientations (render example ≈1.2 s
  each; matching and inference < 0.1 s); held-out sizes cost the same 0.77 s as before. The
  higher acceptance of `learned` (the three `large_logo_long_cta` 300×250 failures disappear)
  is a side effect of the different composition, not evidence of better layouts. Decision
  rule met on the proxy (−70% ≥ −30%), so learning is on by default in the product (approved
  variants of a project are the examples; `GET /api/projects/{id}/learned` shows the rules).
  Not measured: human correction counts, real designer examples with manual edits (matching
  here is trivial because example and master share assets and copy), more than one example
  per orientation (confidence stays 0.5 with one).
- 2026-09-09 — H1 follow-up, slot expansion (`results/ablations_h1_slots_2026-09-09.md`; same
  protocol, corpus and sizes; run from the working tree committed as `d76fa00`). Tight
  unions from one example shrank text columns and logo slots on other sizes of the same
  orientation. Learned text columns and logo slots now expand symmetrically into the free
  space around the example's elements (up to neighbours and canvas margins; a left-aligned
  column mirrors its left margin); the subject keeps its tight region so it cannot grow past
  the example.

  | Config | Agreement, tuning | Agreement, holdout | Held-out sizes with agreement < 0.5 | Accepted (tuning / holdout) | Mean s on example sizes |
  |---|---:|---:|---:|---:|---:|
  | joint (no example) | 0.24 | 0.24 | 60/60 | 59/63 / 40/42 | 1.56 |
  | learned, tight regions (previous entry) | 0.65 | 0.66 | 18/60 | 62/63 / 42/42 | 5.29 |
  | learned, slots | 0.85 | 0.87 | 6/60 (tuning 4, holdout 2) | 61/63 / 41/42 | 5.22 |
  | designer_ref (ceiling) | 0.99 | 0.99 | 0/60 | 59/63 / 40/42 | 1.57 |

  Per orientation (slots vs tight): landscape 0.88 vs 0.85, square 0.80 vs 0.58, portrait
  0.95 vs 0.59; median agreement 0.945. The six remaining low-agreement runs are the three
  `long_text` cases at 600×600 and 300×250, where the learned square family scored more
  than the learned bonus below `square_text_top` (its long copy overflows the learned
  column) and the planner fell back to the hand-written family — the intended behaviour
  when following the example would produce a worse layout. The two `learned` runs that are
  not accepted are `busy_bg` 300×250 reviews (OCR agreement 0%), the same honest reviews as
  every other config. Decision rule met on the proxy (−90% ≥ −30%); correction counts with
  humans remain unmeasured.
