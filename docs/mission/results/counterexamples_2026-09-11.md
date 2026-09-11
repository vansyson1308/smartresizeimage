# Counterexample search (2026-09-11)

Command: `.venv/bin/python backend/tools/find_counterexamples.py --cases 15 --ocr`
(15 synthetic cases × 7 perturbations — none, long copy, short copy, hidden subject, hard
order rule, hard clear-space rule, random creative direction — × 4 sizes = 420 renders,
seed 42, OCR on; 826 s). The code under test is the phase C tree before the repair guard
(`ff28129`); the guard added afterwards is measured separately below.

Independent oracles (no code shared with the checks they audit): rendered-mask overlap
between content elements (> 5 % of the smaller mask, no allowed_overlap rule), canvas
bounds (> 1 px outside), hard `order_below` on the final boxes, hard `clear_space` on the
final boxes, subject correlation with the master asset.

## Result

- verdicts: 355 accepted, 23 needs_review, 42 failed
- runs where an oracle fired: 20 — bounds 15, order 9, clear 3 (some runs fire more than one)
- **counterexamples (verdict accepted while an oracle fired): 0**
- possible false failures (failed, no oracle, no text-fit or glyph-coverage issue): 0
- every oracle firing coincided with a non-accepted verdict: the contract and the oracles
  disagree on nothing in this sweep

Runs where an oracle fired, by perturbation:

- `long_copy`: 3 run(s) — case_02_long_text 300x250 [failed] bounds:headline, bounds:sub, bounds:cta; case_07_long_text 300x250 [failed] bounds:sub, bounds:cta; case_12_long_text 300x250 [failed] bounds:headline, bounds:sub, bounds:cta
- `hard_order`: 13 run(s) — case_02_long_text 1080x1080 [failed] order:sub>cta; case_04_busy_bg 1080x1920 [failed] order:sub>cta; case_04_busy_bg 300x250 [failed] order:sub>cta; case_05_offcenter_hero 1080x1920 [failed] order:sub>cta; case_06_hero_headline_cta_logo 1080x1920 [failed] order:sub>cta; case_07_long_text 1200x628 [failed] bounds:headline
- `hard_clear`: 3 run(s) — case_02_long_text 300x250 [failed] clear:logo~headline; case_07_long_text 300x250 [failed] clear:logo~headline; case_12_long_text 300x250 [failed] clear:logo~headline
- `direction`: 1 run(s) — case_07_long_text 300x250 [failed] bounds:sub, bounds:cta

## Reading

- The 15 `bounds` firings come from long copy, hard order rules and one direction on small
  formats (mostly 300×250): when the text stack cannot fit its column even at the planner's
  smallest scale factor, the last elements extend past the canvas; the contract reports
  them (`inside_canvas` / `element_visible` fail) and never accepts them. A planner
  improvement (clip the stack to the canvas and report the overflow, or drop to the copy's
  minimum size earlier) would remove the cause; recorded as a follow-up, not fixed here.
- The 9 `order` firings are hard `order_below` rules the final boxes break: the planner
  reordered the elements to honour the rule and the overlap repair then moved one of them
  back. The contract failed every one of them (finding F2 holds). Because a repair should not
  create the violation in the first place, repair moves that would break a hard order rule
  are now undone (`acfeebd`); the targeted re-run of the `hard_order` perturbation with the
  guard is reported below.
- The 3 `clear` firings are hard clear-space rules (ratio 1.0) the layout cannot satisfy on
  small formats; the contract fails them.
- A first sweep with OCR off could not produce accepted verdicts at all (legibility
  `not_checked` is critical by contract) and read `order_below` the wrong way round; it is
  superseded by this one, and the oracle's semantics are pinned by
  `test_counterexample_tool.py`.

## Re-run of the `hard_order` perturbation with the repair guard (`acfeebd`)

`--perturbations hard_order --ocr`, 60 renders (15 cases × 4 sizes), 115 s: **order oracle
firings 9 → 0**; 53 accepted, 3 needs_review, 4 failed. The four failures are the contract's
honest calls: two "CTA does not fit at 14 px" on 300×250, one 300×250 where the headline
and subheadline leave the canvas (bounds oracle agrees, `inside_canvas` fails), and one
1080×1080 where the CTA still overlaps the subheadline (`text_overlap` fails) because the
only repair move that would have cleared it was the one that broke the hard order rule —
the layout is now caught between two rules and is handed to a human instead of quietly
breaking either. The mask-overlap oracle stays silent there (overlap under its 5 % floor),
which is the one remaining disagreement, in the safe direction.

## Re-run of `long_copy`, `hard_order` and `direction` with the proportional out-of-canvas penalty (`9b6e0c8`)

The bounds firings came from text stacks planned into narrow columns on small formats
while every box outside the canvas cost the same flat 25 points. The penalty now grows
with the share of the box outside, so the family that keeps the most copy on the canvas
wins. Same 180 renders (15 cases × 3 perturbations × 4 sizes, OCR on), before → after:

| | accepted | needs_review | failed | bounds firings (runs) | counterexamples |
|---|---:|---:|---:|---:|---:|
| before | 145 | 5 | 30 | 15 (8) | 0 |
| after | 153 | 7 | 20 | 10 (6) | 0 |

The six remaining bounds runs are the two long-copy fixtures with the extra 13 words on
300×250 (and one 1200×628 hard-order case): the copy does not fit at the minimum size in
any family and the contract fails them for the right reason. The grammar ablation re-run
after the change is unchanged (32/36 tuning, 22/24 holdout, same six non-acceptances).

Raw runs: `counterexamples_2026-09-11.json` (summary); per-run records are in the tool's
`runs.json` output. Synthetic fixtures; not customer validation.
