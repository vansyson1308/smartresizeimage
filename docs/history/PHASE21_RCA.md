# Phase 2.1 Benchmark RCA (0/36 blocker)

## Scope / constraints
- No product code changes were made in this triage.
- Benchmark was re-run deterministically with seed `42`.

## Commands re-run
```bash
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
```

Observed runner output included repeated:
- `Adaptive scoring fallback to rigid template due to heavy violations`
- `Benchmark completed. Report: backend/tests/fixtures/outputs/bench_phase21/report.md`

## Report summary
From `backend/tests/fixtures/outputs/bench_phase21/report.md` and `summary.json`:
- Phase 2.1 total runs: **36**
- Passed: **0**
- Pass rate: **0.0%**

## Top 3 failure causes by frequency (Phase 2.1 only)
Computed from `summary.json` (`mode == phase21`):
1. `outside_margin_ratio`: **36/36**
2. `total_score`: **33/36**
3. `min_font_size`: **20/36**

Concrete examples:
- `case_01_hero_headline_cta_logo` `1080x1080`
  - fail_reasons: `outside_margin_ratio,min_font_size,total_score`
  - violations: `min_size:headline,outside_margin:logo,max_size:logo,outside_margin:hero,min_size:sub,min_size:cta,overlap:headline:hero,overlap:headline:sub,overlap:hero:sub`
- `case_02_long_text` `1200x628`
  - fail_reasons: `overlap_area_ratio,outside_margin_ratio,total_score`
  - violations: `max_size:logo,outside_margin:hero,max_size:cta,overlap:headline:sub,overlap:headline:cta,overlap:logo:hero,overlap:sub:cta`
- `case_04_busy_bg` `1080x1080`
  - fail_reasons: `outside_margin_ratio,min_font_size,text_plate,total_score`
  - violations: `min_size:headline,outside_margin:logo,max_size:logo,outside_margin:hero,min_size:sub,min_size:cta,overlap:headline:hero,overlap:headline:sub,overlap:hero:sub`

## Two worst-run artifact inspections
Inspected:
- `backend/tests/fixtures/outputs/bench_phase21/case_01_hero_headline_cta_logo/1080x1080/layout_debug.json`
- `backend/tests/fixtures/outputs/bench_phase21/case_04_busy_bg/1080x1080/layout_debug.json`

Findings:
- Both runs clearly show `mode: phase21` and `profile: SQUARE`, so the benchmark runner is invoking Phase 2.1 mode.
- Both have identical bboxes/metrics except `text_plate_applied_when_busy` differs in busy-bg case, suggesting layout path parity and only plate behavior diverges.

## Did Phase 2.1 features actually run?
### Evidence they are wired
- Runner explicitly enables Phase 2.1 toggles in mode `phase21`:
  - `LAYOUT_PROFILE_SCORING_ENABLED = True`
  - `TEXT_SAFE_PLATE_ENABLED = True`
- It then calls `LayoutEngine.calculate_layout(...)` and evaluates via `evaluate_bench_run(...)`.

### Evidence output often collapses back to rigid template
- In `LayoutEngine.calculate_layout`, if `len(best_violations) >= 6`, it logs:
  - `Adaptive scoring fallback to rigid template due to heavy violations`
  - and returns `base_results` (rigid template path).
- Benchmark run emitted that fallback log repeatedly.
- In summary comparison, **32/36** case-size pairs have identical baseline vs phase21 metrics, consistent with fallback dominating outputs.

## Classification of root cause bucket
- **Most likely: C) Phase 2.1 logic producing invalid layouts that trigger hard fallback to rigid template, plus strict evaluator thresholds.**
  - Not B: runner wiring is present and mode/profile are set in artifacts.
  - Not primarily A: evaluator may be strict (`min_total_score=250`), but universal `outside_margin_ratio` and frequent min-size/overlap violations indicate substantive layout quality issues first.

## Most likely cause (short)
The adaptive candidate/solver path appears to generate high-violation layouts, then the guard `if len(best_violations) >= 6: return base_results` forces rigid-template fallback for most runs. Rigid outputs themselves still violate margin/font/overlap checks, yielding 0/36.

## Files/lines to inspect next
1. Fallback gate and candidate scoring:
   - `backend/app/layout/engine.py` around solver loop and fallback threshold.
2. Benchmark threshold strictness and fail reasons:
   - `backend/app/layout/bench_thresholds.py`
   - `backend/app/layout/bench_metrics.py`
3. Runner config/wiring and debug payload completeness:
   - `backend/tools/run_layout_bench.py`
4. Text-plate misses on busy fixtures:
   - `backend/app/composition/engine.py` (text box extraction + plate invocation)
   - `backend/app/generative/text_plate.py` (busy score / thresholds)

## Proposed fix plan (small PRs)
### PR-1: "Unblock adaptive path + improve observability"
- Add richer `layout_debug.json` metadata for phase21 runs:
  - candidate count, chosen candidate index, solver iterations, fallback reason, whether returned baseline.
- Relax/replace hard `>=6 violations => fallback` gate with score-delta-based fallback, so adaptive output can survive when it improves over baseline.
- Acceptance criteria:
  - Benchmark rerun still deterministic.
  - At least one phase21 run differs from baseline due to adaptive keep-path (verified via summary debug metadata).
  - Pass rate increases from `0/36` to `>0/36`.

### PR-2: "Target top violations"
- Tune profile/template zones and typography bounds to reduce:
  - `outside_margin:hero/logo`
  - `min_size:headline/sub/cta`
  - key overlap pairs (`headline:sub`, `headline:hero`).
- Revisit benchmark thresholds only if still clearly misaligned after geometric/layout fixes.
- Acceptance criteria:
  - `outside_margin_ratio` failures < 36/36.
  - `min_font_size` failures < 20/36.
  - Overall pass rate strictly > 0/36 on seed 42 pack.
