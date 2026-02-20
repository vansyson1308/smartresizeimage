# POST-PUBLIC AUDIT

## 1) Executive Summary

### Baseline selection
- No release tags exist, so I selected commit **`4407d4b`** as BASE because it is the commit immediately preceding the public-readiness/doc-polish sequence (`44292c2`, merged in PR #1). Evidence from git history is included in section 2.

### What materially improved (evidence-backed)
1. **Headless reliability improved substantially vs BASE**:
   - At BASE (`4407d4b`), `pytest -q` fails during collection with `ImportError: libGL.so.1` from eager `cv2` import.
   - At HEAD, full test suite passes (`161 passed`) in the same environment.
   - This aligns with code changes introducing lazy OpenCV loading and explicit fallback paths in composition background extension.
2. **Quality gate coverage expanded significantly**:
   - Diffstat shows broad additions across `backend/tests` and benchmark tooling (`backend/tools/*`).
   - CI lint scope expanded from only `backend/app/` to `backend/app backend/tests backend/tools`.

### What did not improve / regressions / gaps
1. **Benchmark quality target currently failing**:
   - Phase 2.1 benchmark report shows **0/36 pass** (0.0% pass rate), with recurring violations: `outside_margin_ratio`, `min_font_size`, overlaps, and score threshold failures.
   - This directly weakens the “designer-like relayout” claim for production readiness.
2. **No before/after benchmark delta is available**:
   - BASE has no benchmark harness scripts (`backend/tools/run_layout_bench.py`, `generate_bench_fixtures.py` absent), so only absolute current metrics can be reported.
3. **Headless fallback works, but quality impact likely remains**:
   - Runtime logs repeatedly show fallback due to unavailable OpenCV/libGL; system remains operational, but image-quality behavior under fallback should be explicitly productized/tested for parity.

## 2) Commands executed + key outputs

### Phase 0: Baseline determination
- `git tag --list`
  - Output: no tags.
- `git log --oneline --decorate --graph -n 80`
- `git log --oneline --grep="Phase" -n 50`
- `git log --oneline --grep="public-ready" -n 50`
  - Found `44292c2 chore: public-ready docs and remove binary benchmark artifacts`.
  - Selected BASE = `4407d4b` (immediately prior commit).

### Phase 1: Objective diff
- `git diff --stat 4407d4b..HEAD`
  - Output summary: 75 files changed, ~6500 insertions, 186 deletions.
- `git diff 4407d4b..HEAD -- .github/workflows backend/app backend/tests README.md .gitignore`
  - Confirms large additions in layout/composition/generative pipelines, tests, benchmark tools, docs, and CI lint scope.

### Phase 2: Quality gates
- `ruff check .`
  - Output: `All checks passed!`
- `pytest -q`
  - Output: `161 passed in 20.32s`
- Determinism sanity (same pipeline run twice, same input/size)
  - Python script executed two runs; SHA256 hashes identical:
    - `20f730767b9954453b6fd5537f01004a07eb6f68b4b33c2ca70256d6db6d870f`
    - `identical True`
- Headless/optional-deps robustness
  - `pytest -q backend/tests/test_composition_headless.py` passed.
  - Runtime logs from deterministic run showed fallback rather than crash:
    - `cv2 unavailable -> fallback edge-repeat fit path: libGL.so.1 ...`
    - `Inpaint extend failed: cv2 unavailable, using edge repeat`

### Baseline reproducibility checks (for comparison)
Executed in a detached worktree at BASE (`/tmp/smartresize_base`):
- `ruff check .`
  - Fails with multiple lint issues in tests (unused imports, import sorting).
- `pytest -q`
  - Fails at collection with `ImportError: libGL.so.1` from `cv2` import chain.

### Phase 3: Benchmark run
- `python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42`
  - Output: `Generated 12 fixtures in backend/tests/bench_fixtures`
- `python backend/tools/run_layout_bench.py --mode both --seed 42`
  - Output completed with report path:
    - `Benchmark completed. Report: backend/tests/fixtures/outputs/bench_phase21/report.md`

### Phase 4: Runtime smoke workflow
- Scripted relayout run:
  - Input: `backend/tests/bench_fixtures/case_01_hero_headline_cta_logo/input.png`
  - Generated:
    - `backend/tests/fixtures/outputs/smoke_phase4/output_1200x628.png`
    - `backend/tests/fixtures/outputs/smoke_phase4/output_1080x1080.png`
    - `backend/tests/fixtures/outputs/smoke_phase4/output_1080x1920.png`
  - Runtime remains functional under missing OpenCV libs (fallback logs emitted, no crash).

### Phase 5: Public readiness checks
- Artifact tracking sanity:
  - `git ls-files | rg -n '(outputs/|\.pytest_cache|__pycache__|\.zip$|bench_phase21|tmp|cache|\.log$)'`
  - Only tracked output placeholder: `backend/tests/fixtures/outputs/.gitkeep`
- Secrets scan:
  - `git grep -nE '(sk-|api[_-]?key|token|secret|BEGIN PRIVATE KEY)' .`
  - Matches are documentation text and benign variable names; no concrete leaked keys found.

## 3) Change Map (BASE..HEAD)

| Area | Files | What changed | Why it matters |
|---|---|---|---|
| CI quality gates | `.github/workflows/ci.yml` | Ruff scope widened to include tests/tools | Prevents regressions in non-app code paths and keeps benchmark tooling healthy |
| Headless resilience | `backend/app/composition/background.py`, `backend/app/composition/content_aware_fit.py` | Lazy/defensive OpenCV handling + fallback paths | Avoids hard crashes in libGL-limited environments |
| Layout engine sophistication | `backend/app/layout/*` (engine, solver, profiles, typography, scoring, constraints) | Added profile-based scoring, solver iteration, typography reflow and constraints | Enables more advanced adaptive relayout behavior |
| Benchmarking framework | `backend/tools/generate_bench_fixtures.py`, `backend/tools/run_layout_bench.py`, `backend/tests/test_bench_tools.py` | Deterministic fixture generation + evaluation/reporting harness | Makes quality claims measurable and regressions detectable |
| Regression tests | multiple `backend/tests/test_*.py` | Added broad test coverage for typography, gates, generative components, content-aware fit, golden checks | Increases reliability confidence and catches edge cases earlier |
| Public readiness docs | `README.md`, `RELEASE_READINESS.md`, `docs/PUBLIC_RELEASE_CHECKLIST.md`, templates | Added quickstart/troubleshooting/release docs and contributor templates | Improves maintainability and external user onboarding |

### Notable risk changes
- Large functional increase in relayout/generative logic without benchmark pass-rate success yet.
- Benchmark report indicates widespread hard-fail constraints despite richer machinery.
- OpenCV fallback prevents crash, but repeated fallback indicates production path may often run in degraded mode in minimal environments.

## 4) Benchmark results summary

Source: `backend/tests/fixtures/outputs/bench_phase21/report.md`

- Total Phase2.1 runs: **36**
- Passed: **0**
- Pass rate: **0.0%**
- Frequent failure themes:
  - `outside_margin_ratio`
  - `min_font_size`
  - overlap violations (`overlap:*`)
  - `total_score`
  - occasional `text_plate` failures on busy-background scenarios

Interpretation:
- The benchmarking infrastructure itself is a major improvement.
- However, benchmark outcomes do **not** currently support a claim of designer-like quality readiness.
- Since BASE lacks benchmark tooling/reports, this is an absolute-state assessment (no formal delta claim).

## 5) Smoke test evidence

Executed scripted pipeline with representative benchmark input (`case_01_hero_headline_cta_logo`) and generated all 3 target aspect outputs successfully.

Validation against guarantees:
- **Pipeline execution reliability**: PASS (no crash; outputs produced).
- **Protected assets preserved / no disappearance**: INCONCLUSIVE from smoke alone; benchmark violations show layout issues involving hero/logo overlap in evaluated outputs.
- **Margins/overlap/readability constraints**: FAIL in benchmark for sampled and broader pack (multiple violations).
- **Text-safe plate only on busy backgrounds**: PARTIAL/INCONCLUSIVE; benchmark includes `text_plate` violations on busy-bg scenarios, requiring targeted tuning and assertions.

## 6) Risk register (top 5) + mitigations

1. **Quality gate mismatch to product claim**
   - Risk: “designer-like relayout” claim overstates current measured output quality.
   - Mitigation: define release threshold (e.g., >=85% pass rate on deterministic benchmark) and gate release on it.

2. **Constraint failures across core scenarios**
   - Risk: margins/overlap/font-size violations cause visibly poor outputs.
   - Mitigation: tune profile scoring weights + solver penalties and add scenario-specific acceptance tests from top failing cases.

3. **Fallback-heavy runtime in headless environments**
   - Risk: degradation under common deployment images lacking libGL/OpenCV support.
   - Mitigation: either ship known-good headless stack defaults or codify expected fallback quality floor with dedicated tests.

4. **No historical benchmark baseline data**
   - Risk: inability to prove trend improvement over time.
   - Mitigation: archive benchmark summaries per release/PR and track score/pass-rate deltas.

5. **Large change surface in one tranche**
   - Risk: hidden regressions in less-traveled paths.
   - Mitigation: split follow-up work into smaller PRs with per-area benchmark slices and stricter CI gates.

## 7) Go/No-Go recommendation

**Recommendation: NO-GO for broad user shipping right now.**

### Conditions required for GO
1. Achieve and sustain benchmark pass-rate threshold (suggest >=85%) on deterministic pack.
2. Resolve recurring hard violations (`outside_margin_ratio`, `min_font_size`, overlap) in top failing scenarios.
3. Add explicit acceptance checks for protected asset persistence and text-safe-plate trigger correctness.
4. Keep headless fallback behavior validated in CI (no libGL env) with quality-floor assertions.

If the team needs a near-term limited release, consider **GO for private beta only** with explicit caveats and manual QA sign-off per output set.
