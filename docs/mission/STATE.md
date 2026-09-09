# Current state

Updated: 2026-09-09. Branch `claude/blissful-archimedes-5n2dvd` (from `main` @ 9d913a0).
This file is a checkpoint, not a completion claim.

## What exists and is verified locally

- **Quality contract v2** (`backend/app/quality`): rendered-output checks with explicit statuses;
  the preserved false positive is rejected for the right reason. 13 dedicated tests.
- **Layout engine defects fixed**: background excluded from the solver, zone stacking, zone
  overflow, side-by-side rhythm, uniform raster-text scaling (`backend/tests/test_layout_fixes.py`).
- **Result semantics fixed**: `gates_passed=None` until evaluated; Phase 3 no longer claims a
  text plate; degraded/last-resort selections set `used_fallback`; OCR placeholder is
  `not_checked`; seeds stable across processes; generative adapter labelled mock.
- **Benchmark parity**: `run_layout_bench.py` drives the production `ReLayoutEngine` and records
  seed, candidates, config flags, environment and commit. New `--mode design` runs the
  document pipeline.
- **Design representation** (`backend/app/design`): typed document (schema 1.0) with native
  text runs, assets by content hash, roles + confidence, constraints, allowed transforms,
  provenance; serialization + migration guard; font registry with substitution disclosure;
  native text fitting/rendering (FreeType, raqm available); master renderer; variant pipeline
  (plan → typeset → render → verify → bounded repair); file-backed projects with history/undo.
- **API + jobs** (`backend/app/api`): FastAPI service with upload/blank/add-element, document
  ops with snapshots + undo/restore, variant jobs (progress, cancel, idempotency, partial
  completion, restart marks interrupted), approval with reasons, exports (PNG/JPEG/WebP zip with
  manifest + reports + font disclosure), editable project export/import. Optional API key.
- **Web UI** (`backend/app/web`, dependency-free): projects, design canvas with select/drag/
  resize/nudge/undo, element panel (role + confidence, text/style, lock, allowed transforms,
  priority, z-order), rules list with proposed/confirmed state, brief (presets with provenance,
  custom sizes, copy overrides, locale), job progress + cancel, review grid with verdict badges,
  detail with issues, compare, approve/reject with reason, per-variant copy override, exports.
- **Gradio UI**: per-session state, verdict summary (legacy path).

## Test and benchmark status (this environment)

Environment: Python 3.11.15, Pillow 10.4.0, NumPy 1.26.4, SciPy 1.17.1, opencv-headless
4.11.0.86, psd-tools 1.19.0, tesseract 5.3.4, FastAPI 0.141.1. 4 CPUs, no GPU.

- `ruff check backend` clean; `pytest backend/tests` → **215 passed** (includes API round trip).
- Benchmark, 12 synthetic cases × 3 sizes, seed 42, contract v2 verdicts:

| Mode | Accepted | Needs review | Failed | Legacy v1 "pass" | Mean s/run |
|---|---:|---:|---:|---:|---:|
| baseline (template, raster text) | 5/36 | 19 | 12 | 0/36 | 2.9 |
| phase21 (adaptive, raster text) | 0/36 | 14 | 22 | 20/36 | 3.2 |
| phase3 (procedural redesign, 8 candidates) | 0/36 | 6 | 30 | 20/36 | ~43 (contended run) |
| **design (native text, verify+repair)** | **32/36** | 4 | 0 | 13/36 | 1.9 |

The 4 design-mode reviews are busy/noise backgrounds where OCR agreement is low (honest
`needs_review`). Raster-text modes fail mostly on hidden/illegible CTA rasters and text
overlaps; those modes remain for flat/legacy inputs only. Numbers are synthetic-fixture
engineering results, not customer validation.

- Browser journey (Playwright + Chromium, real server): see `RESUME.md` for the script and the
  latest run status.

## Known limitations / not done

- No PSD fixture in the repo: PSD import is unit-tested with synthetic layers only; real
  customer PSDs are `BLOCKED_EXTERNAL`.
- Flat images are not decomposed (always `needs_review`).
- Layout planning still uses the template/zone engine; keep_group is enforced post hoc.
- No multi-tenant auth, metering, billing, or durable job queue across restarts (jobs are
  in-process; interrupted variants are marked failed on restart).
- Human calibration of the verdict and operator-time measurements not performed.
- Legacy documents in repo root (`AUDIT_REPORT.md`, `BENCH_DELTA_*.md`, `RELEASE_READINESS.md`)
  describe the pre-v2 evaluator; treat their numbers as historical.

## Next actions (in order)

1. Land this change set as a draft PR; keep CI green (`requirements-ci.txt` now includes
   FastAPI/httpx/pytesseract; CI runners lack tesseract → OCR checks report `not_checked`).
2. Phase C: constraint-aware planner on the document (replace zone templates for the design
   pipeline), localization-aware copy fitting, flat-image decomposition with confidence.
3. Phase D: auth/tenancy, durable jobs, metering; Phase E: held-out evaluation + ablations (H1/H2).
