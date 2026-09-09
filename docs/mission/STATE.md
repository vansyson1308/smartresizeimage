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
| design, zone planner (native text, verify+repair) | 32/36 | 4 | 0 | 13/36 | 1.9 |
| **design, constraint planner + contrast-aware plates** | **36/36** | 0 | 0 | — | 1.5 |

Progression of the design pipeline on the same 36 runs: zone planner 32 accepted (4 busy
backgrounds unreadable) → constraint planner 32 accepted, logo clear-space notes 12 → 0 →
contrast-aware text plates 36 accepted with OCR agreement 1.0 on the busy cases. Raster-text
modes fail mostly on hidden/illegible CTA rasters and text overlaps; they remain for legacy
inputs only. Numbers are synthetic-fixture engineering results, not customer validation.

- Browser journey (Playwright + Chromium, real server): see `RESUME.md` for the script and the
  latest run status.

## Known limitations / not done

- No PSD fixture in the repo: PSD import is unit-tested with synthetic layers only; real
  customer PSDs are `BLOCKED_EXTERNAL`.
- Flat-image decomposition is validated on a synthetic banner only; real photography untested.
- The constraint planner uses three hand-written layout families per aspect class plus, per
  project, families learned from approved variants (H1). Learning is validated on synthetic
  examples only (same assets and copy as the master); real designer examples with manual
  edits are untested, and one example per orientation stays at confidence 0.5.
- No multi-tenant auth, metering, billing, or durable job queue across restarts (jobs are
  in-process; interrupted variants are marked failed on restart).
- Human calibration of the verdict and operator-time measurements not performed.
- Legacy documents in repo root (`AUDIT_REPORT.md`, `BENCH_DELTA_*.md`, `RELEASE_READINESS.md`)
  describe the pre-v2 evaluator; treat their numbers as historical.

## Phase C progress (2026-09-09, later the same day)

- Constraint-aware planner (`backend/app/design/planner.py`): layout families per aspect,
  reading-order text stack with hierarchy from the master, uniform subjects/logos, logo clear
  space, content pressure (copy takes room from the subject down to a floor), constraints
  `order_below` / `anchor_edge` / `scale_range` / `keep_visible`; conflicts reported.
  `Config.DESIGN_PLANNER` = "constraints" (default) | "zones". Tests: `test_planner.py`.
- Approved translations per element (`TextContent.translations`), selected by the brief's
  locale; protected copy ignores overrides. Glyph coverage via fontTools with automatic
  fallback to a face that has the glyphs (disclosed) and a critical `font_coverage` check.
- Contrast-aware, resolution-independent text plates (light/dark panel chosen from the text
  colour, merged per text stack).
- PR #7 CI: backend-tests and GitGuardian green on the first push.

- Flat-image decomposition wired into import: OCR text blocks (alpha cut-outs, colour,
  role guess), unreadable small blocks as logo marks, salient subject via edge energy +
  GrabCut (other recovered regions excluded), inpainted background; every recovered element
  carries confidence and `recovered_text`; `convert_to_text` op + UI button; variants stay
  `needs_review` (`recovered_unconfirmed`) until confirmed. `test_decompose.py` (5 tests, one
  end-to-end through the API).

## Phase D progress

- API keys map to owners (`AUTOBANNER_API_KEYS`); projects, jobs, usage and exports are
  scoped per owner (foreign resources read as 404); imported projects belong to the importer.
- Durable job records (`data/jobs/*.json`) reloaded on start as `interrupted`; per-owner
  usage meter (`GET /api/usage`) and quotas (`AUTOBANNER_QUOTA_*`, HTTP 429).
- Streamed upload limit (early 413). `docs/OPERATIONS.md` covers run, config, data layout,
  backup/restore, deploy/rollback, monitoring and what is not provided.
- Tests: `test_ownership_and_jobs.py` (6 tests).

## Phase E progress

- Ablation harness with frozen holdout (`backend/tools/run_ablations.py`); results in
  `EXPERIMENTS.md` and `results/ablations_2026-09-09.md`: constraint planner 44/48 accepted
  (holdout 12/12) vs zone planner 36/48 (holdout 9/12); repair only helps the zone planner;
  plates +6 accepted on busy backgrounds; 433 s wall for 240 runs.
- Local pilot instruments: append-only event log per owner, `GET /api/pilot/summary`
  (first-pass acceptance, median time to decision, corrections per project, rejection
  reasons). No external telemetry.
- `ECONOMICS.md`: measured compute (≈1.3–1.5 s CPU per accepted variant) and storage
  (≈350 KB per synthetic 4-variant project) with explicit, labelled pricing scenarios.

- H2 joint family planning shipped and ablated: `choose_families` (one family per
  orientation for a size set) is the default for multi-size jobs; cross-variant checks
  (`quality/family.py`) attach to every variant; corpus grown to 15 cases (holdout 10–15,
  busy background included). Family-consistency issues 9 → 3 of 60 runs at equal
  acceptance and compute.

- H1 learning from approved examples shipped and ablated (`design/examples.py`,
  `results/ablations_h1_2026-09-09.md`): approved variants of a project become examples;
  per orientation the planner infers a family, a hierarchy scale and soft constraint
  proposals with confidence; `GET /api/projects/{id}/learned` and the rules panel show them.
  On held-out sizes agreement with the designer's composition rises 0.24 → 0.85 (slot
  expansion, `results/ablations_h1_slots_2026-09-09.md`) and sizes needing a re-layout drop
  60/60 → 6/60 at +3.7 s setup per case. Test suite: 248 tests.

## Next actions (in order)

1. Real-design corpus when access exists (licensed PSDs/photos); decomposition on
   photographs; human calibration of verdicts and correction-time measurement via the pilot
   instruments (this is also what turns the H1 proxy into a measured correction rate).
2. H3 local-edit regression set (pixel diff outside the edited scope must be zero on
   campaign revisions); H5 retrieval from correction history; H1 with more than one example
   per orientation (confidence above 0.5) and with real designer examples.
3. Roles within an owner (viewer/approver), retention policy, rate limiting; Docker image
   build verification on a machine with a Docker daemon.
