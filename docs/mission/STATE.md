# Current state

Updated: 2026-09-11. Branch `claude/blissful-archimedes-5n2dvd` (from `main` @ 9d913a0).
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

- `ruff check backend` clean; `pytest backend/tests` → **337 passed** (unit) + 1 browser journey (41 steps) (API round trips, roles,
  rate limit, retention, planner, examples, local edits, corrections, decomposition).
- Measurement tools (all synthetic, seed 42): `run_layout_bench.py` (modes), `run_ablations.py`
  (planner/repair/plates, joint, H1 protocol), `run_local_edits.py` (H3), `run_corrections.py`
  (H5); verbatim outputs under `results/`.
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
- Local auth (users, sessions, tokens, roles) exists; no SSO, billing, or multi-node job
  queue (jobs are in-process; interrupted variants are marked failed on restart).
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

- H3 local edits shipped and measured (`design/variant.py` `reference`, `keep_layout` on
  regenerate, detail-view checkbox; `results/local_edits_2026-09-09.md`): a copy, asset,
  style or hidden-element revision keeps every other element's pixels in 180/180 corpus
  revisions with pinned text plates (174/180 before pinning, 156/180 when re-planning) and
  in all five regression-set edits; copy that no longer fits falls back to a fresh plan with
  a `layout_change` warning.

- H5 rules from correction history shipped and measured (`design/corrections.py`,
  rejection snapshots per project, `GET .../learned` corrections + unresolved,
  one-click apply; `results/corrections_2026-09-09.md`): with a rule-based reviewer,
  carried rules cut repeat rejections 28 → 0 over 14 later campaigns. Human reasons and
  brand taste unmeasured. Test suite: 259 tests.

- Commercial completeness: roles within an owner (viewer/editor/approver/admin via
  `AUTOBANNER_API_KEYS=key:owner:role`, 403 on disallowed actions), per-owner rate limiting
  on mutating requests (`AUTOBANNER_RATE_LIMIT`, 429 + `Retry-After`, logged), retention
  policy (`AUTOBANNER_RETENTION_DAYS`, purge at startup and daily, logged, no undo).
  `test_ownership_and_jobs.py` 13 tests; `docs/OPERATIONS.md` updated. Test suite: 264 tests.

- Brand-level rules (H5 follow-up, `design/brand.py`): confirmed `scale_range` /
  `min_text_size` / `clear_space` rules are unioned across the owner's projects of one brand
  (stricter wins, keyed by role) and proposed into a project of that brand when it is
  uploaded or first planned; `GET /api/brands/{brand}/rules`; unconfirmed proposals never
  propagate. API + unit tests.

- PDF export (`format=pdf`, one page per variant at 150 dpi inside the deliverables zip,
  page numbers in the manifest). Docker CLI exists here but no daemon: image build stays
  BLOCKED_EXTERNAL.

## Independent audit closure (2026-09-11, Mission V2 phase A)

Machine-readable ledger: `docs/mission/LEDGER.json`.

| Finding | Status | Fix | Proving tests |
|---|---|---|---|
| F1 archive paths/metadata trusted on import | FIXED (24be1f8) | every archive-controlled path and id validated at import and at use (`safe_relative_path`, `is_safe_id`, `Project.variant_file`); member cap, symlink refusal; owner/id overwritten; claimed approvals reset unless the importer may approve | `test_archive_safety.py` (8) |
| F2 hard rule violated after kept-reference regeneration but accepted | FIXED (18f848f) | `constraint_checks` after all transformations; hard rule FAIL/NOT_CHECKED is CRITICAL and blocks acceptance; violating reference plans re-planned; all seven constraint types covered | `test_hard_rules.py` (5) |
| F3 multi-run text rendered with the first run's style | FIXED | per-run shaping and rendering on one baseline (size ratio, face, colour, tracking); PSD style runs extracted; runs kept through editing (`edit_text`, explicit `runs` op), serialization, typography and font disclosure; unsupported attributes disclosed | `test_text_runs.py` (12) |
| F4 no UI auth journey | FIXED | local users, sessions (HttpOnly cookie, CSRF header), setup token, roles, members, personal tokens; images/downloads authenticated by the session; `AUTOBANNER_AUTH=local` in compose | `test_auth_journey.py` (7), `tests/e2e/test_journey.py` |
| F5 unportable tests | FIXED | committed Playwright journey + CI job, bundled DejaVu fonts, coverage-based CJK skip, generated H3 fixture | `tests/e2e`, `test_design_document.py`, `test_local_edits.py` |

Browser journey (committed, 34 steps, 0 console/HTTP errors): setup with wrong then right
token → blank canvas → add text/logo → verbatim price → drag/nudge/undo → rule → brief →
generate → approve / reject with reason → per-variant override regenerate → learned rules →
export approved → save/reopen zip → add approver member → personal token → sign out → sign in
as approver (design tools disabled, approve enabled) → approver approves → second workspace
isolated → token lists projects → 900 px without horizontal scroll.

## Mission V2 phase B progress (2026-09-11)

- **Campaign table** (B1): a variants request takes `rows` (id, label, copy per text element,
  locale, hidden elements); every row is rendered in every size in one job (≤ 144 variants);
  rows are read from a pasted CSV/TSV whose header names text elements (name, id or unique
  role) with optional row/label/locale columns, unknown columns and verbatim copy reported and
  ignored; consistency checks run within a row; review filters by row; exports keep one folder
  per row and list the row in the manifest. Measured 12 × 6 = 72 on the synthetic master:
  72/72 rendered in 88 s (2 workers, 1.23 s per variant), 65 accepted, 5 needs_review (OCR
  doubt on one Vietnamese headline), 2 failed (long subheadline does not fit 300×250 at the
  16 px floor). `results/campaign_12x6_2026-09-11.md`.
- **Restart resume** (B2): job records carry the chosen layout families and every variant its
  brief, so a restart continues unfinished variants in a new job (`resumed_from` /
  `resumed_by`, event `job_resumed`) instead of failing them; `AUTOBANNER_RESUME_JOBS=0`
  restores the old behaviour. Finished variants of the interrupted job are untouched.
- **Brand profile** (B3): the brand book as data per workspace (colours with an optional
  strict palette, headline/body fonts, logo clear space and minimum height, minimum text
  size, tone, never-list); projects of the brand get the logo/text rules as reviewable soft
  constraints and every variant carries `brand_palette` / `brand_font` checks that land
  off-brand copy in review. Stored under `data/brands/<workspace>/`, never sent anywhere.
- **Plan entitlements** (B4): each workspace is on a plan (built-in free/team/business/
  unlimited or operator-defined) capping variants per day, projects, members, campaign rows
  per job and storage; `GET /api/usage` shows used/remaining; over-limit requests get 402
  naming the plan before anything is created; operator assignment with the setup token or
  `AUTOBANNER_WORKSPACE_PLANS`; global quotas still win when stricter.
- **Offline guard** (B5): `AUTOBANNER_OFFLINE=1` refuses non-loopback connections at the
  socket level; the committed browser journey runs its server under the guard, so the whole
  declared workflow is exercised without network.
- Browser journey grew to 40 steps (campaign table read from CSV, count, generation, review
  filtered by row) and stays green.

## Mission V2 phase C progress (2026-09-11)

- **Layout grammar** (`design/grammar.py`): families composed from arrangement × text
  side/position × text share × logo corner × alignment (12 / 24 / 36 per orientation) join
  the hand-written families as planner candidates; plans record the winner's traits and the
  candidate count. Ablation (`results/ablations_grammar_2026-09-11.md`): no acceptance change
  on the synthetic corpus (0.89 tuning / 0.92 holdout either way, same six honest 300×250
  non-acceptances), grammar families win 6/60 runs; kept on for direction coverage, not
  claimed as a quality gain.
- **Creative directions** per run and per campaign row (emphasis, text side/position,
  alignment, arrangement, pinned family, mood) narrow the candidates; what a format cannot
  satisfy is recorded as `direction_unmet` in the plan; CSV `direction` column; selector in
  the Variants view; joint family choice per distinct direction.
- **Subject/mascot integrity**: uniform scale enforced (stretching is critical) and an
  independent pixel oracle (mean colour difference on solid pixels, plus grayscale
  correlation for textured assets) sends occluded, recoloured or cropped subjects to review
  with the measured similarity. A first version mis-flagged flat assets; recalibrated and
  documented.
- **Incremental refresh**: after an edit, only variants whose shown elements, rules, fonts or
  canvas changed are re-rendered (layout kept); the rest are marked current with a reason.
- **Counterexample search** (`tools/find_counterexamples.py`,
  `results/counterexamples_2026-09-11.md`): 420 renders under seeded perturbations with OCR
  on; 0 accepted renders rejected by an independent oracle, 0 unexplained failures; the 20
  oracle firings all had non-accepted verdicts. It surfaced that overlap repair could undo a
  hard order rule the planner had honoured (fixed: such moves are reverted; the hard-order
  perturbation re-run with the guard has 0 order firings instead of 9) and that long copy or
  hard rules on small formats can push the text stack past the canvas; the planner now
  penalises boxes by how far they leave the canvas (145 → 153 accepted, bounds firings
  15 → 10 on the same 180 renders; the rest is copy that fits no family and fails honestly).

## Release-criteria check (updated 2026-09-11)

| Criterion (MISSION.md priorities / brief) | Status | Evidence |
|---|---|---|
| Truthful evaluation: score describes what the customer sees | VERIFIED_LOCAL | Contract v2 on rendered pixels; preserved false positive rejected for the right reason; skipped checks never pass. Calibration vs humans BLOCKED_EXTERNAL. |
| One complete, reopenable end-to-end journey | VERIFIED_LOCAL | Committed 41-step browser journey against the real server with local auth and the offline guard (setup → design → brief with a creative direction → campaign table → generate → review per row → approve/reject → regenerate → export → save/reopen → members, tokens, second workspace, approver account); green on CI. |
| Native text and asset fidelity | VERIFIED_LOCAL | Native text fitting/rendering with font disclosure and glyph coverage; assets by content hash; recovered elements never silently converted. |
| Measured differentiation before claims | VERIFIED_LOCAL (synthetic) | Frozen holdout, Wilson intervals, ablations for planner/repair/plates/joint/H1/H3/H5 with verbatim reports; no customer numbers claimed. |
| Operability | VERIFIED_LOCAL | Local users/sessions/tokens with roles, workspace isolation, plans and quotas, rate limiting, retention, durable jobs with restart resume, incremental refresh, offline guard, pilot instruments, Docker/compose, operations doc. Multi-node, billing, SSO not provided; Docker image build unverified here. |
| Real design corpus, human reviewers, provider credentials, Docker daemon | BLOCKED_EXTERNAL | Not available in this environment; every dependent claim is marked as such. |

## Next actions (in order)

1. Phase D as declared in Mission V2: R1–R4 on a *designed* corpus of 12 brands with real
   masters (licensed PSDs or designer-made layered files), baselines and ablations run with
   `tools/run_ablations.py`, `run_campaign.py` and `find_counterexamples.py`; human
   calibration of verdicts and correction-time measurement through the pilot instruments.
   All of this needs designs and reviewers this environment does not have (BLOCKED_EXTERNAL).
2. Phase E: Docker image build on a machine with a daemon (`docker compose build && up`,
   `/api/health`, browser journey against port 8000, record image size and cold start);
   packaging notes; keep the Vietnamese handoff (`docs/mission/HANDOFF_VI.md`) current.
3. Product follow-ups with evidence value: grammar families for extreme strips (728×90),
   directions in the per-variant detail view, run-level text editing in the UI, multi-node
   job records and sessions.
