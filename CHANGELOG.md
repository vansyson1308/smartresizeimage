# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Layout grammar (`design/grammar.py`): layout families composed from arrangement × text
  side/position × text share × logo corner × alignment (12 landscape / 24 portrait / 36
  square) join the hand-written families as planner candidates (`AUTOBANNER_LAYOUT_GRAMMAR`);
  every plan records the winning family's traits and how many candidates competed.
- Creative directions per run and per campaign row (`direction`: copy / subject first, text
  left / right / top, subject on top, centered, stacked / side, or a pinned family): they
  narrow the candidates, what a format cannot satisfy is reported in the plan, never
  silently ignored; selector in the Variants view and a `direction` CSV column. The variant
  detail view summarises the layout: family, traits, direction asked and whether it was
  honoured, and copy left out on that size.
- Hideable copy on small sizes: an element the design marks "May be left out on small
  sizes" is dropped by the planner only when the stack cannot fit at the minimum text sizes
  (least important first; never headline, CTA, logo or keep_visible elements), recorded as a
  decision and flagged by the family consistency check. Subject-integrity tolerance is
  rounding-aware for tiny boxes (728×90 logos); the planner penalises boxes by how far they
  leave the canvas; repair moves never break a hard order rule.
- Incremental refresh (`POST /api/projects/{id}/variants/refresh`, "Refresh stale" in the
  review view): after a design edit only the variants the change touched are re-rendered
  (keeping their layout); the others are marked current without a render, with a reason per
  variant.
- Subject integrity check (`subject_integrity`): subjects and logos must keep their
  proportions (stretching is a critical failure) and their pixels — the rendered region is
  correlated with the master asset scaled to the planned box, so occlusion, recolouring or
  cropping lands the variant in review with the measured correlation.
- Campaign table: one variants request renders every content row in every size
  (`rows` on `POST /api/projects/{id}/variants`, up to 144 variants per job); rows come
  from a pasted CSV/TSV mapped onto the design's text elements
  (`POST /api/projects/{id}/campaign/rows`) or the table in the Variants view; review
  filters by row, exports keep one folder per row and list the row in the manifest;
  consistency checks run within a row. `backend/tools/run_campaign.py` measures a 12 × 6 run.
- Brand profiles (`PUT/GET/DELETE /api/brands/{brand}`, brand card in the UI): colours,
  fonts, logo clear space and minimum size, minimum text size, tone and "never" notes per
  workspace; proposed as soft rules into projects of the brand and checked on every variant
  (`brand_palette`, `brand_font` → needs_review when off-brand).
- Plan entitlements per workspace (`free`/`team`/`business`/`unlimited`, operator-defined
  plans via `AUTOBANNER_PLANS`, assignments via `PUT /api/plans/{workspace}` with the setup
  token or `AUTOBANNER_WORKSPACE_PLANS`): variants per day, projects, members, campaign rows
  per job and storage; shown in `GET /api/usage` and the workspace card; refused with 402
  naming the plan. Global quotas still apply on top.
- Offline guard (`AUTOBANNER_OFFLINE=1`): outbound connections refused at the socket level;
  the browser journey runs its server under the guard as evidence that the product works
  without network after installation.
- Restart resume: variant jobs cut short by a restart continue on the next start from the
  briefs and layout families stored with the job (`resumed_from` / `resumed_by` on job
  records; `AUTOBANNER_RESUME_JOBS=0` keeps the old mark-as-failed behaviour).

### Security
- Imported project archives are no longer trusted: every archive-controlled path and id is
  validated at import and at use (absolute or traversal paths, symlinks, member floods are
  refused with 400), the importer becomes the owner, and claimed approvals are reset unless
  the importer may approve (`test_archive_safety.py`).
- Local authentication for the UI (`AUTOBANNER_AUTH=local`): first-run setup token creates a
  workspace administrator; PBKDF2 passwords; HttpOnly session cookie that also authenticates
  images and downloads; `X-Requested-With` required on cookie-authenticated writes; login
  throttling; member management with roles; personal API tokens (`abt_…`) for automation.
  Open mode is explicit and switches to local once a user exists.

### Fixed
- Restart resume no longer races the resumed worker: the recovery pass mutates and saves
  the project under its lock, and every store (projects, variant index, history, jobs,
  auth, usage, plans, brand profiles) writes through a uniquely named temp file
  (`design/atomic.py`), so two savers of one file never rename each other's temp away
  (`test_atomic_writes.py`, `test_resume_jobs.py`).
- Hard constraints are evaluated after every transformation, including kept reference plans
  and repairs; a violated or unevaluable hard rule blocks automatic acceptance and violating
  reference plans are re-planned (`test_hard_rules.py`).
- Text with several styled runs keeps each run's face, weight, size ratio, colour and
  tracking through PSD import (all style runs extracted), editing (`edit_text`, explicit
  `runs` op), wrapping, rendering on one baseline, serialization, typography and export
  disclosure; underline/strikethrough/baseline shift are disclosed as not reproduced.
- Tests are self-contained: DejaVu Sans/Bold bundled (Bitstream Vera licence), the CJK
  fallback test skips on measured glyph coverage, the H3 busy-background fixture is generated,
  and the Playwright browser journey is committed and run in CI.

### Added
- PDF export (`format=pdf`): one page per variant at pixel size (150 dpi) inside the
  deliverables zip, with page numbers in the manifest.
- Brand-level rules: confirmed `scale_range` / `min_text_size` / `clear_space` rules carry
  across an owner's projects with the same brand as reviewable proposals when a project is
  uploaded or first planned (`GET /api/brands/{brand}/rules`); unconfirmed proposals never
  propagate.
- Revisions pin the previous render's text-plate rectangles (`plan.text_plate_rects`), so a
  longer copy on a busy background no longer moves the plate of the whole text stack; the
  retention sweep also trims event-log lines older than `AUTOBANNER_RETENTION_DAYS`.
- Roles within an owner (`AUTOBANNER_API_KEYS=key:owner:role`, viewer/editor/approver/admin),
  per-owner rate limiting on mutating requests (`AUTOBANNER_RATE_LIMIT`, 429 + `Retry-After`)
  and a retention policy (`AUTOBANNER_RETENTION_DAYS`, purge untouched projects at startup and
  daily, logged as `project_purged`).
- Rules from correction history (H5, `design/corrections.py`): a rejection snapshots the
  rejected plan and reason; once a variant of the same size is approved, the difference
  becomes a reviewable proposal (`scale_range`, `min_text_size` scaled by the size it was
  measured on, `clear_space`) with its evidence, listed at `GET /api/projects/{id}/learned`
  and in the rules panel with "Add as rule"; unparseable or unpaired rejections are reported,
  never turned into rules. Corpus measurement in `tools/run_corrections.py`.
- Local edits on revision (H3): regenerating a variant keeps its previous plan by default
  (`keep_layout`, on by default in the API and the detail view), so a copy, asset, style or
  hidden-element change alters pixels only inside the edited element's box; copy that no
  longer fits its box falls back to a fresh plan and is reported as `layout_change`.
  Regression set in `test_local_edits.py`; corpus measurement in `tools/run_local_edits.py`.
- Learning from approved variants (H1, `design/examples.py`): approved variants of a project
  become examples; per orientation the planner infers a layout family (text column, subject
  slot, logo slot, alignment, stacking order), a hierarchy scale and soft constraint
  proposals with confidence, exposed at `GET /api/projects/{id}/learned` and in the rules
  panel, and used by the next generation (`planner_meta.from_examples`).
- Joint family planning across a size set (H2) with cross-variant consistency checks
  (`quality/family.py`); ablation harness (`tools/run_ablations.py`) with tuning/holdout split,
  Wilson intervals and the H1 designer-agreement protocol.
- Design representation (`backend/app/design`): typed document with native text, assets by
  content hash, roles with confidence, constraints, allowed transforms and provenance;
  constraint-aware planner with layout families; native text fitting/rendering with font
  substitution disclosure and glyph-coverage checks; approved translations per locale;
  variant pipeline (plan → typeset → render → verify → bounded repair); file-backed projects
  with history/undo.
- Honest flat-image decomposition (`design/decompose.py`): OCR text blocks cut with alpha
  masks, salient subject via GrabCut, inpainted background; everything marked `recovered`
  with confidence, kept as raster until "Convert to editable text".
- FastAPI project API (`backend/app/api`) with jobs (progress, cancel, idempotency), approvals
  with reasons, exports with manifest/report/font disclosure, editable project export/import,
  blank canvas + add-element, optional API key; dependency-free web UI (`backend/app/web`).
- Contrast-aware, resolution-independent text plates (light/dark panel by text colour).
- Quality contract v2 (`backend/app/quality`): rendered-output checks (per-element visibility,
  canvas clipping, OCR legibility via tesseract when installed, dropped required elements,
  export dimensions) with explicit `pass / fail / needs_review / not_checked` statuses and an
  `accepted / needs_review / failed` verdict attached to every `CompositionResult`.
- Mission workspace under `docs/mission/` (state, decisions, capabilities, evaluation,
  experiments, resume) and preserved evidence of the historical benchmark false positive.
- `ReLayoutEngine.load_elements()` so benchmarks run the production pipeline.
- Benchmark fixtures now include a clean `background.png` layer.
- Public-facing repo docs and community meta files.
- Release gate report and benchmark/tooling documentation.

### Changed
- Benchmark report now uses contract v2 verdicts; legacy layout-metadata metrics are kept under
  `legacy_v1` for historical comparison only. Run configuration (seed, Phase 3 candidate count,
  environment, commit) is recorded in `summary.json`.
- Layout solver ignores background/overlay elements and only enforces vertical rhythm between
  elements that share a column; template zones stack their members and overflow instead of
  dropping elements to raw coordinates; raster text is scaled uniformly (no glyph distortion).
- Phase 3 no longer reports `text_plate.applied = true`; last-resort/degraded selections set
  `used_fallback` and a pipeline failure reason. `OCRTextValidator` reports `not_checked`.
- `CompositionResult.gates_passed` defaults to `None` (not evaluated) instead of `True`.
- Phase 3 generator seeds derive from `zlib.crc32` (stable across processes) instead of `hash()`.
- Gradio interface keeps engine state per browser session; the status panel shows each
  variant's verdict and top issues. `AUTOBANNER_USE_AI` is honoured.
- Repository hygiene updates to avoid committing generated binary outputs.
