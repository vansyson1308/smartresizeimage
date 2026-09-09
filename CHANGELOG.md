# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
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
