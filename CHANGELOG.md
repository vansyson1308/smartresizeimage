# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
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
