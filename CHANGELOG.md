# Changelog

All notable changes to this project are documented here. Versions follow SemVer.

## [Unreleased]

### Added
- README demo gallery (`docs/demo/`): three layered masters rendered to 12 sizes each,
  flat-PNG plain resize vs auto-layers, v1 vs v2 engine, and a studio screenshot. Boards
  are produced by `backend/tools/make_demo.py` and are byte-for-byte reproducible.

### Changed
- Stack layout: stickers designed onto the hero (e.g. a "-50%" roundel) stay attached to
  it instead of taking a slot in the copy stack; strip logos are capped so they never
  dominate a 728×90 row.
- Auto-layers: flat or gradient backgrounds are refilled from the fitted background model
  instead of inpainting (no more "ghost" of removed elements, ~2x faster); small visual
  fragments touching the hero (steam, sparkles) are merged into it; the corner wordmark
  logo is detected before copy ranking so the sub-headline keeps its role.
- QA warns only when content is removed; dropped decoration is still listed in
  `qa.dropped` but no longer produces a warning.

### Fixed
- Auto-layers lost small glyph parts (the dot of an "i", accents) from text layers and left
  them behind on the background plate.

## [2.0.0] - 2026-09-24

A productisation release: AutoBanner becomes a deployable service with a studio, REST
API and CLI.

### Added
- **Stack layout engine** (default for layered designs): role-aware strip / landscape /
  vertical arrangements, overlap-free by construction, one solved scale per group to
  keep the type hierarchy, profile margins and size limits, legibility-driven dropping
  of secondary copy (never the headline or CTA). Phase 2.1 bench: 75.0% -> 88.9% pass,
  mean score 39.2 -> 56.4, runs with overlapping elements 36/36 -> 0/36.
- **Auto-layers (beta)** for flat PNG/JPG/WEBP: background modelling + block detection
  split a flattened banner into headline, sub-copy, CTA, logo and hero pseudo-layers
  plus an inpainted background plate (95% of elements detected with the right role on
  the sample set); `auto_layers` option in API/CLI/studio.
- `backend/tools/flat_banner_samples.py`: realistic flat/layered banner generator with
  ground truth; `run_layout_bench.py --engine stack|legacy`.
- **Studio** web UI (no Gradio): drag-and-drop upload, packs/presets, live progress,
  per-size previews with file-size budget and QA warnings, ZIP download, dark mode.
- **REST API** (FastAPI): `/v1/render` (sync ZIP), async `/v1/jobs`, `/v1/analyze`,
  `/v1/presets`, `/v1/config`, `/healthz`, `/readyz`, Prometheus `/metrics`, OpenAPI docs.
- API-key auth, per-key job isolation, rate limiting, upload streaming limits, bounded
  render queue with back-pressure, request IDs, JSON logs, security headers.
- **CLI** `autobanner` (`presets`, `analyze`, `render`, `serve`) with CI-friendly exit codes.
- **Size catalog**: 40 presets (IAB, Google Ads, Meta, TikTok, LinkedIn, X, YouTube,
  Pinterest, web, email) with network file-size caps and UI safe zones, grouped in packs.
- **Export budgets**: PNG/JPEG/WebP with automatic quality search / PNG palette fallback
  to fit a KB cap, reported per file.
- **Safe zones**: key elements are kept out of Story/Reels/TikTok overlay areas.
- **QA report** per output (safe-zone violations, too-small text, gates) and
  `manifest.json` in every ZIP.
- Upload hardening: magic-byte sniffing, dimension/byte limits, decompression-bomb guard,
  filename sanitisation, anchor validation.
- Docker image runs non-root with a read-only root filesystem; optional AI build arg.
- pip-installable package with `autobanner` console script; CI on Python 3.10–3.12,
  wheel smoke test and Docker smoke test.

### Changed
- `python -m app.main` now serves the studio + API with uvicorn (same port 7860).
- Default install is the lite CPU stack; AI extras moved to `requirements-ai.txt`.
- Phase 2.1 rendering 5–8x faster (no duplicate composition when generative stages are
  off; low-res inpainting with full-res seam refinement; LUT gamma resize).
- Phase 3 rendering ~2.7x faster (cached background extension, vectorised scoring).
- Engineering history docs moved to `docs/history/`.

### Fixed
- Raster layers (including PSD text layers) were stretched to reflowed text boxes; they
  are now always scaled uniformly.
- Phase 3 on layered input composed foreground layers into its base and pasted them again
  as anchors, leaving ghost copies when positions differed.
- Background grading froze whole bounding boxes around round/irregular elements, leaving
  a visible rectangle; protection now follows each element's silhouette.
- Gradio UI failed to import with current `huggingface_hub` and shared one engine (and
  its loaded file) across all users.
- Flat PNG/JPG sources were reclassified as `photo`, so Phase 3 manual anchors were cut
  from a blank gray canvas and the preview lost the image.
- Phase 3 manual anchors were stretched non-uniformly when the aspect ratio changed.
- Phase 3 output differed between runs (RNG seeded with per-process salted `hash()`).
- `_palette_lock` overflowed int16 when squaring colour differences.
- Batch rendering silently dropped sizes that raised errors.
- Elements removed to keep a size legible are now listed in the QA report instead of
  disappearing silently.

### Removed
- Gradio dependency and UI.

## [Unreleased before 2.0.0]
- Public-facing repo docs, release gate report, benchmark tooling, binary-artifact hygiene.
