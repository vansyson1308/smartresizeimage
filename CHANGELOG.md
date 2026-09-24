# Changelog

All notable changes to this project are documented here. Versions follow SemVer.

## [2.0.0] - 2026-09-24

A productisation release: AutoBanner becomes a deployable service with a studio, REST
API and CLI.

### Added
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
- Gradio UI failed to import with current `huggingface_hub` and shared one engine (and
  its loaded file) across all users.
- Flat PNG/JPG sources were reclassified as `photo`, so Phase 3 manual anchors were cut
  from a blank gray canvas and the preview lost the image.
- Phase 3 manual anchors were stretched non-uniformly when the aspect ratio changed.
- Phase 3 output differed between runs (RNG seeded with per-process salted `hash()`).
- `_palette_lock` overflowed int16 when squaring colour differences.
- Batch rendering silently dropped sizes that raised errors.

### Removed
- Gradio dependency and UI.

## [Unreleased before 2.0.0]
- Public-facing repo docs, release gate report, benchmark tooling, binary-artifact hygiene.
