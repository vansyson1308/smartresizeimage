# Decisions

Format: date, decision, alternatives rejected, reason. Newest first.

## 2026-09-09 — Constraint planner with hand-written layout families replaces zone templates
Rejected: keep the template/zone engine and add more zones; a generic optimizer over free
coordinates.
Reason: families express designer intent per aspect class (text column, subject slot, logo
slot) and let constraints and content pressure decide the rest; a free optimizer had no
prior for what "designed for the size" means and the zone engine dropped/collided elements.
Learned families from approved variants stay a research item (H1). The zone planner remains
behind `Config.DESIGN_PLANNER = "zones"` for comparison.

## 2026-09-09 — Text plates trigger on texture or contrast, resolution-independent
Rejected: keep the single busy threshold measured on the output canvas.
Reason: the same noise scored 0.23 at source and 0.16 after upscaling, so plates silently
disappeared at larger targets; legibility depends on texture at glyph scale and on the
contrast between the text colour and the background.

## 2026-09-09 — Recovered raster text stays raster until a user converts it
Reason: OCR strings are unverified; typesetting them as native text would silently change
approved copy. The element keeps the recognised string as metadata with its confidence.

## 2026-09-09 — Benchmark runs the production `ReLayoutEngine`
Rejected: keep the separate benchmark harness that composed elements directly.
Reason: the harness skipped harmonization/grounding, used 4 Phase 3 candidates while production
used 8, and toggled config globally without recording it. `load_elements()` was added so the
benchmark and users share one path; every run records config, seed, candidate count, commit and
environment.

## 2026-09-09 — Quality contract v2 replaces layout-metadata pass/fail
Rejected: tune thresholds of the legacy metrics (`bench_metrics.py`).
Reason: the legacy metrics cannot see occlusion, clipping or legibility; the preserved false
positive (`evidence/phase3_false_positive_9d913a0`) passed with all text hidden. v2 checks the
rendered pixels (per-element visibility via expected-vs-composite comparison, OCR legibility,
clipping, dropped elements, export dimensions) and reports explicit statuses. Legacy metrics are
kept under `legacy_v1` for historical comparison only.

## 2026-09-09 — A skipped check is `NOT_CHECKED`, never a pass
Rejected: keep `OCRTextValidator` returning success with `ocr_skipped`.
Reason: skipped checks were counted as positive evidence. Contract v2 derives `NEEDS_REVIEW`
whenever a critical check did not run.

## 2026-09-09 — `gates_passed` is `None` until evaluated
Rejected: keep the `True` default on `CompositionResult`.
Reason: defaults implied success for outputs nobody had checked.

## 2026-09-09 — Raster text is scaled uniformly; native reflow only for native text
Rejected: keep fitting native text metrics and then resizing the raster crop to that box.
Reason: it squashed glyphs. Real reflow requires the native text representation (Phase B).

## 2026-09-09 — Stable seeds via `zlib.crc32`, not `hash()`
Reason: Python salts `hash()` per process; benchmark images differed between processes.

## 2026-09-09 — Gradio session state per browser session
Rejected: keep one engine in the interface closure.
Reason: concurrent users overwrote each other's parsed elements and temp ZIPs.

## 2026-09-09 — Keep Python + Pillow domain engine; add a typed API and a web review UI (Phase B)
Rejected for now: React/Vite frontend, vector editor library, new database.
Reason: no requirement yet justifies a build toolchain or a database; a FastAPI service with a
static, dependency-free web UI and file-backed projects delivers the full journey with the
smallest surface. Revisit when collaboration or scale requires it.
