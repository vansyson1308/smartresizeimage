# AutoBanner

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

AutoBanner is a deterministic banner re-layout engine for converting one design into multiple target sizes while preserving core brand assets (logo/text/mascot) as much as possible.

## What it does
- Parses PSD and flat images (PNG/JPG/WEBP).
- Classifies element roles (headline/subheadline/CTA/logo/hero/background).
- Computes adaptive layouts (Phase 2.1: profile scoring + solver + typography reflow).
- Composes output with high-quality resizing and fallback-safe background extension.
- Runs deterministic benchmark packs for regression tracking.

## What it is NOT
- Not a full Photoshop-equivalent renderer for all layer effects.
- Not a generative design-rewrite tool by default.
- Not guaranteed to perfectly match designer hand-redraw output in all edge cases.

## Key features (Phase 2 + 2.1 + 3)
- Adaptive re-layout profiles: LANDSCAPE / SQUARE / PORTRAIT.
- Typography reflow: auto font scaling + wrapping without rewriting text.
- Collision solver + alignment snapping.
- Smart crop/focus for flat-image SMART fit.
- Text-safe background plate on busy text regions.
- Deterministic benchmark harness (fixtures + metrics + report).
- Target-first Redesign (Phase 3): anchored, brand-locked composition for target-native outputs (flat illustration optimized: sky/cityscape/fireworks/confetti).

## Quickstart (<10 minutes)

### Option A: Docker
```bash
docker compose up --build
```
Open http://localhost:7860

### Option B: Local Python
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m app.main
```

For CI/headless:
```bash
pip install -r requirements-ci.txt
```

## Usage examples

### 1) Analyze + generate 3 sizes (CLI smoke)
```bash
python - <<'PY'
from backend.app.relayout import ReLayoutEngine

engine = ReLayoutEngine(use_ai=False)
engine.load_file("/path/to/input.png")
for size in [(1200, 628), (1080, 1080), (1080, 1920)]:
    result = engine.relayout(size)
    result.image.save(f"output_{size[0]}x{size[1]}.png")
PY
```

Expected result:
- `output_1200x628.png`
- `output_1080x1080.png`
- `output_1080x1920.png`

### 2) Run Phase 2.1 benchmark pack
```bash
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
```
Outputs (generated, not committed):
- `backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/before.png`
- `backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/after.png`
- `backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/layout_debug.json`
- `backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/overlay.png`
- `backend/tests/fixtures/outputs/bench_phase21/report.md`

## Configuration + fallback behavior
Key settings are in `backend/app/config.py`:
- `LAYOUT_PROFILE_SCORING_ENABLED`
- `LAYOUT_SOLVER_MAX_ITERS`
- `LAYOUT_DEBUG_ENABLED`, `LAYOUT_DEBUG_DIR`
- `TEXT_SAFE_PLATE_*`
- `GENERATIVE_BG_*`, `GENERATIVE_DECOR_POLICY`

Headless/OpenCV behavior:
- If OpenCV cannot load (`libGL`/`cv2` issues), composition falls back to deterministic edge-repeat paths.
- The app should still render outputs (with warning logs).


## Target-first Redesign (Phase 3)
- **Relayout (Phase 2.1)**: deterministic geometric relayout of existing composition.
- **Target-first Redesign (Phase 3)**: place immutable brand anchors first (logo/text/CTA/mascot-equivalent) and regenerate only non-protected background/decor regions for a target-native look.

Anchor/protected-region guarantees:
- Flat-image workflow supports manual anchors and a flat-banner style preset (Mascot/MainText/CTA).
- Protected anchors are composed as immutable foreground layers.
- Background/decor generation is restricted to non-protected mask only.

Generative adapter:
- Optional and OFF by default.
- Enable with `AUTOBANNER_ENABLE_GENERATIVE_REDESIGN=true`.
- No external AI key is required for tests or default runtime; deterministic generator is always available.

## Development

### Lint + tests
```bash
ruff check backend/app backend/tests backend/tools
pytest -q
```

### Benchmark commands
```bash
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
python backend/tools/run_layout_bench.py --mode phase3 --seed 42
```

> Benchmark artifacts are generated locally under `backend/tests/fixtures/outputs/` and must not be committed.

## Troubleshooting
- **`libGL.so.1` / OpenCV import errors**: use `requirements-ci.txt`; fallback paths should still run (including Phase 3 deterministic generator).
- **Phase 3 quality differences in headless mode**: ensure deterministic mode baseline first; optional generative adapter is disabled by default and requires explicit env enablement.
- **Gradio localhost/proxy issues**: set `AUTOBANNER_SHARE=true` and `GRADIO_ANALYTICS_ENABLED=false`.
- **Coverage flag errors**: ensure dev deps are installed (`requirements-dev.txt`).

## License
See [LICENSE](./LICENSE).

## Attribution
Project code is first-party unless noted otherwise in future third-party attribution docs.
