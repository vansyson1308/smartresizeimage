# AutoBanner

AI-powered banner re-layout engine. Upload a design (PSD, PNG, JPG, WEBP) and generate production-ready ad creatives for any target size.

## Architecture

Python service using Gradio for the web UI. Parses PSD files with full layer extraction (and flat PNG/JPG/WEBP), classifies layer semantics with CLIP, calculates adaptive layouts for any target aspect ratio, and composes the final output with gamma-correct LANCZOS resizing, content-aware fit strategy (SMART mode: auto COVER/CONTAIN), and tiered background extension (LaMa AI inpainting, OpenCV TELEA inpainting, edge-pixel repetition with feathered blending).

Recent rendering upgrades include:
- Headless-safe composition imports (no import-time crash when `cv2`/`libGL` is unavailable).
- Linear-light premultiplied-alpha compositing utilities.
- Blend mode MVP in linear space (`normal`, `multiply`, `screen`, `overlay`).
- Drop shadow renderer MVP (`effects.drop_shadow`) with synthetic golden coverage.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | Required |
| Docker | 24+ | Quick start (optional) |
| CUDA GPU | - | AI classification (optional, falls back to heuristics) |

## Quick Start (Docker)

```bash
docker compose up --build
```

Open http://localhost:7860 in your browser.

## Development Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m app.main
```

> **Note**: For CPU-only environments without torch/CUDA, install the lightweight dependencies instead:
> ```bash
> pip install -r requirements-ci.txt
> ```
> The backend will still work but skip AI-based CLIP classification (falls back to rule-based + heuristic classification).
>
> **Headless note**: `opencv-python-headless` is used for CI/headless runs. If OpenCV cannot load at runtime, composition falls back to edge-repeat extension paths with warnings instead of crashing.
>
> **Proxy/localhost note**: In restricted environments where localhost checks fail, set:
> ```bash
> AUTOBANNER_SHARE=true
> ```
> or configure `NO_PROXY` for localhost.

## Testing

### Full test suite

```bash
pytest -q
```

### Coverage run (CI-style)

```bash
pytest backend/tests/ -v --cov=backend/app --cov-report=term-missing
```

If `--cov` is not recognized, install dev dependencies first:

```bash
pip install -r backend/requirements-dev.txt
```

### Linting

```bash
ruff check backend/app backend/tests
```

## Golden visual regression tests

Golden fixtures and helper docs are in:

- `backend/tests/fixtures/golden/README.md`

Run golden tests:

```bash
pytest -q backend/tests/test_golden_regression.py
```

How to add a golden case (short):
1. Add deterministic inputs/expected under `backend/tests/fixtures/golden/case_*`.
2. Add a test using `backend/tests/image_diff_utils.py::compare_images`.
3. Keep strict thresholds unless cross-platform variance is proven.
4. Diff artifacts are written to `backend/tests/fixtures/outputs/` when comparisons fail.

## Supported Input Formats

| Format | Processing |
|--------|-----------|
| PSD | Full layer parsing + semantic classification |
| PNG | Content-aware fit (SMART mode) |
| JPG | Content-aware fit (SMART mode) |
| WEBP | Content-aware fit (SMART mode) |

## Project Structure

```
autobanner/
├── backend/
│   ├── app/
│   │   ├── main.py              # Gradio entry point
│   │   ├── config.py             # Configuration
│   │   ├── models.py             # Data models
│   │   ├── enums.py              # ElementRole enum
│   │   ├── constants.py          # Shared constants
│   │   ├── exceptions.py         # Custom exceptions
│   │   ├── validators.py         # Input validation
│   │   ├── parser/               # PSD & image parsers
│   │   ├── classifier/           # Semantic classifier (CLIP)
│   │   ├── layout/               # Layout engine & templates
│   │   ├── composition/          # Composition, resize, background, content-aware fit
│   │   └── relayout.py           # Orchestrator
│   ├── tests/                    # pytest test suite (101 tests)
│   ├── requirements.txt          # Production deps (with torch/AI)
│   ├── requirements-ci.txt       # Lightweight deps (CI/testing)
│   └── requirements-dev.txt      # Dev deps (includes production + tools)
├── docker-compose.yml
└── .github/workflows/ci.yml     # CI pipeline
```

## CI/CD

GitHub Actions runs on every push/PR to `main`:

- Lint (ruff) + Tests + Coverage (`.github/workflows/ci.yml`).
