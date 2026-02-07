# AutoBanner

AI-powered banner re-layout engine. Upload a design (PSD, PNG, JPG, WEBP) and generate production-ready ad creatives for any target size.

## Architecture

Python service using Gradio for the web UI. Parses PSD files with full layer extraction (and flat PNG/JPG/WEBP), classifies layer semantics with CLIP, calculates adaptive layouts for any target aspect ratio, and composes the final output with gamma-correct LANCZOS resizing, content-aware fit strategy (SMART mode: auto COVER/CONTAIN), and tiered background extension (LaMa AI inpainting, OpenCV TELEA inpainting, edge-pixel repetition with feathered blending).

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

## Testing

```bash
pytest backend/tests/ -v --cov=backend/app --cov-report=term-missing
```

### Linting

```bash
ruff check backend/app/
```

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

- Lint (ruff) + Test (pytest, 101 tests) + Coverage
