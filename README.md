# AutoBanner

Automatic banner re-layout engine with dual-mode processing.

## Architecture

AutoBanner has two independent processing modes:

- **Frontend (Local WASM)**: Runs entirely in the browser using WebAssembly-based background removal (`@imgly/background-removal`) and Canvas API composition. Supports batch generation across 11 preset ad formats (Social Media, Google Display Ads, Video & Display). Zero server dependency.

- **Backend (AI-Powered PSD Re-Layout)**: Python service using Gradio for the UI. Parses PSD files (and PNG/JPG/WEBP), classifies layer semantics with CLIP, calculates adaptive layouts for any target aspect ratio, and composes the final output with gamma-correct LANCZOS resizing, content-aware fit strategy (SMART mode: auto COVER/CONTAIN), and tiered background extension (LaMa AI inpainting, OpenCV TELEA inpainting, edge-pixel repetition with feathered blending).

## Prerequisites

| Tool | Version | Required for |
|------|---------|-------------|
| Python | 3.10+ | Backend |
| Node.js | 20+ | Frontend |
| Docker | 24+ | Quick start (optional) |
| CUDA GPU | - | AI classification (optional, backend falls back to heuristics) |

## Quick Start (Docker)

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:7860

## Development Setup

### Backend

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

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## Testing

All commands run from the **project root** (`autobanner/`):

### Backend (pytest)

```bash
pytest backend/tests/ -v --cov=backend/app --cov-report=term-missing
```

### Frontend (vitest)

```bash
cd frontend && npm test
```

### Linting

```bash
ruff check backend/app/
cd frontend && npx tsc --noEmit
```

## Supported Input Formats

| Format | Backend | Frontend |
|--------|---------|----------|
| PSD | Full layer parsing | - |
| PNG | Content-aware fit | Blur + segment |
| JPG | Content-aware fit | Blur + segment |
| WEBP | Content-aware fit | Blur + segment |

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
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Main React app
│   │   ├── types.ts              # TypeScript types
│   │   ├── components/           # UI components
│   │   ├── services/             # Processing & caching
│   │   └── utils/                # Validation & image utils
│   └── index.html
├── docker-compose.yml
└── .github/workflows/ci.yml     # CI pipeline
```

## CI/CD

GitHub Actions runs on every push/PR to `main`:

- **Backend**: Lint (ruff) + Test (pytest, 101 tests) + Coverage
- **Frontend**: TypeScript check + Build (Vite)

## Quality Comparison

| Aspect | Frontend (localhost:3000) | Backend (localhost:7860) |
|--------|--------------------------|-------------------------|
| Resizing | Canvas bilinear | PIL LANCZOS + gamma 2.2 |
| Aspect ratio | Fixed COVER (crop) | SMART (auto COVER/CONTAIN) |
| Background fill | Gaussian blur 20px | OpenCV inpaint / edge-repeat |
| Content preservation | May crop text/elements | Never crops >20% |
| Processing | Instant (client-side) | 1-5s (server, higher quality) |
| Use case | Quick preview | Production-ready assets |
