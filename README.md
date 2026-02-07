# AutoBanner

Automatic banner re-layout engine with dual-mode processing.

## Architecture

AutoBanner has two independent processing modes:

- **Frontend (Local WASM)**: Runs entirely in the browser using WebAssembly-based background removal (`@imgly/background-removal`) and Canvas API composition. Supports batch generation across 11 preset ad formats (Social Media, Google Display Ads, Video & Display). Zero server dependency.

- **Backend (AI-Powered PSD Re-Layout)**: Python service using Gradio for the UI. Parses PSD files (and PNG/JPG/WEBP), classifies layer semantics with CLIP, calculates adaptive layouts for any target aspect ratio, and composes the final output with gamma-correct resizing and AI inpainting (LaMa) for background extension.

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
python -m backend.app.main
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Testing

### Backend (pytest)

```bash
pytest backend/tests/ -v --cov=backend/app --cov-report=term-missing
```

### Frontend (vitest)

```bash
cd frontend
npm test
```

### Linting

```bash
ruff check backend/app/
cd frontend && npx tsc --noEmit
```

## Supported Input Formats

| Format | Backend | Frontend |
|--------|---------|----------|
| PSD    | Full layer parsing | - |
| PNG    | Single-layer | Full |
| JPG    | Single-layer | Full |
| WEBP   | Single-layer | Full |

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
│   │   ├── composition/          # Composition, resize, background
│   │   └── relayout.py           # Orchestrator
│   └── tests/                    # pytest test suite
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
- Backend: Lint (ruff) + Test (pytest) + Coverage
- Frontend: TypeScript check + Build
- Docker: Image build verification
