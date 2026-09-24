# Contributing

Thanks for contributing to AutoBanner.

## Setup
1. `cd backend`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements-dev.txt`
4. `python -m app.main` → studio + API on http://localhost:7860

## Local checks (required, run from the repo root)
- `ruff check backend/app backend/tests backend/tools`
- `pytest backend/tests -q`

## Where things live
- `backend/app/service.py` - the one entry point used by API, CLI and studio
- `backend/app/api/` - HTTP layer; `backend/app/web/static/` - studio
- `backend/app/presets.py` - add new platform sizes here (with `max_kb` and safe zone)
- Rendering engines: `relayout.py` (Phase 2.1) and `redesign/` (Phase 3)

## Benchmark (if layout/composition changes)
- `python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42`
- `python backend/tools/run_layout_bench.py --mode both --seed 42`

## Pull Request process
- Keep PRs scoped and reviewable.
- Update docs/tests for behavioral changes.
- Include commands run + results.

## Binary artifact policy (important)
Do **not** commit generated binaries/artifacts:
- `outputs/`, `**/outputs/`
- benchmark before/after/overlay images
- caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`)
- exports (`*.zip`)

Use deterministic generators/tests instead of committing generated artifacts whenever possible.

Exception: the README demo images in `docs/demo/` are committed on purpose. Regenerate them
with `python backend/tools/make_demo.py` whenever rendering changes, and keep the folder
small (JPEG boards, ~1.5 MB total).
