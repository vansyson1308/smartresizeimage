# Contributing

Thanks for contributing to AutoBanner.

## Setup
1. `cd backend`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements-dev.txt`

## Local checks (required)
- `ruff check backend/app backend/tests backend/tools`
- `pytest -q`

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
