# Public Release Checklist

## Pre-release verification
- [ ] `ruff check backend/app backend/tests backend/tools`
- [ ] `pytest -q`
- [ ] `python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42`
- [ ] `python backend/tools/run_layout_bench.py --mode both --seed 42`
- [ ] Runtime smoke (UI + CLI smoke for 3 sizes)
- [ ] Confirm no binary artifacts are staged/tracked

## Repo hygiene
- [ ] `git status --porcelain` clean
- [ ] No `outputs/` artifacts committed
- [ ] No cache files committed
- [ ] LICENSE present
- [ ] Security/Contributing/Code of Conduct present

## Release steps
1. Update `CHANGELOG.md` from Unreleased.
2. Tag release (e.g. `v0.1.0`).
3. Publish release notes with key features + known limitations.
4. Verify CI badge and latest workflow run status.
