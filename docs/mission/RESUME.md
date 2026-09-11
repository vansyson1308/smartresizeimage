# Resume here

## Environment

```bash
cd /path/to/smartresizeimage
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-ci.txt
.venv/bin/pip install "uvicorn[standard]" playwright   # server + browser checks
sudo apt-get install -y tesseract-ocr                   # optional OCR (else not_checked)
export OMP_THREAD_LIMIT=1                               # keeps tesseract from oversubscribing
```

## Verify

```bash
.venv/bin/ruff check backend
.venv/bin/python -m pytest backend/tests -q
.venv/bin/python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
.venv/bin/python backend/tools/run_layout_bench.py --mode design --seed 42 --outdir /tmp/bench_design
.venv/bin/python backend/tools/run_layout_bench.py --mode both   --seed 42 --outdir /tmp/bench_both
```

## Run the product

```bash
AUTOBANNER_DATA_DIR=./data .venv/bin/uvicorn backend.app.api.server:app --port 8000
# open http://localhost:8000  (API docs at /api/docs)
```

## Browser journey check

Committed at `backend/tests/e2e/test_journey.py` (Playwright + Chromium against a real
uvicorn server with `AUTOBANNER_AUTH=local`; CI job `browser-journey`):

```bash
.venv/bin/python -m playwright install --with-deps chromium   # once (CI does the same)
AUTOBANNER_E2E_SHOTS=/tmp/shots .venv/bin/python -m pytest backend/tests/e2e -rs
```

Screenshots and `journey_report.json` (every step with its result) land in
`AUTOBANNER_E2E_SHOTS`. In this environment Chromium is at `/opt/pw-browsers/chromium`
(the test finds it through `PLAYWRIGHT_BROWSERS_PATH`).

## Next executable task

Mission V2 phase A (audit closure) is done locally; see `LEDGER.json`. Continue with phase B,
in this order:

Phase B is done locally: campaign table (`rows`, CSV import, per-row review/export,
`tools/run_campaign.py`), restart resume, brand profiles, plan entitlements, offline guard.

1. Phase C (owned intelligence): creative directions per campaign row as planner inputs;
   adaptive grammar/compiler over the layout families; mascot/subject invariants;
   incremental compilation; counterexample tooling.
2. Then phase D (R1–R4 on a designed 12-brand corpus with baselines and ablations) and
   phase E (Docker build on a machine with a daemon, packaging, Vietnamese handoff).

## Files to know

- `backend/app/quality/` — contract v2 (checks, verdict rules)
- `backend/app/design/` — document, serialize, fonts, text_render, adapter, assets, project,
  render, variant (incl. H3 reference plans), planner, examples (H1), corrections (H5),
  decompose
- `backend/app/api/` — service (domain ops, retention), server (FastAPI, roles, rate limit),
  auth (local users, sessions, personal tokens),
  jobs, presets, events, ratelimit
- `backend/app/web/` — index.html, app.js, styles.css
- `backend/tools/run_layout_bench.py` — modes baseline/phase21/phase3/design
- `docs/mission/` — this workspace; `evidence/` holds the preserved false positive
