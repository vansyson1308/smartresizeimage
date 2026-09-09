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

The Playwright script used for verification lives outside the repo during the session
(`scratchpad/ui_journey.py`); its steps: create blank canvas → add text (prompt) → add logo
(file input) → add elements via API → mark price verbatim → drag/nudge/undo → add rule →
brief with custom size + copy override → generate → review → approve / reject with reason →
per-variant override regenerate → export approved zip → save project zip → reopen via import
→ learned-rules panel shows the approved variant → 900px viewport without horizontal scroll
→ no console errors. Re-create it from this list if
needed; launch Chromium with `executable_path="/opt/pw-browsers/chromium"` in this environment.

## Next executable task

The research track (H1–H5) and the local commercial layer are in place; what remains needs
things this environment does not have (real designs, human reviewers, provider credentials,
a Docker daemon). Next executable items, in order:

1. Brand-level rule carry-over (H5 follow-up): projects share `meta.brand`; add
   `GET /api/brands/{brand}/rules` that unions confirmed correction rules and learned
   families across an owner's projects with that brand (keyed by role), and let
   `request_variants` apply them as proposals for a new project of the same brand. Test:
   two projects, rule confirmed in the first, proposed in the second.
2. Docker build verification: on a machine with a daemon run `docker compose build && docker
   compose up`, hit `/api/health`, run the browser journey against port 8000, and record the
   image size and cold-start time in `docs/OPERATIONS.md`.
3. When real designs or reviewers become available: run the pilot instruments
   (`/api/pilot/summary`) on a real campaign, calibrate the verdict against reviewer decisions,
   and replace the synthetic numbers in `EVALUATION.md` with measured ones.

## Files to know

- `backend/app/quality/` — contract v2 (checks, verdict rules)
- `backend/app/design/` — document, serialize, fonts, text_render, adapter, assets, project,
  render, variant (incl. H3 reference plans), planner, examples (H1), corrections (H5),
  decompose
- `backend/app/api/` — service (domain ops, retention), server (FastAPI, roles, rate limit),
  jobs, presets, events, ratelimit
- `backend/app/web/` — index.html, app.js, styles.css
- `backend/tools/run_layout_bench.py` — modes baseline/phase21/phase3/design
- `docs/mission/` — this workspace; `evidence/` holds the preserved false positive
