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

Two small, independent items, then a consolidation pass:

1. Event-log retention: `EventLog.trim(owner, days)` keeping the last N days of
   `data/events/<owner>.jsonl`, called from `ProjectService.purge_stale_projects` with the
   same `AUTOBANNER_RETENTION_DAYS`; test in `test_ownership_and_jobs.py`; note in
   `docs/OPERATIONS.md`.
2. Per-element text plates (H3 residual): in `generative/text_plate.py`, cluster boxes per
   element instead of per stack when `TextPlateConfig.per_element` is set, and use that
   mode for reference-plan revisions so a longer CTA on a busy background stays inside its
   own padded box; re-run `run_local_edits.py` and update the H3 entry in `EXPERIMENTS.md`.
3. Consolidation: re-run the browser journey and all three measurement tools on the final
   head, refresh `CAPABILITIES.md` statuses and the PR description, and record the release
   criteria check in `STATE.md` (what is VERIFIED_LOCAL vs BLOCKED_EXTERNAL: real corpus,
   human calibration, provider credentials, Docker daemon).

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
