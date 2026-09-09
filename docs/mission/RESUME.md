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

Commercial completeness, in this order, each with tests in `test_ownership_and_jobs.py`:

1. Rate limiting per API key/owner in `backend/app/api/server.py` (token bucket per owner
   for `POST` routes, `429` with `Retry-After`, limits from `AUTOBANNER_RATE_LIMIT` such as
   `60/minute`, disabled when unset); record limit hits in the event log.
2. Retention policy in `backend/app/api/service.py`: `AUTOBANNER_RETENTION_DAYS` removes
   projects (and their variants, corrections, history) untouched for longer than that on
   startup and once a day from the job manager thread; document in `docs/OPERATIONS.md`
   with the restore path (project zip export before deletion is the operator's job — say
   so).
3. Roles within an owner: API keys map to `owner:role` (`editor` | `approver` | `viewer`)
   in `AUTOBANNER_API_KEYS`; viewers get `403` on document edits, generation and
   approvals; only approvers may approve/reject; editors may do everything but approve.
   Keep `LOCAL_OWNER` as editor+approver when no keys are configured.

Then update `CAPABILITIES.md` (auth/roles/retention/rate limiting → VERIFIED_LOCAL),
`docs/OPERATIONS.md`, and `ECONOMICS.md` if limits change the cost model.

## Files to know

- `backend/app/quality/` — contract v2 (checks, verdict rules)
- `backend/app/design/` — document, serialize, fonts, text_render, adapter, assets, project,
  render, variant (incl. H3 reference plans), planner, examples (H1), corrections (H5),
  decompose
- `backend/app/api/` — service (domain ops), server (FastAPI), jobs, presets
- `backend/app/web/` — index.html, app.js, styles.css
- `backend/tools/run_layout_bench.py` — modes baseline/phase21/phase3/design
- `docs/mission/` — this workspace; `evidence/` holds the preserved false positive
