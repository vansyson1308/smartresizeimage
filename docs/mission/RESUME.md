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

H1 follow-up — content pressure for learned stacked families: in
`backend/app/design/examples.py`, when the example is a stacked composition
(`subject_first` or text below subject), record the text stack's share of the canvas height
and let `_plan_family` in `planner.py` grow the learned text region on smaller canvases the
way `_stacked_family` does for hand-written families (square/portrait agreement with the
designer plan is 0.58/0.59 vs 0.85 for landscape in
`docs/mission/results/ablations_h1_2026-09-09.md`). Re-run
`run_ablations.py --configs joint,designer_ref,learned` with the seven sizes listed in
`EVALUATION.md` and record the delta in `EXPERIMENTS.md`. Then start H3 (local-edit
regression set: pixel diff outside the edited scope must be zero).

## Files to know

- `backend/app/quality/` — contract v2 (checks, verdict rules)
- `backend/app/design/` — document, serialize, fonts, text_render, adapter, assets, project,
  render, variant, planner, examples (H1), decompose
- `backend/app/api/` — service (domain ops), server (FastAPI), jobs, presets
- `backend/app/web/` — index.html, app.js, styles.css
- `backend/tools/run_layout_bench.py` — modes baseline/phase21/phase3/design
- `docs/mission/` — this workspace; `evidence/` holds the preserved false positive
