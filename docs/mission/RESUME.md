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
→ 900px viewport without horizontal scroll → no console errors. Re-create it from this list if
needed; launch Chromium with `executable_path="/opt/pw-browsers/chromium"` in this environment.

## Next executable task

H1 — learn from approved examples: add `backend/app/design/examples.py` with
`match_elements(master_doc, approved_variant_doc)` (by asset hash for images, by normalised
text for text elements, by role as fallback) and `infer_family(master_doc, examples)` that
derives, per aspect class, region fractions (text column, subject slot, logo slot) and a
hierarchy scale from the approved variants, emitting a `Family` plus `Constraint`
proposals with `provenance.origin = "recovered"` and a confidence based on agreement across
examples. Feed inferred families into `choose_families` ahead of the hand-written ones.
Evaluate on the synthetic corpus by treating one generated size as the "approved example"
and holding out the others (`run_ablations.py --configs full,learned`). Record H1 in
`EXPERIMENTS.md` with the setup time counted (import + match + confirm).

## Files to know

- `backend/app/quality/` — contract v2 (checks, verdict rules)
- `backend/app/design/` — document, serialize, fonts, text_render, adapter, assets, project,
  render, variant
- `backend/app/api/` — service (domain ops), server (FastAPI), jobs, presets
- `backend/app/web/` — index.html, app.js, styles.css
- `backend/tools/run_layout_bench.py` — modes baseline/phase21/phase3/design
- `docs/mission/` — this workspace; `evidence/` holds the preserved false positive
