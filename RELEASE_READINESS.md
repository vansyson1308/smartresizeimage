# Release Readiness Report

## Proposed version
- **v0.1.0-rc1** (release candidate)

## Scope included in this release gate
- Phase 2 deterministic relayout pipeline with adaptive layout profile/scoring/solver/typography.
- Composition stack upgrades: linear compositing, blend-modes MVP, drop-shadow MVP, text-safe plate.
- Phase 2.1 benchmark pack: fixture generator, benchmark runner, metrics, thresholds, and auto report.

## Not included / known limitations
- Full PSD effect parity (stroke/glow/gradient overlay and full Photoshop-equivalent rendering) is not complete.
- Generative redesign quality is gated and conservative; heavy generative models are intentionally not part of benchmark mode.
- UI runtime currently has a Gradio/OpenAPI schema crash in this environment (see blocker #1).

---

## A) Repo hygiene / safety

### A1. Working tree + artifact safety
**Commands run**
- `git status --short`
- `git ls-files | rg '(outputs/|__pycache__/|\.pyc$|\.zip$|\.psd$|\.ruff_cache|\.pytest_cache|\.DS_Store)'`
- `find . -type f -not -path './.git/*' -size +1M`
- `sed -n '1,220p' .gitignore`

**Results**
- Working tree clean before edits for this gate.
- Tracked suspicious artifact path found only: `backend/tests/fixtures/outputs/.gitkeep` (expected).
- Benchmark outputs are not tracked and are ignored by `.gitignore` (`backend/tests/fixtures/outputs/*` with `.gitkeep` exception).
- Caches and common generated files are ignored (`__pycache__/`, `*.pyc`, `.pytest_cache/`, etc.).

### A2. Secrets scan
**Command run**
- `rg -n --hidden --glob '!.git' '(AKIA...|AIza...|ghp_...|github_pat_...|xox...|sk-...|PRIVATE KEY|api[_-]?key\s*[:=]|token\s*[:=]|secret\s*[:=]|password\s*[:=])' .`

**Results**
- No credential matches.
- Only false positives on variable name `token` in typography wrapping logic.

### A3. License & attribution
**Command run**
- `ls | rg '^LICENSE$|^LICENSE\.md$|^COPYING'`

**Results**
- **BLOCKER:** no repository license file found.
- Third-party attribution references were not found in top-level docs for bundled sample assets.

---

## B) Build & Test Matrix

### B1. Lint
**Command run**
- `ruff check backend/app backend/tests backend/tools`

**Result**
- Pass.

### B2. Unit tests
**Command run**
- `pytest -q`

**Result**
- Pass (`161 passed`).

### B3. Benchmark (Phase 2.1-G)
**Commands run**
- `python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42`
- `python backend/tools/run_layout_bench.py --mode both --seed 42`

**Results**
- Fixture generation pass (12 cases).
- Benchmark run pass; report generated at:
  - `backend/tests/fixtures/outputs/bench_phase21/report.md`
  - `backend/tests/fixtures/outputs/bench_phase21/summary.json`

### B4. Determinism check
**Commands run**
- `python backend/tools/run_layout_bench.py --mode both --seed 42` (second run)
- `sha256sum backend/tests/fixtures/outputs/bench_phase21/report.md backend/tests/fixtures/outputs/bench_phase21/summary.json`

**Result**
- Same hashes across repeated runs with identical seed/config.

### B5. Dependency sanity (clean venv/headless)
**Commands run**
- `python -m venv /tmp/releasegate-venv && source ... && pip install -r backend/requirements-ci.txt`
- Fallback validation: `python -m venv /tmp/releasegate-venv-ssp --system-site-packages && source ... && pytest -q backend/tests/test_composition_headless.py backend/tests/test_content_aware_fit.py::TestHeadlessFallback::test_content_aware_fit_falls_back_without_cv2`

**Results**
- Clean venv dependency install failed due network/proxy restriction (`Tunnel connection failed: 403 Forbidden`, no package download).
- System-site-packages venv fallback succeeded (`2 passed`), confirming headless fallback paths are operational.

---

## C) Runtime smoke tests

### C1. Run app per README
**Command run**
- `cd backend && timeout 20s env GRADIO_ANALYTICS_ENABLED=false AUTOBANNER_SHARE=true python -m app.main; echo EXIT:$?`

**Result**
- **BLOCKER:** Gradio ASGI exception during API schema generation:
  - `TypeError: argument of type 'bool' is not iterable`
  - stack in `gradio_client/utils.py` -> `json_schema_to_python_type`
- Process timed out (`EXIT:124`) and repeatedly errored while serving.

### C2. End-to-end generation (3 sizes)
**Command run**
- Python smoke script with `ReLayoutEngine(use_ai=False)`:
  - load one input image
  - relayout to `(1200,628)`, `(1080,1080)`, `(1080,1920)`

**Result**
- Pass; output files created under `backend/tests/fixtures/outputs/release_gate/`.

### C3. Product guarantees validation
**Command run**
- `pytest -q backend/tests/test_harmonize.py::test_color_grading_keeps_protected_logo_text_unchanged backend/tests/test_generative_engine.py::test_outpaint_background_preserves_protected_pixels_when_mocked_backend backend/tests/test_generative_engine.py::test_outpaint_background_fallback_when_backend_unavailable backend/tests/test_relayout.py::TestReLayoutEngine::test_failed_quality_gate_triggers_fallback`

**Result**
- Pass (`4 passed`): protected pixel invariance and fallback behavior verified.

---

## D) Docs & UX

### D1. README completeness
Verified present:
- quickstart/setup commands,
- troubleshooting (headless/cv2 and proxy notes),
- Phase 2 / 2.1 policy and benchmark sections,
- testing and benchmark commands.

### D2. Examples section
Added a minimal **Examples** section with a CLI-style smoke command and expected outputs.

---

## Risk register (top 5)
1. **UI server crash (Gradio schema path)** — High  
   Mitigation: pin compatible Gradio/FastAPI versions or patch API schema generation path in app interface definitions.
2. **Missing LICENSE** — High (distribution/legal)  
   Mitigation: add explicit project license before public release.
3. **Network-dependent clean venv validation unavailable in constrained envs** — Medium  
   Mitigation: run in CI with open package index and cache wheels.
4. **Adaptive solver often falls back due heavy violations on synthetic extremes** — Medium  
   Mitigation: tune profile thresholds and candidate generation; keep benchmark trend monitoring.
5. **Headless OpenCV unavailable in some targets** — Low/Medium  
   Mitigation: keep existing deterministic edge-repeat fallback and document quality tradeoff.

---

## Go / No-Go decision
- **Decision: NO-GO**

### Blocking issues
1. **Missing LICENSE file** (legal release blocker).
2. **Runtime UI smoke failure** (`python -m app.main`) due Gradio schema exception:
   - `TypeError: argument of type 'bool' is not iterable`
   - reproducible via command in section C1.

### Smallest fix set to reach GO
1. Add `LICENSE` (and optional attribution note for any bundled assets).
2. Fix/pin UI dependency/runtime path to make `python -m app.main` start without recurring ASGI exception.
3. Re-run this release gate checklist after fixes.

---

## Exact command ledger (this gate)
- `git status --short`
- `git ls-files | rg '(outputs/|__pycache__/|\.pyc$|\.zip$|\.psd$|\.ruff_cache|\.pytest_cache|\.DS_Store)'`
- `find . -type f -not -path './.git/*' -size +1M`
- `sed -n '1,220p' .gitignore`
- `rg -n --hidden --glob '!.git' '(AKIA...|AIza...|ghp_...|github_pat_...|xox...|sk-...|PRIVATE KEY|api[_-]?key\s*[:=]|token\s*[:=]|secret\s*[:=]|password\s*[:=])' .`
- `ls | rg '^LICENSE$|^LICENSE\.md$|^COPYING'`
- `ruff check backend/app backend/tests backend/tools`
- `pytest -q`
- `python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42`
- `python backend/tools/run_layout_bench.py --mode both --seed 42`
- `python backend/tools/run_layout_bench.py --mode both --seed 42` (repeat)
- `sha256sum backend/tests/fixtures/outputs/bench_phase21/report.md backend/tests/fixtures/outputs/bench_phase21/summary.json`
- `python -m venv /tmp/releasegate-venv ... pip install -r backend/requirements-ci.txt` (failed: proxy 403)
- `python -m venv /tmp/releasegate-venv-ssp --system-site-packages ... pytest -q backend/tests/test_composition_headless.py ...`
- `cd backend && timeout 20s env GRADIO_ANALYTICS_ENABLED=false AUTOBANNER_SHARE=true python -m app.main; echo EXIT:$?`
- `python - <<'PY' ... ReLayoutEngine 3-size smoke ... PY`
- `pytest -q backend/tests/test_harmonize.py::... backend/tests/test_generative_engine.py::... backend/tests/test_relayout.py::...`

## Post-launch next steps (once GO)
1. Add CI job for benchmark runner + report artifact upload.
2. Track benchmark score trends and violation hot spots over time.
3. Expand fixture set with 2–3 real-world anonymized banners.
4. Add release automation for changelog + signed tag.
