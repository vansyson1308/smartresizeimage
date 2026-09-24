# AutoBanner Repository Audit & Upgrade Plan

## 1) Executive Summary
AutoBanner has a clean, modular pipeline and a practical fallback strategy for AI features, but the current rendering path is much simpler than the README vision for PSD fidelity. The architecture is coherent: UI triggers a single orchestrator (`ReLayoutEngine`) that invokes parser → classifier → layout → composition, and the test suite strongly covers parser/classifier/layout/validators behavior. The layout system is deterministic and template-based, which makes behavior stable and explainable but limited for complex creative intent. Composition quality is mixed: resize uses gamma-aware conversion and LANCZOS, but alpha handling is straight-alpha only (no explicit premultiplication pipeline), and there is no PSD effect renderer beyond storing minimal effect metadata. As a result, designer-critical visual effects (drop shadows, blend modes, glows, strokes, gradient overlays) are not reconstructed during recomposition.

The largest runtime risk discovered is environment fragility around OpenCV import: `cv2` is imported at module import-time in composition modules, and in this environment it fails due to `libGL.so.1`, preventing all composition-related tests and full relayout execution. Lint passes; a large subset of tests pass (52), while composition/content-aware/relayout tests fail during collection due to system library dependency.

Given current code, deterministic relayout can produce usable resized variants and flat-image fit/extension behavior, but it cannot reach “human-redrawn” parity for effect-heavy PSDs. Achieving that quality requires first-class PSD effect rendering + color/blend correctness; designer-redraw-like output likely requires optional generative redesign with strict brand controls.

## 2) System Map (Entry/Exit + Pipeline)

### Entry points
- CLI/web entry: `backend/app/main.py::main()` starts Gradio and builds UI callbacks.
- UI analysis callback: `analyze_file()` → `ReLayoutEngine.load_file()`.
- UI generation callback: `generate_layouts()` → `ReLayoutEngine.batch_relayout()`.

### Functional pipeline call graph
1. **UI (Gradio)**
   - `main.py:create_interface()` wires buttons and handlers.
2. **Orchestrator**
   - `relayout.py:ReLayoutEngine.load_file()`
     - validates file path
     - parser factory selection
     - parse elements
     - classify semantics
   - `relayout.py:ReLayoutEngine.relayout()`
     - validate dimensions
     - compute layout
     - compose output image
3. **Parser layer**
   - `parser/__init__.py:get_parser()` chooses `PSDParser` or `ImageParser`.
   - `ImageParser`: one flat background element + metadata.
   - `PSDParser`: recursive layer extraction + optional text/font/effects metadata capture.
4. **Classifier layer**
   - `SemanticClassifier.classify_all()`:
     - naming rules → optional CLIP → geometric heuristics fallback
5. **Layout layer**
   - `LayoutEngine.calculate_layout()`:
     - select aspect template
     - split background/content
     - assign to zones by role and capacity
     - scale/place elements with min/max clamps
6. **Composition layer**
   - `CompositionEngine.compose()`:
     - flat-image path → `ContentAwareFitStrategy.fit(SMART)`
     - multi-element path → background compose + element alpha paste
   - background extension: LaMa → OpenCV TELEA → edge-repeat fallback
   - resize: `high_quality_resize()` (gamma-aware LANCZOS)
7. **Output**
   - RGB PIL images returned in `CompositionResult`; UI displays gallery + ZIP export.

## 3) Module Structure, Models, Enums, Validators, Exceptions

### `backend/app/*` structure summary
- `main.py`: Gradio UI and user workflow handlers.
- `relayout.py`: orchestrator state + flow control.
- `parser/`: base parser, image parser, PSD parser, parser factory.
- `classifier/`: semantic role classification.
- `layout/`: template selection and zone-based placement.
- `composition/`: fit/resize/background extension/final raster composition.
- `models.py`: dataclass models.
- `enums.py`: `ElementRole`.
- `validators.py`: dimension/file validation.
- `exceptions.py`: app exception hierarchy.
- `constants.py`, `config.py`: role priorities, thresholds, quality constants.

### Data models
- `BoundingBox`, `DesignElement`, `LayoutZone`, `LayoutResult`, `CompositionResult`.

### Enums
- `ElementRole` includes headline/subheadline/cta/logo/hero/background/overlay/etc.

### Validators
- `validate_dimensions(width,height)` enforces type, min/max constraints.
- `validate_file_path(path)` enforces existence + supported extensions.

### Exceptions
- Base: `AutoBannerError`, specialized parse/format/classification/layout/composition/validation errors.

## 4) Top 10 Most Important Functions/Classes
1. `ReLayoutEngine` — central orchestrator and state container.
2. `ReLayoutEngine.load_file()` — parse + classify input and produce UI analysis payload.
3. `ReLayoutEngine.relayout()` — target-size pipeline entry for one output.
4. `get_parser()` — polymorphic parser dispatch.
5. `PSDParser._layer_to_element()` — converts PSD layer metadata/pixels into `DesignElement`.
6. `SemanticClassifier.classify()` — 3-stage semantic role classification logic.
7. `LayoutEngine.calculate_layout()` — template-driven deterministic placement.
8. `CompositionEngine.compose()` — route between flat-image and multi-element paths.
9. `ContentAwareFitStrategy.fit()` — SMART cover/contain choice and extension strategy.
10. `high_quality_resize()` — gamma-aware resampling core utility.

## 5) Technical Verification (Executed)

### Commands run
- `ruff check backend/app/` ✅ pass.
- `pytest backend/tests/ -v --cov=backend/app --cov-report=term-missing` ❌ failed (pytest-cov args unrecognized in current env).
- `pytest backend/tests/ -v` ❌ failed during collection due to `cv2` import error (`libGL.so.1` missing).
- `pytest backend/tests/test_parser.py backend/tests/test_layout.py backend/tests/test_classifier.py backend/tests/test_validators.py -v` ✅ 52 passed.
- Parser/classifier/layout fixture exercise via ad-hoc Python script ✅ 2 cases executed.

### Result summary
- Lint quality: strong (ruff clean).
- Test health: partial pass; composition stack blocked by environment-level OpenCV dynamic dependency.
- Coverage report: **not produced** in this environment (pytest-cov plugin unavailable).

## 6) 4-Axis Evaluation

### A. Functionality / Algorithm (highest priority)
**What currently works (based on code+tests):**
- Parser dispatch by extension and robust errors for unsupported formats.
- Flat image parsing with alpha detection metadata.
- PSD layer traversal with bbox filtering, text extraction attempt, and per-layer image compositing where possible.
- Role assignment pipeline with clear rule→AI→heuristic fallback.
- Deterministic template layout with role-zones, occupancy limits, scale clamping.
- Multi-size batch relayout orchestration and UI ZIP export.

**Current algorithmic limitations:**
- Layout is hard-coded templates; no optimization-based global objective, overlap solver, or typography-aware adaptive reflow.
- Heuristic classifier depends on naming/area thresholds; CLIP optional and not calibrated by domain dataset.
- Flat-image SMART fit uses a single crop-threshold heuristic; lacks saliency/face/text safety maps.
- Background extension methods are generic inpainting/edge repeat, not scene-aware geometry relighting.
- Multi-element composition ignores non-normal blend behaviors and effect synthesis.

**Hard-coded vs adaptable:**
- Hard-coded/template-heavy: templates, priority caps, zone capacities, thresholds.
- Adaptable potential: classifier backend pluggability, inpainting strategy selection, content-aware policy tuning.

### B. Quality (Visual fidelity) — Shadow/Shading focus
- Resize path is gamma-aware but quantizes to 8-bit during linear conversion, which can introduce precision loss in subtle gradients.
- Alpha pipeline uses standard PIL paste/alpha composite; no explicit premultiplied-alpha linear-light compositing for physically-correct blending.
- PSD effects are only *captured as metadata* (`_extract_effects`) and never rendered in composition.
- Blend mode metadata is stored but not applied by compositor.

**Conclusion on wrong shadows/shading:**
- If designers report mismatch, root cause is expected: no dedicated shadow/effect/blend renderer in composition path. Existing code composites flattened layer images and basic opacity only; PSD effect semantics are not reconstructed after relayout.

### C. Performance / Robustness
- Potential bottlenecks: PSD recursive parsing + `layer.composite()` per layer, optional CLIP model loading/inference, inpainting calls, repeated resizes.
- Robustness concern: hard dependency on OpenCV shared libs at import-time in composition modules can break whole app startup/tests in minimal CPU environments.
- CPU fallback claim is partially true for classifier (CLIP optional), but composition currently still needs importable `cv2`; this weakens `requirements-ci` stability in headless systems.
- Memory risk: large PSD with many RGBA layer images in memory concurrently (`DesignElement.image` per layer).

### D. Engineering quality
- Strengths: coherent separation by domain modules, typed dataclasses, readable orchestration, custom exception hierarchy.
- Weak points: some config values unused for advanced constraints, composition package import strategy tightly couples optional deps, and rendering responsibilities are underpowered for PSD parity goals.
- Scalability: adding new layout templates and classifier strategies is straightforward; adding fully-correct renderer requires significant architecture extension (effect graph, blend pipeline, color management).

## 7) Key Issue: Can quality reach human redraw equivalence?

### (i) Deterministic relayout only → **Impossible / not recommended for parity target**
- Why: current system rearranges/scales layers and inpaints extensions; it does not recreate nuanced lighting/shadow artistry or full PSD effect stack.
- Needed changes: n/a if staying deterministic-only; expectation must be “layout adaptation,” not “redesign parity.”
- Risks: stakeholder dissatisfaction due to mismatched quality target.
- Success criteria: geometric consistency, legibility, artifact-free outputs — not artistic equivalence.

### (ii) Approximate parity possible with renderer upgrades → **Conditionally achievable**
- Needed: implement PSD effect rendering (drop/inner shadow, glow, stroke, gradient overlay), proper blend modes, linear-light compositing with premultiplied alpha, ICC/color-space handling.
- Impact: major visual uplift for designer-authored PSDs while preserving deterministic control.
- Risks: high complexity and many edge-cases across PSD spec variants.
- Criteria: visual diff thresholds vs Photoshop reference renders on golden PSD corpus.

### (iii) Designer-redraw-like outputs via generative redesign → **Possible but risky**
- Needed: optional diffusion/relighting/generative fill pipeline plus brand guardrails.
- Impact: can mimic “redesign” beyond deterministic transforms.
- Risks: brand/logo drift, text corruption, artifacts, non-determinism, compliance concerns.
- Criteria: human review scores + brand fidelity checks + OCR/logo consistency metrics.

## 8) Upgrade Backlog (Prioritized, Actionable)

### Quick Wins (low risk, high impact)
1. **Decouple OpenCV import from module import path**
   - Objective: make CPU/headless fallback reliable.
   - Files: `backend/app/composition/content_aware_fit.py`, `backend/app/composition/background.py`, `backend/app/composition/__init__.py`.
   - Method: lazy import cv2 inside methods; graceful fallback if unavailable.
   - Acceptance: full test collection runs without `libGL` present; composition tests skip/fallback deterministically.
   - Effort: S.

2. **Add coverage tooling parity in dev/CI docs**
   - Objective: ensure pytest-cov availability and reproducible coverage command.
   - Files: `backend/requirements-dev.txt`, `README.md`, CI workflow.
   - Method: enforce plugin in dev deps and CI setup check.
   - Acceptance: `pytest ... --cov` works in clean env.
   - Effort: S.

3. **Instrument timing/memory logs per pipeline stage**
   - Objective: expose bottlenecks empirically.
   - Files: `backend/app/relayout.py`, parser/classifier/composition modules.
   - Method: stage timers + optional psutil memory metrics.
   - Acceptance: logs report parse/classify/layout/compose latency in each run.
   - Effort: S.

### Medium (refactor/design)
1. **Implement blend mode support in compositor**
   - Objective: preserve PSD visual intent beyond normal blend.
   - Files: `backend/app/composition/engine.py` (+ new blend utility module).
   - Method: linear-space blend ops for common modes (normal, multiply, screen, overlay).
   - Acceptance: golden image tests against Photoshop exports for supported modes.
   - Effort: M.

2. **Add effect rendering pipeline (drop shadow/stroke/glow/gradient overlay)**
   - Objective: fix primary shadow mismatch complaints.
   - Files: parser effects extraction + new composition effects renderer.
   - Method: parse effect params deeply, render into intermediate layers before final composite.
   - Acceptance: PSD effect fixtures visually match reference within tolerance.
   - Effort: M/L.

3. **Template engine extensibility + conflict solver**
   - Objective: reduce overlap/truncation in dense layouts.
   - Files: `backend/app/layout/engine.py`, `templates.py`.
   - Method: scoring-based zone assignment and collision checks.
   - Acceptance: synthetic stress tests with >N elements keep key roles visible.
   - Effort: M.

### Big Bets (AI/generative R&D)
1. **Saliency/face/text-aware crop policy**
   - Objective: smarter SMART cover decisions.
   - Files: `content_aware_fit.py` + optional model adapters.
   - Method: detect protected regions; optimize crop window.
   - Acceptance: benchmark set improves content-retention metric.
   - Effort: L.

2. **Generative background relighting/fill module**
   - Objective: blend extended backgrounds with subject lighting.
   - Files: new generative subsystem + composition integration.
   - Method: diffusion inpaint/relight with mask constraints.
   - Acceptance: human MOS uplift without brand drift above threshold.
   - Effort: L.

3. **Full “designer-assist” mode (optional)**
   - Objective: produce near-redraw creative variants.
   - Files: new generation orchestration stack.
   - Method: prompt+controlnet+brand locks+post-validation.
   - Acceptance: brand compliance pass + creative quality review.
   - Effort: L.

## 9) First 2–3 PRs I would implement next
1. **PR-1: Composition dependency hardening**
   - Scope: lazy import OpenCV/LaMa + fallback paths + tests for no-cv2 environment.
   - Why first: unblocks reliability and CI confidence immediately.

2. **PR-2: Blend mode foundation + linear compositing utility**
   - Scope: introduce compositing utility module and support 3–4 high-value blend modes.
   - Why: highest visual ROI before tackling full PSD effects.

3. **PR-3: Drop shadow + stroke renderer MVP with golden fixtures**
   - Scope: render common effect params for PSD layers and verify against baseline references.
   - Why: directly addresses designer shadow mismatch complaint with measurable checks.
