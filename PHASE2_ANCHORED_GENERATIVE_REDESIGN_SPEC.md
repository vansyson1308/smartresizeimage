# Phase 2 Technical Specification: Anchored Generative Redesign

## Scope and non-goals
- **Goal:** handle extreme aspect-ratio shifts (for example 16:9 → 9:16) while preserving mascot/logo/text and improving visual quality beyond deterministic edge extension.
- **Constraint:** anchored assets must remain semantically and visually stable (identity-safe).
- **Non-goals for Phase 2:** full PSD parity for all Adobe effects, arbitrary typography regeneration, and unconstrained style transfer.

## 1) Current integration points in the codebase

### 1.1 ReLayoutEngine orchestration entry points
Current orchestration flow:
1. `ReLayoutEngine.load_file()` validates input, chooses parser via `get_parser`, parses elements, and classifies semantics.
2. `ReLayoutEngine.relayout()` validates target size, runs `LayoutEngine.calculate_layout(...)`, then calls `CompositionEngine.compose(...)`.
3. `batch_relayout()` loops over target sizes and calls the same deterministic pipeline.

**Best Phase 2 integration point:** extend `ReLayoutEngine.relayout()` to select a **render policy** (`BG_ONLY`, `BG_PLUS_DECOR`, `CREATIVE_MODE`) and pass a policy config object into composition.

### 1.2 CompositionEngine integration points
Current composition has two pathways:
- **Flat-image path** (`_compose_flat_image`): uses `ContentAwareFitStrategy.fit(..., FitMode.SMART)`.
- **Multi-element path** (`_compose_multi_element`): composes background, then content layers with linear compositing + blend/drop shadow support.

**Best Phase 2 integration points:**
- Add an optional **anchored generative background stage** between `_compose_background` preparation and content compositing.
- Keep existing layer compositing for protected assets (logo/mascot/text) unchanged except geometric layout transforms.

### 1.3 Location of background extension / SMART fit today
- Background extension currently lives in `BackgroundExtender.extend(...)` and is used by composition when target is larger than source background.
- Flat image SMART fit is handled by `ContentAwareFitStrategy.fit(...)` with cover/contain decision and inpaint/edge-repeat fallback.

**Best Phase 2 insertion:**
- Introduce a new strategy selector adjacent to content-aware-fit/background extension:
  - deterministic extension (existing)
  - anchored generative fill (new)
  - deterministic fallback if quality gates fail

---

## 2) Anchored Generative Redesign technical specification

## 2.1 Asset zoning model

### Protected assets (must preserve)
Protected assets are elements that must not be regenerated:
- `LOGO`, `CTA`, `HEADLINE`, `SUBHEADLINE`.
- Any element with explicit lock metadata: `effects["protected"] == true`.
- Optional: mascot/character class from classifier confidence + user override.

Protected asset guarantees:
- Pixel content source-of-truth remains original layer/image content.
- Only allowed transforms: translation, scale (bounded), optional rotation = 0 by default.
- No generative overwrite inside their alpha footprint.

### Editable zones
Editable zones are regions where generation is allowed:
- Primary: uncovered target canvas areas after protected asset placement.
- Secondary: existing background/decorative layers (`BACKGROUND`, `DECORATION`, optional `HERO_IMAGE` only in creative mode).

Forbidden zones:
- Dilated masks around protected assets (`safe_margin_px`) to prevent bleed/halo.

## 2.2 Policy levels

### Policy A: `BG_ONLY` (default production-safe)
- Generation only in editable background holes.
- Protected assets + decorative layers are deterministic.
- Intended for brand-safe automation.

### Policy B: `BG_PLUS_DECOR`
- Background + decoration layers can be regenerated/retextured.
- Protected assets remain frozen.
- Use when deterministic output is structurally correct but visually weak.

### Policy C: `CREATIVE_MODE` (optional, guarded)
- Allows limited regeneration of non-protected hero/decorative content.
- Requires stricter quality gates and explicit opt-in.
- Should be disabled by default in API/UI.

## 2.3 Reproducibility and auditability contract
Each output must persist a sidecar manifest (JSON) with:
- `run_id`, timestamp.
- `policy_level`.
- `seed`.
- model identifiers (`provider`, `model_name`, `model_version`).
- generation params (steps, guidance, denoise strength, scheduler).
- input hashes (`sha256`) of source assets and protected masks.
- prompt payload hashes (positive/negative/system prompt hashes).
- quality-gate metrics and pass/fail reasons.
- fallback reason if deterministic path used.

Determinism expectations:
- Same inputs + seed + model version + params should reproduce near-identical output (within tolerance).
- If model or prompt changes, manifest makes drift explicit.

## 2.4 Quality gates + deterministic fallback
Quality gates run before final accept:
1. **Protected overlap gate:** generated pixels must not alter protected masks above threshold.
2. **Text/logo integrity gate:** OCR/logo hash similarity for protected assets remains within tolerance.
3. **Edge seam gate:** boundary ring metrics (MAE/gradient discontinuity) below threshold.
4. **Global plausibility gate:** simple no-artifact checks (saturation clipping %, dead alpha, NaN).

Fallback behavior:
- If any mandatory gate fails, auto-fallback to deterministic composition path (existing SMART fit/background extension).
- Emit warning and include gate failure details in manifest.

## 2.5 Data contracts (new DTOs)
Proposed additions:
- `ProtectedAsset`: element id, role, mask, bbox, lock reason.
- `EditableZone`: region mask, source role(s), policy constraints.
- `GenerativeRequest`: policy, prompt bundle, seed, constraints, target size.
- `GenerativeResult`: generated image, debug masks, gate metrics, manifest payload.

These should be pure data objects so they are testable independently of inference backends.

---

## 3) Phase 2 PR plan (PR-A..PR-E)

## PR-A — Policy scaffolding + zoning extraction
**Objective**
- Introduce policy enum and extract protected/editable masks from existing classified elements.

**Implementation scope**
- Add policy enum and request config in models/config.
- Add zoning extractor in relayout/composition boundary.
- No model inference yet.

**Acceptance criteria**
- Unit tests validate protected/editable mask generation for synthetic PSD-like inputs.
- `BG_ONLY` default path preserves current behavior when generative backend disabled.
- `ruff check` + `pytest -q` green.

## PR-B — Generative backend abstraction + no-op deterministic adapter
**Objective**
- Add pluggable interface for generation without tying code to one provider.

**Implementation scope**
- `GenerativeBackend` protocol/interface with `generate(request) -> result`.
- Deterministic no-op adapter returns current extended background.
- Wire backend selection via config/env.

**Acceptance criteria**
- Contract tests for backend interface and adapter behavior.
- System runs with backend disabled/enabled(no-op) identically.
- Manifest object emitted for every render.

## PR-C — Anchored inpaint implementation for `BG_ONLY`
**Objective**
- Real anchored generation on background holes while freezing protected assets.

**Implementation scope**
- Build inpaint masks from editable zones minus dilated protected masks.
- Call backend with seed + prompt package.
- Composite generated background under protected layers.

**Acceptance criteria**
- Golden/snapshot tests show preserved protected pixels (strict mask diff threshold).
- At least 2 synthetic extreme-ratio fixtures pass quality gates.
- On backend failure, deterministic fallback triggers without crash.

## PR-D — Quality gate framework + enforced fallback
**Objective**
- Prevent unsafe generative outputs from shipping.

**Implementation scope**
- Implement gate evaluators (protected overlap, seam quality, artifact checks).
- Add gate report to sidecar manifest.
- Enforce fallback in orchestration path.

**Acceptance criteria**
- Tests inject failing gate conditions and verify fallback activation.
- Diff artifacts saved for gate-failure debugging.
- No regression in existing deterministic tests.

## PR-E — `BG_PLUS_DECOR` policy + UI/CLI controls + docs
**Objective**
- Add next policy level and expose controls safely.

**Implementation scope**
- Extend zoning rules to allow decor generation.
- Add UI/API controls for policy, seed, and strictness.
- Update README and test fixture docs for reproducible generative runs.

**Acceptance criteria**
- E2E smoke test: same seed reproduces same output hash window.
- Clear warnings for unsupported backend/model.
- Documentation includes operational limits and rollback instructions.

---

## 4) Risks and mitigations
- **Brand drift risk:** mitigate with protected-mask hard constraints + gate enforcement.
- **Non-deterministic outputs:** mitigate with mandatory seed + manifest + model pinning.
- **Performance cost:** mitigate with policy defaults (`BG_ONLY`) and configurable step limits.
- **Operational fragility:** mitigate via backend abstraction and deterministic fallback path.

## 5) Definition of done for Phase 2 (overall)
- Policy-based anchored generation integrated into `ReLayoutEngine` and `CompositionEngine` without breaking deterministic mode.
- Reproducibility manifest generated per render.
- Quality gates active and tested, with automatic deterministic fallback.
- Golden regression suite expanded for extreme aspect-ratio scenarios with protected-asset invariance checks.
