# Phase 2 PR-A Specification: Protected Assets & Masks

## Integration points in current pipeline

1. **`ReLayoutEngine` orchestration**
   - `load_file()` parses source into `DesignElement` objects and assigns semantic roles.
   - `relayout()` computes layout via `LayoutEngine.calculate_layout(...)`, then calls `CompositionEngine.compose(...)`.
   - PR-A insertion point: build masks immediately after `load_file()` output is available (source image + parsed elements), and/or just before composition for target-size aware masks.

2. **`CompositionEngine` composition layer**
   - Flat-image branch: `_compose_flat_image(...)` → `ContentAwareFitStrategy.fit(..., FitMode.SMART)`.
   - Multi-element branch: `_compose_multi_element(...)` with background then content layer compositing.
   - PR-A insertion point: attach `Masks` object so future generative stage can inpaint only editable regions while preserving protected regions.

3. **Background extension / SMART fit locations**
   - Background extension currently handled by `BackgroundExtender.extend(...)`.
   - SMART fit currently handled by `ContentAwareFitStrategy.fit(...)`.
   - PR-A output (`protected_mask`, `editable_mask`) is designed to feed those areas in Phase 2 (`BG_ONLY` and `BG_PLUS_DECOR`) when generative fill is introduced.

## Protected assets and editable zones

### Protected assets
A pixel region is protected when sourced from:
- PSD-like elements with semantic roles: `logo`, `headline/subheadline/body_text`, `cta`, `badge/label`, and `hero_image`.
- Explicitly protected metadata (`effects.protected = true`).
- Flat image OCR text boxes and saliency-derived hero/mascot region.

### Editable zones
- `editable_mask = NOT protected_mask` over the full canvas.
- This defines the only region allowed for future generative updates.

## Policy levels (for Phase 2 wiring)

- `BG_ONLY`: generate only background holes outside protected mask.
- `BG_PLUS_DECOR`: also allow decorative region replacement while keeping protected assets immutable.
- `CREATIVE_MODE` (optional): broader generation with stricter gates.

## Reproducibility requirements
Each Phase 2 run should persist:
- random seed,
- model/provider/version,
- generation params,
- hash of input image(s), mask(s), and prompt payload.

PR-A establishes deterministic mask generation to make these hashes stable and auditable.

## Quality gates + deterministic fallback
When generative stage is added:
- verify protected-mask preservation,
- verify seam quality around protected regions,
- fail closed to deterministic composition path if gates fail.

## PR-A implementation deliverables
- New module: `backend/app/generative/masks.py`
- API: `build_masks(parsed_elements, image) -> Masks`
- Deterministic behavior and logs for mask stats (protected/editable area ratios)
- Tests:
  1. PSD-like role-based protection
  2. Flat-image OCR text-mask inclusion
  3. `editable_mask = canvas - protected_mask`
