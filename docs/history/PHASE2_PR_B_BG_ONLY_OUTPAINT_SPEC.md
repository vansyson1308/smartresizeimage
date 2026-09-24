# Phase 2 PR-B Specification: Generative Background Outpainting (BG_ONLY)

## Scope
- Apply generation only in `editable_mask` regions.
- Never alter protected pixels.
- Fall back to deterministic background if backend is unavailable.

## Architecture
- `backend/app/generative/engine.py`
  - `GenerativeBackend` protocol for adapter pattern.
  - `GenerativeOutpaintEngine.outpaint_background(base_canvas, editable_mask, policy, seed) -> image`.
  - `NullGenerativeBackend` for deterministic fallback.

## Pipeline integration
- `ReLayoutEngine.relayout()` computes deterministic layout first.
- `CompositionEngine.compose(..., bg_outpaint_fn=...)` invokes outpaint right after deterministic background composition and before content layers are pasted.
- `build_layout_masks(...)` creates target-canvas protected/editable masks from layout results.

## Data / metadata
- Store run metadata in `CompositionResult.metadata["generative"]`:
  - `seed`
  - `policy`
  - `model_id`
  - backend usage/fallback reason
  - protected/editable area ratios

## Test plan
1. Backend unavailable -> deterministic fallback path runs.
2. Mock backend -> output size is correct and protected region is pixel-equal before/after.
3. Relayout result includes run metadata.
