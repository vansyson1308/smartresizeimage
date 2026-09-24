# Phase 2 PR-C: Harmonize & Grounding Shadow (Safe)

## Objective
- Improve final visual cohesion after composition while preserving brand/protected pixels.

## Safe steps
1. **Constrained color grading**
   - Slight channel gain adjustment on editable/background pixels.
   - Protected pixels are restored exactly (pixel-equal).
2. **Grounding shadow**
   - Build a shadow layer from mascot alpha masks.
   - Blur + offset shadow and composite beneath content.
   - Protected pixels restored exactly after shadow composite.

## Integration
- Applied in `ReLayoutEngine` post-compose stage.
- Uses `build_layout_masks(...)` + mascot mask extraction.

## Acceptance checks
- Protected pixels unchanged in tests.
- Shadow MVP deterministic hash test.
- Full test suite and lint pass.
