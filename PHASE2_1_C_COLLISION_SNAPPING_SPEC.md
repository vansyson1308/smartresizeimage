# Phase 2.1-C: Collision Solver + Alignment Snapping

## Objective
Automate designer-like layer dragging for crowded layouts:
- avoid overlaps
- snap to grid
- keep consistent spacing (vertical rhythm)

## Solver loop
Input: candidate placements.
For N iterations:
1. detect pairwise overlap
2. push-apart with role-priority protection
3. snap bbox positions to grid rows/columns
4. enforce baseline spacing and margin clamps

## Debug outputs (flagged)
- Overlay PNG with bbox + role labels.
- `layout_debug.json` with bboxes, score, and violations.
- Controlled by config flag and disabled by default.

## Acceptance mapping
- Synthetic crowded fixture resolves to overlap area below threshold.
- All boxes remain within profile margins.
- Debug output toggle does not change default output when off.
