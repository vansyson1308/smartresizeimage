# Phase 2 PR-D: Optional Decor Synthesis (BG_PLUS_DECOR)

## Objective
- Allow optional AI decor generation (light streaks, gradients, shapes, texture) in decor-only zones.
- Never modify protected zones.

## Policy behavior
- `OFF`: no decor synthesis, behavior identical to PR-C.
- `BG_PLUS_DECOR`: synthesize decor in template negative-space zones excluding protected mask.

## Zone strategy
- Build decor mask from layout template **negative space**.
- Subtract `protected_mask` from decor mask.

## Quality gate
- OCR-negative gate runs on synthesized image.
- If OCR finds text overlapping decor zones, discard synthesis and fallback to pre-decor image.

## Acceptance mapping
- Policy off parity covered by unit test.
- Policy on adds decor in allowed regions only and not on protected pixels.
- OCR negative gate execution and blocking behavior covered by unit test.
