# Phase 2.1-B: Typography Reflow Engine (Auto Font Scale + Line Breaks)

## Objective
Keep text readable and hierarchy-preserving under aspect-ratio changes without rewriting content.

## Scope
- Add typography utilities to measure, wrap, and fit text blocks in constrained zones.
- Wire text sizing into layout candidate creation.
- Preserve text content (no AI rewrite, no truncation).

## Engine functions
- `measure_text(font, text) -> (w, h)`
- `wrap_text_to_width(text, font, max_width) -> lines`
- `fit_text_block(...) -> (font_size, lines, bbox)`

## Layout behavior
- Text roles use typography fit instead of pure bbox scaling.
- Strategy when text is long:
  1) reduce font down to min bound
  2) if still overflowing, expand block height (text-first tendency)
  3) adaptive candidate scoring decides final layout; rigid fallback remains available

## Acceptance mapping
- Font size never below minimum role threshold.
- Wrapped line count bounded by role line limit.
- Text block width stays within allowed zone width cap.
- Debug logs include chosen font size and line count.
