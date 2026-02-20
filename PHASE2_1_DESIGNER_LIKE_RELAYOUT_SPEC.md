# Phase 2.1 SPEC: Designer-like Re-Layout (Adaptive Layout)

## 1) Current system review

### LayoutEngine today
- Uses rigid aspect-ratio templates with predefined zones.
- Assigns roles to zones, scales elements into zone bounds, and applies simple visibility fallback.
- No candidate search/scoring; template is selected once and applied directly.

### Composition today
- Background first, then content sorted by z-index.
- Supports linear compositing, blend modes, drop shadow, safe harmonize/grounding.

### Phase 2 safety policies already available
- Protected/editable masks.
- BG-only / BG+decor generative stages.
- Quality gates (OCR, similarity, color drift) + deterministic fallback.

## 2) Definition: "Designer-like Re-Layout" (style-preserving)
- Preserve brand identity and visual hierarchy: `headline > subheadline > CTA > logo`.
- Enforce grid alignment + margins + safe areas.
- Keep consistent spacing to avoid crowding.
- Keep hero/mascot prominent (minimum area ratio profile-driven).
- Keep text-safe readability (clean plate strategy reserved for future PR if clutter detected).
- Deterministic: same input + same config yields same layout output.

## 3) Measurable acceptance metrics
- Overlap: no protected overlaps (or within small threshold).
- Safe margins: all content boxes stay inside profile margins.
- Min sizes by role: headline/subheadline/CTA/hero obey profile thresholds.
- Hero prominence: hero area ratio above profile minimum target.
- Determinism: scoring-based selection deterministic for same inputs.

## 4) PR-2.1-A scope
- Add layout profiles (`LANDSCAPE/SQUARE/PORTRAIT`) with:
  - safe margins, grid rows/cols, spacing baseline,
  - role priorities,
  - min/max role size constraints,
  - hero target ratio and text width cap.
- Add constraint validator and score function.
- Generate multiple portrait candidates (>=5) and select best score.
- If violations are heavy, fallback to rigid template and log reason.
