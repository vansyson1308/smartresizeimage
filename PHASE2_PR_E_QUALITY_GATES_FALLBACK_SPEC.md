# Phase 2 PR-E: Quality Gates & Auto Fallback

## Objective
Never return generative output when safety/brand gates fail.

## Gates
- OCR gate (reject strange text in editable zones)
- Logo similarity gate
- Mascot similarity gate
- Color drift gate (brand-safe protected regions)

## Fallback behavior
- If any gate fails, return deterministic baseline output.
- No silent fail: always record and log fail reasons.

## Required return fields
- `gates_passed: bool`
- `fail_reasons: list[str]`
- `used_fallback: bool`

## Integration
- Build deterministic baseline and generative candidate in `ReLayoutEngine`.
- Evaluate gates in `backend/app/generative/gates.py`.
- Return candidate if pass, otherwise deterministic fallback with explicit reasons.
