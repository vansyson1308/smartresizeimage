# Golden Fixtures

This folder stores deterministic **text-based** visual regression fixtures.

## Strategy
- Golden inputs/expected images are generated in test code (synthetic, deterministic).
- Golden baselines are stored as SHA-256 digests in `backend/tests/test_golden_regression.py`.
- Debug artifacts are written to `../outputs/` on mismatch.

This avoids committed binary PNG fixtures that can block PR tooling and keeps diffs reviewable.

## Add a new golden test
1. Create deterministic synthetic input(s) in `test_golden_regression.py`.
2. Render output with the target pipeline path.
3. Compute hash (`sha256(image.convert("RGBA").tobytes())`) and add it to `GOLDEN_HASHES`.
4. Assert runtime hash equals baseline hash.
5. For at least one harness proof test, use `compare_images(...)` and assert diff artifact creation on intentional mismatch.

## Determinism guidance
- Use fixed dimensions and fixed colors.
- Avoid randomness unless seed is hardcoded.
- Avoid network/model calls in regression tests.
