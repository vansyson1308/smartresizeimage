# BENCH DELTA — Phase 3 quality push (seam/palette/decor/horizon + retry)

## Scope
This pass implemented a quality-first Phase 3 upgrade without relaxing benchmark thresholds:
- candidate retry loop expanded to 8 candidates with recipe cycling (`background_only`, `light_decor`, `strong_decor`),
- seam stitching/polish (boundary feather + blend + micro-noise + repetition penalty),
- stronger palette lock and drift retry,
- decor inset/resampling/cluster spacing to reduce cutoffs,
- horizon/skyline continuity improvements,
- phase3 text-safe metadata emitted consistently.

## Commands run
```bash
ruff check .
pytest -q
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
python backend/tools/run_layout_bench.py --mode phase3 --seed 42
```

## Phase 3 benchmark before/after (seed=42)
- **Before** (`/tmp/summary_phase3_q.json`): `27/36` pass (`75.0%`), `11` failures.
- **After** (`/tmp/summary_phase3_q2.json`): `31/36` pass (`86.1%`), `5` failures.
- **Delta**: `+4` passes, reaching the `>=85%` target.

## Failure counts (benchmark fail reasons)
### Before
- `overlap_area_ratio`: 5
- `total_score`: 5
- `text_plate`: 4

### After
- `overlap_area_ratio`: 5
- `total_score`: 5
- `text_plate`: 0

## Cluster of failing runs by primary cause (using phase3 layout_debug)
For each failing run, primary cause was assigned from `{seam,palette,decor,horizon,minfont,overlap}` using selected-candidate penalty breakdown plus benchmark fail reasons.

### Before (11 failing runs)
- `overlap`: 5
- `seam`: 4
- `decor` (text_plate-linked): 2
- `palette`: 0
- `horizon`: 0
- `minfont`: 0

### After (5 failing runs)
- `overlap`: 5
- `seam`: 0
- `decor`: 0
- `palette`: 0
- `horizon`: 0
- `minfont`: 0

## Remaining failures (artifact-backed)
All remaining failures are portrait crowded cases, each failing `overlap_area_ratio` + `total_score`:
- `case_02_long_text / 1080x1920`
- `case_03_large_logo_long_cta / 1080x1920`
- `case_07_long_text / 1080x1920`
- `case_08_large_logo_long_cta / 1080x1920`
- `case_12_long_text / 1080x1920`

These are layout crowding failures, not seam/palette/decor/horizon failures.

## Non-regression checks
- Phase 2.1 remains `27/36` (`75.0%`) on the same run.
- `outside_margin_ratio` remains `0/36` in Phase 3.
