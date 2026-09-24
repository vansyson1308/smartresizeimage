# Phase 3 Target-first Bench Delta

## Commands
```bash
python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
python backend/tools/run_layout_bench.py --mode phase3 --seed 42
```

## Results
- Phase 2.1 (from `--mode both` summary):
  - pass: `27/36` (75.0%)
  - failures: overlap_area_ratio=5, total_score=5, text_plate=4
  - outside_margin_ratio failures: `0/36`
- Phase 3 target-first (from `--mode phase3` summary):
  - pass: `25/36` (69.4%)
  - failures: text_plate=6, overlap_area_ratio=5, total_score=5
  - outside_margin_ratio failures: `0/36`

## Acceptance check
- Phase 2.1 non-regression: maintained at `27/36`.
- Phase 3 acceptance (>=60%): **PASS** (`25/36`, 69.4%).
- Hard margin constraint retained in both modes (`outside_margin_ratio` stays 0).
