# Calibration

This folder contains the calibration workflow for fitting the CR-39 etch-rate
model to experimental track-shape and track-length datasets.

## Main files

- `run_Vd_calibration.py`: full calibration driver (plots + model export).
- `benchmark_bragg_corrections.py`: small helper to compare SRIM Bragg variants.
- `calibration_data.py`: loads Doerschel data and builds simulator bundles.
- `lib_optimiser.py`: objective and optimization routines.

## Fast start

From repo root:

```bash
python calibration/run_Vd_calibration.py
```

## Bragg-correction benchmark (simple)

Use this script to compare SRIM Bragg variants against the same experimental
calibration objective:

```bash
python calibration/benchmark_bragg_corrections.py
```

This runs automatically using built-in settings in
`calibration/benchmark_bragg_corrections.py` (variants, particles, seeds,
optimizer settings, and output folder). Edit the constants at the top of that
file if you want different defaults.

Key controls near the top of the script:

- `DEBRIS_DAMPING_RUN_MODES`: include `"without"`, `"fit"`, or both.
- `ANCHOR_COUNT_USER_OPTIONS`: list of anchor counts to test.
- `ANCHOR_EFFECT_HUGE_RATIO_THRESHOLD`: threshold used to flag large
	anchor-count sensitivity in summaries.

Outputs:

- `calibration/results/bragg_benchmark/bragg_<variant>/seed_<seed>/track_shape_*.png`
- `calibration/results/bragg_benchmark/bragg_benchmark_detailed.csv`
- `calibration/results/bragg_benchmark/bragg_benchmark_summary.csv`
- `calibration/results/bragg_benchmark/bragg_benchmark_overview.csv`

Use the PNGs to visually inspect track-shape agreement per run, then use the
overview/summary CSV files to compare variants numerically.

Choose the variant with the lowest `test_cost_both_median`.
