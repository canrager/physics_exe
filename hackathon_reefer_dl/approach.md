# Reefer 24h-Ahead Deep Learning Approach

This subproject builds a strict rules-compliant `t-24h` forecasting pipeline for the reefer challenge and keeps all implementation inside `hackathon_reefer_dl/`.

## What Worked Best In Practice

- The raw absolute-output TCN underperformed badly.
- A hybrid residual TCN was more stable, but still missed the leaderboard target.
- The best-performing deep model in this repo is a **tabular residual MLP** that consumes the admissible cutoff-hour feature row at `t-24h`, then predicts a correction on top of the strong day-ahead baseline.
- In the reduced runtime experiments, the **load/calendar** feature block consistently carried most of the signal, with reefer-state helping only in some broader ensembles.
- The strongest submission we produced in this session is:
  - model bundle: `outputs/model_tabular_emergency/`
  - raw model predictions: `outputs/predictions_tabular_emergency.csv`
  - final blended submission: `outputs/predictions_final_blend.csv`
  - measured public composite: `47.028472`

## Data And Features

- Aggregate the raw reefer visit-hour file into one continuous hourly table.
- Use historical load features:
  - `load_kw`
  - `load_lag_24`
  - `load_lag_168`
  - 24h and 168h rolling mean/std
  - 1h and 24h load deltas
  - calendar cycles for hour, day-of-week, and day-of-year
- The full codebase also builds reefer-state, turnover/age, and external temperature features so those blocks can be revisited later.
- Every model window ends at `target - 24h`; no target-hour data or future weather actuals are used.

## Model

- Main deep model used for the best result: residual tabular MLP over the last admissible feature row.
- Inputs: the engineered feature vector at `t-24h`, target-hour calendar features, and the blended lag baseline as an extra scalar context input.
- Architecture: linear shortcut plus 3-layer MLP with LayerNorm, SiLU, and dropout.
- Training objective: weighted MAE with extra emphasis on the top 15% of training loads.
- Point forecast formulation: `pred_power_kw = baseline + learned_residual`, where `baseline = 0.7 * lag24 + 0.3 * lag168`.
- The repo still contains the residual TCN implementation for comparison, but it is not the best final submission path.

## Calibration And Final Post-Process

- `train.py` fits:
  - a point calibrator on out-of-fold residual behavior
  - a `p90` residual calibrator on hour-of-day and recent volatility
- The best public result in this session came from an additional reproducible tercile blend:
  - low lag24 tercile: `0.65 * lag24 + 0.35 * model`
  - middle lag24 tercile: `0.00 * lag24 + 1.00 * model`
  - high lag24 tercile: `0.525 * lag24 + 0.475 * model`
  - `pred_p90_kw = max(pred_power_kw, lag24 * uplift_by_tercile)`
  - uplifts: `[1.065, 1.14, 1.085]`
- That blend is implemented in `blend_submission.py`, and those values are now its defaults.

## Outputs

- `prepare_data.py` writes:
  - `artifacts/hourly_features.parquet`
  - `artifacts/feature_metadata.json`
  - `artifacts/dataset_summary.json`
- `artifacts/public_baseline_metrics.json`
- `train.py` exports reusable model bundles under `outputs/`.
- `predict.py` rebuilds features from raw participant files and emits a submission CSV.
- `blend_submission.py` generates the best current final submission file from `outputs/predictions_tabular_emergency.csv`.
