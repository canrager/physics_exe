# Reefer 24h-Ahead Deep Learning Approach

This subproject builds a strict rules-compliant `t-24h` forecasting pipeline for the hackathon reefer challenge.

## Data And Features

- Aggregate the raw reefer visit-hour file into one hourly terminal-level table.
- Use historical load plus reefer state features: active visit count, per-visit power, internal temperature aggregates, hardware shares, stack-tier shares, size shares, visit turnover, and active-age statistics.
- Merge only the two temperature sensor files from `wetterdaten.zip`, forward-fill missing values, and retain availability masks so the model can distinguish true weather data from imputed history.
- Build every model input from a 336-hour history window ending exactly at `target - 24h`.

## Model

- Sequence model: residual TCN with dilations `1/2/4/8/16/32`.
- Input processing: per-hour numeric projection plus LayerNorm.
- Heads: attention pooling over the sequence, last-step state, and a target-hour calendar embedding.
- Objective: weighted MAE with extra emphasis on the highest 15% of training loads.

## Validation And Calibration

- Use four chronological validation folds of 223 hours each.
- Run the fixed feature ablation ladder: load/calendar, reefer state, turnover/age, then external temperature.
- Calibrate `pred_p90_kw` from out-of-fold positive residuals using hour-of-day, predicted-load terciles, and recent-volatility terciles.

## Outputs

- `prepare_data.py` writes hourly parquet artifacts and a reproduced public baseline score.
- `train.py` selects feature groups, runs rolling CV, trains the final seed ensemble, and exports the reusable model bundle.
- `predict.py` rebuilds hourly features from raw participant files and emits the required submission CSV.

