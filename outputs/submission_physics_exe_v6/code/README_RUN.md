# Code Bundle

This folder contains the synced code delivered with the final public-winning submission bundle.

## Current winner

The bundled `predictions.csv` now corresponds to the compact peak-weighted residual XGBoost model in:

- `hackathon_reefer_dl/compact_xgb_peak_forecast.py`

That model scored `29.948833` on the original 223-hour public slice.

## Main pieces

- `hackathon_reefer_dl/`
  - participant-package data preparation
  - deep-learning forecasting experiments
  - candidate blending utilities
  - compact XGBoost public winner
- `preprocess_dataset.py`
  - entrypoint for the wide lag-feature dataset build
- `reefer_preprocessing.py`
  - gap repair, weather merge, and day-ahead-safe feature engineering
- `weather_impact_analysis.py`
  - weather aggregation helper used by preprocessing
- `reefer_forecast_dl.py`
  - legacy deep-learning experiment script kept for reference

## Reproduce The Synced Public Winner

```bash
python preprocess_dataset.py

python hackathon_reefer_dl/compact_xgb_peak_forecast.py \
  --train-csv outputs/preprocessed_dataset/trainval_hourly.csv \
  --test-csv outputs/preprocessed_dataset/test_hourly.csv \
  --out outputs/results/predictions_xgb_resid_shallow_peak_public_v2.csv
```

## Older Blend Path

The earlier blend-based submission path is still included for reference:

```bash
python hackathon_reefer_dl/blend_existing_candidates.py \
  --anchor outputs/results/predictions.csv \
  --tail-specialist hackathon_reefer_dl/outputs/predictions_tabular_emergency.csv \
  --top-k 15 \
  --tail-point-anchor-weight 0.275 \
  --tail-p90-anchor-weight 0.95 \
  --out outputs/results/predictions_physics_exe_full_v6.csv
```
