# Approach

This synced submission treats the task as a strict 24-hour-ahead forecast and uses the strongest public-scoring model developed in the repo.

## Main idea

- Build a day-ahead-safe wide lag-feature dataset from the participant package.
- Repair the large internal 2025 history gap before constructing lag windows.
- Predict a residual over `label_power_kw_tminus24h` instead of the full load directly.
- Train a compact XGBoost regressor with extra weight on high-load hours.
- Derive `pred_p90_kw` from the empirical positive training residual uplift.

## Final submission used here

The bundled `predictions.csv` is produced by `code/hackathon_reefer_dl/compact_xgb_peak_forecast.py`.

Model characteristics:

- input: compact subset of the wide preprocessed lag table
- target: `label_power_kw - label_power_kw_tminus24h`
- learner: shallow residual XGBoost
- peak emphasis: rows above the training 85th percentile receive 4x weight
- upper forecast: `pred_power_kw + q90(max(y_train - pred_train, 0))`

Measured public result for this synced file:

- composite: `29.948833`
- mae_all: `32.350326`
- mae_peak: `39.900770`
- pinball_p90: `9.017197`

## Files included

- `predictions.csv`: synced public-winning submission file
- `code/`: code used during development and final public-winning run

## Notes

- The preprocessing path is `code/preprocess_dataset.py` plus `code/reefer_preprocessing.py`.
- The deep-learning forecasting subproject remains in `code/hackathon_reefer_dl/` for reference.
- The earlier blend-based candidate path is still present in the code bundle, but it is no longer the synced default.
