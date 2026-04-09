# Reefer Dataset Analysis

## Challenge Understanding
- Forecast the combined hourly reefer electricity demand for future timestamps.
- The score rewards overall accuracy, peak-hour accuracy, and a sensible p90 upper estimate.
- This EDA uses `sum(AvPowerCons) / 1000` per hour as the label because that matches the challenge target conceptually.

## Dataset Overview
- Raw rows: 3,774,557
- Aggregated hours: 8,403
- Reefer range: 2025-01-01T00:00:00Z to 2026-01-10T06:00:00Z
- Public target timestamps: 223
- Public target range: 2026-01-01T00:00:00Z to 2026-01-10T06:00:00Z

## Raw File Notes
- The reefer file uses semicolons and decimal commas.
- The raw schema differs slightly from the docs, for example `AvPowerCons` and `RemperatureSupply`.
- The generated `hourly_features.csv` is the main modeling-ready table for the next steps.

## Top 12 Correlations To The Label
- `active_container_count`: 0.795
- `share_stack_tier_1`: -0.599
- `mean_stack_tier`: 0.566
- `share_stack_tier_3`: 0.487
- `share_stack_tier_2`: 0.465
- `mean_ambient_minus_setpoint_c`: 0.405
- `mean_temperature_ambient_c`: 0.389
- `month`: 0.378
- `mean_return_minus_supply_c`: 0.376
- `day_of_year`: 0.373
- `share_hardware_decosiiih`: 0.344
- `share_hardware_ml5`: 0.330

## Top 12 Simple Feature Importances
- `active_container_count`: MAE increase 216.197 kW
- `day_of_year`: MAE increase 137.126 kW
- `mean_temperature_setpoint_c`: MAE increase 105.713 kW
- `month`: MAE increase 91.144 kW
- `mean_temperature_return_c`: MAE increase 44.317 kW
- `mean_temperature_supply_c`: MAE increase 32.832 kW
- `mean_stack_tier`: MAE increase 28.518 kW
- `mean_ambient_minus_setpoint_c`: MAE increase 12.897 kW
- `share_stack_tier_2`: MAE increase 6.386 kW
- `share_hardware_decosiiie`: MAE increase 4.134 kW
- `share_hardware_scc6`: MAE increase 3.520 kW
- `mean_return_minus_supply_c`: MAE increase 3.333 kW

## Generated Files
- `report.html`
- `dataset_summary.json`
- `hourly_features.csv`
- `feature_label_correlations.csv`
- `feature_importance.csv`
