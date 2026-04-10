from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from hackathon_reefer_dl.io_utils import write_json
from hackathon_reefer_dl.metrics import composite_metrics


DEFAULT_CONFIG = {
    "peak_quantile": 0.85,
    "peak_weight": 4.0,
    "n_estimators": 320,
    "learning_rate": 0.04,
    "max_depth": 4,
    "min_child_weight": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 23,
    "n_jobs": 8,
}

BASE_KEEP = {
    "sequence_index",
    "was_gap_shifted",
    "was_dst_adjusted",
    "hour_of_day",
    "day_of_week",
    "day_of_year",
    "month",
    "season",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "weather_history_expected",
    "weather_history_complete",
    "weather_history_expected_feature_count",
    "weather_history_available_feature_count",
    "weather_history_available_fraction",
}

LONG_SIGNAL_LAGS = {
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    72,
    96,
    120,
    144,
    168,
}
SHORT_SIGNAL_LAGS = {
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    72,
}
LAG24_COL = "label_power_kw_tminus24h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact peak-weighted XGBoost reefer forecaster.")
    parser.add_argument("--train-csv", type=Path, required=True, help="Preprocessed train/validation CSV")
    parser.add_argument("--test-csv", type=Path, required=True, help="Preprocessed test CSV")
    parser.add_argument("--out", type=Path, required=True, help="Prediction CSV path")
    return parser.parse_args()


def build_compact_feature_names(columns: list[str]) -> list[str]:
    feature_cols = [column for column in columns if column not in {"source_timestamp_utc", "effective_timestamp_utc", "label_power_kw"}]
    keep: list[str] = []
    for column in feature_cols:
        if column in BASE_KEEP:
            keep.append(column)
            continue
        if "tminus" not in column:
            continue
        lag = int(column.split("tminus", 1)[1][:-1])
        if column.startswith(("label_power_kw_tminus", "active_rows_tminus", "mean_power_per_active_reefer_kw_tminus")):
            if lag in LONG_SIGNAL_LAGS:
                keep.append(column)
        elif column.startswith(
            (
                "mean_temperature_",
                "mean_ambient_minus_setpoint_c_tminus",
                "mean_return_minus_supply_c_tminus",
                "share_stack_tier_",
                "share_hardware_",
                "weather_temperature_",
            )
        ):
            if lag in SHORT_SIGNAL_LAGS:
                keep.append(column)
    return keep


def fit_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    y_train = train_df["label_power_kw"].to_numpy(dtype=float)
    lag24_train = train_df[LAG24_COL].to_numpy(dtype=float)
    target = y_train - lag24_train

    peak_threshold = float(np.quantile(y_train, DEFAULT_CONFIG["peak_quantile"]))
    sample_weight = np.where(y_train >= peak_threshold, DEFAULT_CONFIG["peak_weight"], 1.0)

    model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=int(DEFAULT_CONFIG["n_estimators"]),
        learning_rate=float(DEFAULT_CONFIG["learning_rate"]),
        max_depth=int(DEFAULT_CONFIG["max_depth"]),
        min_child_weight=float(DEFAULT_CONFIG["min_child_weight"]),
        subsample=float(DEFAULT_CONFIG["subsample"]),
        colsample_bytree=float(DEFAULT_CONFIG["colsample_bytree"]),
        reg_alpha=float(DEFAULT_CONFIG["reg_alpha"]),
        reg_lambda=float(DEFAULT_CONFIG["reg_lambda"]),
        tree_method="hist",
        random_state=int(DEFAULT_CONFIG["random_state"]),
        n_jobs=int(DEFAULT_CONFIG["n_jobs"]),
    )
    model.fit(train_df[feature_names], target, sample_weight=sample_weight, verbose=False)

    train_point = lag24_train + model.predict(train_df[feature_names])
    q90_uplift = float(np.quantile(np.maximum(y_train - train_point, 0.0), 0.9))

    point_pred = test_df[LAG24_COL].to_numpy(dtype=float) + model.predict(test_df[feature_names])
    pred_p90 = np.maximum(point_pred, point_pred + q90_uplift)
    return point_pred, pred_p90


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    feature_names = build_compact_feature_names(train_df.columns.tolist())

    point_pred, pred_p90 = fit_and_predict(train_df, test_df, feature_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "timestamp_utc": test_df["source_timestamp_utc"],
            "pred_power_kw": point_pred,
            "pred_p90_kw": pred_p90,
        }
    )
    pred_df.to_csv(args.out, index=False)

    summary = {
        "config": DEFAULT_CONFIG,
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    write_json(args.out.with_suffix(".summary.json"), summary)

    if "label_power_kw" in test_df.columns:
        metrics = composite_metrics(
            test_df["label_power_kw"].to_numpy(dtype=float),
            point_pred,
            pred_p90,
        )
        write_json(args.out.with_suffix(".metrics.json"), metrics.to_dict())
        print(f"Observed-target composite: {metrics.composite:.6f}")

    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
