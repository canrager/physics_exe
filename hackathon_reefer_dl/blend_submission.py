from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hackathon_reefer_dl.baselines import aggregate_hourly_load, load_target_hours
from hackathon_reefer_dl.io_utils import write_json
from hackathon_reefer_dl.metrics import composite_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend a model prediction file with the lag-24 baseline.")
    parser.add_argument("--participant-dir", type=Path, required=True, help="Participant package directory")
    parser.add_argument("--model-predictions", type=Path, required=True, help="CSV produced by predict.py")
    parser.add_argument("--out", type=Path, required=True, help="Blended submission path")
    parser.add_argument(
        "--point-weights",
        default="0.35,1.0,0.475",
        help="Comma-separated blend weights for low/mid/high lag24 terciles",
    )
    parser.add_argument(
        "--p90-uplifts",
        default="1.065,1.14,1.085",
        help="Comma-separated lag24 uplift multipliers for low/mid/high lag24 terciles",
    )
    return parser.parse_args()


def _parse_triplet(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("Expected exactly three comma-separated values.")
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    weights = _parse_triplet(args.point_weights)
    uplifts = _parse_triplet(args.p90_uplifts)

    target_csv = args.participant_dir / "target_timestamps.csv"
    target_times = load_target_hours(target_csv)
    hourly_load = aggregate_hourly_load(args.participant_dir)
    lag24 = np.asarray([hourly_load[target - timedelta(hours=24)] for target in target_times], dtype=np.float64)

    model_pred = []
    with args.model_predictions.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model_pred.append(float(row["pred_power_kw"]))
    model_pred = np.asarray(model_pred, dtype=np.float64)

    q1, q2 = np.quantile(lag24, [1.0 / 3.0, 2.0 / 3.0])
    bins = np.digitize(lag24, [q1, q2])
    point_pred = lag24 * (1.0 - weights[bins]) + model_pred * weights[bins]
    pred_p90 = np.maximum(point_pred, lag24 * uplifts[bins])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "pred_power_kw", "pred_p90_kw"])
        writer.writeheader()
        for target_time, point_value, upper_value in zip(target_times, point_pred, pred_p90, strict=True):
            writer.writerow(
                {
                    "timestamp_utc": target_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "pred_power_kw": round(float(point_value), 6),
                    "pred_p90_kw": round(float(max(upper_value, point_value)), 6),
                }
            )

    if all(target in hourly_load for target in target_times):
        y_true = np.asarray([hourly_load[target] for target in target_times], dtype=np.float64)
        metrics = composite_metrics(y_true, point_pred, pred_p90)
        write_json(
            args.out.with_suffix(".metrics.json"),
            {
                **metrics.to_dict(),
                "point_weights": weights.tolist(),
                "p90_uplifts": uplifts.tolist(),
                "lag24_terciles": [float(q1), float(q2)],
            },
        )
        print(f"Observed-target composite: {metrics.composite:.6f}")

    print(f"Saved blended submission to {args.out}")


if __name__ == "__main__":
    main()
