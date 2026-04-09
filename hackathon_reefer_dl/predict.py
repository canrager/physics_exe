from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hackathon_reefer_dl.calibration import PointPredictionCalibrator, ResidualCalibrator
from hackathon_reefer_dl.data import (
    available_observed_targets,
    build_hourly_feature_table,
    build_prediction_arrays,
    load_target_timestamps,
)
from hackathon_reefer_dl.io_utils import read_json, write_json
from hackathon_reefer_dl.metrics import composite_metrics
from hackathon_reefer_dl.model import build_forecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reefer load forecasts from a trained model bundle.")
    parser.add_argument("--participant-dir", type=Path, required=True, help="Participant package directory")
    parser.add_argument("--model-dir", type=Path, required=True, help="Model bundle directory")
    parser.add_argument("--targets", type=Path, required=True, help="CSV of target timestamps")
    parser.add_argument("--out", type=Path, required=True, help="Prediction CSV path")
    return parser.parse_args()


def _device_from_summary(summary: dict[str, object]) -> torch.device:
    if summary.get("device") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _infer(
    model: torch.nn.Module,
    sequence: np.ndarray,
    target_calendar: np.ndarray,
    naive_baseline: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, sequence.shape[0], batch_size):
            end = start + batch_size
            seq_batch = torch.from_numpy(sequence[start:end]).float().to(device, non_blocking=True)
            cal_batch = torch.from_numpy(target_calendar[start:end]).float().to(device, non_blocking=True)
            base_batch = torch.from_numpy(naive_baseline[start:end]).float().to(device, non_blocking=True)
            pred = (base_batch + model(seq_batch, cal_batch, base_batch)).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    summary = read_json(args.model_dir / "training_summary.json")
    scaler = np.load(args.model_dir / "scaler.npz")
    point_calibration_path = args.model_dir / "point_calibration.json"
    point_calibrator = (
        PointPredictionCalibrator.from_dict(read_json(point_calibration_path))
        if point_calibration_path.exists()
        else None
    )
    calibrator = ResidualCalibrator.from_dict(read_json(args.model_dir / "calibration.json"))
    feature_table = build_hourly_feature_table(args.participant_dir)
    target_times = load_target_timestamps(args.targets)
    feature_names = list(summary["selected_feature_names"])
    prediction_arrays = build_prediction_arrays(
        feature_table,
        feature_names,
        target_times,
        scaler["mean"],
        scaler["std"],
        history_hours=int(summary["history_hours"]),
        horizon_hours=int(summary["forecast_horizon_hours"]),
    )

    device = _device_from_summary(summary)
    ensemble_predictions = []
    for checkpoint_name in summary["final_checkpoints"]:
        model = build_forecaster(**summary["model_kwargs"]).to(device)
        checkpoint = torch.load(args.model_dir / checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        ensemble_predictions.append(
            _infer(
                model,
                prediction_arrays["sequence"],
                prediction_arrays["target_calendar"],
                prediction_arrays["naive_baseline"],
                device=device,
            )
        )
    raw_pred = np.mean(np.stack(ensemble_predictions, axis=0), axis=0)
    pred_power = (
        point_calibrator.predict(
            timestamps=prediction_arrays["target_times"],
            raw_pred=raw_pred,
            naive_baseline=prediction_arrays["naive_baseline"],
        )
        if point_calibrator is not None
        else raw_pred
    )
    pred_p90 = calibrator.predict_upper(
        timestamps=prediction_arrays["target_times"],
        point_pred=pred_power,
        recent_volatility=prediction_arrays["recent_volatility"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "pred_power_kw", "pred_p90_kw"])
        writer.writeheader()
        for target_time, point_pred, upper_pred in zip(target_times, pred_power, pred_p90, strict=True):
            writer.writerow(
                {
                    "timestamp_utc": target_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "pred_power_kw": round(float(max(point_pred, 0.0)), 6),
                    "pred_p90_kw": round(float(max(upper_pred, point_pred, 0.0)), 6),
                }
            )

    observed_targets = available_observed_targets(feature_table, target_times)
    if len(observed_targets) == len(target_times):
        y_true = np.asarray(
            [feature_table.target[feature_table.hour_to_idx[target_time]] for target_time in target_times],
            dtype=np.float64,
        )
        metrics = composite_metrics(y_true, pred_power, pred_p90)
        write_json(args.out.with_suffix(".metrics.json"), metrics.to_dict())
        print(f"Observed-target composite: {metrics.composite:.6f}")

    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
