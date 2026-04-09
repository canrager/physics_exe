from __future__ import annotations

import argparse
import csv
import random
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from statistics import median
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hackathon_reefer_dl.calibration import PointPredictionCalibrator, ResidualCalibrator
from hackathon_reefer_dl.common import parse_target_timestamp
from hackathon_reefer_dl.data import (
    HourlyFeatureTable,
    build_training_arrays,
    feature_names_for_groups,
    load_hourly_feature_table,
    make_cv_splits,
    observed_target_times,
    scaler_from_targets,
)
from hackathon_reefer_dl.io_utils import read_json, write_json
from hackathon_reefer_dl.metrics import composite_metrics
from hackathon_reefer_dl.model import build_forecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reefer TCN forecaster.")
    parser.add_argument("--data", type=Path, required=True, help="Path to hourly_features.parquet")
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for model artifacts")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_loader(
    arrays: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    weights: np.ndarray | None = None,
    device: torch.device | None = None,
) -> DataLoader:
    tensors = [
        torch.from_numpy(arrays["sequence"]).float(),
        torch.from_numpy(arrays["target_calendar"]).float(),
        torch.from_numpy(arrays["labels"]).float(),
        torch.from_numpy(arrays["naive_baseline"]).float(),
    ]
    if weights is not None:
        tensors.append(torch.from_numpy(weights).float())
    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device is not None and device.type == "cuda",
    )


def _infer(
    model: torch.nn.Module,
    sequence: np.ndarray,
    target_calendar: np.ndarray,
    naive_baseline: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, sequence.shape[0], batch_size):
            end = start + batch_size
            seq_batch = torch.from_numpy(sequence[start:end]).float().to(device, non_blocking=True)
            cal_batch = torch.from_numpy(target_calendar[start:end]).float().to(device, non_blocking=True)
            base_batch = torch.from_numpy(naive_baseline[start:end]).float().to(device, non_blocking=True)
            pred = (base_batch + model(seq_batch, cal_batch, base_batch)).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def _model_kwargs(config: dict[str, Any], input_dim: int, target_calendar_dim: int) -> dict[str, Any]:
    return {
        "model_type": str(config["model"].get("type", "tcn")),
        "input_dim": input_dim,
        "target_calendar_dim": target_calendar_dim,
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "kernel_size": int(config["model"]["kernel_size"]),
        "dilations": tuple(int(value) for value in config["model"]["dilations"]),
        "dropout": float(config["model"]["dropout"]),
    }


def train_single_seed(
    train_arrays: dict[str, Any],
    val_arrays: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(seed)
    model = build_forecaster(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    peak_threshold = float(np.quantile(train_arrays["labels"], float(config["peak_quantile"])))
    weights = np.where(
        train_arrays["labels"] > peak_threshold,
        float(config["peak_weight"]),
        1.0,
    ).astype(np.float32)
    train_loader = _to_loader(
        train_arrays,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        weights=weights,
        device=device,
    )

    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_train_q90 = 0.0
    patience = int(config["patience"])
    no_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            sequence_batch, calendar_batch, label_batch, baseline_batch, weight_batch = batch
            sequence_batch = sequence_batch.to(device, non_blocking=True)
            calendar_batch = calendar_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            baseline_batch = baseline_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = baseline_batch + model(sequence_batch, calendar_batch, baseline_batch)
                loss = torch.mean(torch.abs(pred - label_batch) * weight_batch)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu()) * int(label_batch.shape[0])
            sample_count += int(label_batch.shape[0])

        train_pred = _infer(
            model,
            train_arrays["sequence"],
            train_arrays["target_calendar"],
            train_arrays["naive_baseline"],
            device=device,
            batch_size=max(int(config["batch_size"]), 1024),
        )
        train_q90 = float(np.quantile(np.maximum(train_arrays["labels"] - train_pred, 0.0), 0.9))
        val_pred = _infer(
            model,
            val_arrays["sequence"],
            val_arrays["target_calendar"],
            val_arrays["naive_baseline"],
            device=device,
            batch_size=max(int(config["batch_size"]), 1024),
        )
        val_p90 = np.maximum(val_pred, val_pred + train_q90)
        val_metrics = composite_metrics(val_arrays["labels"], val_pred, val_p90)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(sample_count, 1),
                "train_q90": train_q90,
                **val_metrics.to_dict(),
            }
        )

        if val_metrics.composite + 1e-6 < best_metric:
            best_metric = val_metrics.composite
            best_epoch = epoch
            best_train_q90 = train_q90
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training never produced a best checkpoint.")
    model.load_state_dict(best_state)
    val_pred = _infer(
        model,
        val_arrays["sequence"],
        val_arrays["target_calendar"],
        val_arrays["naive_baseline"],
        device=device,
        batch_size=max(int(config["batch_size"]), 1024),
    )
    return {
        "model": model,
        "val_pred": val_pred,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "train_q90": best_train_q90,
        "history": history,
    }


def train_fixed_epochs(
    train_arrays: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    model_kwargs: dict[str, Any],
    epochs: int,
) -> torch.nn.Module:
    seed_everything(seed)
    model = build_forecaster(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    peak_threshold = float(np.quantile(train_arrays["labels"], float(config["peak_quantile"])))
    weights = np.where(
        train_arrays["labels"] > peak_threshold,
        float(config["peak_weight"]),
        1.0,
    ).astype(np.float32)
    train_loader = _to_loader(
        train_arrays,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        weights=weights,
        device=device,
    )

    for _ in tqdm(range(epochs), desc=f"Seed {seed}", leave=False):
        model.train()
        for batch in train_loader:
            sequence_batch, calendar_batch, label_batch, baseline_batch, weight_batch = batch
            sequence_batch = sequence_batch.to(device, non_blocking=True)
            calendar_batch = calendar_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            baseline_batch = baseline_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = baseline_batch + model(sequence_batch, calendar_batch, baseline_batch)
                loss = torch.mean(torch.abs(pred - label_batch) * weight_batch)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    return model


def run_cross_validation(
    feature_table: HourlyFeatureTable,
    target_times: list,
    feature_names: list[str],
    config: dict[str, Any],
    device: torch.device,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    if seeds is None:
        seeds = [int(seed) for seed in config["seeds"]]
    folds = make_cv_splits(
        target_times,
        num_folds=int(config["num_cv_folds"]),
        fold_hours=int(config["cv_fold_hours"]),
    )
    full_matrix = feature_table.feature_matrix(feature_names)
    model_kwargs = _model_kwargs(config, input_dim=len(feature_names), target_calendar_dim=7)

    fold_outputs: list[dict[str, Any]] = []
    oof_times: list = []
    oof_pred_parts: list[np.ndarray] = []
    oof_label_parts: list[np.ndarray] = []
    oof_recent_parts: list[np.ndarray] = []
    oof_baseline_parts: list[np.ndarray] = []
    best_epochs: list[int] = []
    seed_summaries: list[dict[str, Any]] = []

    for fold_idx, (train_times, val_times) in enumerate(folds):
        scaler_mean, scaler_std = scaler_from_targets(
            full_matrix,
            train_times,
            feature_table.hour_to_idx,
            history_hours=int(config["history_hours"]),
            horizon_hours=int(config["forecast_horizon_hours"]),
        )
        train_arrays = build_training_arrays(
            feature_table,
            feature_names,
            train_times,
            scaler_mean,
            scaler_std,
            history_hours=int(config["history_hours"]),
            horizon_hours=int(config["forecast_horizon_hours"]),
        )
        val_arrays = build_training_arrays(
            feature_table,
            feature_names,
            val_times,
            scaler_mean,
            scaler_std,
            history_hours=int(config["history_hours"]),
            horizon_hours=int(config["forecast_horizon_hours"]),
        )

        seed_predictions = []
        for seed in seeds:
            result = train_single_seed(
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                config=config,
                device=device,
                seed=int(seed),
                model_kwargs=model_kwargs,
            )
            seed_predictions.append(result["val_pred"])
            best_epochs.append(int(result["best_epoch"]))
            seed_summaries.append(
                {
                    "fold": fold_idx,
                    "seed": int(seed),
                    "best_epoch": int(result["best_epoch"]),
                    "best_metric": float(result["best_metric"]),
                }
            )

        ensemble_pred = np.mean(np.stack(seed_predictions, axis=0), axis=0)
        fold_outputs.append(
            {
                "fold": fold_idx,
                "target_times": val_times,
                "labels": val_arrays["labels"],
                "pred": ensemble_pred,
                "recent_volatility": val_arrays["recent_volatility"],
                "naive_baseline": val_arrays["naive_baseline"],
            }
        )
        oof_times.extend(val_times)
        oof_label_parts.append(val_arrays["labels"])
        oof_pred_parts.append(ensemble_pred)
        oof_recent_parts.append(val_arrays["recent_volatility"])
        oof_baseline_parts.append(val_arrays["naive_baseline"])

    oof_raw_pred = np.concatenate(oof_pred_parts, axis=0)
    oof_labels = np.concatenate(oof_label_parts, axis=0)
    oof_recent = np.concatenate(oof_recent_parts, axis=0)
    oof_baseline = np.concatenate(oof_baseline_parts, axis=0)
    point_calibrator = PointPredictionCalibrator.fit(
        timestamps=oof_times,
        raw_pred=oof_raw_pred,
        naive_baseline=oof_baseline,
        y_true=oof_labels,
    )
    oof_pred = point_calibrator.predict(
        timestamps=oof_times,
        raw_pred=oof_raw_pred,
        naive_baseline=oof_baseline,
    )
    calibrator = ResidualCalibrator.fit(
        timestamps=oof_times,
        point_pred=oof_pred,
        recent_volatility=oof_recent,
        y_true=oof_labels,
        min_bucket_size=int(config["calibration"]["min_bucket_size"]),
    )

    fold_metrics = []
    for fold_output in fold_outputs:
        point_pred = point_calibrator.predict(
            timestamps=fold_output["target_times"],
            raw_pred=fold_output["pred"],
            naive_baseline=fold_output["naive_baseline"],
        )
        pred_p90 = calibrator.predict_upper(
            timestamps=fold_output["target_times"],
            point_pred=point_pred,
            recent_volatility=fold_output["recent_volatility"],
        )
        metrics = composite_metrics(fold_output["labels"], point_pred, pred_p90)
        fold_metrics.append({"fold": fold_output["fold"], **metrics.to_dict()})

    return {
        "feature_names": feature_names,
        "fold_metrics": fold_metrics,
        "mean_composite": float(np.mean([row["composite"] for row in fold_metrics])),
        "best_epochs": best_epochs,
        "seed_summaries": seed_summaries,
        "point_calibrator": point_calibrator,
        "calibrator": calibrator,
        "oof_predictions": {
            "timestamps": oof_times,
            "labels": oof_labels,
            "pred": oof_pred,
            "raw_pred": oof_raw_pred,
            "naive_baseline": oof_baseline,
            "recent_volatility": oof_recent,
            "pred_p90": calibrator.predict_upper(oof_times, oof_pred, oof_recent),
        },
    }


def save_fold_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "fold", "mae_all", "mae_peak", "pinball_p90", "composite"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_progress(
    out_dir: Path,
    stage_results: list[dict[str, Any]],
    selected_groups: list[str],
    last_mean_composite: float | None = None,
) -> None:
    write_json(
        out_dir / "ablation_progress.json",
        {
            "selected_groups": selected_groups,
            "last_mean_composite": last_mean_composite,
            "stages": stage_results,
        },
    )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    feature_table = load_hourly_feature_table(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(str(config.get("device", "cuda")))
    train_target_end = parse_target_timestamp(str(config["train_target_end"]))
    target_times = observed_target_times(feature_table, train_target_end)

    feature_order = list(config["feature_group_order"])
    ablation_threshold = float(config["ablation_improvement_threshold"])
    ablation_seeds = [int(seed) for seed in config.get("ablation_seeds", config["seeds"])]
    stage_results = []
    selected_groups = [feature_order[0]]

    current_result = run_cross_validation(
        feature_table=feature_table,
        target_times=target_times,
        feature_names=feature_names_for_groups(feature_table.metadata, selected_groups),
        config=config,
        device=device,
        seeds=ablation_seeds,
    )
    stage_results.append(
        {
            "stage": "+".join(selected_groups),
            "groups": list(selected_groups),
            "kept": True,
            "mean_composite": current_result["mean_composite"],
        }
    )
    selected_result = current_result
    write_progress(args.out_dir, stage_results, selected_groups, current_result["mean_composite"])
    print(
        f"[ablation] keep {selected_groups} -> mean composite {current_result['mean_composite']:.4f}",
        flush=True,
    )

    for group_name in feature_order[1:]:
        candidate_groups = selected_groups + [group_name]
        print(f"[ablation] evaluating {candidate_groups} ...", flush=True)
        candidate_result = run_cross_validation(
            feature_table=feature_table,
            target_times=target_times,
            feature_names=feature_names_for_groups(feature_table.metadata, candidate_groups),
            config=config,
            device=device,
            seeds=ablation_seeds,
        )
        improvement = selected_result["mean_composite"] - candidate_result["mean_composite"]
        keep = improvement >= ablation_threshold
        stage_results.append(
            {
                "stage": "+".join(candidate_groups),
                "groups": list(candidate_groups),
                "kept": keep,
                "mean_composite": candidate_result["mean_composite"],
                "improvement_vs_previous": improvement,
            }
        )
        if keep:
            selected_groups = candidate_groups
            selected_result = candidate_result
        write_progress(args.out_dir, stage_results, selected_groups, selected_result["mean_composite"])
        print(
            f"[ablation] {'keep' if keep else 'reject'} {candidate_groups} -> "
            f"mean composite {candidate_result['mean_composite']:.4f} "
            f"(delta {improvement:+.4f})",
            flush=True,
        )

    folds = make_cv_splits(
        target_times,
        num_folds=int(config["num_cv_folds"]),
        fold_hours=int(config["cv_fold_hours"]),
    )
    baseline_scores = {"lag24": [], "organizer_baseline": []}
    for _, val_times in folds:
        y_true = np.asarray(
            [feature_table.target[feature_table.hour_to_idx[target_time]] for target_time in val_times],
            dtype=np.float32,
        )
        lag24_pred = np.asarray(
            [feature_table.target[feature_table.hour_to_idx[target_time - timedelta(hours=24)]] for target_time in val_times],
            dtype=np.float32,
        )
        baseline_scores["lag24"].append(composite_metrics(y_true, lag24_pred, lag24_pred * 1.10).composite)
        blended = np.asarray(
            [
                0.7 * feature_table.target[feature_table.hour_to_idx[target_time - timedelta(hours=24)]]
                + 0.3 * feature_table.target[feature_table.hour_to_idx[target_time - timedelta(hours=168)]]
                for target_time in val_times
            ],
            dtype=np.float32,
        )
        baseline_scores["organizer_baseline"].append(composite_metrics(y_true, blended, blended * 1.10).composite)
    baseline_scores = {name: float(np.mean(values)) for name, values in baseline_scores.items()}

    selected_mean = float(selected_result["mean_composite"])
    acceptance_gate = {
        "beats_lag24": selected_mean < baseline_scores["lag24"],
        "beats_organizer_baseline": selected_mean < baseline_scores["organizer_baseline"],
        "under_target_45": selected_mean < 45.0,
    }
    gate_passed = all(acceptance_gate.values())

    selected_feature_names = selected_result["feature_names"]
    full_feature_matrix = feature_table.feature_matrix(selected_feature_names)
    scaler_mean, scaler_std = scaler_from_targets(
        full_feature_matrix,
        target_times,
        feature_table.hour_to_idx,
        history_hours=int(config["history_hours"]),
        horizon_hours=int(config["forecast_horizon_hours"]),
    )
    full_train_arrays = build_training_arrays(
        feature_table,
        selected_feature_names,
        target_times,
        scaler_mean,
        scaler_std,
        history_hours=int(config["history_hours"]),
        horizon_hours=int(config["forecast_horizon_hours"]),
    )

    final_epochs = max(1, int(round(median(selected_result["best_epochs"]))))
    model_kwargs = _model_kwargs(config, input_dim=len(selected_feature_names), target_calendar_dim=7)

    np.savez(
        args.out_dir / "scaler.npz",
        mean=scaler_mean.astype(np.float32),
        std=scaler_std.astype(np.float32),
    )
    write_json(args.out_dir / "point_calibration.json", selected_result["point_calibrator"].to_dict())
    write_json(args.out_dir / "calibration.json", selected_result["calibrator"].to_dict())

    final_checkpoints = []
    for seed in config["seeds"]:
        print(f"[final] training seed {int(seed)} for {final_epochs} epochs ...", flush=True)
        model = train_fixed_epochs(
            train_arrays=full_train_arrays,
            config=config,
            device=device,
            seed=int(seed),
            model_kwargs=model_kwargs,
            epochs=final_epochs,
        )
        checkpoint_path = args.out_dir / f"model_seed_{int(seed)}.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        final_checkpoints.append(str(checkpoint_path.name))
        print(f"[final] saved {checkpoint_path.name}", flush=True)

    fold_metric_rows = []
    for row in selected_result["fold_metrics"]:
        fold_metric_rows.append({"stage": "+".join(selected_groups), **row})
    save_fold_metrics(args.out_dir / "fold_metrics.csv", fold_metric_rows)

    training_summary = {
        "config_path": str(args.config),
        "data_path": str(args.data),
        "device": str(device),
        "selected_groups": selected_groups,
        "selected_feature_names": selected_feature_names,
        "ablation_results": stage_results,
        "cv_mean_composite": selected_mean,
        "cv_fold_metrics": selected_result["fold_metrics"],
        "cv_baselines": baseline_scores,
        "point_calibration": selected_result["point_calibrator"].to_dict(),
        "acceptance_gate": {**acceptance_gate, "passed": gate_passed},
        "final_epochs": final_epochs,
        "model_kwargs": model_kwargs,
        "history_hours": int(config["history_hours"]),
        "forecast_horizon_hours": int(config["forecast_horizon_hours"]),
        "seeds": [int(seed) for seed in config["seeds"]],
        "ablation_seeds": ablation_seeds,
        "final_checkpoints": final_checkpoints,
    }
    write_json(args.out_dir / "training_summary.json", training_summary)
    print(f"Selected groups: {selected_groups}")
    print(f"CV mean composite: {selected_mean:.4f}")
    print(f"Acceptance gate: {acceptance_gate} -> passed={gate_passed}")
    print(f"Final epochs: {final_epochs}", flush=True)
    print(f"Saved model bundle to {args.out_dir}")


if __name__ == "__main__":
    main()
