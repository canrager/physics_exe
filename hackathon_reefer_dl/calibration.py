from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np


def _to_bucket(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right")


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.9))


def _hour_of_day(timestamps: Iterable[datetime]) -> np.ndarray:
    return np.asarray([ts.hour for ts in timestamps], dtype=np.int64)


def _point_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    peak_threshold = float(np.quantile(y_true, 0.9))
    peak_mask = y_true >= peak_threshold
    mae_all = float(np.mean(np.abs(y_true - pred)))
    mae_peak = float(np.mean(np.abs(y_true[peak_mask] - pred[peak_mask])))
    return 0.5 * mae_all + 0.3 * mae_peak


@dataclass
class PointPredictionCalibrator:
    alpha: float
    global_bias: float
    hour_bias: dict[int, float]
    shrinkage: float = 24.0

    @classmethod
    def fit(
        cls,
        timestamps: list[datetime],
        raw_pred: np.ndarray,
        naive_baseline: np.ndarray,
        y_true: np.ndarray,
        alpha_grid: np.ndarray | None = None,
        shrinkage: float = 48.0,
    ) -> "PointPredictionCalibrator":
        raw_pred = np.asarray(raw_pred, dtype=np.float64)
        naive_baseline = np.asarray(naive_baseline, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        if alpha_grid is None:
            alpha_grid = np.linspace(0.0, 0.75, 16, dtype=np.float64)

        best_alpha = 1.0
        best_score = float("inf")
        best_blend = raw_pred
        for alpha in alpha_grid:
            candidate = naive_baseline + alpha * (raw_pred - naive_baseline)
            score = _point_score(y_true, candidate)
            if score + 1e-12 < best_score:
                best_score = score
                best_alpha = float(alpha)
                best_blend = candidate

        residual = y_true - best_blend
        global_bias = float(np.median(residual))
        residual_by_hour: dict[int, list[float]] = defaultdict(list)
        for ts, resid in zip(timestamps, residual, strict=True):
            residual_by_hour[ts.hour].append(float(resid))

        hour_bias: dict[int, float] = {}
        for hour, values in residual_by_hour.items():
            local_bias = float(np.median(np.asarray(values, dtype=np.float64)))
            weight = len(values) / (len(values) + shrinkage)
            hour_bias[hour] = float(weight * local_bias + (1.0 - weight) * global_bias)

        return cls(
            alpha=best_alpha,
            global_bias=global_bias,
            hour_bias=hour_bias,
            shrinkage=shrinkage,
        )

    def predict(
        self,
        timestamps: list[datetime],
        raw_pred: np.ndarray,
        naive_baseline: np.ndarray,
    ) -> np.ndarray:
        raw_pred = np.asarray(raw_pred, dtype=np.float64)
        naive_baseline = np.asarray(naive_baseline, dtype=np.float64)
        blended = naive_baseline + self.alpha * (raw_pred - naive_baseline)
        calibrated = np.empty_like(blended, dtype=np.float64)
        for idx, ts in enumerate(timestamps):
            calibrated[idx] = blended[idx] + self.hour_bias.get(ts.hour, self.global_bias)
        return calibrated

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "global_bias": self.global_bias,
            "hour_bias": {str(key): value for key, value in self.hour_bias.items()},
            "shrinkage": self.shrinkage,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PointPredictionCalibrator":
        return cls(
            alpha=float(payload["alpha"]),
            global_bias=float(payload["global_bias"]),
            hour_bias={int(key): float(value) for key, value in dict(payload["hour_bias"]).items()},
            shrinkage=float(payload.get("shrinkage", 48.0)),
        )


@dataclass
class ResidualCalibrator:
    pred_edges: np.ndarray
    vol_edges: np.ndarray
    global_q90: float
    hour_q90: dict[int, float]
    bucket_q90: dict[str, float]
    min_bucket_size: int = 10

    @classmethod
    def fit(
        cls,
        timestamps: list[datetime],
        point_pred: np.ndarray,
        recent_volatility: np.ndarray,
        y_true: np.ndarray,
        min_bucket_size: int = 10,
    ) -> "ResidualCalibrator":
        point_pred = np.asarray(point_pred, dtype=np.float64)
        recent_volatility = np.asarray(recent_volatility, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        positive_residual = np.maximum(y_true - point_pred, 0.0)

        pred_edges = np.quantile(point_pred, [1.0 / 3.0, 2.0 / 3.0]).astype(np.float64)
        vol_edges = np.quantile(recent_volatility, [1.0 / 3.0, 2.0 / 3.0]).astype(np.float64)
        pred_bucket = _to_bucket(point_pred, pred_edges)
        vol_bucket = _to_bucket(recent_volatility, vol_edges)
        hour_bucket = _hour_of_day(timestamps)

        bucket_values: dict[str, list[float]] = defaultdict(list)
        hour_values: dict[int, list[float]] = defaultdict(list)
        for hour, p_bin, v_bin, resid in zip(hour_bucket, pred_bucket, vol_bucket, positive_residual, strict=True):
            bucket_values[f"{hour}|{p_bin}|{v_bin}"].append(float(resid))
            hour_values[int(hour)].append(float(resid))

        bucket_q90 = {
            bucket: _p90(values)
            for bucket, values in bucket_values.items()
            if len(values) >= min_bucket_size
        }
        hour_q90 = {hour: _p90(values) for hour, values in hour_values.items()}
        global_q90 = _p90(list(map(float, positive_residual.tolist())))
        return cls(
            pred_edges=pred_edges,
            vol_edges=vol_edges,
            global_q90=global_q90,
            hour_q90=hour_q90,
            bucket_q90=bucket_q90,
            min_bucket_size=min_bucket_size,
        )

    def predict_upper(
        self,
        timestamps: list[datetime],
        point_pred: np.ndarray,
        recent_volatility: np.ndarray,
    ) -> np.ndarray:
        point_pred = np.asarray(point_pred, dtype=np.float64)
        recent_volatility = np.asarray(recent_volatility, dtype=np.float64)
        pred_bucket = _to_bucket(point_pred, self.pred_edges)
        vol_bucket = _to_bucket(recent_volatility, self.vol_edges)
        upper = np.empty_like(point_pred, dtype=np.float64)
        for idx, (ts, pred, p_bin, v_bin) in enumerate(
            zip(timestamps, point_pred, pred_bucket, vol_bucket, strict=True)
        ):
            bucket_key = f"{ts.hour}|{p_bin}|{v_bin}"
            uplift = self.bucket_q90.get(bucket_key, self.hour_q90.get(ts.hour, self.global_q90))
            upper[idx] = max(pred, pred + uplift)
        return upper

    def to_dict(self) -> dict[str, object]:
        return {
            "pred_edges": self.pred_edges.tolist(),
            "vol_edges": self.vol_edges.tolist(),
            "global_q90": self.global_q90,
            "hour_q90": {str(key): value for key, value in self.hour_q90.items()},
            "bucket_q90": self.bucket_q90,
            "min_bucket_size": self.min_bucket_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ResidualCalibrator":
        return cls(
            pred_edges=np.asarray(payload["pred_edges"], dtype=np.float64),
            vol_edges=np.asarray(payload["vol_edges"], dtype=np.float64),
            global_q90=float(payload["global_q90"]),
            hour_q90={int(key): float(value) for key, value in dict(payload["hour_q90"]).items()},
            bucket_q90={str(key): float(value) for key, value in dict(payload["bucket_q90"]).items()},
            min_bucket_size=int(payload.get("min_bucket_size", 10)),
        )
