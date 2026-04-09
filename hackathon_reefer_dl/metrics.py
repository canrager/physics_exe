from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MetricBundle:
    mae_all: float
    mae_peak: float
    pinball_p90: float
    composite: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.9) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def composite_metrics(
    y_true: np.ndarray,
    pred_power: np.ndarray,
    pred_p90: np.ndarray,
    peak_quantile: float = 0.9,
) -> MetricBundle:
    y_true = np.asarray(y_true, dtype=np.float64)
    pred_power = np.asarray(pred_power, dtype=np.float64)
    pred_p90 = np.asarray(pred_p90, dtype=np.float64)
    peak_threshold = float(np.quantile(y_true, peak_quantile))
    peak_mask = y_true >= peak_threshold
    mae_all = float(np.mean(np.abs(y_true - pred_power)))
    mae_peak = float(np.mean(np.abs(y_true[peak_mask] - pred_power[peak_mask])))
    pinball_p90 = pinball_loss(y_true, pred_p90, quantile=0.9)
    composite = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90
    return MetricBundle(
        mae_all=mae_all,
        mae_peak=mae_peak,
        pinball_p90=pinball_p90,
        composite=float(composite),
    )

