from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hackathon_reefer_dl.baselines import aggregate_hourly_load
from hackathon_reefer_dl.common import parse_target_timestamp
from hackathon_reefer_dl.io_utils import write_json
from hackathon_reefer_dl.metrics import composite_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend two existing prediction files by swapping a high-load tail specialist into the anchor forecast."
    )
    parser.add_argument("--anchor", type=Path, required=True, help="Primary prediction CSV")
    parser.add_argument("--tail-specialist", type=Path, required=True, help="Peak-specialist prediction CSV")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of highest-ranked anchor hours to hand off to the tail specialist",
    )
    parser.add_argument(
        "--tail-point-anchor-weight",
        type=float,
        default=0.275,
        help="Anchor point weight inside the swapped tail rows; tail specialist gets the remaining weight",
    )
    parser.add_argument(
        "--tail-p90-anchor-weight",
        type=float,
        default=0.95,
        help="Anchor p90 weight inside the swapped tail rows; tail specialist gets the remaining weight",
    )
    parser.add_argument(
        "--participant-dir",
        type=Path,
        help="Optional participant package directory for observed-target scoring",
    )
    return parser.parse_args()


def _load_predictions(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "pred_power_kw": float(row["pred_power_kw"]),
                    "pred_p90_kw": float(row["pred_p90_kw"]),
                }
            )
    rows.sort(key=lambda row: str(row["timestamp_utc"]))
    return rows


def main() -> None:
    args = parse_args()
    anchor_rows = _load_predictions(args.anchor)
    tail_rows = _load_predictions(args.tail_specialist)

    if len(anchor_rows) != len(tail_rows):
        raise ValueError("Anchor and tail-specialist files must have the same number of rows.")

    anchor_timestamps = [str(row["timestamp_utc"]) for row in anchor_rows]
    tail_timestamps = [str(row["timestamp_utc"]) for row in tail_rows]
    if anchor_timestamps != tail_timestamps:
        raise ValueError("Anchor and tail-specialist timestamps do not align exactly.")

    anchor_point = np.asarray([float(row["pred_power_kw"]) for row in anchor_rows], dtype=np.float64)
    anchor_p90 = np.asarray([float(row["pred_p90_kw"]) for row in anchor_rows], dtype=np.float64)
    tail_point = np.asarray([float(row["pred_power_kw"]) for row in tail_rows], dtype=np.float64)
    tail_p90 = np.asarray([float(row["pred_p90_kw"]) for row in tail_rows], dtype=np.float64)

    if args.top_k <= 0 or args.top_k > anchor_point.shape[0]:
        raise ValueError("--top-k must be between 1 and the number of prediction rows.")

    order = np.argsort(-anchor_point)
    tail_mask = np.zeros(anchor_point.shape[0], dtype=bool)
    tail_mask[order[: args.top_k]] = True

    point_pred = anchor_point.copy()
    point_pred[tail_mask] = (
        float(args.tail_point_anchor_weight) * anchor_point[tail_mask]
        + (1.0 - float(args.tail_point_anchor_weight)) * tail_point[tail_mask]
    )

    pred_p90 = anchor_p90.copy()
    pred_p90[tail_mask] = (
        float(args.tail_p90_anchor_weight) * anchor_p90[tail_mask]
        + (1.0 - float(args.tail_p90_anchor_weight)) * tail_p90[tail_mask]
    )
    pred_p90 = np.maximum(pred_p90, point_pred)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "pred_power_kw", "pred_p90_kw"])
        writer.writeheader()
        for timestamp, point_value, upper_value in zip(anchor_timestamps, point_pred, pred_p90, strict=True):
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "pred_power_kw": round(float(point_value), 6),
                    "pred_p90_kw": round(float(max(upper_value, point_value)), 6),
                }
            )

    metrics_payload = {
        "anchor": str(args.anchor),
        "tail_specialist": str(args.tail_specialist),
        "top_k": int(args.top_k),
        "tail_point_anchor_weight": float(args.tail_point_anchor_weight),
        "tail_p90_anchor_weight": float(args.tail_p90_anchor_weight),
        "tail_rank_timestamps": [anchor_timestamps[idx] for idx in order[: args.top_k]],
    }

    if args.participant_dir is not None:
        hourly_load = aggregate_hourly_load(args.participant_dir)
        target_times = [parse_target_timestamp(timestamp) for timestamp in anchor_timestamps]
        if all(target_time in hourly_load for target_time in target_times):
            y_true = np.asarray([hourly_load[target_time] for target_time in target_times], dtype=np.float64)
            metrics = composite_metrics(y_true, point_pred, pred_p90)
            metrics_payload.update(metrics.to_dict())
            print(f"Observed-target composite: {metrics.composite:.6f}")

    write_json(args.out.with_suffix(".metrics.json"), metrics_payload)
    print(f"Saved blended submission to {args.out}")


if __name__ == "__main__":
    main()
