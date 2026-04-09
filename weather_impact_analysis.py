from __future__ import annotations

import argparse
import csv
import html
import json
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from reefer_dataset_analysis import (
    ACCENT,
    BG,
    GRID,
    MUTED,
    PANEL,
    TEXT,
    draw_heatmap,
    draw_horizontal_bars,
    draw_scatter_panels,
    format_timestamp,
    number_label,
    parse_decimal,
    parse_timestamp,
    pearson,
    save_svg,
    scale,
    svg_document,
    write_csv,
    write_json,
)


SCALAR_COLOR_PALETTE = ["#0c6b52", "#a44a3f", "#2d6a9f", "#8c6d1f", "#6d3ba8", "#00838f"]


@dataclass
class ScalarHourAccumulator:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    def mean(self) -> float:
        return float("nan") if self.count == 0 else self.total / self.count


@dataclass
class DirectionHourAccumulator:
    sin_total: float = 0.0
    cos_total: float = 0.0
    count: int = 0

    def update(self, degrees_value: float) -> None:
        radians = math.radians(degrees_value)
        self.sin_total += math.sin(radians)
        self.cos_total += math.cos(radians)
        self.count += 1

    def mean_components(self) -> tuple[float, float, float]:
        if self.count == 0:
            return float("nan"), float("nan"), float("nan")
        sin_mean = self.sin_total / self.count
        cos_mean = self.cos_total / self.count
        consistency = math.sqrt(sin_mean**2 + cos_mean**2)
        return sin_mean, cos_mean, consistency


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    participant_root = repo_root.parent / "participant_package" / "participant_package"
    parser = argparse.ArgumentParser(description="Analyze weather impact on hourly reefer power.")
    parser.add_argument(
        "--reefer-csv",
        type=Path,
        default=participant_root / "reefer_release" / "reefer_release.csv",
    )
    parser.add_argument(
        "--weather-dir",
        type=Path,
        default=participant_root / "wetterdaten",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "weather_analysis",
    )
    parser.add_argument(
        "--max-history-hours",
        type=int,
        default=72,
    )
    parser.add_argument(
        "--forecast-horizon-hours",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--scatter-sample-size",
        type=int,
        default=1800,
    )
    return parser.parse_args()


def iter_reefer_rows(reefer_csv_path: Path):
    with reefer_csv_path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=";")


def aggregate_hourly_label(reefer_csv_path: Path) -> dict[datetime, float]:
    hourly_label_kw: dict[datetime, float] = defaultdict(float)
    for row in iter_reefer_rows(reefer_csv_path):
        raw_timestamp = row.get("EventTime")
        power = parse_decimal(row.get("AvPowerCons"))
        if not raw_timestamp or power is None:
            continue
        timestamp = parse_timestamp(raw_timestamp)
        hourly_label_kw[timestamp] += power / 1000.0
    return dict(hourly_label_kw)


def infer_weather_feature_prefix(file_path: Path) -> tuple[str, str]:
    name = file_path.stem.lower().replace("  ", " ")
    if "temperatur" in name:
        metric = "temperature"
    elif "windrichtung" in name:
        metric = "wind_direction"
    elif "wind" in name:
        metric = "wind_speed"
    else:
        raise ValueError(f"Unsupported weather file name: {file_path.name}")

    if "halle3" in name:
        location = "vc_halle3"
    elif "zentralgate" in name:
        location = "zentralgate"
    else:
        location = "unknown_location"
    return metric, location


def weather_file_summaries(weather_dir: Path) -> list[Path]:
    return sorted(path for path in weather_dir.rglob("*.csv") if path.is_file())


def aggregate_weather_scalar(file_path: Path) -> dict[datetime, float]:
    hourly: dict[datetime, ScalarHourAccumulator] = defaultdict(ScalarHourAccumulator)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            raw_timestamp = row.get("UtcTimestamp")
            value = parse_decimal(row.get("Value"))
            if not raw_timestamp or value is None:
                continue
            timestamp = parse_timestamp(raw_timestamp).replace(minute=0, second=0, microsecond=0)
            hourly[timestamp].update(value)
    return {timestamp: acc.mean() for timestamp, acc in hourly.items()}


def aggregate_weather_direction(file_path: Path) -> dict[datetime, tuple[float, float, float]]:
    hourly: dict[datetime, DirectionHourAccumulator] = defaultdict(DirectionHourAccumulator)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            raw_timestamp = row.get("UtcTimestamp")
            value = parse_decimal(row.get("Value"))
            if not raw_timestamp or value is None:
                continue
            timestamp = parse_timestamp(raw_timestamp).replace(minute=0, second=0, microsecond=0)
            hourly[timestamp].update(value)
    return {timestamp: acc.mean_components() for timestamp, acc in hourly.items()}


def aggregate_weather_features(weather_dir: Path) -> tuple[dict[datetime, dict[str, float]], dict[str, object]]:
    per_hour: dict[datetime, dict[str, float]] = defaultdict(dict)
    file_summaries: list[dict[str, object]] = []

    for file_path in weather_file_summaries(weather_dir):
        metric, location = infer_weather_feature_prefix(file_path)
        if metric == "wind_direction":
            hourly_data = aggregate_weather_direction(file_path)
            for timestamp, (sin_mean, cos_mean, consistency) in hourly_data.items():
                per_hour[timestamp][f"weather_wind_direction_{location}_sin"] = sin_mean
                per_hour[timestamp][f"weather_wind_direction_{location}_cos"] = cos_mean
                per_hour[timestamp][f"weather_wind_direction_{location}_consistency"] = consistency
            observed_values = len(hourly_data)
        else:
            hourly_data = aggregate_weather_scalar(file_path)
            unit_suffix = "_c" if metric == "temperature" else ""
            for timestamp, value in hourly_data.items():
                per_hour[timestamp][f"weather_{metric}_{location}{unit_suffix}"] = value
            observed_values = len(hourly_data)

        file_summaries.append(
            {
                "file_name": file_path.name,
                "metric": metric,
                "location": location,
                "hourly_observations": observed_values,
            }
        )

    for features in per_hour.values():
        add_derived_weather_features(features)

    summary = {
        "source_files": file_summaries,
        "hourly_timestamp_count": len(per_hour),
    }
    return dict(per_hour), summary


def add_derived_weather_features(features: dict[str, float]) -> None:
    add_mean_and_spread(
        features,
        "weather_temperature_vc_halle3_c",
        "weather_temperature_zentralgate_c",
        "weather_temperature_mean_c",
        "weather_temperature_spread_c",
    )
    add_mean_and_spread(
        features,
        "weather_wind_speed_vc_halle3",
        "weather_wind_speed_zentralgate",
        "weather_wind_speed_mean",
        "weather_wind_speed_spread",
    )
    add_mean_feature(
        features,
        ["weather_wind_direction_vc_halle3_sin", "weather_wind_direction_zentralgate_sin"],
        "weather_wind_direction_mean_sin",
    )
    add_mean_feature(
        features,
        ["weather_wind_direction_vc_halle3_cos", "weather_wind_direction_zentralgate_cos"],
        "weather_wind_direction_mean_cos",
    )
    if (
        math.isfinite(features.get("weather_wind_direction_mean_sin", float("nan")))
        and math.isfinite(features.get("weather_wind_direction_mean_cos", float("nan")))
    ):
        features["weather_wind_direction_mean_consistency"] = math.sqrt(
            features["weather_wind_direction_mean_sin"] ** 2
            + features["weather_wind_direction_mean_cos"] ** 2
        )


def add_mean_and_spread(
    features: dict[str, float],
    left_key: str,
    right_key: str,
    mean_key: str,
    spread_key: str,
) -> None:
    left = features.get(left_key, float("nan"))
    right = features.get(right_key, float("nan"))
    values = [value for value in (left, right) if math.isfinite(value)]
    if values:
        features[mean_key] = float(np.mean(values))
    if len(values) == 2:
        features[spread_key] = left - right


def add_mean_feature(features: dict[str, float], keys: list[str], output_key: str) -> None:
    values = [features.get(key, float("nan")) for key in keys]
    finite = [value for value in values if math.isfinite(value)]
    if finite:
        features[output_key] = float(np.mean(finite))


def join_weather_and_label(
    hourly_label_kw: dict[datetime, float],
    hourly_weather: dict[datetime, dict[str, float]],
) -> tuple[list[dict[str, float | str]], list[str], tuple[datetime, datetime] | None]:
    overlapping_timestamps = sorted(set(hourly_label_kw) & set(hourly_weather))
    if not overlapping_timestamps:
        return [], [], None

    feature_names = sorted({feature for feature_map in hourly_weather.values() for feature in feature_map})
    records: list[dict[str, float | str]] = []
    for timestamp in overlapping_timestamps:
        feature_map = hourly_weather[timestamp]
        record: dict[str, float | str] = {
            "timestamp_utc": format_timestamp(timestamp) or "",
            "label_power_kw": hourly_label_kw[timestamp],
        }
        for feature_name in feature_names:
            record[feature_name] = feature_map.get(feature_name, float("nan"))
        records.append(record)
    return records, feature_names, (overlapping_timestamps[0], overlapping_timestamps[-1])


def build_hourly_grid(
    hourly_label_kw: dict[datetime, float],
    hourly_weather: dict[datetime, dict[str, float]],
    feature_names: list[str],
) -> tuple[list[datetime], np.ndarray, dict[str, np.ndarray]]:
    overlap_start = max(min(hourly_label_kw), min(hourly_weather))
    overlap_end = min(max(hourly_label_kw), max(hourly_weather))

    timestamps: list[datetime] = []
    current = overlap_start
    while current <= overlap_end:
        timestamps.append(current)
        current += timedelta(hours=1)

    label_array = np.array([hourly_label_kw.get(timestamp, float("nan")) for timestamp in timestamps], dtype=float)
    feature_arrays = {
        feature_name: np.array(
            [hourly_weather.get(timestamp, {}).get(feature_name, float("nan")) for timestamp in timestamps],
            dtype=float,
        )
        for feature_name in feature_names
    }
    return timestamps, label_array, feature_arrays


def same_hour_correlations(records: list[dict[str, float | str]], feature_names: list[str]) -> list[dict[str, float | str]]:
    label_values = np.array([float(record["label_power_kw"]) for record in records], dtype=float)
    correlations: list[dict[str, float | str]] = []
    for feature_name in feature_names:
        feature_values = np.array([float(record[feature_name]) for record in records], dtype=float)
        correlation = pearson(feature_values, label_values)
        correlations.append(
            {
                "feature": feature_name,
                "pearson_correlation": correlation,
                "absolute_correlation": abs(correlation),
            }
        )
    correlations.sort(key=lambda item: float(item["absolute_correlation"]), reverse=True)
    return correlations


def rolling_nanmean(values: np.ndarray, window_hours: int) -> np.ndarray:
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    cumulative_sum = np.concatenate(([0.0], np.cumsum(filled)))
    cumulative_count = np.concatenate(([0], np.cumsum(valid.astype(int))))
    sums = cumulative_sum[window_hours:] - cumulative_sum[:-window_hours]
    counts = cumulative_count[window_hours:] - cumulative_count[:-window_hours]
    result = np.full(values.shape, np.nan, dtype=float)
    window_means = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=window_means, where=counts > 0)
    result[window_hours - 1 :] = window_means
    return result


def weather_history_window_scores(
    feature_arrays: dict[str, np.ndarray],
    label_array: np.ndarray,
    max_history_hours: int,
    forecast_horizon_hours: int,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    rows: list[dict[str, float | str]] = []
    matrices: dict[str, np.ndarray] = {}

    for feature_name, values in feature_arrays.items():
        horizon_matrix = np.zeros((max_history_hours, forecast_horizon_hours), dtype=float)
        for window_hours in range(1, max_history_hours + 1):
            window_values = rolling_nanmean(values, window_hours)
            horizon_correlations: list[float] = []
            for horizon in range(1, forecast_horizon_hours + 1):
                if horizon >= len(label_array):
                    correlation = 0.0
                else:
                    correlation = pearson(window_values[:-horizon], label_array[horizon:])
                horizon_correlations.append(correlation)
                horizon_matrix[window_hours - 1, horizon - 1] = correlation

            mean_abs_future_correlation = float(np.mean(np.abs(horizon_correlations)))
            rows.append(
                {
                    "feature": feature_name,
                    "history_window_hours": float(window_hours),
                    "mean_abs_future_correlation": mean_abs_future_correlation,
                    "horizon_1_corr": horizon_correlations[0],
                    "horizon_24_corr": horizon_correlations[-1],
                }
            )
        matrices[feature_name] = horizon_matrix

    rows.sort(
        key=lambda item: (str(item["feature"]), float(item["history_window_hours"])),
    )
    return rows, matrices


def best_window_recommendations(
    window_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    by_feature: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in window_rows:
        by_feature[str(row["feature"])].append(row)

    recommendations: list[dict[str, float | str]] = []
    for feature_name, rows in by_feature.items():
        scores = np.array([float(row["mean_abs_future_correlation"]) for row in rows], dtype=float)
        windows = np.array([float(row["history_window_hours"]) for row in rows], dtype=float)
        peak_index = int(np.argmax(scores))
        peak_score = float(scores[peak_index])
        threshold = peak_score * 0.95
        efficient_index = int(np.argmax(scores >= threshold))
        recommendations.append(
            {
                "feature": feature_name,
                "peak_history_window_hours": windows[peak_index],
                "peak_mean_abs_future_correlation": peak_score,
                "efficient_history_window_hours": windows[efficient_index],
                "efficient_window_score": float(scores[efficient_index]),
            }
        )

    recommendations.sort(
        key=lambda item: float(item["peak_mean_abs_future_correlation"]),
        reverse=True,
    )
    return recommendations


def top_window_features(
    recommendations: list[dict[str, float | str]],
    limit: int,
) -> list[str]:
    preferred_order = [
        "weather_temperature_mean_c",
        "weather_wind_speed_mean",
        "weather_wind_direction_mean_consistency",
        "weather_temperature_vc_halle3_c",
        "weather_temperature_zentralgate_c",
        "weather_wind_speed_vc_halle3",
        "weather_wind_speed_zentralgate",
    ]
    chosen: list[str] = []
    available = {str(row["feature"]) for row in recommendations}
    for feature_name in preferred_order:
        if feature_name in available and feature_name not in chosen:
            chosen.append(feature_name)
        if len(chosen) == limit:
            return chosen
    for row in recommendations:
        feature_name = str(row["feature"])
        if feature_name not in chosen:
            chosen.append(feature_name)
        if len(chosen) == limit:
            break
    return chosen


def friendly_feature_name(feature_name: str) -> str:
    name = feature_name.replace("weather_", "").replace("_", " ")
    replacements = {
        "vc halle3": "VC Halle 3",
        "zentralgate": "Zentralgate",
        "mean c": "mean (C)",
        "spread c": "spread (C)",
        "wind speed": "wind speed",
        "wind direction": "wind direction",
        "consistency": "consistency",
        "sin": "sin",
        "cos": "cos",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    return name


def draw_multi_line_chart(
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    x_label: str,
    y_label: str,
) -> str:
    width = 1240
    height = 620
    left = 95
    right = 220
    top = 70
    bottom = 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_values = [value for _, values in series for value in values if math.isfinite(value)]
    y_min = min(y_values) if y_values else 0.0
    y_max = max(y_values) if y_values else 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    padding = max((y_max - y_min) * 0.08, 0.01)
    y_min -= padding
    y_max += padding
    x_min = min(x_values)
    x_max = max(x_values)

    body = [
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="18" fill="{PANEL}"/>',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" fill="{TEXT}" font-size="24" font-family="Segoe UI, Arial, sans-serif">{html.escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 18:.1f}" text-anchor="middle" fill="{MUTED}" font-size="14" font-family="Segoe UI, Arial, sans-serif">{html.escape(x_label)}</text>',
        f'<text x="24" y="{height / 2:.1f}" text-anchor="middle" fill="{MUTED}" font-size="14" font-family="Segoe UI, Arial, sans-serif" transform="rotate(-90 24 {height / 2:.1f})">{html.escape(y_label)}</text>',
    ]

    for tick in range(6):
        y_value = y_min + (y_max - y_min) * tick / 5.0
        y = scale(y_value, y_min, y_max, top + plot_h, top)
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(number_label(y_value))}</text>')

    for tick in range(7):
        x_value = x_min + (x_max - x_min) * tick / 6.0
        x = scale(x_value, x_min, x_max, left, left + plot_w)
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 24:.1f}" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{int(round(x_value))}</text>')

    for series_index, (label, values) in enumerate(series):
        color = SCALAR_COLOR_PALETTE[series_index % len(SCALAR_COLOR_PALETTE)]
        points = []
        for x_value, y_value in zip(x_values, values):
            x = scale(x_value, x_min, x_max, left, left + plot_w)
            y = scale(y_value, y_min, y_max, top + plot_h, top)
            points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{" ".join(points)}"/>')
        legend_y = top + 18 + series_index * 26
        legend_x = left + plot_w + 24
        body.append(f'<line x1="{legend_x}" x2="{legend_x + 28}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        wrapped = textwrap.wrap(label, width=20) or [label]
        for line_index, line in enumerate(wrapped[:2]):
            body.append(f'<text x="{legend_x + 36}" y="{legend_y + 4 + line_index * 13:.1f}" text-anchor="start" fill="{TEXT}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(line)}</text>')

    return svg_document(width, height, title, "".join(body))


def summarize_overlap(
    hourly_label_kw: dict[datetime, float],
    hourly_weather: dict[datetime, dict[str, float]],
    overlap_range: tuple[datetime, datetime] | None,
    feature_names: list[str],
) -> dict[str, object]:
    label_min = min(hourly_label_kw) if hourly_label_kw else None
    label_max = max(hourly_label_kw) if hourly_label_kw else None
    weather_min = min(hourly_weather) if hourly_weather else None
    weather_max = max(hourly_weather) if hourly_weather else None
    return {
        "label_hour_count": len(hourly_label_kw),
        "weather_hour_count": len(hourly_weather),
        "weather_feature_count": len(feature_names),
        "label_range_utc": [format_timestamp(label_min), format_timestamp(label_max)],
        "weather_range_utc": [format_timestamp(weather_min), format_timestamp(weather_max)],
        "overlap_range_utc": [format_timestamp(overlap_range[0]), format_timestamp(overlap_range[1])] if overlap_range else [None, None],
    }


def build_weather_report_html(
    summary_payload: dict[str, object],
    plot_files: list[tuple[str, str]],
) -> str:
    overlap = summary_payload["overlap_summary"]
    actual_rows = "\n".join(
        f"<tr><td>{html.escape(friendly_feature_name(str(row['feature'])))}</td><td>{float(row['pearson_correlation']):.3f}</td></tr>"
        for row in summary_payload["top_same_hour_correlations"]
    )
    history_rows = "\n".join(
        f"<tr><td>{html.escape(friendly_feature_name(str(row['feature'])))}</td><td>{int(float(row['efficient_history_window_hours']))} h</td><td>{int(float(row['peak_history_window_hours']))} h</td><td>{float(row['peak_mean_abs_future_correlation']):.3f}</td></tr>"
        for row in summary_payload["window_recommendations"][:10]
    )
    plot_cards = "\n".join(
        f'<section class="card"><h2>{html.escape(title)}</h2><img src="{html.escape(filename)}" alt="{html.escape(title)}" /></section>'
        for filename, title in plot_files
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Weather Impact Analysis</title>
  <style>
    body {{ margin: 0; background: linear-gradient(180deg, #fffdf9 0%, #eef5f0 100%); color: {TEXT}; font-family: "Segoe UI", Arial, sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 2.4rem; }}
    h2 {{ margin: 0 0 12px; font-size: 1.2rem; }}
    p, td, th {{ line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin: 24px 0 28px; }}
    .card {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 18px; padding: 18px; box-shadow: 0 10px 28px rgba(48, 36, 20, 0.06); margin-bottom: 18px; }}
    .tables {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid {GRID}; text-align: left; }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid {GRID}; background: white; }}
  </style>
</head>
<body>
  <main>
    <h1>Weather Impact Analysis</h1>
    <p>This report focuses only on the weather streams and how they relate to hourly reefer power, both at the same hour and as historical context for predicting the next 24 hours.</p>
    <div class="grid">
      <section class="card"><h2>Overlap Window</h2><p><strong>Label range:</strong> {overlap['label_range_utc'][0]} to {overlap['label_range_utc'][1]}</p><p><strong>Weather range:</strong> {overlap['weather_range_utc'][0]} to {overlap['weather_range_utc'][1]}</p><p><strong>Usable overlap:</strong> {overlap['overlap_range_utc'][0]} to {overlap['overlap_range_utc'][1]}</p></section>
      <section class="card"><h2>Hourly Coverage</h2><p><strong>Label hours:</strong> {overlap['label_hour_count']:,}</p><p><strong>Weather hours:</strong> {overlap['weather_hour_count']:,}</p><p><strong>Weather features:</strong> {overlap['weather_feature_count']}</p></section>
      <section class="card"><h2>Interpretation</h2><p>Same-hour correlations show actual-weather impact. History-window scores show how much observed weather context is useful at forecast time for the next 24 hours.</p></section>
    </div>
    <section class="tables">
      <div class="card"><h2>Top Same-Hour Weather Correlations</h2><table><thead><tr><th>Feature</th><th>Pearson r</th></tr></thead><tbody>{actual_rows}</tbody></table></div>
      <div class="card"><h2>Recommended History Windows</h2><table><thead><tr><th>Feature</th><th>Efficient window</th><th>Peak window</th><th>Peak score</th></tr></thead><tbody>{history_rows}</tbody></table></div>
    </section>
    {plot_cards}
  </main>
</body>
</html>"""


def build_weather_report_markdown(summary_payload: dict[str, object]) -> str:
    overlap = summary_payload["overlap_summary"]
    lines = [
        "# Weather Impact Analysis",
        "",
        "## Scope",
        "- Same-hour weather versus same-hour reefer power shows actual observed weather impact.",
        "- History-window analysis measures how much past observed weather at forecast time is informative for the next 24 hourly power values.",
        "- This is still an upper-bound analysis because a production day-ahead model would use weather forecasts, not the future realized weather.",
        "",
        "## Overlap",
        f"- Label range: {overlap['label_range_utc'][0]} to {overlap['label_range_utc'][1]}",
        f"- Weather range: {overlap['weather_range_utc'][0]} to {overlap['weather_range_utc'][1]}",
        f"- Usable overlap: {overlap['overlap_range_utc'][0]} to {overlap['overlap_range_utc'][1]}",
        "",
        "## Top Same-Hour Weather Correlations",
    ]
    for row in summary_payload["top_same_hour_correlations"]:
        lines.append(f"- `{row['feature']}`: {float(row['pearson_correlation']):.3f}")
    lines.extend(["", "## Recommended Weather History Windows"])
    for row in summary_payload["window_recommendations"][:10]:
        lines.append(
            f"- `{row['feature']}`: efficient window {int(float(row['efficient_history_window_hours']))} h, peak window {int(float(row['peak_history_window_hours']))} h, peak score {float(row['peak_mean_abs_future_correlation']):.3f}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    joined_records: list[dict[str, float | str]],
    actual_correlations: list[dict[str, float | str]],
    window_rows: list[dict[str, float | str]],
    window_recommendations: list[dict[str, float | str]],
    horizon_matrices: dict[str, np.ndarray],
    overlap_summary: dict[str, object],
    weather_summary: dict[str, object],
    top_features: int,
    scatter_sample_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "weather_label_hourly.csv", joined_records)
    write_csv(output_dir / "same_hour_weather_correlations.csv", actual_correlations)
    write_csv(output_dir / "weather_history_window_scores.csv", window_rows)
    write_csv(output_dir / "weather_history_window_recommendations.csv", window_recommendations)

    summary_payload = {
        "overlap_summary": overlap_summary,
        "weather_source_summary": weather_summary,
        "top_same_hour_correlations": actual_correlations[:top_features],
        "window_recommendations": window_recommendations,
    }
    write_json(output_dir / "weather_summary.json", summary_payload)

    plots: list[tuple[str, str]] = []

    same_hour_bar = draw_horizontal_bars(
        "Same-Hour Weather Correlations To Power",
        [friendly_feature_name(str(row["feature"])) for row in actual_correlations[:top_features]],
        [float(row["pearson_correlation"]) for row in actual_correlations[:top_features]],
        "Pearson correlation",
        diverging=True,
    )
    save_svg(output_dir / "same_hour_weather_correlations.svg", same_hour_bar)
    plots.append(("same_hour_weather_correlations.svg", "Same-Hour Weather Correlations To Power"))

    same_hour_scatter = draw_scatter_panels(joined_records, actual_correlations, scatter_sample_size)
    save_svg(output_dir / "same_hour_weather_relationships.svg", same_hour_scatter)
    plots.append(("same_hour_weather_relationships.svg", "Same-Hour Weather Relationships"))

    selected_window_features = top_window_features(window_recommendations, 4)
    line_series: list[tuple[str, list[float]]] = []
    for feature_name in selected_window_features:
        feature_rows = [row for row in window_rows if str(row["feature"]) == feature_name]
        line_series.append(
            (
                friendly_feature_name(feature_name),
                [float(row["mean_abs_future_correlation"]) for row in feature_rows],
            )
        )
    history_line_chart = draw_multi_line_chart(
        "Past Weather Window Usefulness For Predicting The Next 24 Hours",
        list(range(1, len(line_series[0][1]) + 1)),
        line_series,
        "History window size available at forecast time (hours)",
        "Mean absolute correlation to future 1-24h power",
    )
    save_svg(output_dir / "weather_history_window_usefulness.svg", history_line_chart)
    plots.append(("weather_history_window_usefulness.svg", "Past Weather Window Usefulness"))

    best_feature = str(window_recommendations[0]["feature"]) if window_recommendations else selected_window_features[0]
    heatmap_window_sizes = [1, 3, 6, 12, 24, 36, 48, 72]
    matrix = horizon_matrices[best_feature]
    sampled_matrix = np.array([matrix[window_size - 1, :] for window_size in heatmap_window_sizes], dtype=float)
    history_heatmap = draw_heatmap(
        f"Forecast-Horizon Sensitivity For {friendly_feature_name(best_feature)}",
        sampled_matrix,
        [f"h+{horizon}" for horizon in range(1, matrix.shape[1] + 1)],
        [f"{window_size} h" for window_size in heatmap_window_sizes],
        True,
    )
    save_svg(output_dir / "weather_history_horizon_heatmap.svg", history_heatmap)
    plots.append(("weather_history_horizon_heatmap.svg", "Forecast-Horizon Sensitivity"))

    (output_dir / "report.md").write_text(build_weather_report_markdown(summary_payload), encoding="utf-8")
    (output_dir / "report.html").write_text(build_weather_report_html(summary_payload, plots), encoding="utf-8")


def analyze_weather(args: argparse.Namespace) -> Path:
    reefer_csv = args.reefer_csv.resolve()
    weather_dir = args.weather_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not reefer_csv.exists():
        raise FileNotFoundError(f"Reefer CSV not found: {reefer_csv}")
    if not weather_dir.exists():
        raise FileNotFoundError(f"Weather directory not found: {weather_dir}")

    hourly_label_kw = aggregate_hourly_label(reefer_csv)
    hourly_weather, weather_summary = aggregate_weather_features(weather_dir)
    joined_records, feature_names, overlap_range = join_weather_and_label(hourly_label_kw, hourly_weather)
    if not joined_records:
        raise RuntimeError("No overlap between weather data and reefer label data.")

    _, label_array, feature_arrays = build_hourly_grid(hourly_label_kw, hourly_weather, feature_names)
    actual_correlations = same_hour_correlations(joined_records, feature_names)
    window_rows, horizon_matrices = weather_history_window_scores(
        feature_arrays,
        label_array,
        max_history_hours=args.max_history_hours,
        forecast_horizon_hours=args.forecast_horizon_hours,
    )
    window_recommendations = best_window_recommendations(window_rows)
    overlap_summary = summarize_overlap(hourly_label_kw, hourly_weather, overlap_range, feature_names)

    write_outputs(
        output_dir=output_dir,
        joined_records=joined_records,
        actual_correlations=actual_correlations,
        window_rows=window_rows,
        window_recommendations=window_recommendations,
        horizon_matrices=horizon_matrices,
        overlap_summary=overlap_summary,
        weather_summary=weather_summary,
        top_features=args.top_features,
        scatter_sample_size=args.scatter_sample_size,
    )
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = analyze_weather(args)
    print(f"Weather impact analysis complete. Open {output_dir / 'report.html'} for the summary report.")


if __name__ == "__main__":
    main()
