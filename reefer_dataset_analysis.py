from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import textwrap
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

import numpy as np


RAW_DECIMAL_FIELDS = {
    "AvPowerCons",
    "TemperatureSetPoint",
    "TemperatureAmbient",
    "TemperatureReturn",
    "RemperatureSupply",
    "stack_tier",
}
TIMESTAMP_INPUT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TIMESTAMP_OUTPUT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
BG = "#fffdf9"
PANEL = "#fff7ed"
GRID = "#dfcfbb"
TEXT = "#2a2623"
MUTED = "#77685c"
ACCENT = "#0c6b52"
POSITIVE = "#2a7f62"
NEGATIVE = "#b85042"


@dataclass
class HourAccumulator:
    row_count: int = 0
    power_w_sum: float = 0.0
    setpoint_sum: float = 0.0
    setpoint_count: int = 0
    ambient_sum: float = 0.0
    ambient_count: int = 0
    return_sum: float = 0.0
    return_count: int = 0
    supply_sum: float = 0.0
    supply_count: int = 0
    stack_tier_sum: float = 0.0
    stack_tier_count: int = 0
    hardware_counter: Counter[str] = field(default_factory=Counter)
    size_counter: Counter[str] = field(default_factory=Counter)
    tier_counter: Counter[str] = field(default_factory=Counter)

    def update(self, row: dict[str, str]) -> None:
        self.row_count += 1
        power = parse_decimal(row.get("AvPowerCons"))
        if power is not None:
            self.power_w_sum += power

        setpoint = parse_decimal(row.get("TemperatureSetPoint"))
        if setpoint is not None:
            self.setpoint_sum += setpoint
            self.setpoint_count += 1

        ambient = parse_decimal(row.get("TemperatureAmbient"))
        if ambient is not None:
            self.ambient_sum += ambient
            self.ambient_count += 1

        temp_return = parse_decimal(row.get("TemperatureReturn"))
        if temp_return is not None:
            self.return_sum += temp_return
            self.return_count += 1

        supply = parse_decimal(row.get("RemperatureSupply") or row.get("TemperatureSupply"))
        if supply is not None:
            self.supply_sum += supply
            self.supply_count += 1

        stack_tier = parse_decimal(row.get("stack_tier"))
        if stack_tier is not None:
            self.stack_tier_sum += stack_tier
            self.stack_tier_count += 1
            self.tier_counter[str(int(round(stack_tier)))] += 1

        hardware = clean_category(row.get("HardwareType"))
        if hardware:
            self.hardware_counter[hardware] += 1

        container_size = clean_category(row.get("ContainerSize"))
        if container_size:
            self.size_counter[container_size] += 1


@dataclass
class RawDatasetSummary:
    row_count: int = 0
    hourly_count: int = 0
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    field_missing_counts: dict[str, int] = field(default_factory=dict)
    hardware_type_counts: Counter[str] = field(default_factory=Counter)
    container_size_counts: Counter[str] = field(default_factory=Counter)
    stack_tier_counts: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    participant_root = repo_root.parent / "participant_package" / "participant_package"
    parser = argparse.ArgumentParser(description="Analyze the reefer challenge dataset.")
    parser.add_argument("--reefer-zip", type=Path, default=participant_root / "reefer_release.zip")
    parser.add_argument("--target-csv", type=Path, default=participant_root / "target_timestamps.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "dataset_analysis")
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--scatter-sample-size", type=int, default=1800)
    return parser.parse_args()


def parse_decimal(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.upper() == "NULL":
        return None
    return float(stripped.replace(",", "."))


def clean_category(value: str | None) -> str:
    return "" if value is None else value.strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def safe_mean(total: float, count: int) -> float:
    return float("nan") if count == 0 else total / count


def diff_or_nan(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return a - b


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_INPUT_FORMAT)


def format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime(TIMESTAMP_OUTPUT_FORMAT)


def find_csv_entry(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    for entry in archive.infolist():
        if entry.filename.lower().endswith(".csv") and not entry.filename.startswith("__MACOSX/"):
            return entry
    raise FileNotFoundError("No CSV file found in reefer archive.")


def iter_reefer_rows(reefer_zip_path: Path):
    with zipfile.ZipFile(reefer_zip_path) as archive:
        with archive.open(find_csv_entry(archive)) as raw_stream:
            text_stream = TextIOWrapper(raw_stream, encoding="utf-8", newline="")
            yield from csv.DictReader(text_stream, delimiter=";")


def collect_hourly_dataset(reefer_zip_path: Path) -> tuple[list[dict[str, float | str]], RawDatasetSummary]:
    hourly: dict[datetime, HourAccumulator] = {}
    summary = RawDatasetSummary(
        field_missing_counts={field: 0 for field in RAW_DECIMAL_FIELDS | {"HardwareType", "ContainerSize", "EventTime"}},
    )

    for row in iter_reefer_rows(reefer_zip_path):
        summary.row_count += 1

        raw_timestamp = row.get("EventTime")
        if not raw_timestamp:
            summary.field_missing_counts["EventTime"] += 1
            continue

        timestamp = parse_timestamp(raw_timestamp)
        summary.timestamp_min = timestamp if summary.timestamp_min is None else min(summary.timestamp_min, timestamp)
        summary.timestamp_max = timestamp if summary.timestamp_max is None else max(summary.timestamp_max, timestamp)

        for field in RAW_DECIMAL_FIELDS:
            if not row.get(field):
                summary.field_missing_counts[field] += 1

        hardware = clean_category(row.get("HardwareType"))
        if hardware:
            summary.hardware_type_counts[hardware] += 1
        else:
            summary.field_missing_counts["HardwareType"] += 1

        container_size = clean_category(row.get("ContainerSize"))
        if container_size:
            summary.container_size_counts[container_size] += 1
        else:
            summary.field_missing_counts["ContainerSize"] += 1

        stack_tier = parse_decimal(row.get("stack_tier"))
        if stack_tier is not None:
            summary.stack_tier_counts[str(int(round(stack_tier)))] += 1

        hourly.setdefault(timestamp, HourAccumulator()).update(row)

    summary.hourly_count = len(hourly)
    hardware_categories = sorted(summary.hardware_type_counts)
    size_categories = sorted(summary.container_size_counts)
    tier_categories = sorted(summary.stack_tier_counts, key=lambda value: int(value))

    records: list[dict[str, float | str]] = []
    for timestamp in sorted(hourly):
        acc = hourly[timestamp]
        row_count = max(acc.row_count, 1)
        mean_setpoint = safe_mean(acc.setpoint_sum, acc.setpoint_count)
        mean_ambient = safe_mean(acc.ambient_sum, acc.ambient_count)
        mean_return = safe_mean(acc.return_sum, acc.return_count)
        mean_supply = safe_mean(acc.supply_sum, acc.supply_count)
        mean_stack_tier = safe_mean(acc.stack_tier_sum, acc.stack_tier_count)

        record: dict[str, float | str] = {
            "timestamp_utc": timestamp.strftime(TIMESTAMP_OUTPUT_FORMAT),
            "label_power_kw": acc.power_w_sum / 1000.0,
            "active_container_count": float(acc.row_count),
            "mean_temperature_setpoint_c": mean_setpoint,
            "mean_temperature_ambient_c": mean_ambient,
            "mean_temperature_return_c": mean_return,
            "mean_temperature_supply_c": mean_supply,
            "mean_ambient_minus_setpoint_c": diff_or_nan(mean_ambient, mean_setpoint),
            "mean_return_minus_supply_c": diff_or_nan(mean_return, mean_supply),
            "mean_stack_tier": mean_stack_tier,
            "hour_of_day": float(timestamp.hour),
            "day_of_week": float(timestamp.weekday()),
            "day_of_year": float(timestamp.timetuple().tm_yday),
            "month": float(timestamp.month),
            "is_weekend": float(int(timestamp.weekday() >= 5)),
            "hour_sin": math.sin(2.0 * math.pi * timestamp.hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * timestamp.hour / 24.0),
            "dow_sin": math.sin(2.0 * math.pi * timestamp.weekday() / 7.0),
            "dow_cos": math.cos(2.0 * math.pi * timestamp.weekday() / 7.0),
        }

        for hardware in hardware_categories:
            record[f"share_hardware_{slugify(hardware)}"] = acc.hardware_counter[hardware] / row_count
        for container_size in size_categories:
            record[f"share_container_size_{slugify(container_size)}"] = acc.size_counter[container_size] / row_count
        for tier in tier_categories:
            record[f"share_stack_tier_{slugify(tier)}"] = acc.tier_counter[tier] / row_count

        records.append(record)

    return records, summary


def load_target_timestamps(path: Path) -> list[datetime]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [datetime.strptime(row["timestamp_utc"], TIMESTAMP_OUTPUT_FORMAT) for row in reader]


def records_to_matrix(records: list[dict[str, float | str]]) -> tuple[list[str], np.ndarray]:
    feature_names = [name for name in records[0] if name != "timestamp_utc"]
    matrix = np.array([[float(record[name]) for name in feature_names] for record in records], dtype=float)
    return feature_names, matrix


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = float(np.sqrt(np.sum(xv**2) * np.sum(yv**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(xv * yv) / denom)


def feature_correlations(feature_names: list[str], matrix: np.ndarray) -> list[dict[str, float | str]]:
    label_index = feature_names.index("label_power_kw")
    label_values = matrix[:, label_index]
    correlations: list[dict[str, float | str]] = []
    for index, feature_name in enumerate(feature_names):
        if feature_name == "label_power_kw":
            continue
        correlation = pearson(matrix[:, index], label_values)
        correlations.append(
            {
                "feature": feature_name,
                "pearson_correlation": correlation,
                "absolute_correlation": abs(correlation),
            }
        )
    correlations.sort(key=lambda item: float(item["absolute_correlation"]), reverse=True)
    return correlations


def top_correlation_heatmap(
    feature_names: list[str],
    matrix: np.ndarray,
    correlations: list[dict[str, float | str]],
    top_features: int,
) -> tuple[list[str], np.ndarray]:
    selected = [str(item["feature"]) for item in correlations[:top_features]] + ["label_power_kw"]
    indices = [feature_names.index(name) for name in selected]
    values = matrix[:, indices]
    heatmap = np.zeros((len(indices), len(indices)), dtype=float)
    for row in range(len(indices)):
        for col in range(len(indices)):
            heatmap[row, col] = pearson(values[:, row], values[:, col])
    return selected, heatmap


def ridge_permutation_importance(
    feature_names: list[str],
    matrix: np.ndarray,
    alpha: float,
) -> tuple[list[dict[str, float | str]], float]:
    label_index = feature_names.index("label_power_kw")
    predictor_names = [name for name in feature_names if name != "label_power_kw"]
    predictor_indices = [feature_names.index(name) for name in predictor_names]
    x_all = matrix[:, predictor_indices]
    y_all = matrix[:, label_index]

    split = min(max(int(len(matrix) * 0.8), 10), len(matrix) - 1)
    x_train = x_all[:split].copy()
    x_test = x_all[split:].copy()
    y_train = y_all[:split]
    y_test = y_all[split:]

    train_means = np.nanmean(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, train_means)
    x_test = np.where(np.isfinite(x_test), x_test, train_means)

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    stds[stds == 0.0] = 1.0
    x_train = (x_train - means) / stds
    x_test = (x_test - means) / stds

    y_mean = float(np.mean(y_train))
    weights = np.linalg.solve(
        x_train.T @ x_train + alpha * np.eye(x_train.shape[1], dtype=float),
        x_train.T @ (y_train - y_mean),
    )

    def predict(data: np.ndarray) -> np.ndarray:
        return data @ weights + y_mean

    base_predictions = predict(x_test)
    base_mae = float(np.mean(np.abs(base_predictions - y_test)))
    rng = np.random.default_rng(42)
    importances: list[dict[str, float | str]] = []
    for index, feature_name in enumerate(predictor_names):
        permuted = x_test.copy()
        permuted[:, index] = rng.permutation(permuted[:, index])
        mae = float(np.mean(np.abs(predict(permuted) - y_test)))
        importances.append(
            {
                "feature": feature_name,
                "mae_increase_kw": mae - base_mae,
                "absolute_weight": abs(float(weights[index])),
            }
        )
    importances.sort(key=lambda item: float(item["mae_increase_kw"]), reverse=True)
    return importances, base_mae


def label_profiles(records: list[dict[str, float | str]]) -> tuple[list[float], list[float], np.ndarray]:
    hourly = np.zeros(24, dtype=float)
    hourly_counts = np.zeros(24, dtype=float)
    day_hour = np.zeros((7, 24), dtype=float)
    day_hour_counts = np.zeros((7, 24), dtype=float)
    label_values: list[float] = []

    for record in records:
        label = float(record["label_power_kw"])
        hour = int(float(record["hour_of_day"]))
        day = int(float(record["day_of_week"]))
        label_values.append(label)
        hourly[hour] += label
        hourly_counts[hour] += 1.0
        day_hour[day, hour] += label
        day_hour_counts[day, hour] += 1.0

    hourly /= np.where(hourly_counts == 0.0, 1.0, hourly_counts)
    day_hour /= np.where(day_hour_counts == 0.0, 1.0, day_hour_counts)
    return label_values, hourly.tolist(), day_hour


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def number_label(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:,.0f}"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2.0
    return range_min + (value - domain_min) / (domain_max - domain_min) * (range_max - range_min)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb_to_hex(value: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*value)


def blend(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * ratio)) for a, b in zip(start, end))


def heatmap_color(value: float, value_min: float, value_max: float, diverging: bool) -> str:
    if diverging:
        clipped = max(-1.0, min(1.0, value))
        if clipped >= 0:
            return rgb_to_hex(blend(hex_to_rgb("#f6ddd6"), hex_to_rgb(POSITIVE), clipped))
        return rgb_to_hex(blend(hex_to_rgb("#f6ddd6"), hex_to_rgb(NEGATIVE), abs(clipped)))

    if value_max == value_min:
        ratio = 0.5
    else:
        ratio = (value - value_min) / (value_max - value_min)
    ratio = max(0.0, min(1.0, ratio))
    return rgb_to_hex(blend(hex_to_rgb("#fff3df"), hex_to_rgb(ACCENT), ratio))


def svg_document(width: int, height: int, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="{BG}"/>'
        f'<title>{html.escape(title)}</title>'
        f"{body}</svg>"
    )


def save_svg(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def draw_line_chart(title: str, x_values: list[float], y_values: list[float], x_labels: list[str], y_label: str) -> str:
    width = 1200
    height = 520
    left = 90
    right = 30
    top = 65
    bottom = 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    pad = max((y_max - y_min) * 0.08, 1.0)
    y_min -= pad
    y_max += pad
    body = [
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="18" fill="{PANEL}"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" fill="{TEXT}" font-size="24" font-family="Segoe UI, Arial, sans-serif">{html.escape(title)}</text>',
        f'<text x="24" y="{height / 2:.1f}" text-anchor="middle" fill="{MUTED}" font-size="14" font-family="Segoe UI, Arial, sans-serif" transform="rotate(-90 24 {height / 2:.1f})">{html.escape(y_label)}</text>',
    ]
    for tick in range(6):
        y_value = y_min + (y_max - y_min) * tick / 5.0
        y = scale(y_value, y_min, y_max, top + plot_h, top)
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(number_label(y_value))}</text>')
    for tick in range(6):
        index = round((len(x_values) - 1) * tick / 5.0)
        x = scale(x_values[index], x_min, x_max, left, left + plot_w)
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 24:.1f}" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(x_labels[index])}</text>')
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = scale(x_value, x_min, x_max, left, left + plot_w)
        y = scale(y_value, y_min, y_max, top + plot_h, top)
        points.append(f"{x:.1f},{y:.1f}")
    body.append(f'<polyline fill="none" stroke="{ACCENT}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{" ".join(points)}"/>')
    return svg_document(width, height, title, "".join(body))


def draw_vertical_bars(title: str, labels: list[str], values: list[float], y_label: str) -> str:
    width = 1200
    height = 520
    left = 90
    right = 30
    top = 65
    bottom = 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(values) * 1.15 if values else 1.0
    bar_w = plot_w / max(len(values), 1)
    body = [
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="18" fill="{PANEL}"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" fill="{TEXT}" font-size="24" font-family="Segoe UI, Arial, sans-serif">{html.escape(title)}</text>',
        f'<text x="24" y="{height / 2:.1f}" text-anchor="middle" fill="{MUTED}" font-size="14" font-family="Segoe UI, Arial, sans-serif" transform="rotate(-90 24 {height / 2:.1f})">{html.escape(y_label)}</text>',
    ]
    for tick in range(6):
        y_value = y_max * tick / 5.0
        y = scale(y_value, 0.0, y_max, top + plot_h, top)
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(number_label(y_value))}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * bar_w
        bar_h = scale(value, 0.0, y_max, 0.0, plot_h)
        y = top + plot_h - bar_h
        body.append(f'<rect x="{x + bar_w * 0.12:.1f}" y="{y:.1f}" width="{bar_w * 0.76:.1f}" height="{bar_h:.1f}" rx="8" fill="{ACCENT}"/>')
        body.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 22:.1f}" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="Segoe UI, Arial, sans-serif">{html.escape(label)}</text>')
    return svg_document(width, height, title, "".join(body))


def draw_horizontal_bars(title: str, labels: list[str], values: list[float], x_label: str, diverging: bool = False) -> str:
    width = 1250
    height = 680
    left = 300
    right = 40
    top = 65
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    row_h = plot_h / max(len(labels), 1)
    if diverging:
        extent = max(max(abs(value) for value in values), 0.05)
        x_min = -extent
        x_max = extent
    else:
        x_min = 0.0
        x_max = (max(values) * 1.15) if values else 1.0
    body = [
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="18" fill="{PANEL}"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" fill="{TEXT}" font-size="24" font-family="Segoe UI, Arial, sans-serif">{html.escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 16:.1f}" text-anchor="middle" fill="{MUTED}" font-size="14" font-family="Segoe UI, Arial, sans-serif">{html.escape(x_label)}</text>',
    ]
    for tick_value in np.linspace(x_min, x_max, 5):
        x = scale(float(tick_value), x_min, x_max, left, left + plot_w)
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 24:.1f}" text-anchor="middle" fill="{MUTED}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(number_label(float(tick_value)))}</text>')
    zero_x = scale(0.0, x_min, x_max, left, left + plot_w)
    body.append(f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="{TEXT}" stroke-width="1.5"/>')
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_h + row_h * 0.12
        x0 = scale(min(0.0, value), x_min, x_max, left, left + plot_w)
        x1 = scale(max(0.0, value), x_min, x_max, left, left + plot_w)
        fill = POSITIVE if diverging and value >= 0 else NEGATIVE if diverging else ACCENT
        body.append(f'<rect x="{min(x0, x1):.1f}" y="{y:.1f}" width="{abs(x1 - x0):.1f}" height="{row_h * 0.72:.1f}" rx="8" fill="{fill}"/>')
        body.append(f'<text x="{left - 12}" y="{y + row_h * 0.48:.1f}" text-anchor="end" fill="{TEXT}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(label)}</text>')
        body.append(f'<text x="{max(x0, x1) + 8:.1f}" y="{y + row_h * 0.48:.1f}" text-anchor="start" fill="{MUTED}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(number_label(value))}</text>')
    return svg_document(width, height, title, "".join(body))


def draw_heatmap(title: str, matrix: np.ndarray, x_labels: list[str], y_labels: list[str], diverging: bool) -> str:
    width = 1320
    height = 900 if len(y_labels) > 8 else 620
    left = 210
    right = 40
    top = 70
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / max(len(x_labels), 1)
    cell_h = plot_h / max(len(y_labels), 1)
    value_min = float(np.nanmin(matrix))
    value_max = float(np.nanmax(matrix))
    body = [
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="18" fill="{PANEL}"/>',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" fill="{TEXT}" font-size="24" font-family="Segoe UI, Arial, sans-serif">{html.escape(title)}</text>',
    ]
    for row_index, row_label in enumerate(y_labels):
        y = top + row_index * cell_h
        body.append(f'<text x="{left - 12}" y="{y + cell_h * 0.65:.1f}" text-anchor="end" fill="{TEXT}" font-size="12" font-family="Segoe UI, Arial, sans-serif">{html.escape(row_label)}</text>')
        for col_index, _ in enumerate(x_labels):
            x = left + col_index * cell_w
            value = float(matrix[row_index, col_index])
            fill = heatmap_color(value, value_min, value_max, diverging)
            text_color = "#ffffff" if abs(value) > 0.45 or (not diverging and value > (value_min + value_max) / 2.0) else TEXT
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 2:.1f}" height="{cell_h - 2:.1f}" rx="8" fill="{fill}"/>')
            body.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h * 0.60:.1f}" text-anchor="middle" fill="{text_color}" font-size="11" font-family="Segoe UI, Arial, sans-serif">{value:.2f}</text>')
    for col_index, col_label in enumerate(x_labels):
        x = left + col_index * cell_w + cell_w / 2.0
        body.append(f'<text x="{x:.1f}" y="{top + plot_h + 16:.1f}" text-anchor="end" fill="{TEXT}" font-size="12" font-family="Segoe UI, Arial, sans-serif" transform="rotate(-40 {x:.1f} {top + plot_h + 16:.1f})">{html.escape(col_label)}</text>')
    return svg_document(width, height, title, "".join(body))


def draw_scatter_panels(records: list[dict[str, float | str]], correlations: list[dict[str, float | str]], sample_size: int) -> str:
    title = "Top Feature Relationships To The Label"
    width = 1280
    height = 980
    top_features = [str(item["feature"]) for item in correlations[:4]]
    sample = records if len(records) <= sample_size else random.Random(42).sample(records, sample_size)
    body = [f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" fill="{TEXT}" font-size="24" font-family="Segoe UI, Arial, sans-serif">{title}</text>']
    panel_w = 590
    panel_h = 410
    for index, feature_name in enumerate(top_features):
        col = index % 2
        row = index // 2
        ox = 30 + col * 620
        oy = 70 + row * 440
        cx = ox + 72
        cy = oy + 26
        cw = panel_w - 96
        ch = panel_h - 82
        x_values = np.array([float(record[feature_name]) for record in sample], dtype=float)
        y_values = np.array([float(record["label_power_kw"]) for record in sample], dtype=float)
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[mask]
        y_values = y_values[mask]
        if len(x_values) == 0:
            continue
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        body.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" rx="18" fill="{PANEL}"/>')
        wrapped = textwrap.wrap(feature_name.replace("_", " "), width=26) or [feature_name]
        for line_index, line in enumerate(wrapped[:2]):
            body.append(f'<text x="{ox + panel_w / 2:.1f}" y="{oy + 20 + line_index * 14:.1f}" text-anchor="middle" fill="{TEXT}" font-size="13" font-family="Segoe UI, Arial, sans-serif">{html.escape(line)}</text>')
        for x_value, y_value in zip(x_values, y_values):
            px = scale(float(x_value), x_min, x_max, cx, cx + cw)
            py = scale(float(y_value), y_min, y_max, cy + ch, cy)
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="{ACCENT}" fill-opacity="0.35"/>')
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x0 = scale(x_min, x_min, x_max, cx, cx + cw)
        y0 = scale(slope * x_min + intercept, y_min, y_max, cy + ch, cy)
        x1 = scale(x_max, x_min, x_max, cx, cx + cw)
        y1 = scale(slope * x_max + intercept, y_min, y_max, cy + ch, cy)
        body.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{NEGATIVE}" stroke-width="2.4"/>')
        body.append(f'<text x="{ox + 18:.1f}" y="{cy + ch / 2:.1f}" text-anchor="middle" fill="{MUTED}" font-size="11" font-family="Segoe UI, Arial, sans-serif" transform="rotate(-90 {ox + 18:.1f} {cy + ch / 2:.1f})">label_power_kw</text>')
    return svg_document(width, height, title, "".join(body))


def build_summary(
    summary: RawDatasetSummary,
    target_timestamps: list[datetime],
    correlations: list[dict[str, float | str]],
    importances: list[dict[str, float | str]],
    ridge_test_mae: float,
) -> dict[str, object]:
    return {
        "challenge_understanding": {
            "task": "Forecast combined hourly reefer power and provide both a point estimate and an upper p90 estimate.",
            "evaluation_weights": {"mae_all": 0.5, "mae_peak": 0.3, "pinball_p90": 0.2},
        },
        "raw_dataset_summary": {
            "row_count": summary.row_count,
            "hourly_count": summary.hourly_count,
            "timestamp_min_utc": format_timestamp(summary.timestamp_min),
            "timestamp_max_utc": format_timestamp(summary.timestamp_max),
            "hardware_type_cardinality": len(summary.hardware_type_counts),
            "container_size_cardinality": len(summary.container_size_counts),
            "stack_tier_cardinality": len(summary.stack_tier_counts),
            "top_hardware_types": summary.hardware_type_counts.most_common(8),
            "top_container_sizes": summary.container_size_counts.most_common(8),
            "top_stack_tiers": summary.stack_tier_counts.most_common(8),
            "field_missing_counts": summary.field_missing_counts,
            "raw_format_notes": {
                "delimiter": "semicolon",
                "decimal_separator": "comma",
                "raw_column_examples": ["AvPowerCons", "RemperatureSupply"],
                "label_definition": "sum(AvPowerCons) / 1000 by EventTime hour",
            },
        },
        "target_summary": {
            "row_count": len(target_timestamps),
            "timestamp_min_utc": format_timestamp(min(target_timestamps) if target_timestamps else None),
            "timestamp_max_utc": format_timestamp(max(target_timestamps) if target_timestamps else None),
        },
        "top_feature_correlations": correlations[:12],
        "top_feature_importances": importances[:12],
        "ridge_test_mae_kw": ridge_test_mae,
    }


def build_markdown_report(summary_payload: dict[str, object], top_features: int) -> str:
    raw = summary_payload["raw_dataset_summary"]
    target = summary_payload["target_summary"]
    correlations = summary_payload["top_feature_correlations"]
    importances = summary_payload["top_feature_importances"]
    lines = [
        "# Reefer Dataset Analysis",
        "",
        "## Challenge Understanding",
        "- Forecast the combined hourly reefer electricity demand for future timestamps.",
        "- The score rewards overall accuracy, peak-hour accuracy, and a sensible p90 upper estimate.",
        "- This EDA uses `sum(AvPowerCons) / 1000` per hour as the label because that matches the challenge target conceptually.",
        "",
        "## Dataset Overview",
        f"- Raw rows: {raw['row_count']:,}",
        f"- Aggregated hours: {raw['hourly_count']:,}",
        f"- Reefer range: {raw['timestamp_min_utc']} to {raw['timestamp_max_utc']}",
        f"- Public target timestamps: {target['row_count']:,}",
        f"- Public target range: {target['timestamp_min_utc']} to {target['timestamp_max_utc']}",
        "",
        "## Raw File Notes",
        "- The reefer file uses semicolons and decimal commas.",
        "- The raw schema differs slightly from the docs, for example `AvPowerCons` and `RemperatureSupply`.",
        "- The generated `hourly_features.csv` is the main modeling-ready table for the next steps.",
        "",
        f"## Top {top_features} Correlations To The Label",
    ]
    for row in correlations[:top_features]:
        lines.append(f"- `{row['feature']}`: {float(row['pearson_correlation']):.3f}")
    lines.extend(["", f"## Top {top_features} Simple Feature Importances"])
    for row in importances[:top_features]:
        lines.append(f"- `{row['feature']}`: MAE increase {float(row['mae_increase_kw']):.3f} kW")
    lines.extend(
        [
            "",
            "## Generated Files",
            "- `report.html`",
            "- `dataset_summary.json`",
            "- `hourly_features.csv`",
            "- `feature_label_correlations.csv`",
            "- `feature_importance.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_html_report(summary_payload: dict[str, object], plot_files: list[tuple[str, str]]) -> str:
    raw = summary_payload["raw_dataset_summary"]
    target = summary_payload["target_summary"]
    correlations = summary_payload["top_feature_correlations"]
    importances = summary_payload["top_feature_importances"]
    correlation_rows = "\n".join(
        f"<tr><td>{html.escape(str(row['feature']))}</td><td>{float(row['pearson_correlation']):.3f}</td></tr>"
        for row in correlations
    )
    importance_rows = "\n".join(
        f"<tr><td>{html.escape(str(row['feature']))}</td><td>{float(row['mae_increase_kw']):.3f}</td></tr>"
        for row in importances
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
  <title>Reefer Dataset Analysis</title>
  <style>
    body {{ margin: 0; background: linear-gradient(180deg, #fffdf9 0%, #f6efe3 100%); color: {TEXT}; font-family: "Segoe UI", Arial, sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px 60px; }}
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
    <h1>Reefer Dataset Analysis</h1>
    <p>This report converts the participant reefer release into an hourly target-aligned analysis table and highlights the strongest first-pass feature relationships.</p>
    <div class="grid">
      <section class="card"><h2>Raw Data</h2><p><strong>Rows:</strong> {raw['row_count']:,}</p><p><strong>Hours:</strong> {raw['hourly_count']:,}</p><p><strong>Range:</strong> {raw['timestamp_min_utc']} to {raw['timestamp_max_utc']}</p></section>
      <section class="card"><h2>Targets</h2><p><strong>Timestamps:</strong> {target['row_count']:,}</p><p><strong>Range:</strong> {target['timestamp_min_utc']} to {target['timestamp_max_utc']}</p></section>
      <section class="card"><h2>Schema Notes</h2><p>The raw file uses semicolons, decimal commas, and field names that differ slightly from the challenge markdown.</p></section>
    </div>
    <section class="tables">
      <div class="card"><h2>Top Correlations</h2><table><thead><tr><th>Feature</th><th>Pearson r</th></tr></thead><tbody>{correlation_rows}</tbody></table></div>
      <div class="card"><h2>Top Importances</h2><table><thead><tr><th>Feature</th><th>MAE Increase (kW)</th></tr></thead><tbody>{importance_rows}</tbody></table></div>
    </section>
    {plot_cards}
  </main>
</body>
</html>"""


def write_outputs(
    output_dir: Path,
    records: list[dict[str, float | str]],
    summary: RawDatasetSummary,
    target_timestamps: list[datetime],
    correlations: list[dict[str, float | str]],
    importances: list[dict[str, float | str]],
    heatmap_labels: list[str],
    heatmap_values: np.ndarray,
    label_values: list[float],
    hourly_profile: list[float],
    day_hour: np.ndarray,
    scatter_sample_size: int,
    ridge_test_mae: float,
    top_features: int,
) -> None:
    write_csv(output_dir / "hourly_features.csv", records)
    write_csv(output_dir / "feature_label_correlations.csv", correlations)
    write_csv(output_dir / "feature_importance.csv", importances)

    summary_payload = build_summary(summary, target_timestamps, correlations, importances, ridge_test_mae)
    write_json(output_dir / "dataset_summary.json", summary_payload)

    plots: list[tuple[str, str]] = []
    timestamps = [str(record["timestamp_utc"])[0:10] for record in records]
    plot_specs = [
        ("label_timeseries.svg", "Hourly Aggregate Reefer Power", draw_line_chart("Hourly Aggregate Reefer Power", list(range(len(records))), label_values, timestamps, "Power (kW)")),
        ("label_distribution.svg", "Distribution Of Hourly Reefer Load", draw_vertical_bars("Distribution Of Hourly Reefer Load", [number_label(v) for v in ((np.histogram(np.array(label_values), bins=24)[1][:-1] + np.histogram(np.array(label_values), bins=24)[1][1:]) / 2.0)], np.histogram(np.array(label_values), bins=24)[0].astype(float).tolist(), "Hours")),
        ("hour_of_day_profile.svg", "Average Load By Hour Of Day", draw_vertical_bars("Average Load By Hour Of Day", [f"{hour:02d}" for hour in range(24)], hourly_profile, "Average power (kW)")),
        ("day_hour_heatmap.svg", "Day-Hour Heatmap Of Average Load", draw_heatmap("Day-Hour Heatmap Of Average Load", day_hour, [f"{hour:02d}" for hour in range(24)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], False)),
        ("feature_label_correlations.svg", "Top Feature Correlations To The Label", draw_horizontal_bars("Top Feature Correlations To The Label", [str(row["feature"]) for row in correlations[:top_features]], [float(row["pearson_correlation"]) for row in correlations[:top_features]], "Pearson correlation", True)),
        ("feature_importance.svg", "Permutation Importance On Holdout", draw_horizontal_bars("Permutation Importance On Holdout", [str(row["feature"]) for row in importances[:top_features]], [float(row["mae_increase_kw"]) for row in importances[:top_features]], "MAE increase after shuffling feature (kW)")),
        ("feature_correlation_heatmap.svg", "Correlation Matrix For Top Features", draw_heatmap("Correlation Matrix For Top Features", heatmap_values, heatmap_labels, heatmap_labels, True)),
        ("top_feature_relationships.svg", "Top Feature Relationships To The Label", draw_scatter_panels(records, correlations, scatter_sample_size)),
    ]
    for filename, title, payload in plot_specs:
        save_svg(output_dir / filename, payload)
        plots.append((filename, title))

    (output_dir / "report.md").write_text(build_markdown_report(summary_payload, top_features), encoding="utf-8")
    (output_dir / "report.html").write_text(build_html_report(summary_payload, plots), encoding="utf-8")


def analyze_dataset(args: argparse.Namespace) -> Path:
    reefer_zip = args.reefer_zip.resolve()
    target_csv = args.target_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reefer_zip.exists():
        raise FileNotFoundError(f"Reefer zip not found: {reefer_zip}")
    if not target_csv.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_csv}")

    records, summary = collect_hourly_dataset(reefer_zip)
    target_timestamps = load_target_timestamps(target_csv)
    feature_names, matrix = records_to_matrix(records)
    correlations = feature_correlations(feature_names, matrix)
    heatmap_labels, heatmap_values = top_correlation_heatmap(feature_names, matrix, correlations, args.top_features)
    importances, ridge_test_mae = ridge_permutation_importance(feature_names, matrix, args.ridge_alpha)
    label_values, hourly_profile, day_hour = label_profiles(records)

    write_outputs(
        output_dir=output_dir,
        records=records,
        summary=summary,
        target_timestamps=target_timestamps,
        correlations=correlations,
        importances=importances,
        heatmap_labels=heatmap_labels,
        heatmap_values=heatmap_values,
        label_values=label_values,
        hourly_profile=hourly_profile,
        day_hour=day_hour,
        scatter_sample_size=args.scatter_sample_size,
        ridge_test_mae=ridge_test_mae,
        top_features=args.top_features,
    )
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = analyze_dataset(args)
    print(f"Reefer dataset analysis complete. Open {output_dir / 'report.html'} for the summary report.")
