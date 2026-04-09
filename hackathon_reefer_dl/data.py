from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hackathon_reefer_dl.common import (
    CV_FOLD_HOURS,
    FORECAST_HORIZON_HOURS,
    HISTORY_HOURS,
    REEFER_EXTRACTED_CSV,
    REEFER_ZIP,
    TOP_HARDWARE,
    TOP_SIZES,
    TOP_STACK_TIERS,
    WEATHER_ZIP,
    iso_utc,
    iter_hour_range,
    parse_decimal,
    parse_hour_timestamp,
    parse_target_timestamp,
    target_cutoff_hour,
    to_hour,
    window_bounds_for_target,
)
from hackathon_reefer_dl.io_utils import read_json, read_parquet, write_json, write_parquet

SUPPLY_COLUMN = "RemperatureSupply"


@dataclass
class HourlyAccumulator:
    active_visits: int = 0
    load_kw: float = 0.0
    power_sum: float = 0.0
    power_sumsq: float = 0.0
    power_count: int = 0
    temp_ambient_sum: float = 0.0
    temp_ambient_sumsq: float = 0.0
    temp_ambient_count: int = 0
    temp_setpoint_sum: float = 0.0
    temp_setpoint_sumsq: float = 0.0
    temp_setpoint_count: int = 0
    temp_return_sum: float = 0.0
    temp_return_sumsq: float = 0.0
    temp_return_count: int = 0
    temp_supply_sum: float = 0.0
    temp_supply_sumsq: float = 0.0
    temp_supply_count: int = 0
    hardware_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in TOP_HARDWARE})
    stack_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in TOP_STACK_TIERS})
    size_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in TOP_SIZES})

    def add_power(self, value_kw: float) -> None:
        self.load_kw += value_kw
        self.power_sum += value_kw
        self.power_sumsq += value_kw * value_kw
        self.power_count += 1

    def add_temperature(self, prefix: str, value: float) -> None:
        sum_name = f"{prefix}_sum"
        sumsq_name = f"{prefix}_sumsq"
        count_name = f"{prefix}_count"
        setattr(self, sum_name, getattr(self, sum_name) + value)
        setattr(self, sumsq_name, getattr(self, sumsq_name) + value * value)
        setattr(self, count_name, getattr(self, count_name) + 1)


@dataclass
class HourlyFeatureTable:
    timestamps: list[datetime]
    values: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.hour_to_idx = {timestamp: idx for idx, timestamp in enumerate(self.timestamps)}

    @property
    def target_col(self) -> str:
        return str(self.metadata["target_col"])

    @property
    def target(self) -> np.ndarray:
        return self.values[self.target_col]

    def feature_matrix(self, feature_names: list[str]) -> np.ndarray:
        return np.stack([self.values[name] for name in feature_names], axis=1).astype(np.float32)


def _mean(sum_value: float, count: int) -> float:
    return sum_value / count if count else 0.0


def _std(sum_value: float, sumsq_value: float, count: int) -> float:
    if count <= 1:
        return 0.0
    mean_value = sum_value / count
    variance = max((sumsq_value / count) - (mean_value * mean_value), 0.0)
    return variance ** 0.5


def _lag(values: np.ndarray, lag_hours: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[:lag_hours] = values[0]
    out[lag_hours:] = values[:-lag_hours]
    return out


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumsum = np.cumsum(values, dtype=np.float64)
    out = cumsum.copy()
    out[window:] = cumsum[window:] - cumsum[:-window]
    return out


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    cumsum = np.cumsum(values, dtype=np.float64)
    cumsum_sq = np.cumsum(values * values, dtype=np.float64)
    counts = np.minimum(np.arange(1, values.shape[0] + 1, dtype=np.float64), float(window))
    sums = cumsum.copy()
    sums_sq = cumsum_sq.copy()
    sums[window:] = cumsum[window:] - cumsum[:-window]
    sums_sq[window:] = cumsum_sq[window:] - cumsum_sq[:-window]
    mean = sums / counts
    variance = np.maximum((sums_sq / counts) - (mean * mean), 0.0)
    return mean, np.sqrt(variance)


def _forward_fill(values: np.ndarray, default: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    last = default
    for idx in range(out.shape[0]):
        if np.isnan(out[idx]):
            out[idx] = last
        else:
            last = out[idx]
    return out


def _calendar_features(hours: list[datetime]) -> dict[str, np.ndarray]:
    hour_of_day = np.asarray([hour.hour for hour in hours], dtype=np.float64)
    day_of_week = np.asarray([hour.weekday() for hour in hours], dtype=np.float64)
    day_of_year = np.asarray([hour.timetuple().tm_yday for hour in hours], dtype=np.float64)
    return {
        "hour_sin": np.sin(2.0 * np.pi * hour_of_day / 24.0).astype(np.float32),
        "hour_cos": np.cos(2.0 * np.pi * hour_of_day / 24.0).astype(np.float32),
        "dow_sin": np.sin(2.0 * np.pi * day_of_week / 7.0).astype(np.float32),
        "dow_cos": np.cos(2.0 * np.pi * day_of_week / 7.0).astype(np.float32),
        "doy_sin": np.sin(2.0 * np.pi * day_of_year / 366.0).astype(np.float32),
        "doy_cos": np.cos(2.0 * np.pi * day_of_year / 366.0).astype(np.float32),
        "is_weekend": (day_of_week >= 5.0).astype(np.float32),
    }


def target_calendar_matrix(target_times: list[datetime]) -> np.ndarray:
    features = _calendar_features(target_times)
    names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
    return np.stack([features[name] for name in names], axis=1).astype(np.float32)


@contextmanager
def open_reefer_rows(participant_dir: Path) -> Iterator[csv.DictReader]:
    extracted = participant_dir / REEFER_EXTRACTED_CSV
    if extracted.exists():
        with extracted.open("r", encoding="utf-8-sig", newline="") as handle:
            yield csv.DictReader(handle, delimiter=";")
        return

    archive_path = participant_dir / REEFER_ZIP
    if not archive_path.exists():
        raise FileNotFoundError(f"Could not find reefer data in {participant_dir}")
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("reefer_release.csv") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                yield csv.DictReader(text, delimiter=";")


def scan_reefer_rows(reader: csv.DictReader) -> tuple[dict[datetime, HourlyAccumulator], dict[str, tuple[datetime, datetime]]]:
    hourly: dict[datetime, HourlyAccumulator] = {}
    visit_bounds: dict[str, tuple[datetime, datetime]] = {}
    for row in reader:
        hour = to_hour(parse_hour_timestamp(row["EventTime"]))
        accumulator = hourly.setdefault(hour, HourlyAccumulator())
        accumulator.active_visits += 1

        power = parse_decimal(row["AvPowerCons"])
        if power is not None:
            accumulator.add_power(power / 1000.0)

        temp_ambient = parse_decimal(row.get("TemperatureAmbient"))
        if temp_ambient is not None:
            accumulator.add_temperature("temp_ambient", temp_ambient)

        temp_setpoint = parse_decimal(row.get("TemperatureSetPoint"))
        if temp_setpoint is not None:
            accumulator.add_temperature("temp_setpoint", temp_setpoint)

        temp_return = parse_decimal(row.get("TemperatureReturn"))
        if temp_return is not None:
            accumulator.add_temperature("temp_return", temp_return)

        temp_supply = parse_decimal(row.get(SUPPLY_COLUMN) or row.get("TemperatureSupply"))
        if temp_supply is not None:
            accumulator.add_temperature("temp_supply", temp_supply)

        hardware = (row.get("HardwareType") or "").strip()
        if hardware in accumulator.hardware_counts:
            accumulator.hardware_counts[hardware] += 1

        stack_tier = (row.get("stack_tier") or "").strip()
        if stack_tier in accumulator.stack_counts:
            accumulator.stack_counts[stack_tier] += 1

        size = (row.get("ContainerSize") or "").strip()
        if size in accumulator.size_counts:
            accumulator.size_counts[size] += 1

        visit_id = row["container_visit_uuid"]
        if visit_id in visit_bounds:
            start, end = visit_bounds[visit_id]
            visit_bounds[visit_id] = (min(start, hour), max(end, hour))
        else:
            visit_bounds[visit_id] = (hour, hour)
    return hourly, visit_bounds


def _align_hourly_accumulators(hourly: dict[datetime, HourlyAccumulator]) -> tuple[list[datetime], dict[str, np.ndarray]]:
    min_hour = min(hourly)
    max_hour = max(hourly)
    hours = list(iter_hour_range(min_hour, max_hour))
    hour_to_idx = {hour: idx for idx, hour in enumerate(hours)}
    size = len(hours)

    load_kw = np.zeros(size, dtype=np.float32)
    active_visits = np.zeros(size, dtype=np.float32)
    power_per_visit_mean_kw = np.zeros(size, dtype=np.float32)
    power_per_visit_std_kw = np.zeros(size, dtype=np.float32)
    temp_ambient_mean = np.zeros(size, dtype=np.float32)
    temp_ambient_std = np.zeros(size, dtype=np.float32)
    temp_setpoint_mean = np.zeros(size, dtype=np.float32)
    temp_setpoint_std = np.zeros(size, dtype=np.float32)
    temp_return_mean = np.zeros(size, dtype=np.float32)
    temp_return_std = np.zeros(size, dtype=np.float32)
    temp_supply_mean = np.zeros(size, dtype=np.float32)
    temp_supply_std = np.zeros(size, dtype=np.float32)
    hardware_share = {key: np.zeros(size, dtype=np.float32) for key in TOP_HARDWARE}
    stack_share = {key: np.zeros(size, dtype=np.float32) for key in TOP_STACK_TIERS}
    size_share = {key: np.zeros(size, dtype=np.float32) for key in TOP_SIZES}

    for hour, accumulator in hourly.items():
        idx = hour_to_idx[hour]
        active = float(accumulator.active_visits)
        active_visits[idx] = active
        load_kw[idx] = accumulator.load_kw
        power_per_visit_mean_kw[idx] = _mean(accumulator.power_sum, accumulator.power_count)
        power_per_visit_std_kw[idx] = _std(accumulator.power_sum, accumulator.power_sumsq, accumulator.power_count)
        temp_ambient_mean[idx] = _mean(accumulator.temp_ambient_sum, accumulator.temp_ambient_count)
        temp_ambient_std[idx] = _std(
            accumulator.temp_ambient_sum,
            accumulator.temp_ambient_sumsq,
            accumulator.temp_ambient_count,
        )
        temp_setpoint_mean[idx] = _mean(accumulator.temp_setpoint_sum, accumulator.temp_setpoint_count)
        temp_setpoint_std[idx] = _std(
            accumulator.temp_setpoint_sum,
            accumulator.temp_setpoint_sumsq,
            accumulator.temp_setpoint_count,
        )
        temp_return_mean[idx] = _mean(accumulator.temp_return_sum, accumulator.temp_return_count)
        temp_return_std[idx] = _std(
            accumulator.temp_return_sum,
            accumulator.temp_return_sumsq,
            accumulator.temp_return_count,
        )
        temp_supply_mean[idx] = _mean(accumulator.temp_supply_sum, accumulator.temp_supply_count)
        temp_supply_std[idx] = _std(
            accumulator.temp_supply_sum,
            accumulator.temp_supply_sumsq,
            accumulator.temp_supply_count,
        )
        if active > 0:
            for key in TOP_HARDWARE:
                hardware_share[key][idx] = accumulator.hardware_counts[key] / active
            for key in TOP_STACK_TIERS:
                stack_share[key][idx] = accumulator.stack_counts[key] / active
            for key in TOP_SIZES:
                size_share[key][idx] = accumulator.size_counts[key] / active

    payload = {
        "load_kw": load_kw,
        "active_visits": active_visits,
        "power_per_visit_mean_kw": power_per_visit_mean_kw,
        "power_per_visit_std_kw": power_per_visit_std_kw,
        "temp_ambient_mean": temp_ambient_mean,
        "temp_ambient_std": temp_ambient_std,
        "temp_setpoint_mean": temp_setpoint_mean,
        "temp_setpoint_std": temp_setpoint_std,
        "temp_return_mean": temp_return_mean,
        "temp_return_std": temp_return_std,
        "temp_supply_mean": temp_supply_mean,
        "temp_supply_std": temp_supply_std,
    }
    for key, values in hardware_share.items():
        payload[f"hardware_share_{key}"] = values
    for key, values in stack_share.items():
        payload[f"stack_share_{key}"] = values
    for key, values in size_share.items():
        payload[f"size_share_{key}"] = values
    return hours, payload


def _visit_turnover_and_age_features(
    visit_bounds: dict[str, tuple[datetime, datetime]],
    hours: list[datetime],
) -> dict[str, np.ndarray]:
    hour_to_idx = {hour: idx for idx, hour in enumerate(hours)}
    size = len(hours)
    arrivals = np.zeros(size, dtype=np.float64)
    departures = np.zeros(size, dtype=np.float64)
    age_lists: list[list[float]] = [[] for _ in range(size)]

    for start, end in visit_bounds.values():
        start_idx = hour_to_idx[start]
        end_idx = hour_to_idx[end]
        arrivals[start_idx] += 1.0
        departures[end_idx] += 1.0
        for offset, idx in enumerate(range(start_idx, end_idx + 1)):
            age_lists[idx].append(float(offset))

    age_mean = np.zeros(size, dtype=np.float32)
    age_median = np.zeros(size, dtype=np.float32)
    age_p90 = np.zeros(size, dtype=np.float32)
    age_share_lt24 = np.zeros(size, dtype=np.float32)
    age_share_gt72 = np.zeros(size, dtype=np.float32)
    age_share_gt168 = np.zeros(size, dtype=np.float32)
    for idx, ages in enumerate(age_lists):
        if not ages:
            continue
        age_arr = np.asarray(ages, dtype=np.float64)
        age_mean[idx] = float(age_arr.mean())
        age_median[idx] = float(np.median(age_arr))
        age_p90[idx] = float(np.quantile(age_arr, 0.9))
        age_share_lt24[idx] = float(np.mean(age_arr < 24.0))
        age_share_gt72[idx] = float(np.mean(age_arr > 72.0))
        age_share_gt168[idx] = float(np.mean(age_arr > 168.0))

    return {
        "arrivals_last_1h": _rolling_sum(arrivals, 1).astype(np.float32),
        "arrivals_last_6h": _rolling_sum(arrivals, 6).astype(np.float32),
        "arrivals_last_24h": _rolling_sum(arrivals, 24).astype(np.float32),
        "departures_last_1h": _rolling_sum(departures, 1).astype(np.float32),
        "departures_last_6h": _rolling_sum(departures, 6).astype(np.float32),
        "departures_last_24h": _rolling_sum(departures, 24).astype(np.float32),
        "active_age_mean_h": age_mean,
        "active_age_median_h": age_median,
        "active_age_p90_h": age_p90,
        "active_age_share_lt24h": age_share_lt24,
        "active_age_share_gt72h": age_share_gt72,
        "active_age_share_gt168h": age_share_gt168,
    }


def _load_temperature_weather(participant_dir: Path, hours: list[datetime]) -> dict[str, np.ndarray]:
    weather_zip = participant_dir / WEATHER_ZIP
    if not weather_zip.exists():
        raise FileNotFoundError(f"Could not find weather zip in {participant_dir}")

    hour_to_idx = {hour: idx for idx, hour in enumerate(hours)}
    size = len(hours)
    members = {
        "ext_temp_halle3": None,
        "ext_temp_zentralgate": None,
    }

    with zipfile.ZipFile(weather_zip) as archive:
        for member in archive.namelist():
            if not member.endswith(".csv") or "Temperatur" not in member:
                continue
            if "Halle3" in member:
                members["ext_temp_halle3"] = member
            elif "Zentralgate" in member:
                members["ext_temp_zentralgate"] = member

        weather_features: dict[str, np.ndarray] = {}
        for feature_name, member in members.items():
            if member is None:
                raise FileNotFoundError(f"Missing expected weather file for {feature_name}")
            sums: dict[datetime, float] = defaultdict(float)
            counts: dict[datetime, int] = defaultdict(int)
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                    reader = csv.DictReader(text, delimiter=";")
                    for row in reader:
                        value = parse_decimal(row.get("Value"))
                        if value is None:
                            continue
                        hour = to_hour(parse_hour_timestamp(row["UtcTimestamp"]))
                        if hour in hour_to_idx:
                            sums[hour] += value
                            counts[hour] += 1

            raw_values = np.full(size, np.nan, dtype=np.float64)
            availability = np.zeros(size, dtype=np.float32)
            for hour, total in sums.items():
                idx = hour_to_idx[hour]
                raw_values[idx] = total / counts[hour]
                availability[idx] = 1.0
            observed = raw_values[np.isfinite(raw_values)]
            if observed.size:
                clip_low = max(-30.0, float(np.quantile(observed, 0.005)))
                clip_high = min(45.0, float(np.quantile(observed, 0.995)))
                raw_values = np.where(
                    np.isfinite(raw_values),
                    np.clip(raw_values, clip_low, clip_high),
                    raw_values,
                )
            default = float(np.nanmean(raw_values)) if np.isfinite(np.nanmean(raw_values)) else 0.0
            weather_features[feature_name] = _forward_fill(raw_values, default=default).astype(np.float32)
            weather_features[f"{feature_name}_available"] = availability

    avg = (weather_features["ext_temp_halle3"] + weather_features["ext_temp_zentralgate"]) / 2.0
    roll_mean_24, roll_std_24 = _rolling_mean_std(avg, 24)
    weather_features["ext_temp_avg"] = avg.astype(np.float32)
    weather_features["ext_temp_diff"] = (
        weather_features["ext_temp_halle3"] - weather_features["ext_temp_zentralgate"]
    ).astype(np.float32)
    weather_features["ext_temp_roll_mean_24"] = roll_mean_24.astype(np.float32)
    weather_features["ext_temp_roll_std_24"] = roll_std_24.astype(np.float32)
    weather_features["ext_temp_any_available"] = np.maximum(
        weather_features["ext_temp_halle3_available"],
        weather_features["ext_temp_zentralgate_available"],
    ).astype(np.float32)
    return weather_features


def build_hourly_feature_table(participant_dir: Path) -> HourlyFeatureTable:
    participant_dir = participant_dir.resolve()
    with open_reefer_rows(participant_dir) as reader:
        hourly, visit_bounds = scan_reefer_rows(reader)

    hours, base = _align_hourly_accumulators(hourly)
    turnover = _visit_turnover_and_age_features(visit_bounds, hours)
    weather = _load_temperature_weather(participant_dir, hours)
    calendar = _calendar_features(hours)

    load_values = np.asarray(base["load_kw"], dtype=np.float64)
    load_roll_mean_24, load_roll_std_24 = _rolling_mean_std(load_values, 24)
    load_roll_mean_168, load_roll_std_168 = _rolling_mean_std(load_values, 168)
    derived = {
        "load_lag_24": _lag(load_values, 24).astype(np.float32),
        "load_lag_168": _lag(load_values, 168).astype(np.float32),
        "load_roll_mean_24": load_roll_mean_24.astype(np.float32),
        "load_roll_std_24": load_roll_std_24.astype(np.float32),
        "load_roll_mean_168": load_roll_mean_168.astype(np.float32),
        "load_roll_std_168": load_roll_std_168.astype(np.float32),
        "load_delta_1": (load_values - _lag(load_values, 1)).astype(np.float32),
        "load_delta_24": (load_values - _lag(load_values, 24)).astype(np.float32),
    }

    values: dict[str, np.ndarray] = {}
    values.update(base)
    values.update(derived)
    values.update(turnover)
    values.update(weather)
    values.update(calendar)

    timestamp_strings = np.asarray([iso_utc(hour) for hour in hours], dtype=object)
    feature_groups = {
        "load_calendar": [
            "load_kw",
            "load_lag_24",
            "load_lag_168",
            "load_roll_mean_24",
            "load_roll_std_24",
            "load_roll_mean_168",
            "load_roll_std_168",
            "load_delta_1",
            "load_delta_24",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
        ],
        "reefer_state": [
            "active_visits",
            "power_per_visit_mean_kw",
            "power_per_visit_std_kw",
            "temp_ambient_mean",
            "temp_ambient_std",
            "temp_setpoint_mean",
            "temp_setpoint_std",
            "temp_return_mean",
            "temp_return_std",
            "temp_supply_mean",
            "temp_supply_std",
            *[f"hardware_share_{key}" for key in TOP_HARDWARE],
            *[f"stack_share_{key}" for key in TOP_STACK_TIERS],
            *[f"size_share_{key}" for key in TOP_SIZES],
        ],
        "turnover_age": [
            "arrivals_last_1h",
            "arrivals_last_6h",
            "arrivals_last_24h",
            "departures_last_1h",
            "departures_last_6h",
            "departures_last_24h",
            "active_age_mean_h",
            "active_age_median_h",
            "active_age_p90_h",
            "active_age_share_lt24h",
            "active_age_share_gt72h",
            "active_age_share_gt168h",
        ],
        "external_temperature": [
            "ext_temp_halle3",
            "ext_temp_zentralgate",
            "ext_temp_avg",
            "ext_temp_diff",
            "ext_temp_roll_mean_24",
            "ext_temp_roll_std_24",
            "ext_temp_halle3_available",
            "ext_temp_zentralgate_available",
            "ext_temp_any_available",
        ],
    }
    metadata = {
        "participant_dir": str(participant_dir),
        "timestamp_col": "timestamp_utc",
        "target_col": "load_kw",
        "history_hours": HISTORY_HOURS,
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "cv_fold_hours": CV_FOLD_HOURS,
        "feature_groups": feature_groups,
        "target_calendar_names": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"],
    }
    values["timestamp_utc"] = timestamp_strings
    return HourlyFeatureTable(timestamps=hours, values=values, metadata=metadata)


def save_hourly_feature_table(feature_table: HourlyFeatureTable, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "hourly_features.parquet"
    metadata_path = out_dir / "feature_metadata.json"
    summary_path = out_dir / "dataset_summary.json"

    write_parquet(parquet_path, feature_table.values)
    write_json(metadata_path, feature_table.metadata)
    summary = {
        "rows": len(feature_table.timestamps),
        "time_min": iso_utc(feature_table.timestamps[0]),
        "time_max": iso_utc(feature_table.timestamps[-1]),
        "feature_group_sizes": {
            key: len(value) for key, value in dict(feature_table.metadata["feature_groups"]).items()
        },
    }
    write_json(summary_path, summary)
    return parquet_path, metadata_path, summary_path


def load_hourly_feature_table(parquet_path: Path) -> HourlyFeatureTable:
    raw = read_parquet(parquet_path)
    metadata = read_json(parquet_path.with_name("feature_metadata.json"))
    timestamps = [parse_target_timestamp(value) for value in raw.pop(metadata["timestamp_col"]).tolist()]
    values = {name: np.asarray(column, dtype=np.float32) for name, column in raw.items()}
    return HourlyFeatureTable(timestamps=timestamps, values=values, metadata=metadata)


def load_target_timestamps(path: Path) -> list[datetime]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_target_timestamp(row["timestamp_utc"]) for row in reader]


def observed_target_times(feature_table: HourlyFeatureTable, max_target_time: datetime) -> list[datetime]:
    min_target_index = feature_table.metadata["history_hours"] + feature_table.metadata["forecast_horizon_hours"] - 1
    return [
        timestamp
        for timestamp in feature_table.timestamps[min_target_index:]
        if timestamp <= max_target_time
    ]


def make_cv_splits(
    target_times: list[datetime],
    num_folds: int = 4,
    fold_hours: int = CV_FOLD_HOURS,
) -> list[tuple[list[datetime], list[datetime]]]:
    if len(target_times) < num_folds * fold_hours:
        raise ValueError("Not enough target hours for the requested rolling CV setup.")
    validation_pool = target_times[-(num_folds * fold_hours) :]
    splits: list[tuple[list[datetime], list[datetime]]] = []
    for fold_idx in range(num_folds):
        start = fold_idx * fold_hours
        end = start + fold_hours
        val_times = validation_pool[start:end]
        train_times = [timestamp for timestamp in target_times if timestamp < val_times[0]]
        splits.append((train_times, val_times))
    return splits


def feature_names_for_groups(metadata: dict[str, Any], group_names: list[str]) -> list[str]:
    names: list[str] = []
    for group_name in group_names:
        for feature_name in metadata["feature_groups"][group_name]:
            if feature_name not in names:
                names.append(feature_name)
    return names


def scaler_from_targets(
    feature_matrix: np.ndarray,
    target_times: list[datetime],
    hour_to_idx: dict[datetime, int],
    history_hours: int = HISTORY_HOURS,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    used_rows = np.zeros(feature_matrix.shape[0], dtype=bool)
    for target_time in target_times:
        start_idx, end_idx = window_bounds_for_target(
            target_time,
            hour_to_idx,
            history_hours=history_hours,
            horizon_hours=horizon_hours,
        )
        used_rows[start_idx : end_idx + 1] = True
    subset = feature_matrix[used_rows]
    mean = subset.mean(axis=0)
    std = subset.std(axis=0)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def build_training_arrays(
    feature_table: HourlyFeatureTable,
    feature_names: list[str],
    target_times: list[datetime],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    history_hours: int = HISTORY_HOURS,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> dict[str, Any]:
    feature_matrix = (feature_table.feature_matrix(feature_names) - scaler_mean) / scaler_std
    target_calendar = target_calendar_matrix(target_times)
    sequence = np.empty((len(target_times), history_hours, len(feature_names)), dtype=np.float32)
    labels = np.empty(len(target_times), dtype=np.float32)
    recent_volatility = np.empty(len(target_times), dtype=np.float32)
    naive_baseline = np.empty(len(target_times), dtype=np.float32)
    for idx, target_time in enumerate(target_times):
        start_idx, end_idx = window_bounds_for_target(
            target_time,
            feature_table.hour_to_idx,
            history_hours=history_hours,
            horizon_hours=horizon_hours,
        )
        sequence[idx] = feature_matrix[start_idx : end_idx + 1]
        label_idx = feature_table.hour_to_idx[target_time]
        labels[idx] = feature_table.target[label_idx]
        recent_window = feature_table.target[max(0, end_idx - 23) : end_idx + 1]
        recent_volatility[idx] = float(np.std(recent_window))
        lag_24_idx = feature_table.hour_to_idx[target_time - timedelta(hours=24)]
        naive_baseline[idx] = feature_table.target[lag_24_idx]
    return {
        "sequence": sequence,
        "target_calendar": target_calendar,
        "labels": labels,
        "recent_volatility": recent_volatility,
        "naive_baseline": naive_baseline,
        "target_times": target_times,
    }


def build_prediction_arrays(
    feature_table: HourlyFeatureTable,
    feature_names: list[str],
    target_times: list[datetime],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    history_hours: int = HISTORY_HOURS,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> dict[str, Any]:
    feature_matrix = (feature_table.feature_matrix(feature_names) - scaler_mean) / scaler_std
    target_calendar = target_calendar_matrix(target_times)
    sequence = np.empty((len(target_times), history_hours, len(feature_names)), dtype=np.float32)
    recent_volatility = np.empty(len(target_times), dtype=np.float32)
    naive_baseline = np.empty(len(target_times), dtype=np.float32)
    for idx, target_time in enumerate(target_times):
        start_idx, end_idx = window_bounds_for_target(
            target_time,
            feature_table.hour_to_idx,
            history_hours=history_hours,
            horizon_hours=horizon_hours,
        )
        sequence[idx] = feature_matrix[start_idx : end_idx + 1]
        recent_window = feature_table.target[max(0, end_idx - 23) : end_idx + 1]
        recent_volatility[idx] = float(np.std(recent_window))
        lag_24_idx = feature_table.hour_to_idx[target_time - timedelta(hours=24)]
        naive_baseline[idx] = feature_table.target[lag_24_idx]
    return {
        "sequence": sequence,
        "target_calendar": target_calendar,
        "recent_volatility": recent_volatility,
        "naive_baseline": naive_baseline,
        "target_times": target_times,
    }


def available_observed_targets(feature_table: HourlyFeatureTable, target_times: list[datetime]) -> list[datetime]:
    return [timestamp for timestamp in target_times if timestamp in feature_table.hour_to_idx]
