from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
from weather_impact_analysis import aggregate_weather_features


TIMESTAMP_INPUT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TIMESTAMP_OUTPUT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FORECAST_HORIZON_HOURS = 24
TOP_HARDWARE_FAMILIES = (
    "SCC6",
    "DecosVb",
    "ML5",
    "DecosIIIh",
    "DecosIIIj",
    "DecosVa",
    "MP4000",
    "ML3",
)
RAW_DROPPED_COLUMNS = [
    "container_visit_uuid",
    "container_uuid",
    "customer_uuid",
    "ContainerSize",
    "AvPowerCons",
    "TtlEnergyConsHour",
    "TtlEnergyCons",
]
OBSERVED_STRUCTURAL_FIELDS = [
    "label_power_kw",
    "active_rows",
    "unique_containers",
    "unique_visits",
    "distinct_customers",
    "mean_power_per_active_reefer_kw",
    "mean_stack_tier",
    "share_stack_tier_1",
    "share_stack_tier_2",
    "share_stack_tier_3",
    "share_hardware_scc6",
    "share_hardware_decosvb",
    "share_hardware_ml5",
    "share_hardware_decosiiih",
    "share_hardware_decosiiij",
    "share_hardware_decosva",
    "share_hardware_mp4000",
    "share_hardware_ml3",
    "share_hardware_other",
    "mean_temperature_setpoint_c",
    "mean_temperature_ambient_c",
    "mean_temperature_return_c",
    "mean_temperature_supply_c",
    "mean_ambient_minus_setpoint_c",
    "mean_return_minus_supply_c",
]
TEMPERATURE_FIELDS = [
    "mean_temperature_setpoint_c",
    "mean_temperature_ambient_c",
    "mean_temperature_return_c",
    "mean_temperature_supply_c",
    "mean_ambient_minus_setpoint_c",
    "mean_return_minus_supply_c",
]
REEFER_HISTORY_WINDOWS = {
    "label_power_kw": 168,
    "active_rows": 168,
    "unique_containers": 168,
    "unique_visits": 168,
    "distinct_customers": 168,
    "mean_power_per_active_reefer_kw": 168,
    "mean_stack_tier": 48,
    "share_stack_tier_1": 48,
    "share_stack_tier_2": 48,
    "share_stack_tier_3": 48,
    "share_hardware_scc6": 48,
    "share_hardware_decosvb": 48,
    "share_hardware_ml5": 48,
    "share_hardware_decosiiih": 48,
    "share_hardware_decosiiij": 48,
    "share_hardware_decosva": 48,
    "share_hardware_mp4000": 48,
    "share_hardware_ml3": 48,
    "share_hardware_other": 48,
    "mean_temperature_setpoint_c": 48,
    "mean_temperature_ambient_c": 48,
    "mean_temperature_return_c": 48,
    "mean_temperature_supply_c": 48,
    "mean_ambient_minus_setpoint_c": 48,
    "mean_return_minus_supply_c": 48,
}
WEATHER_HISTORY_WINDOWS = {
    "weather_temperature_vc_halle3_c": 51,
    "weather_temperature_mean_c": 44,
    "weather_temperature_zentralgate_c": 46,
    "weather_wind_direction_vc_halle3_cos": 60,
    "weather_wind_direction_mean_cos": 62,
    "weather_wind_direction_zentralgate_cos": 63,
    "weather_wind_direction_zentralgate_consistency": 26,
    "weather_wind_speed_vc_halle3": 57,
    "weather_wind_speed_spread": 43,
    "weather_temperature_spread_c": 43,
    "weather_wind_speed_zentralgate": 42,
    "weather_wind_speed_mean": 42,
    "weather_wind_direction_zentralgate_sin": 35,
    "weather_wind_direction_vc_halle3_consistency": 37,
    "weather_wind_direction_mean_sin": 36,
    "weather_wind_direction_mean_consistency": 30,
    "weather_wind_direction_vc_halle3_sin": 37,
}
WEATHER_FILL_FIELDS = list(WEATHER_HISTORY_WINDOWS)


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
    visit_ids: set[str] = field(default_factory=set)
    container_ids: set[str] = field(default_factory=set)
    customer_ids: set[str] = field(default_factory=set)
    hardware_counter: Counter[str] = field(default_factory=Counter)
    tier_counter: Counter[str] = field(default_factory=Counter)

    def update(self, row: dict[str, str]) -> None:
        self.row_count += 1

        visit_id = clean_category(row.get("container_visit_uuid"))
        if visit_id:
            self.visit_ids.add(visit_id)

        container_id = clean_category(row.get("container_uuid"))
        if container_id:
            self.container_ids.add(container_id)

        customer_id = clean_category(row.get("customer_uuid"))
        if customer_id:
            self.customer_ids.add(customer_id)

        hardware = clean_category(row.get("HardwareType"))
        if hardware:
            self.hardware_counter[hardware] += 1

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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    participant_root = repo_root.parent / "participant_package" / "participant_package"
    extracted_csv = participant_root / "reefer_release" / "reefer_release.csv"
    default_reefer_source = extracted_csv if extracted_csv.exists() else participant_root / "reefer_release.zip"

    parser = argparse.ArgumentParser(description="Build a day-ahead-safe derived reefer dataset.")
    parser.add_argument("--reefer-source", type=Path, default=default_reefer_source)
    parser.add_argument("--target-csv", type=Path, default=participant_root / "target_timestamps.csv")
    parser.add_argument("--weather-dir", type=Path, default=participant_root / "wetterdaten")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "preprocessed_dataset")
    parser.add_argument("--local-timezone", type=str, default="Europe/Berlin")
    parser.add_argument("--small-gap-hours", type=int, default=3)
    return parser.parse_args()


def parse_decimal(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.upper() == "NULL":
        return None
    return float(stripped.replace(",", "."))


def clean_category(value: str | None) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    return "" if not stripped or stripped.upper() == "NULL" else stripped


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_INPUT_FORMAT)


def format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime(TIMESTAMP_OUTPUT_FORMAT)


def safe_mean(total: float, count: int) -> float:
    return float("nan") if count == 0 else total / count


def diff_or_nan(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return left - right


def find_csv_entry(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    for entry in archive.infolist():
        if entry.filename.lower().endswith(".csv") and not entry.filename.startswith("__MACOSX/"):
            return entry
    raise FileNotFoundError("No CSV file found in reefer archive.")


def iter_reefer_rows(reefer_source: Path) -> Iterable[dict[str, str]]:
    if reefer_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(reefer_source) as archive:
            with archive.open(find_csv_entry(archive)) as raw_stream:
                text_stream = TextIOWrapper(raw_stream, encoding="utf-8", newline="")
                yield from csv.DictReader(text_stream, delimiter=";")
        return

    with reefer_source.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=";")


def load_target_timestamps(path: Path) -> list[datetime]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        timestamps = [datetime.strptime(row["timestamp_utc"], TIMESTAMP_OUTPUT_FORMAT) for row in reader]
    return sorted(timestamps)


def normalize_hardware_label(label: str) -> str:
    return "".join(character for character in label.lower() if character.isalnum())


def season_from_month(month: int) -> int:
    if month in (12, 1, 2):
        return 1
    if month in (3, 4, 5):
        return 2
    if month in (6, 7, 8):
        return 3
    return 4


def daypart_from_hour(hour: int) -> int:
    if hour <= 5:
        return 0
    if hour <= 11:
        return 1
    if hour <= 17:
        return 2
    return 3


def load_hourly_reefer_observations(reefer_source: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    hourly: dict[datetime, HourAccumulator] = {}
    previous_raw_timestamp: datetime | None = None
    raw_rows_monotonic = True
    row_count = 0
    invalid_timestamp_rows = 0
    missing_power_rows = 0
    negative_power_rows = 0
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None

    for row in iter_reefer_rows(reefer_source):
        row_count += 1
        raw_timestamp = row.get("EventTime")
        if not raw_timestamp:
            invalid_timestamp_rows += 1
            continue

        try:
            timestamp = parse_timestamp(raw_timestamp)
        except ValueError:
            invalid_timestamp_rows += 1
            continue

        if previous_raw_timestamp is not None and timestamp < previous_raw_timestamp:
            raw_rows_monotonic = False
        previous_raw_timestamp = timestamp

        timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
        timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)

        power = parse_decimal(row.get("AvPowerCons"))
        if power is None:
            missing_power_rows += 1
        elif power < 0:
            negative_power_rows += 1

        hourly.setdefault(timestamp, HourAccumulator()).update(row)

    records: list[dict[str, object]] = []
    negative_hourly_label_count = 0
    for timestamp in sorted(hourly):
        accumulator = hourly[timestamp]
        active_rows = float(accumulator.row_count)
        mean_setpoint = safe_mean(accumulator.setpoint_sum, accumulator.setpoint_count)
        mean_ambient = safe_mean(accumulator.ambient_sum, accumulator.ambient_count)
        mean_return = safe_mean(accumulator.return_sum, accumulator.return_count)
        mean_supply = safe_mean(accumulator.supply_sum, accumulator.supply_count)
        mean_stack_tier = safe_mean(accumulator.stack_tier_sum, accumulator.stack_tier_count)
        label_power_kw = accumulator.power_w_sum / 1000.0
        if label_power_kw < 0:
            negative_hourly_label_count += 1

        record: dict[str, object] = {
            "source_timestamp": timestamp,
            "effective_timestamp": timestamp,
            "label_power_kw": label_power_kw,
            "active_rows": active_rows,
            "unique_containers": float(len(accumulator.container_ids)),
            "unique_visits": float(len(accumulator.visit_ids)),
            "distinct_customers": float(len(accumulator.customer_ids)),
            "mean_power_per_active_reefer_kw": float("nan") if active_rows == 0.0 else label_power_kw / active_rows,
            "mean_stack_tier": mean_stack_tier,
            "share_stack_tier_1": accumulator.tier_counter["1"] / active_rows if active_rows else float("nan"),
            "share_stack_tier_2": accumulator.tier_counter["2"] / active_rows if active_rows else float("nan"),
            "share_stack_tier_3": accumulator.tier_counter["3"] / active_rows if active_rows else float("nan"),
            "mean_temperature_setpoint_c": mean_setpoint,
            "mean_temperature_ambient_c": mean_ambient,
            "mean_temperature_return_c": mean_return,
            "mean_temperature_supply_c": mean_supply,
            "mean_ambient_minus_setpoint_c": diff_or_nan(mean_ambient, mean_setpoint),
            "mean_return_minus_supply_c": diff_or_nan(mean_return, mean_supply),
            "was_gap_shifted": 0,
            "was_dst_adjusted": 0,
        }

        hardware_share_total = 0.0
        for family in TOP_HARDWARE_FAMILIES:
            normalized = normalize_hardware_label(family)
            share = accumulator.hardware_counter[family] / active_rows if active_rows else float("nan")
            record[f"share_hardware_{normalized}"] = share
            hardware_share_total += 0.0 if not math.isfinite(share) else share
        record["share_hardware_other"] = float("nan") if active_rows == 0.0 else max(0.0, 1.0 - hardware_share_total)

        records.append(record)

    summary = {
        "row_count": row_count,
        "hourly_count": len(records),
        "timestamp_min_utc": format_timestamp(timestamp_min),
        "timestamp_max_utc": format_timestamp(timestamp_max),
        "raw_rows_monotonic_non_decreasing": raw_rows_monotonic,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "missing_power_rows": missing_power_rows,
        "negative_power_rows": negative_power_rows,
        "negative_hourly_label_count": negative_hourly_label_count,
    }
    return records, summary


def collect_time_gaps(hours: list[datetime]) -> list[dict[str, object]]:
    gaps: list[dict[str, object]] = []
    for previous, current in zip(hours, hours[1:]):
        diff_hours = int((current - previous).total_seconds() // 3600)
        if diff_hours != 1:
            gaps.append(
                {
                    "previous_timestamp_utc": format_timestamp(previous),
                    "current_timestamp_utc": format_timestamp(current),
                    "diff_hours": diff_hours,
                    "missing_hours": max(diff_hours - 1, 0),
                }
            )
    return gaps


def find_timezone_transitions(hours: list[datetime], timezone_name: str) -> list[dict[str, object]]:
    if not hours:
        return []
    tz = ZoneInfo(timezone_name)
    start = hours[0].replace(tzinfo=timezone.utc)
    end = hours[-1].replace(tzinfo=timezone.utc)
    transitions: list[dict[str, object]] = []
    current = start
    previous_offset = current.astimezone(tz).utcoffset()
    while current <= end:
        current_offset = current.astimezone(tz).utcoffset()
        if current_offset != previous_offset:
            transitions.append(
                {
                    "transition_utc": format_timestamp(current.replace(tzinfo=None)),
                    "offset_before_hours": previous_offset.total_seconds() / 3600 if previous_offset else 0.0,
                    "offset_after_hours": current_offset.total_seconds() / 3600 if current_offset else 0.0,
                }
            )
            previous_offset = current_offset
        current += timedelta(hours=1)
    return transitions


def audit_dst_behavior(hours: list[datetime], timezone_name: str) -> dict[str, object]:
    transitions = find_timezone_transitions(hours, timezone_name)
    gaps = collect_time_gaps(hours)
    transition_datetimes = [
        datetime.strptime(transition["transition_utc"], TIMESTAMP_OUTPUT_FORMAT)
        for transition in transitions
        if transition["transition_utc"] is not None
    ]
    nearby_anomalies = [
        gap
        for gap in gaps
        if any(
            abs((datetime.strptime(gap["previous_timestamp_utc"], TIMESTAMP_OUTPUT_FORMAT) - transition).total_seconds()) <= 6 * 3600
            or abs((datetime.strptime(gap["current_timestamp_utc"], TIMESTAMP_OUTPUT_FORMAT) - transition).total_seconds()) <= 6 * 3600
            for transition in transition_datetimes
        )
    ]
    requires_adjustment = any(gap["diff_hours"] in (0, 2) for gap in nearby_anomalies)
    return {
        "timezone": timezone_name,
        "transition_count": len(transitions),
        "transitions": transitions,
        "non_hourly_gap_count": len(gaps),
        "non_hourly_gaps_near_dst_windows": nearby_anomalies,
        "requires_adjustment": requires_adjustment,
        "assessment": "source_hours_already_use_one_consistent_basis" if not requires_adjustment else "source_hours_look_dst_local_and_need_normalization",
    }


def apply_dst_adjustment_if_needed(records: list[dict[str, object]], audit: dict[str, object], timezone_name: str) -> int:
    if not audit["requires_adjustment"]:
        return 0

    tz = ZoneInfo(timezone_name)
    adjusted_count = 0
    for record in records:
        source_timestamp = record["source_timestamp"]
        normalized = source_timestamp.replace(tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None)
        if normalized != source_timestamp:
            adjusted_count += 1
            record["effective_timestamp"] = normalized
            record["was_dst_adjusted"] = 1
    return adjusted_count


def split_observed_records(
    records: list[dict[str, object]],
    test_timestamps: set[datetime],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trainval = [dict(record) for record in records if record["source_timestamp"] not in test_timestamps]
    test = [dict(record) for record in records if record["source_timestamp"] in test_timestamps]
    return trainval, test


def add_years_safe(value: datetime, years: int) -> datetime:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value + timedelta(days=365 * years)


def reorder_trainval_records(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not records:
        return [], {
            "gap_found": False,
            "largest_gap_hours": 0,
            "gap_start_utc": None,
            "gap_end_utc": None,
            "pre_gap_count": 0,
            "post_gap_count": 0,
        }

    ordered = sorted(records, key=lambda record: record["effective_timestamp"])
    gaps = []
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]["effective_timestamp"]
        current = ordered[index]["effective_timestamp"]
        diff_hours = int((current - previous).total_seconds() // 3600)
        if diff_hours > 1:
            gaps.append((diff_hours, index - 1, previous, current))

    if not gaps:
        return ordered, {
            "gap_found": False,
            "largest_gap_hours": 0,
            "gap_start_utc": None,
            "gap_end_utc": None,
            "pre_gap_count": len(ordered),
            "post_gap_count": 0,
        }

    largest_gap_hours, split_index, gap_start, gap_end = max(gaps, key=lambda item: item[0])
    pre_gap = [dict(record) for record in ordered[: split_index + 1]]
    post_gap = [dict(record) for record in ordered[split_index + 1 :]]
    shifted_pre_gap = []
    for record in pre_gap:
        shifted = dict(record)
        shifted["effective_timestamp"] = add_years_safe(record["effective_timestamp"], 1)
        shifted["was_gap_shifted"] = 1
        shifted_pre_gap.append(shifted)

    reordered = post_gap + shifted_pre_gap
    return reordered, {
        "gap_found": True,
        "largest_gap_hours": largest_gap_hours,
        "gap_start_utc": format_timestamp(gap_start),
        "gap_end_utc": format_timestamp(gap_end),
        "pre_gap_count": len(pre_gap),
        "post_gap_count": len(post_gap),
        "moved_block_source_start_utc": format_timestamp(pre_gap[0]["source_timestamp"]) if pre_gap else None,
        "moved_block_source_end_utc": format_timestamp(pre_gap[-1]["source_timestamp"]) if pre_gap else None,
        "moved_block_effective_start_utc": format_timestamp(shifted_pre_gap[0]["effective_timestamp"]) if shifted_pre_gap else None,
        "moved_block_effective_end_utc": format_timestamp(shifted_pre_gap[-1]["effective_timestamp"]) if shifted_pre_gap else None,
    }


def apply_small_gap_fill(records: list[dict[str, object]], field_names: list[str], max_gap_hours: int) -> dict[str, int]:
    fill_counts = {field_name: 0 for field_name in field_names}
    effective_timestamps = [record["effective_timestamp"] for record in records]

    for field_name in field_names:
        values = np.array([float(record[field_name]) for record in records], dtype=float)
        index = 0
        while index < len(values):
            if math.isfinite(values[index]):
                index += 1
                continue

            gap_start = index
            while index < len(values) and not math.isfinite(values[index]):
                index += 1
            gap_end = index
            if gap_end - gap_start > max_gap_hours:
                continue

            for gap_index in range(gap_start, gap_end):
                target_daypart = daypart_from_hour(effective_timestamps[gap_index].hour)
                replacement = float("nan")

                for previous_index in range(gap_index - 1, -1, -1):
                    if daypart_from_hour(effective_timestamps[previous_index].hour) != target_daypart:
                        continue
                    if math.isfinite(values[previous_index]):
                        replacement = values[previous_index]
                        break

                if not math.isfinite(replacement):
                    for next_index in range(gap_index + 1, len(values)):
                        if daypart_from_hour(effective_timestamps[next_index].hour) != target_daypart:
                            continue
                        if math.isfinite(values[next_index]):
                            replacement = values[next_index]
                            break

                if math.isfinite(replacement):
                    values[gap_index] = replacement
                    fill_counts[field_name] += 1

        for record, value in zip(records, values):
            record[field_name] = float(value)

    return fill_counts


def merge_weather_observations(
    records: list[dict[str, object]],
    weather_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not weather_dir.exists():
        return [dict(record) for record in records], {
            "available": False,
            "selected_weather_features": list(WEATHER_HISTORY_WINDOWS),
            "merged_hour_count": 0,
            "weather_hourly_start_utc": None,
            "weather_hourly_end_utc": None,
        }

    hourly_weather, weather_summary = aggregate_weather_features(weather_dir)
    weather_timestamps = sorted(hourly_weather)
    merged_records: list[dict[str, object]] = []
    for record in records:
        merged = dict(record)
        weather_values = hourly_weather.get(record["source_timestamp"], {})
        for feature_name in WEATHER_HISTORY_WINDOWS:
            merged[feature_name] = float(weather_values.get(feature_name, float("nan")))
        merged_records.append(merged)

    merge_summary = {
        "available": True,
        "selected_weather_features": list(WEATHER_HISTORY_WINDOWS),
        "merged_hour_count": len(hourly_weather),
        "weather_hourly_start_utc": format_timestamp(weather_timestamps[0]) if weather_timestamps else None,
        "weather_hourly_end_utc": format_timestamp(weather_timestamps[-1]) if weather_timestamps else None,
        "raw_weather_feature_count": len({feature for feature_map in hourly_weather.values() for feature in feature_map}),
        "weather_source_summary": weather_summary,
    }
    return merged_records, merge_summary


def build_history_matrix(values: np.ndarray, start_lag_hours: int, window_hours: int) -> np.ndarray:
    history = np.full((len(values), window_hours), np.nan, dtype=float)
    for offset in range(window_hours):
        lag_hours = start_lag_hours + offset
        if lag_hours < len(values):
            history[lag_hours:, offset] = values[: len(values) - lag_hours]
    return history


def build_calendar_features(timestamp: datetime) -> dict[str, float | int]:
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()
    day_of_year = timestamp.timetuple().tm_yday
    month = timestamp.month
    season = season_from_month(month)
    return {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "month": month,
        "season": season,
        "is_weekend": int(day_of_week >= 5),
        "hour_sin": math.sin(2.0 * math.pi * hour_of_day / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour_of_day / 24.0),
        "dow_sin": math.sin(2.0 * math.pi * day_of_week / 7.0),
        "dow_cos": math.cos(2.0 * math.pi * day_of_week / 7.0),
        "day_of_year_sin": math.sin(2.0 * math.pi * day_of_year / 366.0),
        "day_of_year_cos": math.cos(2.0 * math.pi * day_of_year / 366.0),
        "month_sin": math.sin(2.0 * math.pi * month / 12.0),
        "month_cos": math.cos(2.0 * math.pi * month / 12.0),
    }


def engineer_dataset(
    ordered_records: list[dict[str, object]],
    sequence_index_start: int,
    small_gap_hours: int,
    weather_start_timestamp: datetime | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not ordered_records:
        return [], {
            "filled_small_gaps": {},
            "dropped_rows_by_reason": {},
            "generated_feature_count": 0,
            "history_windows_hours": {},
        }

    working_records = [dict(record) for record in ordered_records]
    fill_counts = apply_small_gap_fill(
        working_records,
        TEMPERATURE_FIELDS + WEATHER_FILL_FIELDS,
        small_gap_hours,
    )

    all_history_windows = {**REEFER_HISTORY_WINDOWS, **WEATHER_HISTORY_WINDOWS}
    base_arrays = {
        field_name: np.array([float(record[field_name]) for record in working_records], dtype=float)
        for field_name in all_history_windows
    }

    export_records: list[dict[str, object]] = []
    for local_index, record in enumerate(working_records):
        export_record: dict[str, object] = {
            "source_timestamp_utc": format_timestamp(record["source_timestamp"]),
            "effective_timestamp_utc": format_timestamp(record["effective_timestamp"]),
            "sequence_index": sequence_index_start + local_index,
            "label_power_kw": float(record["label_power_kw"]),
            "was_gap_shifted": int(record["was_gap_shifted"]),
            "was_dst_adjusted": int(record["was_dst_adjusted"]),
        }
        export_record.update(build_calendar_features(record["effective_timestamp"]))
        export_records.append(export_record)

    generated_feature_names: list[str] = []
    weather_generated_feature_names: dict[str, list[str]] = {}
    reefer_generated_feature_names: list[str] = []
    weather_required_from: dict[str, datetime] = {}
    for field_name, values in base_arrays.items():
        window_hours = all_history_windows[field_name]
        history_matrix = build_history_matrix(values, FORECAST_HORIZON_HOURS, window_hours)
        feature_columns: list[str] = []
        for offset in range(window_hours):
            actual_lag = FORECAST_HORIZON_HOURS + offset
            column_name = f"{field_name}_tminus{actual_lag}h"
            feature_columns.append(column_name)
            column_values = history_matrix[:, offset]
            for record, value in zip(export_records, column_values):
                record[column_name] = float(value)
        generated_feature_names.extend(feature_columns)
        if field_name in WEATHER_HISTORY_WINDOWS:
            weather_generated_feature_names[field_name] = feature_columns
            if weather_start_timestamp is not None:
                weather_required_from[field_name] = weather_start_timestamp + timedelta(
                    hours=FORECAST_HORIZON_HOURS + window_hours - 1
                )
        else:
            reefer_generated_feature_names.extend(feature_columns)

    warmup_limit = FORECAST_HORIZON_HOURS + max(REEFER_HISTORY_WINDOWS.values()) - 1
    dropped_rows_by_reason = Counter()
    retained: list[dict[str, object]] = []
    rows_with_incomplete_weather_history = 0
    rows_before_full_weather_history = 0
    for local_index, record in enumerate(export_records):
        missing_reefer = [
            feature_name
            for feature_name in reefer_generated_feature_names
            if not math.isfinite(float(record[feature_name]))
        ]
        if missing_reefer:
            if local_index < warmup_limit:
                dropped_rows_by_reason["warmup_incomplete_history"] += 1
            else:
                dropped_rows_by_reason["missing_required_reefer_history_after_fill"] += 1
            continue

        source_timestamp = datetime.strptime(record["source_timestamp_utc"], TIMESTAMP_OUTPUT_FORMAT)
        weather_expected_feature_count = 0
        weather_available_feature_count = 0
        for field_name, feature_columns in weather_generated_feature_names.items():
            required_from = weather_required_from.get(field_name)
            if required_from is None or source_timestamp < required_from:
                continue
            weather_expected_feature_count += len(feature_columns)
            weather_available_feature_count += sum(
                1 for column_name in feature_columns if math.isfinite(float(record[column_name]))
            )

        if weather_expected_feature_count == 0:
            rows_before_full_weather_history += 1
        elif weather_available_feature_count < weather_expected_feature_count:
            rows_with_incomplete_weather_history += 1

        record["weather_history_expected"] = int(weather_expected_feature_count > 0)
        record["weather_history_complete"] = int(
            weather_expected_feature_count > 0
            and weather_available_feature_count == weather_expected_feature_count
        )
        record["weather_history_expected_feature_count"] = weather_expected_feature_count
        record["weather_history_available_feature_count"] = weather_available_feature_count
        record["weather_history_available_fraction"] = (
            float(weather_available_feature_count / weather_expected_feature_count)
            if weather_expected_feature_count > 0
            else float("nan")
        )
        retained.append(record)

    summary = {
        "filled_small_gaps": fill_counts,
        "dropped_rows_by_reason": dict(dropped_rows_by_reason),
        "generated_feature_count": len(generated_feature_names),
        "generated_feature_name_sample": generated_feature_names[:30],
        "history_windows_hours": all_history_windows,
        "rows_before_full_weather_history": rows_before_full_weather_history,
        "rows_with_incomplete_weather_history": rows_with_incomplete_weather_history,
        "weather_history_availability_columns": [
            "weather_history_expected",
            "weather_history_complete",
            "weather_history_expected_feature_count",
            "weather_history_available_feature_count",
            "weather_history_available_fraction",
        ],
    }
    return retained, summary


def verify_contiguous_timestamps(records: list[dict[str, object]], timestamp_key: str) -> dict[str, object]:
    if not records:
        return {"gap_count": 0, "largest_gap_hours": 0, "gaps": []}
    timestamps = [record[timestamp_key] for record in records]
    gaps = collect_time_gaps(timestamps)
    largest_gap_hours = max((gap["diff_hours"] for gap in gaps), default=0)
    return {
        "gap_count": len(gaps),
        "largest_gap_hours": largest_gap_hours,
        "gaps": gaps,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def audit_weather_file(file_path: Path, timezone_name: str) -> dict[str, object]:
    hourly_timestamps: set[datetime] = set()
    row_count = 0
    null_value_rows = 0
    invalid_timestamp_rows = 0
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            row_count += 1
            raw_timestamp = row.get("UtcTimestamp")
            raw_value = row.get("Value")
            if not raw_timestamp:
                invalid_timestamp_rows += 1
                continue
            try:
                timestamp = parse_timestamp(raw_timestamp)
            except ValueError:
                invalid_timestamp_rows += 1
                continue

            if parse_decimal(raw_value) is None:
                null_value_rows += 1
                continue

            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)
            hourly_timestamps.add(timestamp.replace(minute=0, second=0, microsecond=0))

    sorted_hours = sorted(hourly_timestamps)
    gaps = collect_time_gaps(sorted_hours)
    return {
        "file_name": file_path.name,
        "row_count": row_count,
        "null_value_rows": null_value_rows,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "hourly_timestamp_count": len(sorted_hours),
        "timestamp_min_utc": format_timestamp(timestamp_min),
        "timestamp_max_utc": format_timestamp(timestamp_max),
        "missing_hour_count": sum(gap["missing_hours"] for gap in gaps),
        "largest_gap_hours": max((gap["diff_hours"] for gap in gaps), default=0),
        "dst_audit": audit_dst_behavior(sorted_hours, timezone_name),
    }


def audit_weather_directory(weather_dir: Path, timezone_name: str) -> dict[str, object]:
    if not weather_dir.exists():
        return {"available": False, "files": []}
    files = sorted(path for path in weather_dir.rglob("*.csv") if path.is_file())
    return {
        "available": True,
        "file_count": len(files),
        "files": [audit_weather_file(file_path, timezone_name) for file_path in files],
    }


def build_summary(
    reefer_summary: dict[str, object],
    reefer_dst_audit: dict[str, object],
    dst_adjusted_record_count: int,
    gap_summary: dict[str, object],
    weather_merge_summary: dict[str, object],
    trainval_engineering_summary: dict[str, object],
    test_engineering_summary: dict[str, object],
    trainval_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    target_timestamps: list[datetime],
    weather_summary: dict[str, object],
) -> dict[str, object]:
    target_timestamps_set = {format_timestamp(timestamp) for timestamp in target_timestamps}
    exported_test_timestamps = [record["source_timestamp_utc"] for record in test_rows]
    return {
        "raw_reefer_summary": reefer_summary,
        "dropped_raw_columns": RAW_DROPPED_COLUMNS,
        "reefer_dst_audit": {
            **reefer_dst_audit,
            "adjusted_record_count": dst_adjusted_record_count,
        },
        "gap_repair": gap_summary,
        "weather_feature_merge": weather_merge_summary,
        "feature_engineering": {
            "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
            "reefer_history_windows_hours": REEFER_HISTORY_WINDOWS,
            "weather_history_windows_hours": WEATHER_HISTORY_WINDOWS,
            "trainval": trainval_engineering_summary,
            "test": test_engineering_summary,
        },
        "exports": {
            "trainval_row_count": len(trainval_rows),
            "test_row_count": len(test_rows),
            "trainval_effective_continuity": verify_contiguous_timestamps(
                [
                    {
                        "effective_timestamp": datetime.strptime(record["effective_timestamp_utc"], TIMESTAMP_OUTPUT_FORMAT)
                    }
                    for record in trainval_rows
                ],
                "effective_timestamp",
            ),
            "test_source_continuity": verify_contiguous_timestamps(
                [
                    {
                        "source_timestamp": datetime.strptime(record["source_timestamp_utc"], TIMESTAMP_OUTPUT_FORMAT)
                    }
                    for record in test_rows
                ],
                "source_timestamp",
            ),
            "test_matches_target_exactly": set(exported_test_timestamps) == target_timestamps_set,
            "test_first_timestamp_utc": exported_test_timestamps[0] if exported_test_timestamps else None,
            "test_last_timestamp_utc": exported_test_timestamps[-1] if exported_test_timestamps else None,
            "trainval_first_effective_timestamp_utc": trainval_rows[0]["effective_timestamp_utc"] if trainval_rows else None,
            "trainval_last_effective_timestamp_utc": trainval_rows[-1]["effective_timestamp_utc"] if trainval_rows else None,
        },
        "weather_audit": weather_summary,
    }


def run_preprocessing(args: argparse.Namespace) -> Path:
    reefer_source = args.reefer_source.resolve()
    target_csv = args.target_csv.resolve()
    output_dir = args.output_dir.resolve()
    weather_dir = args.weather_dir.resolve()

    if not reefer_source.exists():
        raise FileNotFoundError(f"Reefer source not found: {reefer_source}")
    if not target_csv.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_csv}")

    target_timestamps = load_target_timestamps(target_csv)
    if not target_timestamps:
        raise RuntimeError("Target timestamp file is empty.")

    observed_records, reefer_summary = load_hourly_reefer_observations(reefer_source)
    observed_records, weather_merge_summary = merge_weather_observations(observed_records, weather_dir)
    source_hours = [record["source_timestamp"] for record in observed_records]
    reefer_dst_audit = audit_dst_behavior(source_hours, args.local_timezone)
    dst_adjusted_record_count = apply_dst_adjustment_if_needed(observed_records, reefer_dst_audit, args.local_timezone)

    test_timestamp_set = set(target_timestamps)
    target_timestamp_strings = {format_timestamp(timestamp) for timestamp in target_timestamps}
    trainval_observed, _ = split_observed_records(observed_records, test_timestamp_set)
    reordered_trainval, gap_summary = reorder_trainval_records(trainval_observed)

    trainval_rows, trainval_engineering_summary = engineer_dataset(
        reordered_trainval,
        sequence_index_start=0,
        small_gap_hours=args.small_gap_hours,
        weather_start_timestamp=(
            datetime.strptime(weather_merge_summary["weather_hourly_start_utc"], TIMESTAMP_OUTPUT_FORMAT)
            if weather_merge_summary.get("weather_hourly_start_utc")
            else None
        ),
    )
    actual_ordered_records = sorted((dict(record) for record in observed_records), key=lambda record: record["source_timestamp"])
    actual_engineered_rows, test_engineering_summary = engineer_dataset(
        actual_ordered_records,
        sequence_index_start=0,
        small_gap_hours=args.small_gap_hours,
        weather_start_timestamp=(
            datetime.strptime(weather_merge_summary["weather_hourly_start_utc"], TIMESTAMP_OUTPUT_FORMAT)
            if weather_merge_summary.get("weather_hourly_start_utc")
            else None
        ),
    )
    test_rows = [dict(row) for row in actual_engineered_rows if row["source_timestamp_utc"] in target_timestamp_strings]
    for index, row in enumerate(test_rows):
        row["sequence_index"] = len(trainval_rows) + index
    test_engineering_summary["selected_test_row_count"] = len(test_rows)

    if len(test_rows) != len(target_timestamps):
        raise RuntimeError(
            f"Expected {len(target_timestamps)} test rows after feature engineering, found {len(test_rows)}."
        )

    weather_summary = audit_weather_directory(weather_dir, args.local_timezone)
    summary = build_summary(
        reefer_summary=reefer_summary,
        reefer_dst_audit=reefer_dst_audit,
        dst_adjusted_record_count=dst_adjusted_record_count,
        gap_summary=gap_summary,
        weather_merge_summary=weather_merge_summary,
        trainval_engineering_summary=trainval_engineering_summary,
        test_engineering_summary=test_engineering_summary,
        trainval_rows=trainval_rows,
        test_rows=test_rows,
        target_timestamps=target_timestamps,
        weather_summary=weather_summary,
    )

    if not summary["exports"]["test_matches_target_exactly"]:
        raise RuntimeError("The exported test timestamps do not exactly match target_timestamps.csv.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trainval_hourly.csv", trainval_rows)
    write_csv(output_dir / "test_hourly.csv", test_rows)
    write_json(output_dir / "preprocessing_summary.json", summary)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = run_preprocessing(args)
    print(
        f"Preprocessing complete. Created {output_dir / 'trainval_hourly.csv'}, "
        f"{output_dir / 'test_hourly.csv'}, and {output_dir / 'preprocessing_summary.json'}."
    )


if __name__ == "__main__":
    main()
