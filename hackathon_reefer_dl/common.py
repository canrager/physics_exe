from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

HISTORY_HOURS = 336
FORECAST_HORIZON_HOURS = 24
CV_FOLD_HOURS = 223

TARGET_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
HOUR_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

TOP_HARDWARE = ("SCC6", "ML3", "DecosVb", "DecosVa", "DecosIIIj", "MP4000")
TOP_STACK_TIERS = ("1", "2", "3")
TOP_SIZES = ("20", "40")

REEFER_EXTRACTED_CSV = Path("reefer_release/reefer_release.csv")
REEFER_ZIP = Path("reefer_release.zip")
WEATHER_ZIP = Path("wetterdaten.zip")


def parse_decimal(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().replace("\ufeff", "")
    if not text or text.upper() == "NULL":
        return None
    return float(text.replace(",", "."))


def parse_hour_timestamp(value: str) -> datetime:
    text = value.strip().replace("T", " ").rstrip("Z")
    if "." in text:
        text = text.split(".", 1)[0]
    return datetime.strptime(text, HOUR_TIMESTAMP_FORMAT)


def to_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def parse_target_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        return datetime.strptime(text, TARGET_TIMESTAMP_FORMAT)
    return parse_hour_timestamp(text)


def iso_utc(value: datetime) -> str:
    return value.strftime(TARGET_TIMESTAMP_FORMAT)


def iter_hour_range(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(hours=1)


def target_cutoff_hour(target_time: datetime, horizon_hours: int = FORECAST_HORIZON_HOURS) -> datetime:
    return target_time - timedelta(hours=horizon_hours)


def window_bounds_for_target(
    target_time: datetime,
    hour_to_idx: dict[datetime, int],
    history_hours: int = HISTORY_HOURS,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> tuple[int, int]:
    cutoff = target_cutoff_hour(target_time, horizon_hours=horizon_hours)
    if cutoff not in hour_to_idx:
        raise KeyError(f"Target cutoff hour {iso_utc(cutoff)} is missing from hourly history.")
    end_idx = hour_to_idx[cutoff]
    start_idx = end_idx - history_hours + 1
    if start_idx < 0:
        raise IndexError(
            f"Not enough history for target {iso_utc(target_time)}; "
            f"need {history_hours} hours ending at {iso_utc(cutoff)}."
        )
    return start_idx, end_idx
