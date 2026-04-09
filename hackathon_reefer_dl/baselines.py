from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

from hackathon_reefer_dl.common import (
    REEFER_EXTRACTED_CSV,
    REEFER_ZIP,
    iso_utc,
    parse_decimal,
    parse_hour_timestamp,
    parse_target_timestamp,
    to_hour,
)
from hackathon_reefer_dl.metrics import composite_metrics


def _open_reefer_reader(participant_dir: Path):
    extracted = participant_dir / REEFER_EXTRACTED_CSV
    if extracted.exists():
        handle = extracted.open("r", encoding="utf-8-sig", newline="")
        return handle, csv.DictReader(handle, delimiter=";")

    reefer_zip = participant_dir / REEFER_ZIP
    if not reefer_zip.exists():
        raise FileNotFoundError(f"Could not find reefer data inside {participant_dir}")
    archive = zipfile.ZipFile(reefer_zip)
    raw = archive.open("reefer_release.csv")
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return archive, csv.DictReader(text, delimiter=";")


def aggregate_hourly_load(participant_dir: Path) -> dict:
    closeable, reader = _open_reefer_reader(participant_dir)
    hourly_load = defaultdict(float)
    try:
        for row in reader:
            timestamp = to_hour(parse_hour_timestamp(row["EventTime"]))
            power = parse_decimal(row["AvPowerCons"])
            if power is not None:
                hourly_load[timestamp] += power / 1000.0
    finally:
        closeable.close()
    return dict(hourly_load)


def load_target_hours(target_csv: Path) -> list:
    with target_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_target_timestamp(row["timestamp_utc"]) for row in reader]


def blended_public_baseline(hourly_load: dict, target_hours: list) -> list[dict[str, float | str]]:
    predictions: list[dict[str, float | str]] = []
    for target_time in target_hours:
        lag_24 = hourly_load.get(target_time - timedelta(hours=24), 0.0)
        lag_168 = hourly_load.get(target_time - timedelta(hours=168), 0.0)
        pred = 0.7 * lag_24 + 0.3 * lag_168
        predictions.append(
            {
                "timestamp_utc": iso_utc(target_time),
                "pred_power_kw": pred,
                "pred_p90_kw": pred * 1.10,
            }
        )
    return predictions


def score_public_baseline(participant_dir: Path, target_csv: Path) -> dict[str, float]:
    hourly_load = aggregate_hourly_load(participant_dir)
    target_hours = load_target_hours(target_csv)
    predictions = blended_public_baseline(hourly_load, target_hours)
    y_true = np.asarray([hourly_load[parse_target_timestamp(row["timestamp_utc"])] for row in predictions], dtype=np.float64)
    pred_power = np.asarray([row["pred_power_kw"] for row in predictions], dtype=np.float64)
    pred_p90 = np.asarray([row["pred_p90_kw"] for row in predictions], dtype=np.float64)
    return composite_metrics(y_true, pred_power, pred_p90).to_dict()
