from __future__ import annotations

import csv
import io
import unittest
from datetime import datetime, timedelta

from hackathon_reefer_dl.common import parse_decimal, parse_hour_timestamp, window_bounds_for_target
from hackathon_reefer_dl.data import scan_reefer_rows


class ParseAndAggregationTests(unittest.TestCase):
    def test_parse_decimal_handles_commas_and_nulls(self) -> None:
        self.assertAlmostEqual(parse_decimal("887,79348807004"), 887.79348807004)
        self.assertIsNone(parse_decimal("NULL"))
        self.assertIsNone(parse_decimal(""))

    def test_scan_reefer_rows_handles_bom_semicolons_and_hourly_aggregation(self) -> None:
        text = (
            "\ufeffcontainer_visit_uuid;EventTime;AvPowerCons;TemperatureAmbient;TemperatureSetPoint;"
            "TemperatureReturn;RemperatureSupply;HardwareType;stack_tier;ContainerSize\n"
            "visit-a;2025-01-01 00:05:00.000;1000,0;5,0;-18,0;-17,0;-19,0;SCC6;1;40\n"
            "visit-b;2025-01-01 00:55:00.000;500,0;7,0;-19,0;-18,0;-20,0;ML3;2;20\n"
            "visit-a;2025-01-01 01:10:00.000;750,0;6,0;-18,0;-17,5;-19,5;SCC6;1;40\n"
        )
        wrapper = io.TextIOWrapper(io.BytesIO(text.encode("utf-8")), encoding="utf-8-sig", newline="")
        reader = csv.DictReader(wrapper, delimiter=";")
        hourly, visit_bounds = scan_reefer_rows(reader)

        first_hour = parse_hour_timestamp("2025-01-01 00:00:00")
        second_hour = parse_hour_timestamp("2025-01-01 01:00:00")
        self.assertEqual(len(hourly), 2)
        self.assertAlmostEqual(hourly[first_hour].load_kw, 1.5)
        self.assertEqual(hourly[first_hour].active_visits, 2)
        self.assertEqual(hourly[first_hour].hardware_counts["SCC6"], 1)
        self.assertEqual(hourly[first_hour].hardware_counts["ML3"], 1)
        self.assertAlmostEqual(hourly[second_hour].load_kw, 0.75)
        self.assertEqual(visit_bounds["visit-a"][0], first_hour)
        self.assertEqual(visit_bounds["visit-a"][1], second_hour)


class AdmissibilityTests(unittest.TestCase):
    def test_window_bounds_end_at_target_minus_24h(self) -> None:
        start = datetime(2025, 1, 1, 0, 0, 0)
        hours = [start + timedelta(hours=offset) for offset in range(500)]
        hour_to_idx = {hour: idx for idx, hour in enumerate(hours)}
        target_time = datetime(2025, 1, 20, 0, 0, 0)
        start_idx, end_idx = window_bounds_for_target(
            target_time,
            hour_to_idx,
            history_hours=336,
            horizon_hours=24,
        )
        self.assertEqual(hours[end_idx], target_time - timedelta(hours=24))
        self.assertEqual(hours[start_idx], target_time - timedelta(hours=24 + 335))
        self.assertLess(hours[end_idx], target_time)


if __name__ == "__main__":
    unittest.main()

