from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hackathon_reefer_dl.baselines import score_public_baseline
from hackathon_reefer_dl.data import build_hourly_feature_table, save_hourly_feature_table
from hackathon_reefer_dl.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hourly reefer features from the participant package.")
    parser.add_argument("--participant-dir", type=Path, required=True, help="Participant package directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output artifact directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_table = build_hourly_feature_table(args.participant_dir)
    parquet_path, metadata_path, summary_path = save_hourly_feature_table(feature_table, args.out_dir)

    target_csv = args.participant_dir / "target_timestamps.csv"
    if target_csv.exists():
        baseline_metrics = score_public_baseline(args.participant_dir, target_csv)
        write_json(args.out_dir / "public_baseline_metrics.json", baseline_metrics)
        print(f"Public organizer baseline composite: {baseline_metrics['composite']:.6f}")

    print(f"Saved hourly features to {parquet_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
