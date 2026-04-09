from __future__ import annotations

import unittest
from pathlib import Path

from hackathon_reefer_dl.baselines import score_public_baseline


class PublicBaselineRegressionTests(unittest.TestCase):
    def test_public_baseline_matches_reference_score(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        participant_dir = repo_root.parent / "FSL-assests" / "participant_package" / "participant_package"
        if not participant_dir.exists():
            self.skipTest("Participant package not available in the expected sibling directory.")

        metrics = score_public_baseline(participant_dir, participant_dir / "target_timestamps.csv")
        self.assertAlmostEqual(metrics["composite"], 63.539563, delta=0.05)
        self.assertAlmostEqual(metrics["mae_all"], 55.730514, delta=0.05)
        self.assertAlmostEqual(metrics["pinball_p90"], 12.461674, delta=0.05)


if __name__ == "__main__":
    unittest.main()
