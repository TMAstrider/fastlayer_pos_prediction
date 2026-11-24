import unittest

import numpy as np

from tools.evaluate_history_preference_fusion import (
    evaluate_fusion,
    history_geometry_summary,
    zscore,
)


class HistoryPreferenceFusionTests(unittest.TestCase):
    def test_zscore_centres_and_scales_scores(self) -> None:
        scores = zscore(np.asarray([1.0, 2.0, 3.0]))

        self.assertAlmostEqual(float(scores.mean()), 0.0)
        self.assertAlmostEqual(float(scores.std()), 1.0)
        np.testing.assert_allclose(zscore(np.ones(4)), np.zeros(4))

    def test_history_fusion_changes_top1_and_recovers_nearby_top5(self) -> None:
        angle = np.deg2rad(60.0)
        opposite_side_angle = np.deg2rad(120.0)
        directions = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [np.cos(opposite_side_angle), np.sin(opposite_side_angle), 0.0],
            [np.cos(angle), np.sin(angle), 0.0],
        ])
        part = {
            "directions": directions,
            "history_direction": directions[-1],
            "areas": np.asarray(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            ),
        }
        area_scores = np.asarray(
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        )
        history_scores = np.asarray(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0]
        )

        area_only = evaluate_fusion(
            [part],
            [(area_scores, history_scores)],
            alpha=0.0,
            preference="learned",
        )
        fused = evaluate_fusion(
            [part],
            [(area_scores, history_scores)],
            alpha=10.0,
            preference="learned",
        )

        self.assertAlmostEqual(
            area_only["history_top5_min_angle_median_deg"],
            30.0,
        )
        self.assertAlmostEqual(fused["history_top1_angle_median_deg"], 0.0)
        self.assertAlmostEqual(fused["history_top5_min_angle_median_deg"], 0.0)

    def test_history_geometry_uses_synthetic_area_oracle(self) -> None:
        directions = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        part = {
            "directions": directions,
            "history_direction": directions[0],
            "targets": np.asarray([1.0, 0.4, 0.0]),
            "areas": np.asarray([1.0, 2.0, 3.0]),
        }

        summary = history_geometry_summary([part])

        self.assertAlmostEqual(
            summary["synthetic_gt_to_history_angle_median_deg"],
            0.0,
        )
        self.assertAlmostEqual(summary["history_area_over_oracle_median"], 1.0)
        self.assertEqual(summary["history_area_within_10pct"], 1)
        self.assertEqual(summary["history_area_rank_median"], 1.0)


if __name__ == "__main__":
    unittest.main()
