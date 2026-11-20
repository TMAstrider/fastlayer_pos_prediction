import unittest

import numpy as np

from tools.evaluate_geometry_prior_fusion import (
    fuse_predictions,
    fusion_selection_score,
)


class GeometryPriorFusionTests(unittest.TestCase):
    def test_fusion_promotes_lower_30deg_area(self) -> None:
        prediction = np.asarray([0.9, 0.7, 0.2])
        part = {
            "projected_area_30": np.asarray([10.0, 1.0, 5.0]),
        }
        no_fusion = fuse_predictions([prediction], [part], 0.0)[0]
        with_fusion = fuse_predictions([prediction], [part], 2.0)[0]
        self.assertEqual(int(np.argmax(no_fusion)), 0)
        self.assertEqual(int(np.argmax(with_fusion)), 1)

    def test_selection_score_penalizes_top1_regret(self) -> None:
        strong = {
            "spearman_median": 0.8,
            "ndcg_at_5_median": 0.9,
            "top1_regret_median": 0.01,
        }
        weak = {**strong, "top1_regret_median": 0.09}
        self.assertGreater(
            fusion_selection_score(strong),
            fusion_selection_score(weak),
        )


if __name__ == "__main__":
    unittest.main()
