import unittest

import numpy as np
import torch

from tools.train_pointnet_direction_scorer import (
    PointNetDirectionScorer,
    make_model_inputs,
)


class PointNetDirectionScorerTests(unittest.TestCase):
    def test_forward_scores_each_direction_from_xyz(self) -> None:
        model = PointNetDirectionScorer(
            point_channels=6,
            point_embedding_dim=64,
            shape_hidden_dim=32,
            direction_hidden_dim=24,
            dropout=0.0,
        ).eval()
        points = torch.randn(2, 32, 6)
        directions = torch.randn(2, 17, 3)
        with torch.no_grad():
            scores = model(points, directions)
        self.assertEqual(tuple(scores.shape), (2, 17))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fixed_inputs_contain_only_points_normals_and_directions(self) -> None:
        points = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        part = {
            "points": points,
            "normals": np.eye(3, dtype=np.float32)[[0, 1, 2, 0]],
            "directions": np.asarray([
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, -3.0],
            ], dtype=np.float32),
            "targets": np.asarray([0.2, 0.8, 0.5], dtype=np.float32),
        }
        cloud, directions, targets = make_model_inputs(
            [part],
            point_count=4,
            device=torch.device("cpu"),
            augment=False,
            generator=None,
        )
        self.assertEqual(tuple(cloud.shape), (1, 4, 6))
        self.assertEqual(tuple(directions.shape), (1, 3, 3))
        self.assertEqual(tuple(targets.shape), (1, 3))
        np.testing.assert_allclose(
            torch.linalg.vector_norm(directions, dim=-1).numpy(),
            1.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
