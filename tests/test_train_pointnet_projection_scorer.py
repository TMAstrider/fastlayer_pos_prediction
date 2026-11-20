import unittest

import numpy as np
import torch

from tools.train_pointnet_projection_scorer import (
    PointNetCandidateScorer,
    normalise_cloud,
    random_rotations,
)


class PointNetProjectionScorerTests(unittest.TestCase):
    def test_forward_returns_one_score_per_candidate(self) -> None:
        model = PointNetCandidateScorer(
            feature_dim=18,
            point_channels=6,
            point_embedding_dim=128,
            shape_hidden_dim=32,
            candidate_hidden_dim=32,
            dropout=0.0,
        ).eval()
        points = torch.randn(2, 32, 6)
        features = torch.randn(2, 17, 18)
        with torch.no_grad():
            scores = model(points, features)
        self.assertEqual(tuple(scores.shape), (2, 17))
        self.assertTrue(torch.isfinite(scores).all())

    def test_cloud_normalisation_centres_and_scales_points(self) -> None:
        points = np.asarray([
            [10.0, 20.0, 30.0],
            [11.0, 20.0, 30.0],
            [10.0, 22.0, 30.0],
        ], dtype=np.float32)
        normals = np.asarray([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ], dtype=np.float32)
        centered, normalized_normals = normalise_cloud(points, normals)
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-6)
        self.assertAlmostEqual(
            float(np.linalg.norm(centered, axis=1).max()),
            1.0,
            places=5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(normalized_normals, axis=1),
            1.0,
            atol=1e-6,
        )

    def test_random_rotations_are_orthonormal(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(7)
        rotations = random_rotations(8, generator, torch.device("cpu"))
        identity = torch.eye(3).expand(8, -1, -1)
        products = torch.bmm(rotations, rotations.transpose(1, 2))
        self.assertTrue(torch.allclose(products, identity, atol=1e-5))
        determinants = torch.linalg.det(rotations)
        self.assertTrue(torch.allclose(determinants, torch.ones(8), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
