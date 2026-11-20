import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from tools.train_pointnet_direction_scorer import (
    PointNetDirectionScorer,
    load_part_rows,
    make_model_inputs,
    stacked_feature_standardisation,
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
            "candidate_prior": np.empty((3, 0), dtype=np.float32),
            "global_features": np.empty(0, dtype=np.float32),
        }
        cloud, directions, global_features, targets = make_model_inputs(
            [part],
            point_count=4,
            prior_mean=np.empty(0, dtype=np.float32),
            prior_std=np.empty(0, dtype=np.float32),
            global_mean=np.empty(0, dtype=np.float32),
            global_std=np.empty(0, dtype=np.float32),
            device=torch.device("cpu"),
            augment=False,
            generator=None,
        )
        self.assertEqual(tuple(cloud.shape), (1, 4, 6))
        self.assertEqual(tuple(directions.shape), (1, 3, 3))
        self.assertEqual(tuple(global_features.shape), (1, 0))
        self.assertEqual(tuple(targets.shape), (1, 3))
        np.testing.assert_allclose(
            torch.linalg.vector_norm(directions, dim=-1).numpy(),
            1.0,
            atol=1e-6,
        )

    def test_forward_uses_prior_features_and_global_shape_context(self) -> None:
        model = PointNetDirectionScorer(
            point_channels=6,
            point_embedding_dim=64,
            shape_hidden_dim=32,
            direction_hidden_dim=24,
            candidate_feature_dim=5,
            global_feature_dim=4,
            dropout=0.0,
        ).eval()
        points = torch.randn(2, 32, 6)
        candidates = torch.randn(2, 17, 8)
        global_features = torch.randn(2, 4)
        with torch.no_grad():
            scores = model(points, candidates, global_features)
        self.assertEqual(tuple(scores.shape), (2, 17))
        self.assertTrue(torch.isfinite(scores).all())

    def test_load_part_rows_joins_30deg_sidecar_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directions = np.asarray([
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, -3.0],
            ], dtype=np.float32)
            label_path = root / "label.npz"
            np.savez(
                label_path,
                points=np.arange(4 * 3, dtype=np.float32).reshape(4, 3),
                normals=np.tile(
                    np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
                    (4, 1),
                ),
                directions=directions,
                projected_area_mm2=np.asarray([3.0, 2.0, 1.0]),
                score=np.asarray([0.0, 0.5, 1.0]),
                retained_surface_mm2=np.asarray([6.0, 4.0, 2.0]),
            )
            geometry_path = root / "geometry.npz"
            np.savez(
                geometry_path,
                directions=directions,
                candidate_features=np.arange(
                    3 * 4,
                    dtype=np.float32,
                ).reshape(3, 4),
                global_features=np.asarray(
                    [1.0, 2.0, 3.0],
                    dtype=np.float32,
                ),
                projected_area_mm2_30=np.asarray([4.0, 2.0, 0.0]),
            )
            parts = load_part_rows(
                [{
                    "job_id": "shape_test",
                    "label_file": str(label_path),
                    "mesh_hash_sha256": "a" * 64,
                    "source_project": "project",
                    "source_part_id": 1,
                }],
                point_count=4,
                geometry_feature_records={
                    "shape_test": {
                        "geometry_feature_file": str(geometry_path),
                    }
                },
            )

        part = parts[0]
        self.assertEqual(part["candidate_prior"].shape, (3, 4))
        self.assertEqual(part["global_features"].shape, (3,))
        self.assertEqual(part["projected_area_30"].shape, (3,))
        np.testing.assert_allclose(
            np.linalg.norm(part["directions"], axis=1),
            1.0,
            atol=1e-6,
        )
        cloud, candidates, global_features, targets = make_model_inputs(
            [part],
            point_count=4,
            prior_mean=np.zeros(4, dtype=np.float32),
            prior_std=np.ones(4, dtype=np.float32),
            global_mean=np.zeros(3, dtype=np.float32),
            global_std=np.ones(3, dtype=np.float32),
            device=torch.device("cpu"),
            augment=False,
            generator=None,
        )
        self.assertEqual(tuple(cloud.shape), (1, 4, 6))
        self.assertEqual(tuple(candidates.shape), (1, 3, 7))
        self.assertEqual(tuple(global_features.shape), (1, 3))
        self.assertEqual(tuple(targets.shape), (1, 3))

    def test_standardisation_handles_candidate_and_global_shapes(self) -> None:
        parts = [
            {
                "candidate_prior": np.asarray(
                    [[1.0, 2.0], [3.0, 4.0]],
                    dtype=np.float32,
                ),
                "global_features": np.asarray([5.0, 9.0]),
            },
            {
                "candidate_prior": np.asarray(
                    [[5.0, 6.0], [7.0, 8.0]],
                    dtype=np.float32,
                ),
                "global_features": np.asarray([7.0, 11.0]),
            },
        ]
        candidate_mean, candidate_std = stacked_feature_standardisation(
            parts,
            "candidate_prior",
        )
        global_mean, global_std = stacked_feature_standardisation(
            parts,
            "global_features",
        )
        np.testing.assert_allclose(candidate_mean, [4.0, 5.0])
        np.testing.assert_allclose(candidate_std, np.sqrt([5.0, 5.0]))
        np.testing.assert_allclose(global_mean, [6.0, 10.0])
        np.testing.assert_allclose(global_std, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
