import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.build_30deg_geometry_features import (
    CANDIDATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    compute_candidate_features,
    compute_global_features,
)


def tetrahedron():
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    faces = np.asarray([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ])
    return vertices, faces


class ThirtyDegreeGeometryFeatureTests(unittest.TestCase):
    def test_feature_shapes_and_values_are_finite(self) -> None:
        vertices, faces = tetrahedron()
        directions = np.asarray([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        global_features = compute_global_features(vertices, faces)
        candidate_features, projected_area = compute_candidate_features(
            vertices,
            faces,
            directions,
            global_features,
            30.0,
        )
        self.assertEqual(
            global_features.shape,
            (len(GLOBAL_FEATURE_NAMES),),
        )
        self.assertEqual(
            candidate_features.shape,
            (len(directions), len(CANDIDATE_FEATURE_NAMES)),
        )
        self.assertEqual(projected_area.shape, (len(directions),))
        self.assertTrue(np.isfinite(global_features).all())
        self.assertTrue(np.isfinite(candidate_features).all())
        self.assertTrue((projected_area >= 0.0).all())

    def test_features_are_invariant_to_simultaneous_global_rotation(self) -> None:
        vertices, faces = tetrahedron()
        directions = np.asarray([
            [0.2, -0.3, 0.9],
            [-0.8, 0.1, 0.4],
            [0.5, 0.7, -0.2],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        rotation = Rotation.from_quat(
            [0.2, -0.3, 0.4, 0.8]
        ).as_matrix()

        original_global = compute_global_features(vertices, faces)
        rotated_global = compute_global_features(
            vertices @ rotation.T,
            faces,
        )
        original_candidates, original_area = compute_candidate_features(
            vertices,
            faces,
            directions,
            original_global,
            30.0,
        )
        rotated_candidates, rotated_area = compute_candidate_features(
            vertices @ rotation.T,
            faces,
            directions @ rotation.T,
            rotated_global,
            30.0,
        )
        np.testing.assert_allclose(
            rotated_global,
            original_global,
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            rotated_candidates,
            original_candidates,
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            rotated_area,
            original_area,
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
