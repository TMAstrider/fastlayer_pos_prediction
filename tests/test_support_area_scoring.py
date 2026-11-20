import unittest

import numpy as np

from tools.support_area_scoring import (
    dense_area_ranks,
    evaluate_support_projection,
    normalise_projected_areas,
    projected_triangle_area_sum,
    rasterize_projected_union,
)


def limits():
    return {
        "self_support_angle_threshold_deg": 45.0,
        "horizontal_hole_diameter_threshold_mm": 6.0,
        "overhang_bridge_span_threshold_mm": 2.0,
        "overhang_boss_size_threshold_mm": 1.0,
    }


def suspended_plate_mesh():
    vertices = np.array([
        [0, 0, 4], [2, 0, 4], [2, 2, 4], [0, 2, 4],
        [0, 0, 1], [2, 0, 1], [2, 2, 1], [0, 2, 1],
    ], dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
    ], dtype=int)
    return vertices, faces


class SupportAreaScoringTests(unittest.TestCase):
    def test_projected_union_deduplicates_faces_at_different_heights(self):
        triangles = np.array([
            [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
            [[0.0, 0.0, 4.0], [2.0, 0.0, 4.0], [0.0, 2.0, 4.0]],
        ])
        indices = np.array([0, 1])
        area_sum = projected_triangle_area_sum(triangles, indices)
        projected = rasterize_projected_union(
            triangles, indices, cell_size_mm=0.1
        )
        self.assertAlmostEqual(area_sum, 4.0, places=8)
        self.assertAlmostEqual(projected.area_mm2, 2.0, delta=0.2)

    def test_projected_area_applies_triangle_orientation(self):
        triangles = np.array([
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        ])
        self.assertAlmostEqual(
            projected_triangle_area_sum(triangles, np.array([0])),
            2.0,
        )

    def test_small_feature_exemption_reduces_projected_union(self):
        vertices, faces = suspended_plate_mesh()
        without = evaluate_support_projection(
            vertices, faces, limits(), projection_grid_mm=0.1
        )
        with_exemption = evaluate_support_projection(
            vertices,
            faces,
            limits(),
            annotations={
                "features": [{
                    "kind": "horizontal_hole",
                    "face_indices": [0],
                    "diameter_mm": 5.0,
                }]
            },
            projection_grid_mm=0.1,
        )
        self.assertAlmostEqual(
            without["report"]["areas_mm2"]["projected_union"],
            4.0,
            delta=0.2,
        )
        self.assertAlmostEqual(
            with_exemption["report"]["areas_mm2"]["projected_union"],
            2.0,
            delta=0.2,
        )

    def test_score_is_higher_for_lower_area_and_ties_share_rank(self):
        scores, minimum, maximum = normalise_projected_areas(
            np.array([10.0, 20.0, 20.0, 30.0])
        )
        self.assertEqual(minimum, 10.0)
        self.assertEqual(maximum, 30.0)
        self.assertTrue(np.allclose(scores, [1.0, 0.5, 0.5, 0.0]))
        self.assertEqual(
            dense_area_ranks(np.array([10.0, 20.0, 20.0, 30.0])).tolist(),
            [1, 2, 2, 3],
        )

if __name__ == "__main__":
    unittest.main()
