import unittest

import numpy as np

from tools.generate_simplified_supports import (
    annotation_masks,
    classify_overhang_faces,
    evaluate_supports,
    process_config,
    rotation_align_to_z,
)
from tools.recognize_part_features import (
    merge_annotations,
    reclassify_features_for_build_direction,
    recognize_features,
    resolve_support_query,
    segment_segment_distance,
)
from tools.search_support_orientations import ranking_key


def suspended_plate_mesh():
    """Two open rectangular surfaces: an underside at z=4 over a deck at z=1."""
    vertices = np.array([
        [0, 0, 4], [2, 0, 4], [2, 2, 4], [0, 2, 4],
        [0, 0, 1], [2, 0, 1], [2, 2, 1], [0, 2, 1],
    ], dtype=float)
    # Upper triangles wind downward; lower triangles wind upward.
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
    ], dtype=int)
    return vertices, faces


def horizontal_cylinder_mesh(radius=3.0, length=10.0, sides=64, internal=True):
    """Open cylinder side wall with axis along input X."""
    vertices = []
    for x in (-length / 2.0, length / 2.0):
        for side in range(sides):
            angle = 2.0 * np.pi * side / sides
            vertices.append([x, radius * np.cos(angle), radius * np.sin(angle)])
    faces = []
    for side in range(sides):
        next_side = (side + 1) % sides
        a, b = side, next_side
        c, d = sides + side, sides + next_side
        if internal:
            faces.extend([[a, c, b], [c, d, b]])
        else:
            faces.extend([[a, b, c], [c, b, d]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


class SimplifiedSupportTests(unittest.TestCase):
    def setUp(self):
        self.limits = {
            "self_support_angle_threshold_deg": 45.0,
            "horizontal_hole_diameter_threshold_mm": 6.0,
            "overhang_bridge_span_threshold_mm": 2.0,
            "overhang_boss_size_threshold_mm": 1.0,
        }

    def test_angle_is_measured_from_horizontal_and_includes_equal_boundary(self):
        angles = np.array([0.0, 25.0, 25.01, 40.0, 55.0, 89.0])
        normals = np.array([[0.0, np.sin(np.deg2rad(a)), -np.cos(np.deg2rad(a))] for a in angles])
        areas = np.ones(len(angles))
        selected, _, measured = classify_overhang_faces(normals, areas, 25.0)
        self.assertTrue(np.allclose(measured, angles))
        self.assertEqual(np.flatnonzero(selected).tolist(), [0, 1])

    def test_rule_values_are_loaded_per_material_and_mode(self):
        import json
        from pathlib import Path
        rules = json.loads((Path(__file__).resolve().parents[1] / "configs/support_rules_v0.json").read_text())
        self.assertEqual(process_config(rules, "TC4", "am_build")["self_support_angle_threshold_deg"], 25.0)
        self.assertEqual(process_config(rules, "316L", "standard")["self_support_angle_threshold_deg"], 55.0)

    def test_build_up_axis_rotation_maps_requested_direction_to_z(self):
        up = np.array([1.0, -2.0, 3.0])
        rotation = rotation_align_to_z(up)
        aligned = rotation @ (up / np.linalg.norm(up))
        self.assertTrue(np.allclose(aligned, [0.0, 0.0, 1.0], atol=1e-10))

    def test_infeasible_directions_rank_by_violations_before_support_cost(self):
        rows = [
            {
                "feasible": False,
                "violation_count": 6,
                "support_volume_mm3": 3000.0,
                "support_column_count": 300,
            },
            {
                "feasible": False,
                "violation_count": 93,
                "support_volume_mm3": 500.0,
                "support_column_count": 20,
            },
            {
                "feasible": True,
                "violation_count": 0,
                "support_volume_mm3": 9000.0,
                "support_column_count": 500,
            },
        ]
        ordered = sorted(rows, key=ranking_key)
        self.assertTrue(ordered[0]["feasible"])
        self.assertEqual(ordered[1]["violation_count"], 6)
        self.assertEqual(ordered[2]["violation_count"], 93)

    def test_feature_exemptions_use_strict_less_than_boundary(self):
        annotations = {"features": [
            {"kind": "horizontal_hole", "face_indices": [0], "diameter_mm": 5.99},
            {"kind": "bridge", "face_indices": [1], "span_mm": 2.0},
            {"kind": "boss", "face_indices": [2], "size_mm": 0.99},
        ]}
        no_support, exempt, _ = annotation_masks(3, annotations, self.limits)
        self.assertEqual(np.flatnonzero(no_support).tolist(), [])
        self.assertEqual(np.flatnonzero(exempt).tolist(), [0, 2])

    def test_support_columns_land_on_first_part_surface(self):
        vertices, faces = suspended_plate_mesh()
        result = evaluate_supports(vertices, faces, self.limits, sample_spacing_mm=0.8)
        report = result["report"]
        self.assertGreater(report["support_summary"]["count"], 0)
        self.assertEqual(report["support_summary"]["part_surface_landings"], report["support_summary"]["count"])
        self.assertTrue(all(abs(row["height_mm"] - 3.0) < 1e-7 for row in report["supports"]))
        self.assertEqual(report["hard_constraint_violations"], [])

    def test_forbidden_source_and_landing_faces_are_hard_violations(self):
        vertices, faces = suspended_plate_mesh()
        source_result = evaluate_supports(
            vertices, faces, self.limits,
            {"no_support_face_indices": [0]}, sample_spacing_mm=0.8,
        )
        self.assertIn("overhang_on_no_support_face", {
            row["type"] for row in source_result["report"]["hard_constraint_violations"]
        })
        landing_result = evaluate_supports(
            vertices, faces, self.limits,
            {"no_support_face_indices": [2, 3]}, sample_spacing_mm=0.8,
        )
        self.assertIn("support_landing_on_no_support_face", {
            row["type"] for row in landing_result["report"]["hard_constraint_violations"]
        })
        self.assertEqual(landing_result["report"]["support_summary"]["count"], 0)

    def test_small_tagged_hole_region_is_exempt_without_masking_neighbors(self):
        vertices, faces = suspended_plate_mesh()
        result = evaluate_supports(
            vertices, faces, self.limits,
            {"features": [{"kind": "horizontal_hole", "face_indices": [0], "diameter_mm": 5.0}]},
            sample_spacing_mm=0.8,
        )
        self.assertEqual(result["report"]["face_counts"]["feature_exempt"], 1)
        self.assertEqual(result["report"]["face_counts"]["support_candidate_after_exemptions"], 1)

    def test_internal_horizontal_cylinder_is_recognized_and_external_is_not(self):
        vertices, faces = horizontal_cylinder_mesh()
        recognized = recognize_features(vertices, faces, small_hole_diameter_mm=6.0)
        self.assertEqual(recognized["summary"]["recognized_internal_cylinders"], 1)
        feature = recognized["features"][0]
        self.assertAlmostEqual(feature["diameter_mm"], 6.0, delta=0.05)
        self.assertAlmostEqual(feature["axis_length_mm"], 10.0, delta=1e-8)
        self.assertTrue(feature["is_horizontal_to_build"])
        self.assertTrue(feature["small_horizontal_exempt"])
        self.assertEqual(feature["support_policy"], "size_exempt")
        self.assertEqual(recognized["suggested_annotation"]["no_support_face_indices"], [])
        self.assertAlmostEqual(abs(feature["axis_input"][0]), 1.0, delta=1e-8)

        external_vertices, external_faces = horizontal_cylinder_mesh(internal=False)
        external = recognize_features(external_vertices, external_faces)
        self.assertEqual(external["summary"]["recognized_internal_cylinders"], 0)

    def test_large_horizontal_hole_is_support_allowed_without_user_query(self):
        vertices, faces = horizontal_cylinder_mesh(radius=3.5)
        recognized = recognize_features(vertices, faces, small_hole_diameter_mm=6.0)
        feature = recognized["features"][0]
        self.assertFalse(feature["small_horizontal_exempt"])
        self.assertEqual(feature["support_policy"], "support_allowed")
        self.assertEqual(recognized["suggested_annotation"]["no_support_face_indices"], [])

    def test_user_query_turns_selected_recognized_hole_into_hard_constraint(self):
        recognized = {
            "features": [
                {
                    "feature_id": "hole_a", "kind": "internal_cylinder",
                    "face_indices": [2, 3], "is_horizontal_to_build": True,
                    "diameter_mm": 8.0,
                },
                {
                    "feature_id": "hole_b", "kind": "internal_cylinder",
                    "face_indices": [4, 5], "is_horizontal_to_build": False,
                    "diameter_mm": 4.0,
                },
            ],
        }
        query = {
            "query_text": "不要把支撑放进 A 孔",
            "no_support": {
                "feature_ids": ["hole_a"],
                "face_indices": [1],
                "clearance_mm": 0.3,
            },
        }
        resolved = resolve_support_query(recognized, query, 6)
        self.assertEqual(resolved["selected_feature_ids"], ["hole_a"])
        self.assertEqual(resolved["no_support_face_indices"], [1, 2, 3])
        self.assertEqual(resolved["clearance_mm"], 0.3)

    def test_feature_exemption_is_reclassified_for_new_build_direction(self):
        recognized = {
            "features": [{
                "feature_id": "hole_a",
                "axis_input": [1.0, 0.0, 0.0],
                "diameter_mm": 5.0,
                "face_indices": [1],
                "confidence": 0.9,
            }],
            "parameters": {},
            "summary": {},
        }
        horizontal = reclassify_features_for_build_direction(
            recognized, np.array([0.0, 0.0, 1.0]), 6.0
        )
        vertical = reclassify_features_for_build_direction(
            recognized, np.array([1.0, 0.0, 0.0]), 6.0
        )
        self.assertTrue(horizontal["features"][0]["small_horizontal_exempt"])
        self.assertFalse(vertical["features"][0]["small_horizontal_exempt"])
        self.assertEqual(vertical["features"][0]["support_policy"], "support_allowed")

    def test_manual_annotations_are_merged_with_recognized_holes(self):
        recognized = {
            "suggested_annotation": {
                "no_support_face_indices": [3, 4],
                "features": [{
                    "kind": "horizontal_hole",
                    "face_indices": [3, 4],
                    "diameter_mm": 4.0,
                    "source": "automatic_geometry",
                }],
            }
        }
        merged = merge_annotations(
            {
                "no_support_face_indices": [1],
                "features": [{"kind": "bridge", "face_indices": [1], "span_mm": 1.0}],
            },
            recognized,
            5,
        )
        self.assertEqual(merged["no_support_face_indices"], [1, 3, 4])
        self.assertEqual([feature["kind"] for feature in merged["features"]],
                         ["bridge", "horizontal_hole"])

    def test_support_shaft_is_rejected_when_it_intersects_recognized_cylinder(self):
        vertices, faces = suspended_plate_mesh()
        recognized = {
            "features": [{
                "feature_id": "test_cylinder",
                "kind": "internal_cylinder",
                "axis_input": [1.0, 0.0, 0.0],
                "axis_start_input_mm": [0.0, 1.0, 2.0],
                "axis_end_input_mm": [2.0, 1.0, 2.0],
                "radius_mm": 0.25,
            }],
            "suggested_annotation": {"no_support_face_indices": [], "features": []},
        }
        query = {
            "selected_feature_ids": ["test_cylinder"],
            "no_support_face_indices": [],
            "clearance_mm": 0.1,
        }
        result = evaluate_supports(
            vertices,
            faces,
            self.limits,
            sample_spacing_mm=0.5,
            recognized_features=recognized,
            support_query=query,
        )
        self.assertIn(
            "support_shaft_intersects_recognized_internal_cylinder",
            {row["type"] for row in result["report"]["hard_constraint_violations"]},
        )

    def test_recognized_cylinder_does_not_block_support_without_user_query(self):
        vertices, faces = suspended_plate_mesh()
        recognized = {
            "features": [{
                "feature_id": "test_cylinder",
                "kind": "internal_cylinder",
                "axis_input": [1.0, 0.0, 0.0],
                "axis_start_input_mm": [0.0, 1.0, 2.0],
                "axis_end_input_mm": [2.0, 1.0, 2.0],
                "radius_mm": 0.25,
            }],
            "suggested_annotation": {"no_support_face_indices": [], "features": []},
        }
        result = evaluate_supports(
            vertices,
            faces,
            self.limits,
            sample_spacing_mm=0.5,
            recognized_features=recognized,
        )
        self.assertEqual(result["report"]["face_counts"]["no_support_faces"], 0)
        self.assertNotIn(
            "support_shaft_intersects_recognized_internal_cylinder",
            {row["type"] for row in result["report"]["hard_constraint_violations"]},
        )

    def test_segment_distance_handles_crossing_and_nonintersection(self):
        self.assertAlmostEqual(
            segment_segment_distance(
                [0.0, 0.0, 0.0], [0.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            ),
            0.0,
        )
        self.assertAlmostEqual(
            segment_segment_distance(
                [0.0, 0.0, 0.0], [0.0, 0.0, 2.0],
                [-1.0, 0.5, 1.0], [1.0, 0.5, 1.0],
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
