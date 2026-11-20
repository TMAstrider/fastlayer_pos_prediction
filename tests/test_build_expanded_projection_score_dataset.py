import json
import tempfile
import unittest
from pathlib import Path

from tools.build_expanded_projection_score_dataset import build_jobs


class ExpandedProjectionDatasetTests(unittest.TestCase):
    def test_jobs_preserve_existing_shape_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shapes = [
                {
                    "geometry_sha256": "a" * 64,
                    "shape_group_id": "shape_a",
                    "pointcloud_file": "shape_a.npz",
                    "source_stl_file": "outputs/shape_a.stl",
                    "source_project": "project-a",
                    "source_record_id": "magics:" + "a" * 64,
                    "triangle_count": 10,
                },
                {
                    "geometry_sha256": "b" * 64,
                    "shape_group_id": "shape_b",
                    "pointcloud_file": "shape_b.npz",
                    "source_stl_file": "outputs/shape_b.stl",
                    "source_project": "project-b",
                    "source_record_id": "magics:" + "b" * 64,
                    "triangle_count": 20,
                },
            ]
            shapes_path = root / "shapes.json"
            shapes_path.write_text(
                json.dumps({"records": shapes}),
                encoding="utf-8",
            )
            split = {
                "train_records": [{"geometry_sha256": "a" * 64}],
                "validation_records": [{"geometry_sha256": "b" * 64}],
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            normals_dir = root / "normals"
            normals_dir.mkdir()
            (root / "outputs").mkdir()
            (root / "outputs/shape_a.stl").touch()
            (root / "outputs/shape_b.stl").touch()
            (normals_dir / "shape_a.npz").touch()
            (normals_dir / "shape_b.npz").touch()

            jobs = build_jobs(
                root,
                shapes_path,
                normals_dir,
                split_path,
            )

        self.assertEqual([job["split"] for job in jobs], ["train", "validation"])
        self.assertEqual(jobs[0]["source_part_id"], 0)
        self.assertEqual(jobs[1]["mesh_hash_sha256"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
