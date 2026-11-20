import unittest

from tools.build_project_isolated_split import build_split


class ProjectIsolatedSplitTests(unittest.TestCase):
    def test_split_holds_out_entire_projects(self) -> None:
        records = []
        for project, count in {
            "a": 5,
            "b": 5,
            "c": 4,
            "d": 6,
        }.items():
            for index in range(count):
                geometry = f"{project}{index:02d}"
                records.append({
                    "job_id": geometry,
                    "mesh_hash_sha256": geometry,
                    "source_project": project,
                })

        split = build_split(records, target_parts=10, seed=30)

        train_projects = {
            record["source_project"]
            for record in split["train_records"]
        }
        validation_projects = {
            record["source_project"]
            for record in split["validation_records"]
        }
        self.assertEqual(split["validation_count"], 10)
        self.assertTrue(train_projects.isdisjoint(validation_projects))


if __name__ == "__main__":
    unittest.main()
