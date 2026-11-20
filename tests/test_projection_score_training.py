import json
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.train_projection_score_mlp import (
    CandidateMLP,
    choose_development_project,
    ndcg_at_k,
    rank_fusion,
    score_from_lower_is_better,
    zero_last_linear_layer,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "labels/projection_support_area_v0_256"


class ProjectionScoreTrainingTests(unittest.TestCase):
    def test_lower_surface_cost_maps_to_high_score(self):
        scores = score_from_lower_is_better(np.array([4.0, 2.0, 2.0]))
        self.assertTrue(np.allclose(scores, [0.0, 1.0, 1.0]))

    def test_constant_cost_is_represented_by_tied_best_scores(self):
        scores = score_from_lower_is_better(np.full(5, 3.0))
        self.assertTrue(np.allclose(scores, 1.0))

    def test_ndcg_rewards_oracle_order(self):
        targets = np.array([1.0, 0.5, 0.1, 0.0])
        self.assertAlmostEqual(ndcg_at_k(targets, targets, 2), 1.0)
        self.assertLess(ndcg_at_k(targets, -targets, 2), 1.0)

    def test_development_project_is_selected_as_a_whole(self):
        records = [
            {"source_project": "large", "source_part_id": i}
            for i in range(20)
        ] + [
            {"source_project": "small", "source_part_id": i}
            for i in range(3)
        ] + [
            {"source_project": "middle", "source_part_id": i}
            for i in range(11)
        ]
        self.assertEqual(choose_development_project(records), "middle")

    def test_anchor_initialisation_starts_from_zero_residual(self):
        torch.manual_seed(1)
        model = CandidateMLP(5, hidden_dim=8)
        zero_last_linear_layer(model)
        output = model(torch.randn(3, 5))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_rank_fusion_prefers_anchor_when_anchor_weight_dominates(self):
        prediction = np.array([4.0, 3.0, 2.0, 1.0])
        anchor = np.array([0.0, 1.0, 0.5, 0.25])
        fused = rank_fusion(prediction, anchor, anchor_weight=100.0)
        self.assertEqual(int(np.argmax(fused)), int(np.argmax(anchor)))

    def test_dataset_features_do_not_contain_target_fields(self):
        manifest = json.loads((DATASET / "manifest.json").read_text())
        forbidden = {
            "projected_union",
            "projected_area",
            "projected_area_mm2",
            "score",
            "rank",
        }
        self.assertTrue(forbidden.isdisjoint(manifest["feature_names"]))

    def test_training_development_and_test_projects_are_disjoint(self):
        split = json.loads((DATASET / "split.json").read_text())
        train_projects = {
            record["source_project"] for record in split["train_records"]
        }
        test_projects = set(split["validation_projects"])
        development_project = choose_development_project(split["train_records"])
        self.assertIn(development_project, train_projects)
        self.assertTrue(train_projects.isdisjoint(test_projects))


if __name__ == "__main__":
    unittest.main()
