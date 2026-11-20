#!/usr/bin/env python3
"""Select a network/30-degree rank-fusion weight on development parts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.train_pointnet_direction_scorer import (
        PointNetDirectionScorer,
        choose_development_project,
        enriched_rows,
        load_geometry_feature_records,
        load_part_rows,
        predict_parts,
        stacked_feature_standardisation,
        summarise_rows,
        write_csv,
    )
    from tools.train_pointnet_projection_scorer import evaluate_parts
    from tools.train_projection_score_mlp import (
        rank_fusion,
        score_from_lower_is_better,
    )
except ModuleNotFoundError:
    from train_pointnet_direction_scorer import (
        PointNetDirectionScorer,
        choose_development_project,
        enriched_rows,
        load_geometry_feature_records,
        load_part_rows,
        predict_parts,
        stacked_feature_standardisation,
        summarise_rows,
        write_csv,
    )
    from train_pointnet_projection_scorer import evaluate_parts
    from train_projection_score_mlp import (
        rank_fusion,
        score_from_lower_is_better,
    )


def fusion_selection_score(metrics: dict[str, Any]) -> float:
    """Favor top-five quality while retaining global ranking robustness."""
    return (
        0.20 * metrics["spearman_median"]
        + 0.50 * metrics["ndcg_at_5_median"]
        + 0.30 * (
            1.0 - min(metrics["top1_regret_median"] * 20.0, 1.0)
        )
    )


def fuse_predictions(
    predictions: list[np.ndarray],
    parts: list[dict[str, Any]],
    weight: float,
) -> list[np.ndarray]:
    fused = []
    for prediction, part in zip(predictions, parts):
        anchor = score_from_lower_is_better(part["projected_area_30"])
        fused.append(rank_fusion(prediction, anchor, weight))
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256"
        ),
    )
    parser.add_argument(
        "--geometry-features",
        type=Path,
        default=Path(
            "labels/geometry_prior_30deg_expanded_932_100_v1"
        ),
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=Path(
            "runs/pointnet_geometry_prior_30deg_4096_expanded_932_100_2026-11-20"
        ),
    )
    parser.add_argument(
        "--name",
        default=(
            "pointnet_geometry_prior_30deg_4096_fusion_"
            "expanded_932_100_2026-11-20"
        ),
    )
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
            5.0,
        ],
    )
    args = parser.parse_args()
    if any(weight < 0 for weight in args.weights):
        parser.error("fusion weights must be non-negative")

    root = Path(__file__).resolve().parents[1]
    dataset_dir = (root / args.dataset).resolve()
    geometry_dir = (root / args.geometry_features).resolve()
    run_dir = (root / args.run).resolve()
    report_dir = root / "reports" / args.name
    if report_dir.exists():
        parser.error(f"Existing report directory: {report_dir}")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    split = json.loads((dataset_dir / "split.json").read_text(encoding="utf-8"))
    development_project = choose_development_project(
        split["train_records"]
    )
    fit_records = [
        record
        for record in split["train_records"]
        if record["source_project"] != development_project
    ]
    development_records = [
        record
        for record in split["train_records"]
        if record["source_project"] == development_project
    ]
    test_records = split["validation_records"]
    geometry_records, _, _ = load_geometry_feature_records(geometry_dir)
    fit_parts = load_part_rows(
        fit_records,
        args.point_count,
        geometry_records,
    )
    development_parts = load_part_rows(
        development_records,
        args.point_count,
        geometry_records,
    )
    test_parts = load_part_rows(
        test_records,
        args.point_count,
        geometry_records,
    )
    prior_mean, prior_std = stacked_feature_standardisation(
        fit_parts,
        "candidate_prior",
    )
    global_mean, global_std = stacked_feature_standardisation(
        fit_parts,
        "global_features",
    )

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    model = PointNetDirectionScorer(
        point_channels=6,
        point_embedding_dim=config["point_embedding_dim"],
        shape_hidden_dim=config["shape_hidden_dim"],
        direction_hidden_dim=config["direction_hidden_dim"],
        candidate_feature_dim=len(prior_mean),
        global_feature_dim=len(global_mean),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    development_predictions = predict_parts(
        model,
        development_parts,
        args.point_count,
        prior_mean,
        prior_std,
        global_mean,
        global_std,
        device,
    )
    test_predictions = predict_parts(
        model,
        test_parts,
        args.point_count,
        prior_mean,
        prior_std,
        global_mean,
        global_std,
        device,
    )

    trials = []
    selected = None
    for weight in args.weights:
        development_metrics = summarise_rows(enriched_rows(
            fuse_predictions(
                development_predictions,
                development_parts,
                weight,
            ),
            development_parts,
        ))
        test_metrics, _ = evaluate_parts(
            test_parts,
            fuse_predictions(test_predictions, test_parts, weight),
        )
        selection_score = fusion_selection_score(development_metrics)
        trial = {
            "weight": float(weight),
            "development": development_metrics,
            "test": test_metrics,
            "selection_score": selection_score,
        }
        trials.append(trial)
        if selected is None or selection_score > selected["selection_score"]:
            selected = trial

    if selected is None:
        raise RuntimeError("No fusion weight was evaluated")
    selected_predictions = fuse_predictions(
        test_predictions,
        test_parts,
        selected["weight"],
    )
    selected_metrics, selected_rows = evaluate_parts(
        test_parts,
        selected_predictions,
    )
    direct30_predictions = [
        score_from_lower_is_better(part["projected_area_30"])
        for part in test_parts
    ]
    direct30_metrics, _ = evaluate_parts(
        test_parts,
        direct30_predictions,
    )

    report_dir.mkdir(parents=True)
    write_csv(report_dir / "per_part.csv", selected_rows)
    document = {
        "name": args.name,
        "dataset": str(dataset_dir),
        "geometry_features": str(geometry_dir),
        "run": str(run_dir),
        "development_project": development_project,
        "selection_rule": (
            "0.20*spearman_median + 0.50*ndcg_at_5_median + "
            "0.30*(1-min(top1_regret_median*20,1))"
        ),
        "selected_weight": selected["weight"],
        "selected_test": selected_metrics,
        "direct_30deg_test": direct30_metrics,
        "trials": trials,
        "limitations": [
            "The fusion weight is selected on only 10 development parts.",
            "The 100-part test contains substantial source-project overlap with training.",
            "The 30-degree proxy and 45-degree target are related by construction.",
        ],
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "name": args.name,
        "selected_weight": selected["weight"],
        "test": selected_metrics,
        "direct_30deg": direct30_metrics,
        "report": str(report_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
