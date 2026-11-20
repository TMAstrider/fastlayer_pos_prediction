#!/usr/bin/env python3
"""Evaluate residual strengths around a fixed threshold anchor."""
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
        load_geometry_feature_records,
        load_part_rows,
        predict_parts,
        stacked_feature_standardisation,
    )
    from tools.train_pointnet_projection_scorer import evaluate_parts
except ModuleNotFoundError:
    from train_pointnet_direction_scorer import (
        PointNetDirectionScorer,
        choose_development_project,
        load_geometry_feature_records,
        load_part_rows,
        predict_parts,
        stacked_feature_standardisation,
    )
    from train_pointnet_projection_scorer import evaluate_parts


def apply_residual_scale(
    residuals: list[np.ndarray],
    parts: list[dict[str, Any]],
    scale: float,
) -> list[np.ndarray]:
    predictions = []
    for residual, part in zip(residuals, parts):
        predictions.append(
            scale * np.asarray(residual)
            + np.asarray(part["anchor_scores"])
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--geometry-features", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--residual-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0],
    )
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_dir = (root / args.dataset).resolve()
    split_path = (root / args.split_file).resolve()
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
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development_project = choose_development_project(
        split["train_records"]
    )
    fit_records = [
        record for record in split["train_records"]
        if record["source_project"] != development_project
    ]
    development_records = [
        record for record in split["train_records"]
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

    development_residuals = predict_parts(
        model,
        development_parts,
        args.point_count,
        prior_mean,
        prior_std,
        global_mean,
        global_std,
        device,
        anchor_weight=0.0,
        residual_scale=1.0,
    )
    test_residuals = predict_parts(
        model,
        test_parts,
        args.point_count,
        prior_mean,
        prior_std,
        global_mean,
        global_std,
        device,
        anchor_weight=0.0,
        residual_scale=1.0,
    )

    trials = []
    for scale in args.residual_scales:
        development_metrics, _ = evaluate_parts(
            development_parts,
            apply_residual_scale(
                development_residuals,
                development_parts,
                scale,
            ),
        )
        test_metrics, _ = evaluate_parts(
            test_parts,
            apply_residual_scale(
                test_residuals,
                test_parts,
                scale,
            ),
        )
        trials.append({
            "scale": scale,
            "development": development_metrics,
            "test": test_metrics,
        })

    report_dir.mkdir(parents=True)
    output = {
        "name": args.name,
        "run": str(run_dir),
        "dataset": str(dataset_dir),
        "split_file": str(split_path),
        "development_project": development_project,
        "trials": trials,
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "name": args.name,
        "report": str(report_dir),
        "scales": {
            f"{trial['scale']:.2f}": {
                "test_spearman_median": trial["test"]["spearman_median"],
                "test_ndcg5_median": trial["test"]["ndcg_at_5_median"],
                "test_top5_exact": trial["test"]["top5_exact_count"],
                "test_top5_within_5pct": trial["test"]["top5_within_5pct_count"],
                "test_top10_exact": trial["test"]["top10_exact_count"],
                "test_top10_within_5pct": trial["test"]["top10_within_5pct_count"],
            }
            for trial in trials
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
