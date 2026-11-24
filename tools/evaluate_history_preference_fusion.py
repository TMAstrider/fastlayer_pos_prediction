#!/usr/bin/env python3
"""Evaluate area/history score fusion against stored historical build axes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.train_history_preference_head import (
        choose_development_project,
        load_part_rows,
        predict_dual,
    )
    from tools.train_pointnet_direction_scorer import (
        PointDirectionInteractionScorer,
    )
except ModuleNotFoundError:
    from train_history_preference_head import (
        choose_development_project,
        load_part_rows,
        predict_dual,
    )
    from train_pointnet_direction_scorer import (
        PointDirectionInteractionScorer,
    )


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + 1e-12)


def history_geometry_summary(
    parts: list[dict[str, Any]],
) -> dict[str, float | int]:
    angles = []
    grid_angles = []
    area_ratios = []
    area_ranks = []
    for part in parts:
        directions = np.asarray(part["directions"], dtype=np.float64)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True),
            1e-12,
        )
        history = np.asarray(
            part["history_direction"],
            dtype=np.float64,
        )
        history /= np.linalg.norm(history)
        cosine = np.clip(directions @ history, -1.0, 1.0)
        nearest_index = int(np.argmax(cosine))
        grid_angles.append(float(np.degrees(np.arccos(cosine[nearest_index]))))
        oracle_index = int(np.argmax(part["targets"]))
        oracle_direction = directions[oracle_index]
        angles.append(float(np.degrees(np.arccos(np.clip(
            oracle_direction @ history,
            -1.0,
            1.0,
        )))))
        oracle_area = float(part["areas"][oracle_index])
        history_area = float(part["areas"][nearest_index])
        if oracle_area > 1e-9:
            area_ratios.append(history_area / oracle_area)
        area_ranks.append(
            1 + int(np.sum(
                np.asarray(part["areas"]) < history_area - 1e-9
            ))
        )
    angles_array = np.asarray(angles)
    ratios = np.asarray(area_ratios)
    ranks = np.asarray(area_ranks)
    return {
        "count": len(parts),
        "history_axis_to_nearest_grid_angle_median_deg": float(
            np.median(grid_angles)
        ),
        "synthetic_gt_to_history_angle_mean_deg": float(angles_array.mean()),
        "synthetic_gt_to_history_angle_median_deg": float(
            np.median(angles_array)
        ),
        "synthetic_gt_to_history_angle_p90_deg": float(
            np.percentile(angles_array, 90)
        ),
        "synthetic_gt_within_20deg": int((angles_array <= 20.0).sum()),
        "synthetic_gt_within_45deg": int((angles_array <= 45.0).sum()),
        "history_area_over_oracle_median": float(np.median(ratios)),
        "history_area_over_oracle_p90": float(np.percentile(ratios, 90)),
        "history_area_within_5pct": int((ratios <= 1.05).sum()),
        "history_area_within_10pct": int((ratios <= 1.10).sum()),
        "history_area_rank_median": float(np.median(ranks)),
    }


def evaluate_fusion(
    parts: list[dict[str, Any]],
    predictions: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    preference: str,
) -> dict[str, float | int]:
    top1_angles = []
    top5_angles = []
    area_ratios = []
    exact_count = 0
    for part, (area_scores, history_logits) in zip(parts, predictions):
        directions = np.asarray(part["directions"], dtype=np.float64)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True),
            1e-12,
        )
        history = np.asarray(
            part["history_direction"],
            dtype=np.float64,
        )
        history /= np.linalg.norm(history)
        if preference == "fixed":
            history_scores = directions @ history
        elif preference == "learned":
            history_scores = history_logits[:len(directions)]
        else:
            raise ValueError(f"Unknown preference source: {preference}")
        fused = zscore(area_scores) + alpha * zscore(history_scores)
        order = np.argsort(-fused)
        angles = np.degrees(np.arccos(np.clip(
            directions[order] @ history,
            -1.0,
            1.0,
        )))
        top1_angles.append(float(angles[0]))
        top5_angles.append(float(angles[:5].min()))
        oracle_area = float(part["areas"].min())
        selected_area = float(part["areas"][order[0]])
        if abs(selected_area - oracle_area) <= 1e-9:
            exact_count += 1
        if oracle_area > 1e-9:
            area_ratios.append(selected_area / oracle_area)

    top1 = np.asarray(top1_angles)
    top5 = np.asarray(top5_angles)
    ratios = np.asarray(area_ratios)
    return {
        "alpha": alpha,
        "preference": preference,
        "count": len(parts),
        "history_top1_angle_mean_deg": float(top1.mean()),
        "history_top1_angle_median_deg": float(np.median(top1)),
        "history_top1_within_10deg": int((top1 <= 10.0).sum()),
        "history_top1_within_20deg": int((top1 <= 20.0).sum()),
        "history_top1_within_45deg": int((top1 <= 45.0).sum()),
        "history_top5_min_angle_mean_deg": float(top5.mean()),
        "history_top5_min_angle_median_deg": float(np.median(top5)),
        "history_top5_within_10deg": int((top5 <= 10.0).sum()),
        "history_top5_within_20deg": int((top5 <= 20.0).sum()),
        "history_top5_within_45deg": int((top5 <= 45.0).sum()),
        "area_ratio_median": float(np.median(ratios)),
        "area_ratio_p90": float(np.percentile(ratios, 90)),
        "area_within_5pct": int((ratios <= 1.05).sum()),
        "area_within_10pct": int((ratios <= 1.10).sum()),
        "exact_oracle_count": exact_count,
    }


def markdown_table(rows: list[dict[str, float | int]]) -> str:
    lines = [
        "| 偏好来源 | alpha | 历史Top-1角度中位数 | Top-1 ≤20° | "
        "历史Top-5最小角中位数 | Top-5 ≤20° | 面积/oracle中位数 | "
        "面积≤10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['preference']} | {row['alpha']:g} | "
            f"{row['history_top1_angle_median_deg']:.2f}° | "
            f"{row['history_top1_within_20deg']}/{row['count']} | "
            f"{row['history_top5_min_angle_median_deg']:.2f}° | "
            f"{row['history_top5_within_20deg']}/{row['count']} | "
            f"{row['area_ratio_median']:.3f} | "
            f"{row['area_within_10pct']}/{row['count']} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.write_text(
        f"""# 历史姿态偏好融合实验

## 设置

- 历史姿态：输入网格保存坐标系的 `+Z`
- 合成 GT：256 个采样方向中支撑投影面积最小者
- 历史 head：在冻结的点云/方向交互编码器上训练
- 融合：对每件零件的面积分数和历史偏好分数分别做 z-score，
  再计算 `area_z + alpha * history_z`
- development 零件：{config["development_count"]}
- test 零件：{config["test_count"]}

## 合成 GT 与历史姿态

| 指标 | development | test |
|---|---:|---:|
| 合成 GT 与历史轴夹角中位数 | {metrics["development_history_geometry"]["synthetic_gt_to_history_angle_median_deg"]:.2f}° | {metrics["test_history_geometry"]["synthetic_gt_to_history_angle_median_deg"]:.2f}° |
| 夹角 ≤20° | {metrics["development_history_geometry"]["synthetic_gt_within_20deg"]}/{metrics["development_history_geometry"]["count"]} | {metrics["test_history_geometry"]["synthetic_gt_within_20deg"]}/{metrics["test_history_geometry"]["count"]} |
| 历史方向面积/oracle 中位数 | {metrics["development_history_geometry"]["history_area_over_oracle_median"]:.3f} | {metrics["test_history_geometry"]["history_area_over_oracle_median"]:.3f} |
| 历史方向面积 ≤10% | {metrics["development_history_geometry"]["history_area_within_10pct"]}/{metrics["development_history_geometry"]["count"]} | {metrics["test_history_geometry"]["history_area_within_10pct"]}/{metrics["test_history_geometry"]["count"]} |

## Development 融合曲线

{markdown_table(metrics["development_rows"])}

## Test 融合曲线

{markdown_table(metrics["test_rows"])}

固定 `+Z` 偏好是当前数据的显式历史坐标先验，不代表可以从任意输入坐标泛化的
形状模型。学习 head 使用同一冻结几何编码器，用于检查这种历史信号能否由网络自行
学出。Test 曲线用于观察取舍，不应根据 test 最优 alpha 反向宣称模型泛化。
""",
        encoding="utf-8",
    )


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
        "--split-file",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256/"
            "split_project_isolated_100_seed30.json"
        ),
    )
    parser.add_argument(
        "--history-model",
        type=Path,
        default=Path(
            "runs/history_preference_head_direction_2026-11-24/"
            "history_head_model.pt"
        ),
    )
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--batch-shapes", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--alphas",
        default="0,0.05,0.1,0.25,0.5,1,2,5,10",
    )
    parser.add_argument(
        "--name",
        default="history_preference_fusion_evaluation_2026-11-24",
    )
    args = parser.parse_args()
    alphas = [
        float(value)
        for value in args.alphas.split(",")
        if value.strip()
    ]
    if not alphas or any(value < 0 for value in alphas):
        parser.error("alphas must contain non-negative values")

    torch.set_float32_matmul_precision("high")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    root = Path(__file__).resolve().parents[1]
    split_path = (root / args.split_file).resolve()
    history_model_path = (root / args.history_model).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_train = split["train_records"]
    development_project = choose_development_project(source_train)
    development_records = [
        record
        for record in source_train
        if record["source_project"] == development_project
    ]
    test_records = split["validation_records"]
    development_parts = load_part_rows(
        development_records,
        args.point_count,
    )
    test_parts = load_part_rows(test_records, args.point_count)

    checkpoint = torch.load(
        history_model_path,
        map_location="cpu",
        weights_only=False,
    )
    history_config = checkpoint["config"]
    base_checkpoint = torch.load(
        history_config["init_model"],
        map_location="cpu",
        weights_only=False,
    )
    base_config = base_checkpoint["config"]
    model = PointDirectionInteractionScorer(
        point_channels=6,
        point_embedding_dim=int(base_config["interaction_point_dim"]),
        interaction_hidden_dim=int(base_config["interaction_hidden_dim"]),
        direction_hidden_dim=int(base_config["direction_hidden_dim"]),
        candidate_chunk_size=int(base_config["candidate_chunk_size"]),
        dropout=float(base_config["dropout"]),
        enable_history_head=True,
        history_hidden_dim=int(history_config["history_hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    development_predictions = predict_dual(
        model,
        development_parts,
        args.point_count,
        device,
        args.batch_shapes,
    )
    test_predictions = predict_dual(
        model,
        test_parts,
        args.point_count,
        device,
        args.batch_shapes,
    )

    development_rows = []
    test_rows = []
    for preference in ("fixed", "learned"):
        for alpha in alphas:
            development_rows.append(evaluate_fusion(
                development_parts,
                development_predictions,
                alpha,
                preference,
            ))
            test_rows.append(evaluate_fusion(
                test_parts,
                test_predictions,
                alpha,
                preference,
            ))
    config = {
        "dataset": str((root / args.dataset).resolve()),
        "split_file": str(split_path),
        "history_model": str(history_model_path),
        "development_project": development_project,
        "development_count": len(development_parts),
        "test_count": len(test_parts),
        "alphas": alphas,
        "fusion": "per-part zscore(area) + alpha * zscore(history)",
    }
    metrics = {
        "config": config,
        "development_history_geometry": history_geometry_summary(
            development_parts
        ),
        "test_history_geometry": history_geometry_summary(test_parts),
        "development_rows": development_rows,
        "test_rows": test_rows,
    }
    run_dir = root / "runs" / args.name
    report_dir = root / "reports" / args.name
    if run_dir.exists() or report_dir.exists():
        parser.error("Existing run/report directory; choose a new --name")
    run_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    for directory in (run_dir, report_dir):
        (directory / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(report_dir / "README.md", config, metrics)
    print(json.dumps({
        "report": str(report_dir),
        "development_history_geometry": (
            metrics["development_history_geometry"]
        ),
        "test_history_geometry": metrics["test_history_geometry"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
