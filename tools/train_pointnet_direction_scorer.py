#!/usr/bin/env python3
"""Train PointNet direction scorers with optional geometric prior features."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tools.train_pointnet_projection_scorer import (
        choose_development_project,
        enriched_rows,
        evaluate_parts,
        fixed_point_indices,
        normalise_cloud,
        normalise_features,
        random_rotations,
        seed_everything,
        summarise_rows,
        write_csv,
    )
    from tools.train_projection_score_mlp import score_from_lower_is_better
except ModuleNotFoundError:
    from train_pointnet_projection_scorer import (
        choose_development_project,
        enriched_rows,
        evaluate_parts,
        fixed_point_indices,
        normalise_cloud,
        normalise_features,
        random_rotations,
        seed_everything,
        summarise_rows,
        write_csv,
    )
    from train_projection_score_mlp import score_from_lower_is_better


class PointNetDirectionScorer(nn.Module):
    """Encode a shape once, then score directions with optional prior features."""

    def __init__(
        self,
        point_channels: int = 6,
        point_embedding_dim: int = 1024,
        shape_hidden_dim: int = 256,
        direction_hidden_dim: int = 128,
        candidate_feature_dim: int = 0,
        global_feature_dim: int = 0,
        global_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(point_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Conv1d(512, point_embedding_dim, 1, bias=False),
            nn.BatchNorm1d(point_embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.shape_head = nn.Sequential(
            nn.Linear(point_embedding_dim, shape_hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(shape_hidden_dim),
        )
        self.global_feature_dim = global_feature_dim
        self.global_hidden_dim = global_hidden_dim if global_feature_dim else 0
        self.global_head = (
            nn.Sequential(
                nn.Linear(global_feature_dim, self.global_hidden_dim),
                nn.SiLU(inplace=True),
                nn.LayerNorm(self.global_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
                nn.SiLU(inplace=True),
            )
            if global_feature_dim
            else None
        )
        self.direction_head = nn.Sequential(
            nn.Linear(3 + candidate_feature_dim, direction_hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(direction_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(direction_hidden_dim, direction_hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(direction_hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(
                shape_hidden_dim
                + self.global_hidden_dim
                + direction_hidden_dim,
                256,
            ),
            nn.SiLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        candidate_features: torch.Tensor,
        global_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = points.shape[0]
        direction_count = candidate_features.shape[1]
        point_features = self.point_mlp(
            points.transpose(1, 2).contiguous()
        )
        shape_embedding = self.shape_head(point_features.amax(dim=2))
        if self.global_head is not None:
            if global_features is None:
                raise ValueError(
                    "global_features are required by this model"
                )
            shape_embedding = torch.cat(
                [shape_embedding, self.global_head(global_features)],
                dim=-1,
            )
        normalized_direction = F.normalize(
            candidate_features[:, :, :3],
            dim=-1,
        )
        if candidate_features.shape[-1] > 3:
            direction_input = torch.cat(
                [
                    normalized_direction,
                    candidate_features[:, :, 3:],
                ],
                dim=-1,
            )
        else:
            direction_input = normalized_direction
        direction_embedding = self.direction_head(
            direction_input
        )
        shape_context = shape_embedding[:, None, :].expand(
            -1,
            direction_count,
            -1,
        )
        scores = self.score_head(
            torch.cat([shape_context, direction_embedding], dim=-1)
        )
        return scores.squeeze(-1)


def load_geometry_feature_records(
    directory: Path,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Load a 30-degree prior manifest and index records by job ID."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_values_used_as_inputs"):
        raise ValueError(
            "Geometry prior manifest unexpectedly declares target leakage"
        )
    records = {
        str(record["job_id"]): record
        for record in manifest["records"]
    }
    candidate_names = [str(name) for name in manifest["candidate_feature_names"]]
    global_names = [str(name) for name in manifest["global_feature_names"]]
    return records, candidate_names, global_names


def stacked_feature_standardisation(
    parts: list[dict[str, Any]],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = [part[field] for part in parts if part[field].size]
    if not values:
        return (
            np.zeros(0, dtype=np.float32),
            np.ones(0, dtype=np.float32),
        )
    stacked = (
        np.stack(values, axis=0)
        if values[0].ndim == 1
        else np.concatenate(values, axis=0)
    )
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def load_part_rows(
    records: list[dict[str, Any]],
    point_count: int,
    geometry_feature_records: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    parts = []
    for record in records:
        with np.load(record["label_file"]) as data:
            if "points" not in data.files or "normals" not in data.files:
                raise ValueError(
                    f"Missing point cloud or normals in {record['label_file']}"
                )
            raw_points = np.asarray(data["points"], dtype=np.float32)
            raw_normals = np.asarray(data["normals"], dtype=np.float32)
            if len(raw_points) < point_count or len(raw_normals) != len(raw_points):
                raise ValueError(
                    f"Invalid point cloud shape in {record['label_file']}"
                )
            directions = np.asarray(data["directions"], dtype=np.float32)
            if directions.ndim != 2 or directions.shape[1] != 3:
                raise ValueError(
                    f"Invalid direction shape {directions.shape} in "
                    f"{record['label_file']}"
                )
            areas = np.asarray(data["projected_area_mm2"], dtype=np.float32)
            targets = np.asarray(data["score"], dtype=np.float32)
            retained_surface = np.asarray(
                data["retained_surface_mm2"],
                dtype=np.float32,
            )
        candidate_prior = np.empty(
            (len(directions), 0),
            dtype=np.float32,
        )
        global_features = np.empty(0, dtype=np.float32)
        projected_area_30 = None
        anchor_scores = np.empty(0, dtype=np.float32)
        if geometry_feature_records is not None:
            try:
                geometry_record = geometry_feature_records[record["job_id"]]
            except KeyError as error:
                raise KeyError(
                    f"Missing geometry prior for {record['job_id']}"
                ) from error
            with np.load(
                geometry_record["geometry_feature_file"]
            ) as data:
                prior_directions = np.asarray(
                    data["directions"],
                    dtype=np.float32,
                )
                if not np.allclose(
                    prior_directions,
                    directions,
                    atol=1e-7,
                ):
                    raise ValueError(
                        f"Direction mismatch for {record['job_id']}"
                    )
                candidate_prior = np.asarray(
                    data["candidate_features"],
                    dtype=np.float32,
                )
                global_features = np.asarray(
                    data["global_features"],
                    dtype=np.float32,
                )
                projected_area_30 = np.asarray(
                    data["projected_area_mm2_30"],
                    dtype=np.float32,
                )
                anchor_scores = score_from_lower_is_better(
                    projected_area_30
                ).astype(np.float32)
        points, normals = normalise_cloud(raw_points, raw_normals)
        directions /= np.clip(
            np.linalg.norm(directions, axis=1, keepdims=True),
            1e-9,
            None,
        )
        parts.append({
            "record": record,
            "points": points,
            "normals": normals,
            "directions": directions.astype(np.float32),
            "areas": areas,
            "targets": targets,
            "retained_surface": retained_surface,
            "candidate_prior": candidate_prior,
            "global_features": global_features,
            "projected_area_30": projected_area_30,
            "anchor_scores": anchor_scores,
        })
    if not parts:
        raise ValueError("No labelled parts found")
    direction_count = len(parts[0]["directions"])
    if any(len(part["directions"]) != direction_count for part in parts):
        raise ValueError("Inconsistent candidate direction counts")
    return parts


def make_model_inputs(
    parts: list[dict[str, Any]],
    point_count: int,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    device: torch.device,
    augment: bool,
    generator: torch.Generator | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch = len(parts)
    direction_count = len(parts[0]["directions"])
    prior_dim = len(prior_mean)
    global_dim = len(global_mean)
    points = np.empty((batch, point_count, 6), dtype=np.float32)
    candidate_features = np.empty(
        (batch, direction_count, 3 + prior_dim),
        dtype=np.float32,
    )
    global_features = np.empty(
        (batch, global_dim),
        dtype=np.float32,
    )
    targets = np.empty((batch, direction_count), dtype=np.float32)
    anchors = np.empty((batch, direction_count), dtype=np.float32)
    for index, part in enumerate(parts):
        if augment:
            ids = np.random.choice(
                len(part["points"]),
                point_count,
                replace=False,
            )
        else:
            ids = fixed_point_indices(point_count, len(part["points"]))
        points[index, :, :3] = part["points"][ids]
        points[index, :, 3:] = part["normals"][ids]
        part_directions = np.asarray(part["directions"], dtype=np.float32)
        normalized_directions = part_directions / np.clip(
            np.linalg.norm(part_directions, axis=1, keepdims=True),
            1e-9,
            None,
        )
        if prior_dim:
            candidate_features[index, :, :3] = normalized_directions
            candidate_features[index, :, 3:] = normalise_features(
                part["candidate_prior"],
                prior_mean,
                prior_std,
            )
        else:
            candidate_features[index] = normalized_directions
        if global_dim:
            global_features[index] = normalise_features(
                part["global_features"],
                global_mean,
                global_std,
            )
        targets[index] = part["targets"]
        anchors[index] = part["anchor_scores"]

    points_tensor = torch.from_numpy(points).to(device)
    candidate_tensor = torch.from_numpy(candidate_features).to(device)
    global_tensor = torch.from_numpy(global_features).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
    anchors_tensor = torch.from_numpy(anchors).to(device)
    if augment:
        if generator is None:
            raise ValueError("A random generator is required for training")
        rotations = random_rotations(batch, generator, device)
        points_tensor[:, :, :3] = torch.bmm(
            points_tensor[:, :, :3],
            rotations.transpose(1, 2),
        )
        points_tensor[:, :, 3:] = torch.bmm(
            points_tensor[:, :, 3:],
            rotations.transpose(1, 2),
        )
        candidate_tensor[:, :, :3] = torch.bmm(
            candidate_tensor[:, :, :3],
            rotations.transpose(1, 2),
        )
    return (
        points_tensor,
        candidate_tensor,
        global_tensor,
        targets_tensor,
        anchors_tensor,
    )


def zero_last_linear_layer(model: nn.Module) -> None:
    """Make an anchor-residual model start from an exact zero residual."""
    last = model.score_head[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("PointNetDirectionScorer score head must end in Linear")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


def combine_anchor_residual(
    residual: torch.Tensor,
    anchors: torch.Tensor,
    anchor_weight: float,
    residual_scale: float,
) -> torch.Tensor:
    """Apply the explicit lower-threshold anchor before residual correction."""
    if residual.shape != anchors.shape:
        raise ValueError("Residual and anchor tensors must have matching shapes")
    return residual_scale * residual + anchor_weight * anchors


def listwise_softmax_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Match the target score distribution over each part's candidates."""
    if temperature <= 0:
        raise ValueError("Listwise temperature must be positive")
    target_probabilities = F.softmax(targets / temperature, dim=1)
    log_prediction_probabilities = F.log_softmax(
        predictions / temperature,
        dim=1,
    )
    return -(
        target_probabilities * log_prediction_probabilities
    ).sum(dim=1).mean()


def topk_pairwise_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    positive_count: int,
    samples_per_shape: int,
) -> torch.Tensor:
    """Rank 45-degree target top-k candidates above sampled negatives."""
    if positive_count < 1 or samples_per_shape < 1:
        raise ValueError("Top-k loss counts must be positive")
    batch, candidate_count = targets.shape
    k = min(positive_count, candidate_count)
    positive_indices = torch.topk(
        targets,
        k=k,
        dim=1,
        sorted=False,
    ).indices
    positive_indices = torch.gather(
        positive_indices,
        1,
        torch.randint(
            0,
            k,
            (batch, samples_per_shape),
            device=predictions.device,
        ),
    )
    negative_indices = torch.randint(
        0,
        candidate_count,
        (batch, samples_per_shape),
        device=predictions.device,
    )
    positive_targets = torch.gather(targets, 1, positive_indices)
    negative_targets = torch.gather(targets, 1, negative_indices)
    gaps = (positive_targets - negative_targets).clamp_min(0.0)
    valid = gaps > 1e-6
    if not valid.any():
        return torch.zeros((), device=predictions.device)
    positive_scores = torch.gather(
        predictions,
        1,
        positive_indices,
    )
    negative_scores = torch.gather(
        predictions,
        1,
        negative_indices,
    )
    losses = F.softplus(negative_scores - positive_scores) * gaps
    return losses[valid].sum() / gaps[valid].sum().clamp_min(1e-9)


def development_selection_score(metrics: dict[str, Any]) -> float:
    """Balance global ranking with practical Top-5/Top-10 recommendation."""
    return (
        0.25 * metrics["spearman_median"]
        + 0.25 * metrics["ndcg_at_5_median"]
        + 0.15 * metrics["ndcg_at_10_median"]
        + 0.15 * (
            1.0 - min(metrics["top5_regret_median"] * 20.0, 1.0)
        )
        + 0.10 * (
            1.0 - min(metrics["top10_regret_median"] * 20.0, 1.0)
        )
        + 0.10 * (
            metrics["top5_within_20pct_count"] / metrics["count"]
        )
    )


@torch.no_grad()
def predict_parts(
    model: nn.Module,
    parts: list[dict[str, Any]],
    point_count: int,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    device: torch.device,
    batch_shapes: int = 8,
    anchor_weight: float = 0.0,
    residual_scale: float = 1.0,
) -> list[np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(parts), batch_shapes):
        selected = parts[start:start + batch_shapes]
        (
            points,
            candidate_features,
            global_features,
            _,
            anchors,
        ) = make_model_inputs(
            selected,
            point_count,
            prior_mean,
            prior_std,
            global_mean,
            global_std,
            device,
            augment=False,
            generator=None,
        )
        residual = model(
                points,
                candidate_features,
                global_features if len(global_mean) else None,
            )
        prediction = combine_anchor_residual(
            residual,
            anchors,
            anchor_weight,
            residual_scale,
        )
        predictions.extend(prediction.float().cpu().numpy())
    return predictions


def load_reference_model(path: Path) -> dict[str, Any]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    test = reference["test"]
    for key in (
        "model",
        "pointnet_direction_only",
        "pointnet",
    ):
        if key in test:
            return test[key]
    raise KeyError(f"No supported model metrics found in {path}")


def write_report(
    path: Path,
    arguments: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    network = metrics["test"]["network"]
    direction_only = metrics["test"]["direction_only_reference"]
    prior30 = metrics["test"]["geometry_prior_30deg_baseline"]
    baseline = metrics["test"]["retained_surface_baseline"]
    anchor_weight = metrics["model"]["anchor_weight"]
    residual_scale = metrics["model"]["residual_scale"]
    if prior30 is None:
        title = "4096 点 PointNet 方向-only 评分器"
        comparison_header = "方向-only PointNet"
        comparison_values = direction_only
        input_lines = (
            "- 每个候选方向只输入 3 维 XYZ。\n"
            "- 不使用候选几何特征或几何 baseline。"
        )
        results_table = f"""| 指标 | 当前网络 | 方向-only参考 | 45°保留面规则 |
|---|---:|---:|---:|
| Spearman 均值 | {network["spearman_mean"]:.4f} | {direction_only["spearman_mean"]:.4f} | {baseline["spearman_mean"]:.4f} |
| Spearman 中位数 | {network["spearman_median"]:.4f} | {direction_only["spearman_median"]:.4f} | {baseline["spearman_median"]:.4f} |
| Kendall 中位数 | {network["kendall_median"]:.4f} | {direction_only["kendall_median"]:.4f} | {baseline["kendall_median"]:.4f} |
| NDCG@5 均值 | {network["ndcg_at_5_mean"]:.4f} | {direction_only["ndcg_at_5_mean"]:.4f} | {baseline["ndcg_at_5_mean"]:.4f} |
| NDCG@5 中位数 | {network["ndcg_at_5_median"]:.4f} | {direction_only["ndcg_at_5_median"]:.4f} | {baseline["ndcg_at_5_median"]:.4f} |
| NDCG@10 中位数 | {network["ndcg_at_10_median"]:.4f} | {direction_only["ndcg_at_10_median"]:.4f} | {baseline["ndcg_at_10_median"]:.4f} |
| Top-1 regret 中位数 | {network["top1_regret_median"]:.4f} | {direction_only["top1_regret_median"]:.4f} | {baseline["top1_regret_median"]:.4f} |
| Top-1 面积/oracle 中位数 | {network["top1_area_ratio_median"]:.4f} | {direction_only["top1_area_ratio_median"]:.4f} | {baseline["top1_area_ratio_median"]:.4f} |
| Top-5 精确命中 | {network["top5_exact_count"]}/{network["count"]} | {direction_only["top5_exact_count"]}/{direction_only["count"]} | {baseline["top5_exact_count"]}/{baseline["count"]} |"""
    else:
        title = "4096 点 PointNet＋30° 几何先验评分器"
        comparison_header = "PointNet＋30°先验"
        comparison_values = network
        input_lines = (
            f"- 每个候选方向输入 3 维方向、"
            f"{len(metrics['dataset']['candidate_input']) - 3} 维 "
            "30°候选几何特征。\n"
            f"- PointNet 形状向量额外拼接 "
            f"{metrics['dataset']['global_input_dim']} 维旋转不变全局形状特征。\n"
            "- 30°先验不应用小孔豁免，也不读取 45°目标面积。\n"
            f"- 预测为 `{anchor_weight:g} × 30°分数 + "
            f"{residual_scale:g} × 网络残差`。"
        )
        results_table = f"""| 指标 | {comparison_header} | 方向-only PointNet | 30°规则直接排序 | 45°保留面规则 |
|---|---:|---:|---:|---:|
| Spearman 均值 | {comparison_values["spearman_mean"]:.4f} | {direction_only["spearman_mean"]:.4f} | {prior30["spearman_mean"]:.4f} | {baseline["spearman_mean"]:.4f} |
| Spearman 中位数 | {comparison_values["spearman_median"]:.4f} | {direction_only["spearman_median"]:.4f} | {prior30["spearman_median"]:.4f} | {baseline["spearman_median"]:.4f} |
| Kendall 中位数 | {comparison_values["kendall_median"]:.4f} | {direction_only["kendall_median"]:.4f} | {prior30["kendall_median"]:.4f} | {baseline["kendall_median"]:.4f} |
| NDCG@5 均值 | {comparison_values["ndcg_at_5_mean"]:.4f} | {direction_only["ndcg_at_5_mean"]:.4f} | {prior30["ndcg_at_5_mean"]:.4f} | {baseline["ndcg_at_5_mean"]:.4f} |
| NDCG@5 中位数 | {comparison_values["ndcg_at_5_median"]:.4f} | {direction_only["ndcg_at_5_median"]:.4f} | {prior30["ndcg_at_5_median"]:.4f} | {baseline["ndcg_at_5_median"]:.4f} |
| NDCG@10 中位数 | {comparison_values["ndcg_at_10_median"]:.4f} | {direction_only["ndcg_at_10_median"]:.4f} | {prior30["ndcg_at_10_median"]:.4f} | {baseline["ndcg_at_10_median"]:.4f} |
| Top-1 regret 中位数 | {comparison_values["top1_regret_median"]:.4f} | {direction_only["top1_regret_median"]:.4f} | {prior30["top1_regret_median"]:.4f} | {baseline["top1_regret_median"]:.4f} |
| Top-1 面积/oracle 中位数 | {comparison_values["top1_area_ratio_median"]:.4f} | {direction_only["top1_area_ratio_median"]:.4f} | {prior30["top1_area_ratio_median"]:.4f} | {baseline["top1_area_ratio_median"]:.4f} |
| Top-5 精确命中 | {comparison_values["top5_exact_count"]}/{comparison_values["count"]} | {direction_only["top5_exact_count"]}/{direction_only["count"]} | {prior30["top5_exact_count"]}/{prior30["count"]} | {baseline["top5_exact_count"]}/{baseline["count"]} |"""
    path.write_text(
        f"""# {title}

## 输入与网络

- 每件零件 {arguments.point_count} 点，输入 XYZ＋法向，共 6 个通道。
- PointNet 对每件形状只编码一次，得到 {arguments.point_embedding_dim} 维 embedding。
{input_lines}
- 形状上下文与候选方向 embedding 拼接后输出一个标量分数。
- 训练时对点云、法向和候选方向施加同一个随机 SO(3) 旋转。
- `anchor_weight={anchor_weight:g}`，`residual_scale={residual_scale:g}`。
- 残差头零初始化；启用 anchor 时，训练前输出精确等于30°规则分数。
- 45°投影面积目标只进入损失和评价，不进入输入。

## Test 结果

{results_table}

## 训练

- 参数更新零件：{metrics["dataset"]["fit_count"]}
- development 零件：{metrics["dataset"]["development_count"]}
- test 零件：{metrics["dataset"]["test_count"]}
- 最佳 epoch：{metrics["training"]["best_epoch"]}
- 完成 epoch：{metrics["training"]["epochs_completed"]}
- 参数量：{metrics["model"]["parameters"]}
- 30°几何先验：{metrics["dataset"]["uses_30deg_geometry_prior"]}

## 产物

- `metrics.json`
- `per_part.csv`
- `history.json`
- `config.json`

生成数据不提交到 Git。
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("labels/projection_support_area_v0_256"),
    )
    parser.add_argument(
        "--pointnet-reference-report",
        type=Path,
        default=Path(
            "reports/pointnet_direction_only_4096_expanded_932_100_2026-11-20/metrics.json"
        ),
    )
    parser.add_argument(
        "--name",
        default="pointnet_direction_only_4096_2026-11-20",
    )
    parser.add_argument(
        "--geometry-features",
        type=Path,
        help="30-degree prior feature directory with manifest.json",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help=(
            "Optional split JSON inside the dataset. Defaults to the "
            "dataset's split.json."
        ),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-shapes", type=int, default=16)
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--point-embedding-dim", type=int, default=1024)
    parser.add_argument("--shape-hidden-dim", type=int, default=256)
    parser.add_argument("--direction-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-samples", type=int, default=128)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.0)
    parser.add_argument("--listwise-temperature", type=float, default=0.1)
    parser.add_argument("--topk-weight", type=float, default=0.0)
    parser.add_argument("--topk-positive-count", type=int, default=10)
    parser.add_argument("--topk-pairwise-samples", type=int, default=64)
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the lower-threshold anchor score. Set to 1 with a "
            "geometry prior to train an anchored residual model."
        ),
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Multiplier on the network residual when an anchor is enabled.",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_shapes < 1 or args.point_count < 16:
        parser.error("epochs, batch-shapes, and point-count must be positive")
    if args.mse_weight < 0 or args.pairwise_weight < 0:
        parser.error("MSE and pairwise weights must be non-negative")
    if args.listwise_weight < 0 or args.topk_weight < 0:
        parser.error("Listwise and Top-K weights must be non-negative")
    if args.listwise_temperature <= 0:
        parser.error("Listwise temperature must be positive")
    if (
        args.topk_positive_count < 1
        or args.topk_pairwise_samples < 1
    ):
        parser.error("Top-K loss counts must be positive")
    if args.anchor_weight and args.geometry_features is None:
        parser.error("--anchor-weight requires --geometry-features")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    dataset_dir = (root / args.dataset).resolve()
    split_path = (
        (root / args.split_file).resolve()
        if args.split_file is not None
        else dataset_dir / "split.json"
    )
    geometry_features_dir = (
        (root / args.geometry_features).resolve()
        if args.geometry_features is not None
        else None
    )
    pointnet_reference_path = (root / args.pointnet_reference_report).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_train_records = split["train_records"]
    test_records = split["validation_records"]
    development_project = choose_development_project(source_train_records)
    fit_records = [
        record
        for record in source_train_records
        if record["source_project"] != development_project
    ]
    development_records = [
        record
        for record in source_train_records
        if record["source_project"] == development_project
    ]
    fit_hashes = {record["mesh_hash_sha256"] for record in fit_records}
    development_hashes = {
        record["mesh_hash_sha256"] for record in development_records
    }
    test_hashes = {record["mesh_hash_sha256"] for record in test_records}
    if fit_hashes & development_hashes or (
        fit_hashes | development_hashes
    ) & test_hashes:
        raise ValueError("Geometry hash overlap detected between data splits")

    geometry_feature_records = None
    candidate_prior_names: list[str] = []
    global_feature_names: list[str] = []
    if geometry_features_dir is not None:
        (
            geometry_feature_records,
            candidate_prior_names,
            global_feature_names,
        ) = load_geometry_feature_records(geometry_features_dir)

    fit_parts = load_part_rows(
        fit_records,
        args.point_count,
        geometry_feature_records,
    )
    development_parts = load_part_rows(
        development_records,
        args.point_count,
        geometry_feature_records,
    )
    test_parts = load_part_rows(
        test_records,
        args.point_count,
        geometry_feature_records,
    )

    prior_mean, prior_std = stacked_feature_standardisation(
        fit_parts,
        "candidate_prior",
    )
    global_mean, global_std = stacked_feature_standardisation(
        fit_parts,
        "global_features",
    )
    if len(prior_mean) != len(candidate_prior_names):
        raise ValueError(
            "Candidate prior feature names and values have inconsistent sizes"
        )
    if len(global_mean) != len(global_feature_names):
        raise ValueError(
            "Global feature names and values have inconsistent sizes"
        )

    model = PointNetDirectionScorer(
        point_channels=6,
        point_embedding_dim=args.point_embedding_dim,
        shape_hidden_dim=args.shape_hidden_dim,
        direction_hidden_dim=args.direction_hidden_dim,
        candidate_feature_dim=len(prior_mean),
        global_feature_dim=len(global_mean),
        dropout=args.dropout,
    ).to(device)
    if args.anchor_weight:
        zero_last_linear_layer(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.02,
    )

    run_dir = root / "runs" / args.name
    report_dir = root / "reports" / args.name
    if run_dir.exists() or report_dir.exists():
        parser.error("Existing run/report directory; choose a new --name")
    run_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    history: list[dict[str, float]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(fit_parts))
        totals = {
            "total": 0.0,
            "mse": 0.0,
            "rank": 0.0,
            "listwise": 0.0,
            "topk": 0.0,
        }
        batches = 0
        for start in range(0, len(order), args.batch_shapes):
            selected = [
                fit_parts[int(index)]
                for index in order[start:start + args.batch_shapes]
            ]
            (
                points,
                candidate_features,
                global_features,
                targets,
                anchors,
            ) = make_model_inputs(
                selected,
                args.point_count,
                prior_mean,
                prior_std,
                global_mean,
                global_std,
                device,
                augment=True,
                generator=generator,
            )
            residual = model(
                points,
                candidate_features,
                global_features if len(global_mean) else None,
            )
            predictions = combine_anchor_residual(
                residual,
                anchors,
                args.anchor_weight,
                args.residual_scale,
            )
            mse = F.mse_loss(predictions, targets)
            batch, candidates = targets.shape
            left = torch.randint(
                0,
                candidates,
                (batch, args.pairwise_samples),
                device=device,
            )
            right = torch.randint(
                0,
                candidates,
                (batch, args.pairwise_samples),
                device=device,
            )
            target_difference = (
                torch.gather(targets, 1, left)
                - torch.gather(targets, 1, right)
            )
            valid = target_difference.abs() > 1e-5
            if valid.any():
                rank_loss = F.softplus(
                    -torch.sign(target_difference[valid])
                    * (
                        torch.gather(predictions, 1, left)[valid]
                        - torch.gather(predictions, 1, right)[valid]
                    )
                ).mean()
            else:
                rank_loss = torch.zeros((), device=device)
            listwise_loss = (
                listwise_softmax_loss(
                    predictions,
                    targets,
                    args.listwise_temperature,
                )
                if args.listwise_weight
                else torch.zeros((), device=device)
            )
            topk_loss = (
                topk_pairwise_loss(
                    predictions,
                    targets,
                    args.topk_positive_count,
                    args.topk_pairwise_samples,
                )
                if args.topk_weight
                else torch.zeros((), device=device)
            )
            loss = (
                args.mse_weight * mse
                + args.pairwise_weight * rank_loss
                + args.listwise_weight * listwise_loss
                + args.topk_weight * topk_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            totals["total"] += float(loss.item())
            totals["mse"] += float(mse.item())
            totals["rank"] += float(rank_loss.item())
            totals["listwise"] += float(listwise_loss.item())
            totals["topk"] += float(topk_loss.item())
            batches += 1
        scheduler.step()

        development_predictions = predict_parts(
            model,
            development_parts,
            args.point_count,
            prior_mean,
            prior_std,
            global_mean,
            global_std,
            device,
            anchor_weight=args.anchor_weight,
            residual_scale=args.residual_scale,
        )
        development_rows = enriched_rows(
            development_predictions,
            development_parts,
        )
        development_metrics = summarise_rows(development_rows)
        current_score = development_selection_score(development_metrics)
        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / batches,
            "train_mse": totals["mse"] / batches,
            "train_rank_loss": totals["rank"] / batches,
            "train_listwise_loss": totals["listwise"] / batches,
            "train_topk_loss": totals["topk"] / batches,
            "validation_spearman": development_metrics["spearman_median"],
            "validation_ndcg_at_5": development_metrics["ndcg_at_5_median"],
            "validation_top1_regret": development_metrics["top1_regret_median"],
            "validation_top5_regret": development_metrics["top5_regret_median"],
            "validation_top10_regret": development_metrics["top10_regret_median"],
            "validation_top5_within_20pct": (
                development_metrics["top5_within_20pct_count"]
                / development_metrics["count"]
            ),
            "validation_selection_score": current_score,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch:03d} "
            f"loss={row['train_total_loss']:.5f} "
            f"val_spearman={row['validation_spearman']:.4f} "
            f"val_ndcg5={row['validation_ndcg_at_5']:.4f} "
            f"val_top5_regret={row['validation_top5_regret']:.4f} "
            f"val_top5_20={row['validation_top5_within_20pct']:.2f} "
            f"best={best_epoch}",
            flush=True,
        )
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_predictions = predict_parts(
        model,
        test_parts,
        args.point_count,
        prior_mean,
        prior_std,
        global_mean,
        global_std,
        device,
        anchor_weight=args.anchor_weight,
        residual_scale=args.residual_scale,
    )
    baseline_predictions = [
        score_from_lower_is_better(part["retained_surface"])
        for part in test_parts
    ]
    test_network, test_rows = evaluate_parts(
        test_parts,
        test_predictions,
    )
    test_baseline, baseline_rows = evaluate_parts(
        test_parts,
        baseline_predictions,
    )
    test_prior30 = None
    prior30_rows = None
    if geometry_feature_records is not None:
        prior30_predictions = [
            score_from_lower_is_better(part["projected_area_30"])
            for part in test_parts
        ]
        test_prior30, prior30_rows = evaluate_parts(
            test_parts,
            prior30_predictions,
    )
    pointnet_reference = load_reference_model(pointnet_reference_path)
    network_name = (
        "PointNetGeometryPrior30Scorer"
        if geometry_feature_records is not None
        else "PointNetDirectionScorer"
    )

    metrics = {
        "dataset": {
            "path": str(dataset_dir),
            "split_file": str(split_path),
            "fit_count": len(fit_parts),
            "development_count": len(development_parts),
            "test_count": len(test_parts),
            "direction_count": int(len(fit_parts[0]["targets"])),
            "candidate_input_dim": 3 + len(prior_mean),
            "candidate_input": [
                "direction_x",
                "direction_y",
                "direction_z",
                *candidate_prior_names,
            ],
            "global_input_dim": len(global_mean),
            "global_input": global_feature_names,
            "point_count": args.point_count,
            "point_channels": ["x", "y", "z", "normal_x", "normal_y", "normal_z"],
            "uses_18_candidate_geometry_features": False,
            "uses_30deg_geometry_prior": bool(prior_mean.size),
            "geometry_features": (
                str(geometry_features_dir)
                if geometry_features_dir is not None
                else None
            ),
            "development_project": development_project,
            "test_projects": sorted({
                record["source_project"] for record in test_records
            }),
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_selection_score": best_score,
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
        },
        "model": {
            "name": network_name,
            "parameters": parameter_count,
            "anchor_weight": args.anchor_weight,
            "residual_scale": args.residual_scale,
            "mse_weight": args.mse_weight,
            "pairwise_weight": args.pairwise_weight,
            "listwise_weight": args.listwise_weight,
            "listwise_temperature": args.listwise_temperature,
            "topk_weight": args.topk_weight,
            "topk_positive_count": args.topk_positive_count,
            "topk_pairwise_samples": args.topk_pairwise_samples,
            "geometry_fusion": bool(prior_mean.size),
            "uses_18_candidate_geometry_features": False,
            "uses_30deg_geometry_prior": bool(prior_mean.size),
        },
        "test": {
            "network": test_network,
            "network_name": network_name,
            "direction_only_reference": pointnet_reference,
            "geometry_prior_30deg_baseline": test_prior30,
            "retained_surface_baseline": test_baseline,
            "per_part": test_rows,
            "baseline_per_part": baseline_rows,
            "prior30_per_part": prior30_rows,
        },
        "limitations": [
            "The target is a deterministic projected-area proxy, not Magics output.",
            "The expanded split contains 100 validation parts, but 95 come from projects also represented in training.",
            "The model scores 256 sampled directions rather than a continuous sphere.",
            "PointNet rotation robustness is encouraged by augmentation but is not mathematically guaranteed.",
            "The 30-degree prior is a geometric proxy and is strongly related to the 45-degree target by construction.",
        ],
    }
    config = {
        "name": args.name,
        "dataset": str(dataset_dir),
        "split_file": str(split_path),
        "epochs": args.epochs,
        "batch_shapes": args.batch_shapes,
        "point_count": args.point_count,
        "point_channels": 6,
        "candidate_input_dim": 3 + len(prior_mean),
        "candidate_input": [
            "direction_x",
            "direction_y",
            "direction_z",
            *candidate_prior_names,
        ],
        "candidate_feature_names": candidate_prior_names,
        "global_input_dim": len(global_mean),
        "global_feature_names": global_feature_names,
        "uses_18_candidate_geometry_features": False,
        "uses_30deg_geometry_prior": bool(prior_mean.size),
        "geometry_features": (
            str(geometry_features_dir)
            if geometry_features_dir is not None
            else None
        ),
        "point_embedding_dim": args.point_embedding_dim,
        "shape_hidden_dim": args.shape_hidden_dim,
        "direction_hidden_dim": args.direction_hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pairwise_weight": args.pairwise_weight,
        "pairwise_samples": args.pairwise_samples,
        "mse_weight": args.mse_weight,
        "listwise_weight": args.listwise_weight,
        "listwise_temperature": args.listwise_temperature,
        "topk_weight": args.topk_weight,
        "topk_positive_count": args.topk_positive_count,
        "topk_pairwise_samples": args.topk_pairwise_samples,
        "anchor_weight": args.anchor_weight,
        "residual_scale": args.residual_scale,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "anchor_weight": args.anchor_weight,
        "residual_scale": args.residual_scale,
        "geometry_fusion": bool(prior_mean.size),
        "candidate_prior_mean": prior_mean.tolist(),
        "candidate_prior_std": prior_std.tolist(),
        "global_feature_mean": global_mean.tolist(),
        "global_feature_std": global_std.tolist(),
    }
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "config": config,
    }, run_dir / "best_model.pt")
    for directory in (run_dir, report_dir):
        (directory / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "split.json").write_text(
            json.dumps(split, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(report_dir / "per_part.csv", test_rows)
    write_report(report_dir / "README.md", args, metrics)
    print(json.dumps({
        "name": args.name,
        "best_epoch": best_epoch,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "test": metrics["test"]["network"],
        "prior30_baseline": metrics["test"][
            "geometry_prior_30deg_baseline"
        ],
        "baseline": metrics["test"]["retained_surface_baseline"],
        "report": str(report_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
