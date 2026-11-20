#!/usr/bin/env python3
"""Train a PointNet scorer directly on support-projection target scores."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

try:
    from tools.train_projection_score_mlp import (
        aggregate_metrics,
        evaluate_part,
        ndcg_at_k,
        score_from_lower_is_better,
    )
except ModuleNotFoundError:
    from train_projection_score_mlp import (
        aggregate_metrics,
        evaluate_part,
        ndcg_at_k,
        score_from_lower_is_better,
    )


class PointNetCandidateScorer(nn.Module):
    """Encode a shape once, then score every candidate feature vector."""

    def __init__(
        self,
        feature_dim: int = 18,
        point_channels: int = 6,
        point_embedding_dim: int = 1024,
        shape_hidden_dim: int = 256,
        candidate_hidden_dim: int = 128,
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
        self.candidate_head = nn.Sequential(
            nn.Linear(feature_dim, candidate_hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(candidate_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(candidate_hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(shape_hidden_dim + candidate_hidden_dim, 256),
            nn.SiLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, points: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        batch = points.shape[0]
        candidate_count = candidate_features.shape[1]
        point_features = self.point_mlp(points.transpose(1, 2).contiguous())
        shape_embedding = self.shape_head(point_features.amax(dim=2))
        candidate_embedding = self.candidate_head(candidate_features)
        shape_context = shape_embedding[:, None, :].expand(-1, candidate_count, -1)
        scores = self.score_head(torch.cat([shape_context, candidate_embedding], dim=-1))
        return scores.squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalise_cloud(
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    scale = max(float(np.linalg.norm(centered, axis=1).max()), 1e-9)
    centered = centered / scale
    normals = normals / np.clip(
        np.linalg.norm(normals, axis=1, keepdims=True),
        1e-9,
        None,
    )
    return centered.astype(np.float32), normals.astype(np.float32)


def load_part_rows(
    records: list[dict[str, Any]],
    point_count: int,
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
            features = np.asarray(data["features"], dtype=np.float32)
            if features.shape != (len(data["directions"]), 18):
                raise ValueError(
                    f"Invalid feature shape {features.shape} in {record['label_file']}"
                )
            directions = np.asarray(data["directions"], dtype=np.float32)
            areas = np.asarray(data["projected_area_mm2"], dtype=np.float32)
            targets = np.asarray(data["score"], dtype=np.float32)
            retained_surface = np.asarray(
                data["retained_surface_mm2"],
                dtype=np.float32,
            )
        points, normals = normalise_cloud(raw_points, raw_normals)
        parts.append({
            "record": record,
            "points": points,
            "normals": normals,
            "features": features,
            "directions": directions,
            "areas": areas,
            "targets": targets,
            "retained_surface": retained_surface,
        })
    if not parts:
        raise ValueError("No labelled parts found")
    return parts


def choose_development_project(
    records: list[dict[str, Any]],
    target_count: int = 10,
) -> str:
    from collections import Counter

    counts = Counter(record["source_project"] for record in records)
    if len(counts) < 2:
        raise ValueError("Development split requires at least two projects")
    return min(
        counts,
        key=lambda project: (abs(counts[project] - target_count), project),
    )


def feature_standardisation(
    parts: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([part["features"] for part in parts], axis=0)
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def normalise_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.clip((features - mean) / std, -10.0, 10.0).astype(np.float32)


def random_rotations(
    count: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    quaternion = torch.randn(
        (count, 4),
        generator=generator,
        device=device,
    )
    quaternion = F.normalize(quaternion, dim=1)
    x, y, z, w = quaternion.unbind(dim=1)
    matrices = torch.empty((count, 3, 3), dtype=quaternion.dtype, device=device)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def fixed_point_indices(point_count: int, pool_size: int) -> np.ndarray:
    return np.linspace(
        0,
        pool_size - 1,
        point_count,
        dtype=np.int64,
    )


def make_model_inputs(
    parts: list[dict[str, Any]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    point_count: int,
    device: torch.device,
    augment: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(parts)
    candidate_count = parts[0]["features"].shape[0]
    points = np.empty((batch, point_count, 6), dtype=np.float32)
    features = np.empty((batch, candidate_count, 18), dtype=np.float32)
    targets = np.empty((batch, candidate_count), dtype=np.float32)
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
        features[index] = normalise_features(
            part["features"],
            feature_mean,
            feature_std,
        )
        targets[index] = part["targets"]

    points_tensor = torch.from_numpy(points).to(device)
    features_tensor = torch.from_numpy(features).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
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
        # The first 15 candidate features are computed in the candidate-aligned
        # frame and are rotation invariant. The final three store direction XYZ.
        features_tensor[:, :, 15:18] = torch.bmm(
            features_tensor[:, :, 15:18],
            rotations.transpose(1, 2),
        )
    return points_tensor, features_tensor, targets_tensor


@torch.no_grad()
def predict_parts(
    model: nn.Module,
    parts: list[dict[str, Any]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    point_count: int,
    device: torch.device,
    batch_shapes: int = 8,
) -> list[np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(parts), batch_shapes):
        selected = parts[start:start + batch_shapes]
        points, features, _ = make_model_inputs(
            selected,
            feature_mean,
            feature_std,
            point_count,
            device,
            augment=False,
            generator=None,
        )
        predictions.extend(
            model(points, features).float().cpu().numpy()
        )
    return predictions


def enriched_rows(
    predictions: list[np.ndarray],
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for prediction, part in zip(predictions, parts):
        predicted_index = int(np.argmax(prediction))
        oracle_index = int(np.argmax(part["targets"]))
        top5 = np.argsort(-prediction)[:5]
        predicted_area = float(part["areas"][predicted_index])
        oracle_area = float(part["areas"][oracle_index])
        positive_oracle = oracle_area > 1e-9
        evaluation = evaluate_part(
            prediction,
            part["targets"],
            part["areas"],
            part["directions"],
        )
        evaluation.pop("top1_area_ratio")
        row = {
            "source_project": part["record"]["source_project"],
            "source_part_id": part["record"]["source_part_id"],
            **evaluation,
            "predicted_area_mm2": predicted_area,
            "oracle_area_mm2": oracle_area,
            "top1_area_ratio": (
                predicted_area / oracle_area if positive_oracle else None
            ),
            "area_excess_pct": (
                (predicted_area / oracle_area - 1.0) * 100.0
                if positive_oracle
                else None
            ),
            "top1_exact": int(predicted_index == oracle_index),
            "top5_exact": int(oracle_index in top5),
        }
        for k in (1, 3, 5, 10, 20):
            top = np.argsort(-prediction)[:k]
            best_area = float(part["areas"][top].min())
            best_target = float(part["targets"][top].max())
            row[f"top{k}_regret"] = float(max(
                part["targets"][oracle_index] - best_target,
                0.0,
            ))
            row[f"top{k}_area_ratio"] = (
                best_area / oracle_area if positive_oracle else None
            )
            row[f"top{k}_exact"] = int(
                np.any(np.isclose(
                    part["areas"][top],
                    oracle_area,
                    rtol=1e-7,
                    atol=1e-7,
                ))
            )
        rows.append(row)
    return rows


def summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_metrics([
        {
            key: value
            for key, value in row.items()
            if not key.endswith("_area_ratio")
            and key not in {"area_excess_pct"}
        }
        for row in rows
    ])
    positive = [
        row for row in rows if row["oracle_area_mm2"] > 1e-9
    ]
    zero = [row for row in rows if row["oracle_area_mm2"] <= 1e-9]
    excess = np.asarray([
        row["area_excess_pct"] for row in positive
    ], dtype=np.float64)
    ratio = np.asarray([
        row["top1_area_ratio"] for row in positive
    ], dtype=np.float64)
    zero_exact = int(sum(
        row["predicted_area_mm2"] <= 1e-9 for row in zero
    ))
    result = {
        **aggregate,
        "count": len(rows),
        "top1_exact_count": int(sum(row["top1_exact"] for row in rows)),
        "top1_exact_rate": float(np.mean([row["top1_exact"] for row in rows])),
        "top5_exact_count": int(sum(row["top5_exact"] for row in rows)),
        "top5_exact_rate": float(np.mean([row["top5_exact"] for row in rows])),
        "positive_oracle_count": len(positive),
        "zero_oracle_count": len(zero),
        "zero_oracle_exact_count": zero_exact,
        "within_1pct_count": int(np.sum(excess <= 1.0)),
        "within_5pct_count": int(np.sum(excess <= 5.0)),
        "within_10pct_count": int(np.sum(excess <= 10.0)),
        "within_15pct_count": int(np.sum(excess <= 15.0)),
        "area_excess_pct_p50": float(np.percentile(excess, 50)),
        "area_excess_pct_p90": float(np.percentile(excess, 90)),
        "area_excess_pct_max": float(excess.max()),
        "top1_area_ratio_mean": float(np.mean(ratio)),
        "top1_area_ratio_median": float(np.median(ratio)),
    }
    for k in (1, 3, 5, 10, 20):
        positive_ratios = np.asarray([
            row[f"top{k}_area_ratio"]
            for row in positive
        ], dtype=np.float64)
        accepted = []
        for row in rows:
            value = row[f"top{k}_area_ratio"]
            if value is None:
                accepted.append(bool(row[f"top{k}_exact"]))
            else:
                accepted.append(float(value))
        result.update({
            f"top{k}_area_ratio_mean": float(np.mean(positive_ratios)),
            f"top{k}_area_ratio_median": float(np.median(positive_ratios)),
            f"top{k}_area_ratio_p90": float(np.percentile(
                positive_ratios,
                90,
            )),
            f"top{k}_exact_count": int(sum(
                row[f"top{k}_exact"] for row in rows
            )),
            f"top{k}_within_5pct_count": int(sum(
                float(value) <= 1.05 + 1e-12
                for value in accepted
            )),
            f"top{k}_within_10pct_count": int(sum(
                float(value) <= 1.10 + 1e-12
                for value in accepted
            )),
            f"top{k}_within_20pct_count": int(sum(
                float(value) <= 1.20 + 1e-12
                for value in accepted
            )),
        })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_parts(
    parts: list[dict[str, Any]],
    predictions: list[np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = enriched_rows(predictions, parts)
    return summarise_rows(rows), rows


def write_report(
    path: Path,
    arguments: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    pointnet = metrics["test"]["pointnet"]
    baseline = metrics["test"]["retained_surface_baseline"]
    candidate_mlp = metrics["test"]["candidate_mlp_reference"]
    path.write_text(
        f"""# PointNet 投影面积评分器

## 目的

保持相同的 171/12/10 零件划分和 256 方向投影面积 GT，验证 PointNet
是否能利用点云结构改善纯候选特征 MLP 的直接评分能力。

## 输入与网络

- 每件点云采样 {arguments.point_count} 点，输入 XYZ＋法向，共 6 个通道。
- PointNet 对每件形状编码一次，得到 {arguments.point_embedding_dim} 维形状向量。
- 每个候选方向的 18 维几何特征独立编码为 {arguments.candidate_hidden_dim} 维。
- 两者拼接后由打分头输出该方向的标量分数。
- `anchor_weight=0`；几何 baseline 只作为独立对照，不进入预测。
- 训练时对点云、法向和方向向量施加同一个随机 SO(3) 旋转。

## Test 结果

| 指标 | PointNet | CandidateMLP | 几何 baseline |
|---|---:|---:|---:|
| Spearman 中位数 | {pointnet["spearman_median"]:.4f} | {candidate_mlp["spearman_median"]:.4f} | {baseline["spearman_median"]:.4f} |
| Kendall 中位数 | {pointnet["kendall_median"]:.4f} | {candidate_mlp["kendall_median"]:.4f} | {baseline["kendall_median"]:.4f} |
| NDCG@5 中位数 | {pointnet["ndcg_at_5_median"]:.4f} | {candidate_mlp["ndcg_at_5_median"]:.4f} | {baseline["ndcg_at_5_median"]:.4f} |
| NDCG@10 中位数 | {pointnet["ndcg_at_10_median"]:.4f} | {candidate_mlp["ndcg_at_10_median"]:.4f} | {baseline["ndcg_at_10_median"]:.4f} |
| Top-1 精确命中 | {pointnet["top1_exact_count"]}/{pointnet["count"]} | — | {baseline["top1_exact_count"]}/{baseline["count"]} |
| Top-5 精确命中 | {pointnet["top5_exact_count"]}/{pointnet["count"]} | — | {baseline["top5_exact_count"]}/{baseline["count"]} |
| Top-1 面积/oracle 中位数 | {pointnet["top1_area_ratio_median"]:.4f} | {candidate_mlp["top1_area_ratio_median"]:.4f} | {baseline["top1_area_ratio_median"]:.4f} |
| Top-1 归一化损失中位数 | {pointnet["top1_regret_median"]:.4f} | {candidate_mlp["top1_regret_median"]:.4f} | {baseline["top1_regret_median"]:.4f} |

CandidateMLP 的历史逐零件文件没有保存精确方向索引，因此它的 Top-1/Top-5
精确命中数不能从中位得分反推；最可靠比较应以 Spearman、Kendall、NDCG、
归一化 regret 和面积/oracle 比为准。

## 训练

- 参数更新零件：{metrics["dataset"]["fit_count"]}
- development 零件：{metrics["dataset"]["development_count"]}
- test 零件：{metrics["dataset"]["test_count"]}
- 最佳 epoch：{metrics["training"]["best_epoch"]}
- 完成 epoch：{metrics["training"]["epochs_completed"]}
- 参数量：{metrics["model"]["parameters"]}

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
    parser.add_argument("--dataset", type=Path, default=Path("labels/projection_support_area_v0_256"))
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=Path("reports/projection_score_mlp_unanchored_2026-11-20/metrics.json"),
    )
    parser.add_argument("--name", default="pointnet_projection_scorer_2026-11-20")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-shapes", type=int, default=24)
    parser.add_argument("--point-count", type=int, default=2048)
    parser.add_argument("--point-embedding-dim", type=int, default=1024)
    parser.add_argument("--shape-hidden-dim", type=int, default=256)
    parser.add_argument("--candidate-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-samples", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_shapes < 1 or args.point_count < 16:
        parser.error("epochs, batch-shapes, and point-count must be positive")

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
    reference_path = (root / args.reference_report).resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads((dataset_dir / "split.json").read_text(encoding="utf-8"))
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

    fit_parts = load_part_rows(fit_records, args.point_count)
    development_parts = load_part_rows(development_records, args.point_count)
    test_parts = load_part_rows(test_records, args.point_count)
    feature_mean, feature_std = feature_standardisation(fit_parts)

    model = PointNetCandidateScorer(
        feature_dim=18,
        point_channels=6,
        point_embedding_dim=args.point_embedding_dim,
        shape_hidden_dim=args.shape_hidden_dim,
        candidate_hidden_dim=args.candidate_hidden_dim,
        dropout=args.dropout,
    ).to(device)
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
        totals = {"total": 0.0, "mse": 0.0, "rank": 0.0}
        batches = 0
        for start in range(0, len(order), args.batch_shapes):
            selected = [
                fit_parts[int(index)]
                for index in order[start:start + args.batch_shapes]
            ]
            points, features, targets = make_model_inputs(
                selected,
                feature_mean,
                feature_std,
                args.point_count,
                device,
                augment=True,
                generator=generator,
            )
            predictions = model(points, features)
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
            loss = mse + args.pairwise_weight * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            totals["total"] += float(loss.item())
            totals["mse"] += float(mse.item())
            totals["rank"] += float(rank_loss.item())
            batches += 1
        scheduler.step()

        development_predictions = predict_parts(
            model,
            development_parts,
            feature_mean,
            feature_std,
            args.point_count,
            device,
        )
        development_rows = enriched_rows(
            development_predictions,
            development_parts,
        )
        development_metrics = summarise_rows(development_rows)
        current_score = (
            0.45 * development_metrics["spearman_median"]
            + 0.35 * development_metrics["ndcg_at_5_median"]
            + 0.20 * (
                1.0
                - min(development_metrics["top1_regret_median"] * 20.0, 1.0)
            )
        )
        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / batches,
            "train_mse": totals["mse"] / batches,
            "train_rank_loss": totals["rank"] / batches,
            "validation_spearman": development_metrics["spearman_median"],
            "validation_ndcg_at_5": development_metrics["ndcg_at_5_median"],
            "validation_top1_regret": development_metrics["top1_regret_median"],
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
            f"val_regret={row['validation_top1_regret']:.4f} "
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
        feature_mean,
        feature_std,
        args.point_count,
        device,
    )
    baseline_predictions = [
        score_from_lower_is_better(part["retained_surface"])
        for part in test_parts
    ]
    test_pointnet, test_rows = evaluate_parts(test_parts, test_predictions)
    test_baseline, baseline_rows = evaluate_parts(
        test_parts,
        baseline_predictions,
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_model = reference["test"]["model"]
    reference_model = {
        **reference_model,
        "top1_area_ratio_median": reference_model["top1_area_ratio_median"],
        "top5_best_target_score_median": reference_model[
            "top5_best_target_score_median"
        ],
    }

    metrics = {
        "dataset": {
            "path": str(dataset_dir),
            "fit_count": len(fit_parts),
            "development_count": len(development_parts),
            "test_count": len(test_parts),
            "direction_count": int(len(fit_parts[0]["targets"])),
            "feature_dim": 18,
            "point_count": args.point_count,
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
            "name": "PointNetCandidateScorer",
            "parameters": parameter_count,
            "anchor_weight": 0.0,
            "geometry_fusion": False,
        },
        "test": {
            "pointnet": test_pointnet,
            "retained_surface_baseline": test_baseline,
            "candidate_mlp_reference": reference_model,
            "per_part": test_rows,
            "baseline_per_part": baseline_rows,
        },
        "limitations": [
            "The target is a deterministic projected-area proxy, not Magics output.",
            "The final test contains one project and only 10 parts.",
            "Projected area is not a complete support-volume or manufacturability model.",
            "PointNet rotation robustness is encouraged by augmentation but is not mathematically guaranteed.",
        ],
    }
    config = {
        "name": args.name,
        "dataset": str(dataset_dir),
        "epochs": args.epochs,
        "batch_shapes": args.batch_shapes,
        "point_count": args.point_count,
        "point_channels": 6,
        "point_embedding_dim": args.point_embedding_dim,
        "shape_hidden_dim": args.shape_hidden_dim,
        "candidate_hidden_dim": args.candidate_hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pairwise_weight": args.pairwise_weight,
        "pairwise_samples": args.pairwise_samples,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "parameters": parameter_count,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "best_epoch": best_epoch,
        "anchor_weight": 0.0,
        "geometry_fusion": False,
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
        "test": metrics["test"]["pointnet"],
        "baseline": metrics["test"]["retained_surface_baseline"],
        "report": str(report_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
