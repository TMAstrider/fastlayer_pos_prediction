#!/usr/bin/env python3
"""Train PointNet to score directions directly from 3D direction coordinates."""
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
        random_rotations,
        seed_everything,
        summarise_rows,
        write_csv,
    )
    from train_projection_score_mlp import score_from_lower_is_better


class PointNetDirectionScorer(nn.Module):
    """Encode a shape once, then score candidate directions from XYZ only."""

    def __init__(
        self,
        point_channels: int = 6,
        point_embedding_dim: int = 1024,
        shape_hidden_dim: int = 256,
        direction_hidden_dim: int = 128,
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
        self.direction_head = nn.Sequential(
            nn.Linear(3, direction_hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(direction_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(direction_hidden_dim, direction_hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(direction_hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(shape_hidden_dim + direction_hidden_dim, 256),
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
        directions: torch.Tensor,
    ) -> torch.Tensor:
        batch = points.shape[0]
        direction_count = directions.shape[1]
        point_features = self.point_mlp(
            points.transpose(1, 2).contiguous()
        )
        shape_embedding = self.shape_head(point_features.amax(dim=2))
        direction_embedding = self.direction_head(
            F.normalize(directions, dim=-1)
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
    device: torch.device,
    augment: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(parts)
    direction_count = len(parts[0]["directions"])
    points = np.empty((batch, point_count, 6), dtype=np.float32)
    directions = np.empty((batch, direction_count, 3), dtype=np.float32)
    targets = np.empty((batch, direction_count), dtype=np.float32)
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
        directions[index] = part_directions / np.clip(
            np.linalg.norm(part_directions, axis=1, keepdims=True),
            1e-9,
            None,
        )
        targets[index] = part["targets"]

    points_tensor = torch.from_numpy(points).to(device)
    directions_tensor = torch.from_numpy(directions).to(device)
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
        directions_tensor = torch.bmm(
            directions_tensor,
            rotations.transpose(1, 2),
        )
    return points_tensor, directions_tensor, targets_tensor


@torch.no_grad()
def predict_parts(
    model: nn.Module,
    parts: list[dict[str, Any]],
    point_count: int,
    device: torch.device,
    batch_shapes: int = 8,
) -> list[np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(parts), batch_shapes):
        selected = parts[start:start + batch_shapes]
        points, directions, _ = make_model_inputs(
            selected,
            point_count,
            device,
            augment=False,
            generator=None,
        )
        predictions.extend(
            model(points, directions).float().cpu().numpy()
        )
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
    direction_only = metrics["test"]["pointnet_direction_only"]
    pointnet_features = metrics["test"]["pointnet_with_18_features"]
    candidate_mlp = metrics["test"]["candidate_mlp"]
    baseline = metrics["test"]["retained_surface_baseline"]
    path.write_text(
        f"""# 4096 点 PointNet 方向评分器

## 输入与网络

- 每件零件 {arguments.point_count} 点，输入 XYZ＋法向，共 6 个通道。
- PointNet 对每件形状只编码一次，得到 {arguments.point_embedding_dim} 维 embedding。
- 每个候选方向只输入 3 维 XYZ，不使用任何 18 维候选几何特征。
- 形状 embedding 与方向 embedding 拼接后输出一个标量分数。
- 训练时对点云、法向和候选方向施加同一个随机 SO(3) 旋转。
- `anchor_weight=0`；几何 baseline 不进入预测、损失或模型选择。

## Test 结果

| 指标 | 方向-only PointNet | PointNet＋18 特征 | CandidateMLP |
|---|---:|---:|---:|
| Spearman 中位数 | {direction_only["spearman_median"]:.4f} | {pointnet_features["spearman_median"]:.4f} | {candidate_mlp["spearman_median"]:.4f} |
| Kendall 中位数 | {direction_only["kendall_median"]:.4f} | {pointnet_features["kendall_median"]:.4f} | {candidate_mlp["kendall_median"]:.4f} |
| NDCG@5 中位数 | {direction_only["ndcg_at_5_median"]:.4f} | {pointnet_features["ndcg_at_5_median"]:.4f} | {candidate_mlp["ndcg_at_5_median"]:.4f} |
| NDCG@10 中位数 | {direction_only["ndcg_at_10_median"]:.4f} | {pointnet_features["ndcg_at_10_median"]:.4f} | {candidate_mlp["ndcg_at_10_median"]:.4f} |
| Top-1 面积/oracle 中位数 | {direction_only["top1_area_ratio_median"]:.4f} | {pointnet_features["top1_area_ratio_median"]:.4f} | {candidate_mlp["top1_area_ratio_median"]:.4f} |
| Top-5 精确命中 | {direction_only["top5_exact_count"]}/{direction_only["count"]} | {pointnet_features["top5_exact_count"]}/{pointnet_features["count"]} | — |

独立几何 baseline 的 Spearman 中位数为
{baseline["spearman_median"]:.4f}，NDCG@5 中位数为
{baseline["ndcg_at_5_median"]:.4f}。它与当前候选特征高度相关，因此只用于
检查标签可预测性，不作为网络输入。

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
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("labels/projection_support_area_v0_256"),
    )
    parser.add_argument(
        "--pointnet-reference-report",
        type=Path,
        default=Path(
            "reports/pointnet_projection_scorer_2026-11-20/metrics.json"
        ),
    )
    parser.add_argument(
        "--mlp-reference-report",
        type=Path,
        default=Path(
            "reports/projection_score_mlp_unanchored_2026-11-20/metrics.json"
        ),
    )
    parser.add_argument(
        "--name",
        default="pointnet_direction_only_4096_2026-11-20",
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
    pointnet_reference_path = (root / args.pointnet_reference_report).resolve()
    mlp_reference_path = (root / args.mlp_reference_report).resolve()
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
    development_parts = load_part_rows(
        development_records,
        args.point_count,
    )
    test_parts = load_part_rows(test_records, args.point_count)

    model = PointNetDirectionScorer(
        point_channels=6,
        point_embedding_dim=args.point_embedding_dim,
        shape_hidden_dim=args.shape_hidden_dim,
        direction_hidden_dim=args.direction_hidden_dim,
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
            points, directions, targets = make_model_inputs(
                selected,
                args.point_count,
                device,
                augment=True,
                generator=generator,
            )
            predictions = model(points, directions)
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
        args.point_count,
        device,
    )
    baseline_predictions = [
        score_from_lower_is_better(part["retained_surface"])
        for part in test_parts
    ]
    test_direction_only, test_rows = evaluate_parts(
        test_parts,
        test_predictions,
    )
    test_baseline, baseline_rows = evaluate_parts(
        test_parts,
        baseline_predictions,
    )
    pointnet_reference = load_reference_model(pointnet_reference_path)
    candidate_mlp = load_reference_model(mlp_reference_path)

    metrics = {
        "dataset": {
            "path": str(dataset_dir),
            "fit_count": len(fit_parts),
            "development_count": len(development_parts),
            "test_count": len(test_parts),
            "direction_count": int(len(fit_parts[0]["targets"])),
            "candidate_input_dim": 3,
            "candidate_input": ["direction_x", "direction_y", "direction_z"],
            "point_count": args.point_count,
            "point_channels": ["x", "y", "z", "normal_x", "normal_y", "normal_z"],
            "uses_18_candidate_geometry_features": False,
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
            "name": "PointNetDirectionScorer",
            "parameters": parameter_count,
            "anchor_weight": 0.0,
            "geometry_fusion": False,
            "uses_18_candidate_geometry_features": False,
        },
        "test": {
            "pointnet_direction_only": test_direction_only,
            "pointnet_with_18_features": pointnet_reference,
            "candidate_mlp": candidate_mlp,
            "retained_surface_baseline": test_baseline,
            "per_part": test_rows,
            "baseline_per_part": baseline_rows,
        },
        "limitations": [
            "The target is a deterministic projected-area proxy, not Magics output.",
            "The final test contains one project and only 10 parts.",
            "The model scores 256 sampled directions rather than a continuous sphere.",
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
        "candidate_input_dim": 3,
        "candidate_input": ["direction_x", "direction_y", "direction_z"],
        "uses_18_candidate_geometry_features": False,
        "point_embedding_dim": args.point_embedding_dim,
        "shape_hidden_dim": args.shape_hidden_dim,
        "direction_hidden_dim": args.direction_hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pairwise_weight": args.pairwise_weight,
        "pairwise_samples": args.pairwise_samples,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "parameters": parameter_count,
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
        "test": metrics["test"]["pointnet_direction_only"],
        "baseline": metrics["test"]["retained_surface_baseline"],
        "report": str(report_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
