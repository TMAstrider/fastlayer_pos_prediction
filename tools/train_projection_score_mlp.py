#!/usr/bin/env python3
"""Train and evaluate an MLP ranker for deterministic support projected area."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, rankdata, spearmanr


class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def score_from_lower_is_better(values: np.ndarray) -> np.ndarray:
    """Map a per-part cost to the same [0, 1] convention as area scores."""
    values = np.asarray(values, dtype=np.float64)
    minimum = float(values.min()) if len(values) else 0.0
    maximum = float(values.max()) if len(values) else 0.0
    if maximum - minimum <= 1e-12:
        return np.ones_like(values, dtype=np.float64)
    return 1.0 - (values - minimum) / (maximum - minimum)


def load_part_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts = []
    feature_names = None
    for record in records:
        with np.load(record["label_file"]) as data:
            names = list(data["features"].shape)
            if len(names) != 2:
                raise ValueError(f"Invalid feature shape {names}")
            row = {
                "record": record,
                "features": np.asarray(data["features"], dtype=np.float32),
                "directions": np.asarray(data["directions"], dtype=np.float32),
                "areas": np.asarray(data["projected_area_mm2"], dtype=np.float32),
                "targets": np.asarray(data["score"], dtype=np.float32),
                "retained_surface": np.asarray(
                    data["retained_surface_mm2"], dtype=np.float32
                ),
            }
            row["anchor_scores"] = score_from_lower_is_better(
                row["retained_surface"]
            ).astype(np.float32)
            parts.append(row)
        feature_names = names[1]
    if not parts:
        raise ValueError("No labelled parts found")
    expected = parts[0]["features"].shape[1]
    if any(part["features"].shape[1] != expected for part in parts):
        raise ValueError("Inconsistent feature dimensions")
    return parts


def choose_development_project(
    records: list[dict[str, Any]],
    target_count: int = 10,
) -> str:
    """Choose one whole project for model selection, never individual records."""
    counts = Counter(record["source_project"] for record in records)
    if len(counts) < 2:
        raise ValueError("Project-isolated development split requires at least two projects")
    return min(
        counts,
        key=lambda project: (abs(counts[project] - target_count), project),
    )


def standardisation(parts: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
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


def ndcg_at_k(targets: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    ideal = np.argsort(-targets)
    discounts = 1.0 / np.log2(np.arange(2, min(k, len(scores)) + 2))
    actual = np.asarray(targets[order[:len(discounts)]], dtype=np.float64)
    best = np.asarray(targets[ideal[:len(discounts)]], dtype=np.float64)
    denominator = float((best * discounts).sum())
    if denominator <= 1e-12:
        return 1.0
    return float((actual * discounts).sum() / denominator)


def evaluate_part(
    predictions: np.ndarray,
    targets: np.ndarray,
    areas: np.ndarray,
    directions: np.ndarray,
) -> dict[str, float]:
    predicted_index = int(np.argmax(predictions))
    oracle_index = int(np.argmax(targets))
    top5 = np.argsort(-predictions)[:5]
    spearman = spearmanr(predictions, targets).statistic
    kendall = kendalltau(predictions, targets).statistic
    angle = float(np.degrees(np.arccos(np.clip(
        np.dot(directions[predicted_index], directions[oracle_index]),
        -1.0,
        1.0,
    ))))
    return {
        "mae_score": float(np.mean(np.abs(predictions - targets))),
        "rmse_score": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "spearman": float(spearman),
        "kendall": float(kendall),
        "ndcg_at_5": ndcg_at_k(targets, predictions, 5),
        "ndcg_at_10": ndcg_at_k(targets, predictions, 10),
        "top1_target_score": float(targets[predicted_index]),
        "top1_regret": float(max(targets[oracle_index] - targets[predicted_index], 0.0)),
        "top5_best_target_score": float(targets[top5].max()),
        "top5_regret": float(max(targets[oracle_index] - targets[top5].max(), 0.0)),
        "top1_area_ratio": float(
            areas[predicted_index] / max(float(areas[oracle_index]), 1e-9)
        ),
        "top1_angle_to_oracle_deg": angle,
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in rows[0] if isinstance(rows[0][key], (float, int))]
    return {
        f"{key}_{stat}": float(function([row[key] for row in rows]))
        for key in keys
        for stat, function in (("mean", np.mean), ("median", np.median))
    }


def evaluate_parts(
    predictions: list[np.ndarray],
    parts: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    rows = []
    for prediction, part in zip(predictions, parts):
        rows.append({
            "source_project": part["record"]["source_project"],
            "source_part_id": part["record"]["source_part_id"],
            **evaluate_part(
                prediction,
                part["targets"],
                part["areas"],
                part["directions"],
            ),
        })
    return rows


def evaluate_baseline(
    parts: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    predictions = [
        score_from_lower_is_better(part["retained_surface"])
        for part in parts
    ]
    return evaluate_parts(predictions, parts)


def rank_fusion(
    prediction: np.ndarray,
    anchor: np.ndarray,
    anchor_weight: float,
) -> np.ndarray:
    """Fuse candidate rankings on a per-part scale, preserving all geometry."""
    prediction_rank = rankdata(prediction, method="average") / len(prediction)
    anchor_rank = rankdata(anchor, method="average") / len(anchor)
    return prediction_rank + anchor_weight * anchor_rank


def ranking_selection_score(rows: list[dict[str, float | str]]) -> float:
    metrics = aggregate_metrics(rows)
    return (
        0.45 * metrics["spearman_median"]
        + 0.35 * metrics["ndcg_at_5_median"]
        + 0.20 * (1.0 - min(metrics["top1_regret_median"] * 20.0, 1.0))
    )


@torch.no_grad()
def predict_parts(
    model: nn.Module,
    parts: list[dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    anchor_weight: float = 0.0,
    residual_scale: float = 1.0,
) -> list[np.ndarray]:
    model.eval()
    predictions = []
    for part in parts:
        features = torch.from_numpy(
            normalise_features(part["features"], mean, std)
        ).to(device)
        prediction = residual_scale * model(features)
        if anchor_weight:
            anchor = torch.from_numpy(part["anchor_scores"]).to(device)
            prediction = prediction + anchor_weight * anchor
        predictions.append(prediction.float().cpu().numpy())
    return predictions


def zero_last_linear_layer(model: CandidateMLP) -> None:
    """Start anchored training from the deterministic baseline exactly."""
    last = model.network[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("CandidateMLP must end with a Linear layer")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


def write_report(
    path: Path,
    arguments: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    model_metrics = metrics["test"]["model"]
    hybrid_metrics = metrics["test"]["hybrid"]
    surface_metrics = metrics["test"]["retained_surface_baseline"]
    development_model = metrics["development"]["model"]
    development_hybrid = metrics["development"]["hybrid"]
    development_surface = metrics["development"]["retained_surface_baseline"]
    rows = "\n".join(
        "| {part} | {id} | {spearman:.3f} | {top1:.3f} | {top5:.3f} | {ratio:.3f} |".format(
            part=row["source_project"],
            id=row["source_part_id"],
            spearman=row["spearman"],
            top1=row["top1_regret"],
            top5=row["top5_regret"],
            ratio=row["top1_area_ratio"],
        )
        for row in metrics["test"]["hybrid_per_part"]
    )
    development_rows = "\n".join(
        "| {part} | {id} | {spearman:.3f} | {top1:.3f} | {top5:.3f} | {ratio:.3f} |".format(
            part=row["source_project"],
            id=row["source_part_id"],
            spearman=row["spearman"],
            top1=row["top1_regret"],
            top5=row["top5_regret"],
            ratio=row["top1_area_ratio"],
        )
        for row in metrics["development"]["hybrid_per_part"]
    )
    path.write_text(
        f"""# 投影面积评分 MLP：项目隔离训练、选模与测试

## 目的

验证 MLP 能否从候选方向的几何特征预测确定性代理支撑投影面积，并在项目中未出现的零件上产生正确排序。用户 query 不参与训练。

## 配置

| 项目 | 值 |
|---|---:|
| 参数更新零件 | {metrics["dataset"]["fit_count"]} |
| development 零件 | {metrics["dataset"]["development_count"]} |
| test 零件 | {metrics["dataset"]["test_count"]} |
| 每件方向 | {metrics["dataset"]["direction_count"]} |
| 投影栅格 | {metrics["dataset"]["projection_grid_mm"]} mm |
| development 项目 | `{", ".join(metrics["dataset"]["development_projects"])}` |
| test 项目 | `{", ".join(metrics["dataset"]["test_projects"])}` |
| 模型 | CandidateMLP |
| 隐藏维度 | {arguments.hidden_dim} |
| epoch | {metrics["training"]["best_epoch"]} |
| batch shapes | {arguments.batch_shapes} |
| pairwise weight | {arguments.pairwise_weight} |
| retained-surface anchor weight | {arguments.anchor_weight} |
| residual scale | {arguments.residual_scale} |
| development-selected fusion weight | {metrics["fusion"]["anchor_weight"]} |

## Test 总体结果

| 指标 | 融合后中位数 | MLP 中位数 | 保留悬垂面积基线中位数 |
|---|---:|---:|---:|
| Spearman | {hybrid_metrics["spearman_median"]:.4f} | {model_metrics["spearman_median"]:.4f} | {surface_metrics["spearman_median"]:.4f} |
| Kendall | {hybrid_metrics["kendall_median"]:.4f} | {model_metrics["kendall_median"]:.4f} | {surface_metrics["kendall_median"]:.4f} |
| NDCG@5 | {hybrid_metrics["ndcg_at_5_median"]:.4f} | {model_metrics["ndcg_at_5_median"]:.4f} | {surface_metrics["ndcg_at_5_median"]:.4f} |
| NDCG@10 | {hybrid_metrics["ndcg_at_10_median"]:.4f} | {model_metrics["ndcg_at_10_median"]:.4f} | {surface_metrics["ndcg_at_10_median"]:.4f} |
| Top-1 归一化面积损失 | {hybrid_metrics["top1_regret_median"]:.4f} | {model_metrics["top1_regret_median"]:.4f} | {surface_metrics["top1_regret_median"]:.4f} |
| Top-5 归一化面积损失 | {hybrid_metrics["top5_regret_median"]:.4f} | {model_metrics["top5_regret_median"]:.4f} | {surface_metrics["top5_regret_median"]:.4f} |
| Top-1 面积 / oracle 面积 | {hybrid_metrics["top1_area_ratio_median"]:.4f} | {model_metrics["top1_area_ratio_median"]:.4f} | {surface_metrics["top1_area_ratio_median"]:.4f} |

几何特征基线按 `-retained_surface_mm2` 排序，再映射到与目标相同的
逐零件 `[0, 1]` 分数口径。它不包含目标投影面积字段。

## Development 总体结果

| 指标 | 融合后中位数 | MLP 中位数 | 保留悬垂面积基线中位数 |
|---|---:|---:|---:|
| Spearman | {development_hybrid["spearman_median"]:.4f} | {development_model["spearman_median"]:.4f} | {development_surface["spearman_median"]:.4f} |
| NDCG@5 | {development_hybrid["ndcg_at_5_median"]:.4f} | {development_model["ndcg_at_5_median"]:.4f} | {development_surface["ndcg_at_5_median"]:.4f} |
| Top-1 归一化面积损失 | {development_hybrid["top1_regret_median"]:.4f} | {development_model["top1_regret_median"]:.4f} | {development_surface["top1_regret_median"]:.4f} |
| Top-1 面积 / oracle 面积 | {development_hybrid["top1_area_ratio_median"]:.4f} | {development_model["top1_area_ratio_median"]:.4f} | {development_surface["top1_area_ratio_median"]:.4f} |

## Development 零件

| 项目 | 零件 | Spearman | Top-1 损失 | Top-5 损失 | Top-1 面积比 |
|---|---:|---:|---:|---:|---:|
{development_rows}

## Test 零件

| 项目 | 零件 | Spearman | Top-1 损失 | Top-5 损失 | Top-1 面积比 |
|---|---:|---:|---:|---:|---:|
{rows}

## 解释

- 参数更新、早停选模和最终测试使用三个按源项目隔离的子集；test 不参与训练或早停。
- 输入特征描述角度候选、豁免面积、构建高度和投影包络，不包含 `projected_union_area` 目标。
- 分数是每个零件候选集内相对投影面积的 min-max 归一化。
- `anchor_weight > 0` 时，MLP 只学习保留悬垂面积分数之上的残差；
  `residual_scale < 1` 限制网络修正幅度，初始化使训练前输出精确等于锚点。
- 最终候选分数在同一零件内部融合 MLP 排名与保留悬垂面积排名；
  融合权重只由 development 集选择，test 不参与选择。
- 最终 test 只有 10 件，用于判断方案有没有学习信号，不能当作最终泛化性能。

## 产物

- `metrics.json`
- `per_part.csv`
- `history.json`
- `training_curve.png`
- `runs/` 中保存最佳模型与配置
""",
        encoding="utf-8",
    )


def write_curve(path: Path, history: list[dict[str, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(epochs, [row["train_total_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_mse"] for row in history], label="validation MSE")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(epochs, [row["validation_spearman"] for row in history], color="#2c8a57")
    axes[1].set_ylabel("validation Spearman")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Projection-area candidate MLP")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="#f5f7fa")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", default="projection_score_mlp_2026-11-20")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-shapes", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-samples", type=int, default=64)
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Add this weight times the retained-surface score to the MLP output. "
            "The MLP then learns only the projection-overlap residual."
        ),
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Multiplier on the MLP residual when an anchor is enabled.",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--development-project",
        help=(
            "Whole source project used for early stopping. By default, choose "
            "the training project whose part count is closest to 10."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_dir = args.dataset.resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads((dataset_dir / "split.json").read_text(encoding="utf-8"))
    source_train_records = split["train_records"]
    test_records = split["validation_records"]
    development_project = (
        args.development_project
        or choose_development_project(source_train_records)
    )
    train_projects = {record["source_project"] for record in source_train_records}
    test_projects = {record["source_project"] for record in test_records}
    if development_project not in train_projects:
        parser.error("--development-project must exist in split train_records")
    if development_project in test_projects:
        parser.error("development and test projects must be disjoint")
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
    train_parts = load_part_rows(fit_records)
    development_parts = load_part_rows(development_records)
    test_parts = load_part_rows(test_records)
    mean, std = standardisation(train_parts)
    feature_dim = train_parts[0]["features"].shape[1]
    model = CandidateMLP(feature_dim, args.hidden_dim).to(device)
    if args.anchor_weight:
        zero_last_linear_layer(model)
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

    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / args.name
    report_dir = root / "reports" / args.name
    if run_dir.exists() or report_dir.exists():
        parser.error("Existing run/report directory; choose a new --name")
    run_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_parts))
        totals = {"total": 0.0, "mse": 0.0, "rank": 0.0}
        batches = 0
        for start in range(0, len(order), args.batch_shapes):
            selected = [train_parts[int(index)] for index in order[start:start + args.batch_shapes]]
            features = torch.from_numpy(np.stack([
                normalise_features(part["features"], mean, std) for part in selected
            ])).to(device)
            targets = torch.from_numpy(np.stack([
                part["targets"] for part in selected
            ])).to(device)
            anchors = torch.from_numpy(np.stack([
                part["anchor_scores"] for part in selected
            ])).to(device)
            predictions = (
                args.residual_scale * model(features)
                + args.anchor_weight * anchors
            )
            mse = F.mse_loss(predictions, targets)
            batch, candidates = targets.shape
            left = torch.randint(
                0, candidates, (batch, args.pairwise_samples), device=device
            )
            right = torch.randint(
                0, candidates, (batch, args.pairwise_samples), device=device
            )
            left_targets = torch.gather(targets, 1, left)
            right_targets = torch.gather(targets, 1, right)
            left_scores = torch.gather(predictions, 1, left)
            right_scores = torch.gather(predictions, 1, right)
            target_difference = left_targets - right_targets
            valid_pairs = target_difference.abs() > 1e-5
            if valid_pairs.any():
                rank_loss = F.softplus(
                    -torch.sign(target_difference[valid_pairs])
                    * (left_scores[valid_pairs] - right_scores[valid_pairs])
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

        predictions = predict_parts(
            model,
            development_parts,
            mean,
            std,
            device,
            args.anchor_weight,
            args.residual_scale,
        )
        validation_mse = float(np.mean([
            np.mean((prediction - part["targets"]) ** 2)
            for prediction, part in zip(predictions, development_parts)
        ]))
        validation_spearman = float(np.median([
            spearmanr(prediction, part["targets"]).statistic
            for prediction, part in zip(predictions, development_parts)
        ]))
        validation_ndcg5 = float(np.median([
            ndcg_at_k(part["targets"], prediction, 5)
            for prediction, part in zip(predictions, development_parts)
        ]))
        validation_top1_regret = float(np.median([
            max(
                float(part["targets"].max() - part["targets"][np.argmax(prediction)]),
                0.0,
            )
            for prediction, part in zip(predictions, development_parts)
        ]))
        selection_score = (
            0.45 * validation_spearman
            + 0.35 * validation_ndcg5
            + 0.20 * (1.0 - min(validation_top1_regret * 20.0, 1.0))
        )
        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / max(batches, 1),
            "train_mse": totals["mse"] / max(batches, 1),
            "train_rank_loss": totals["rank"] / max(batches, 1),
            "validation_mse": validation_mse,
            "validation_spearman": validation_spearman,
            "validation_ndcg_at_5": validation_ndcg5,
            "validation_top1_regret": validation_top1_regret,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps({key: round(value, 5) if isinstance(value, float) else value
                          for key, value in row.items()}), flush=True)
        if selection_score > best_score + 1e-5:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training produced no model state")
    model.load_state_dict(best_state)
    development_predictions = predict_parts(
        model,
        development_parts,
        mean,
        std,
        device,
        args.anchor_weight,
        args.residual_scale,
    )
    test_predictions = predict_parts(
        model,
        test_parts,
        mean,
        std,
        device,
        args.anchor_weight,
        args.residual_scale,
    )
    development_rows = evaluate_parts(development_predictions, development_parts)
    development_baseline_rows = evaluate_baseline(development_parts)
    test_rows = evaluate_parts(test_predictions, test_parts)
    test_baseline_rows = evaluate_baseline(test_parts)
    fusion_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    development_hybrid_rows_by_weight: dict[float, list[dict[str, Any]]] = {}
    for fusion_weight in fusion_grid:
        fused_predictions = [
            rank_fusion(prediction, part["anchor_scores"], fusion_weight)
            for prediction, part in zip(development_predictions, development_parts)
        ]
        development_hybrid_rows_by_weight[fusion_weight] = evaluate_parts(
            fused_predictions, development_parts
        )
    fusion_weight = max(
        fusion_grid,
        key=lambda weight: ranking_selection_score(
            development_hybrid_rows_by_weight[weight]
        ),
    )
    development_hybrid_rows = development_hybrid_rows_by_weight[fusion_weight]
    test_hybrid_predictions = [
        rank_fusion(prediction, part["anchor_scores"], fusion_weight)
        for prediction, part in zip(test_predictions, test_parts)
    ]
    test_hybrid_rows = evaluate_parts(test_hybrid_predictions, test_parts)

    metrics = {
        "dataset": {
            "path": str(dataset_dir),
            "fit_count": len(train_parts),
            "development_count": len(development_parts),
            "test_count": len(test_parts),
            "direction_count": int(manifest["direction_count"]),
            "projection_grid_mm": float(manifest["projection_grid_mm"]),
            "feature_names": manifest["feature_names"],
            "development_projects": [development_project],
            "test_projects": split["validation_projects"],
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_selection_score": float(best_score),
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
        },
        "development": {
            "model": aggregate_metrics(development_rows),
            "hybrid": aggregate_metrics(development_hybrid_rows),
            "retained_surface_baseline": aggregate_metrics(
                development_baseline_rows
            ),
            "per_part": development_rows,
            "hybrid_per_part": development_hybrid_rows,
            "baseline_per_part": development_baseline_rows,
        },
        "test": {
            "model": aggregate_metrics(test_rows),
            "hybrid": aggregate_metrics(test_hybrid_rows),
            "retained_surface_baseline": aggregate_metrics(test_baseline_rows),
            "per_part": test_rows,
            "hybrid_per_part": test_hybrid_rows,
            "baseline_per_part": test_baseline_rows,
        },
        "fusion": {
            "method": "per-part average-rank fusion",
            "anchor_weight": fusion_weight,
            "development_selection_score": ranking_selection_score(
                development_hybrid_rows
            ),
        },
        "limitations": [
            "Only one project with 10 parts is used for final test.",
            "The target is a rasterized geometric proxy, not Materialise Magics output.",
            "No user no-support query is included in training.",
        ],
    }
    config = {
        **vars(args),
        "dataset": str(dataset_dir),
        "feature_dim": feature_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "fusion_anchor_weight": fusion_weight,
    }
    torch.save({
        "model_state_dict": best_state,
        "config": config,
        "best_epoch": best_epoch,
    }, run_dir / "best_model.pt")
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import csv
    with (report_dir / "per_part.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    write_curve(report_dir / "training_curve.png", history)
    write_report(report_dir / "README.md", args, metrics)
    print(json.dumps({
        "best_epoch": best_epoch,
        "fusion_anchor_weight": fusion_weight,
        "hybrid_median": {
            "spearman": metrics["test"]["hybrid"]["spearman_median"],
            "ndcg_at_5": metrics["test"]["hybrid"]["ndcg_at_5_median"],
            "top1_regret": metrics["test"]["hybrid"]["top1_regret_median"],
            "top5_regret": metrics["test"]["hybrid"]["top5_regret_median"],
        },
        "model_median": {
            "spearman": metrics["test"]["model"]["spearman_median"],
            "ndcg_at_5": metrics["test"]["model"]["ndcg_at_5_median"],
            "top1_regret": metrics["test"]["model"]["top1_regret_median"],
            "top5_regret": metrics["test"]["model"]["top5_regret_median"],
        },
        "baseline_median": {
            "spearman": metrics["test"]["retained_surface_baseline"]["spearman_median"],
            "ndcg_at_5": metrics["test"]["retained_surface_baseline"]["ndcg_at_5_median"],
            "top1_regret": metrics["test"]["retained_surface_baseline"]["top1_regret_median"],
            "top5_regret": metrics["test"]["retained_surface_baseline"]["top5_regret_median"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
