#!/usr/bin/env python3
"""Train a history-preference head on frozen point-direction geometry features."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tools.train_pointnet_direction_scorer import (
        PointDirectionInteractionScorer,
        choose_development_project,
        load_part_rows,
        make_model_inputs,
    )
except ModuleNotFoundError:
    from train_pointnet_direction_scorer import (
        PointDirectionInteractionScorer,
        choose_development_project,
        load_part_rows,
        make_model_inputs,
    )


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_dual(
    model: PointDirectionInteractionScorer,
    parts: list[dict[str, Any]],
    point_count: int,
    device: torch.device,
    batch_shapes: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    model.eval()
    empty_prior = np.empty(0, dtype=np.float32)
    empty_global = np.empty(0, dtype=np.float32)
    outputs = []
    for start in range(0, len(parts), batch_shapes):
        selected = parts[start:start + batch_shapes]
        (
            points,
            candidate_features,
            global_features,
            _,
            _,
            history_directions,
        ) = make_model_inputs(
            selected,
            point_count,
            empty_prior,
            np.ones(0, dtype=np.float32),
            empty_global,
            np.ones(0, dtype=np.float32),
            device,
            augment=False,
            generator=None,
        )
        del global_features
        area_scores, history_logits, exact_history = model.forward_dual(
            points,
            candidate_features,
            history_directions,
        )
        for area, history, exact in zip(
            area_scores.cpu().numpy(),
            history_logits.cpu().numpy(),
            exact_history.cpu().numpy(),
        ):
            outputs.append((
                area.astype(np.float32),
                np.concatenate([
                    history.astype(np.float32),
                    np.asarray([exact], dtype=np.float32),
                ]),
            ))
    return outputs


def history_metrics(
    parts: list[dict[str, Any]],
    predictions: list[tuple[np.ndarray, np.ndarray]],
    temperature: float,
) -> dict[str, float | int]:
    exact_ranks = []
    history_angles = []
    losses = []
    for part, (_, history_logits) in zip(parts, predictions):
        candidate_count = len(part["directions"])
        exact_target = int(candidate_count)
        target = torch.tensor([exact_target], dtype=torch.long)
        logits = torch.from_numpy(history_logits[None, :])
        losses.append(float(
            F.cross_entropy(logits / temperature, target).item()
        ))
        exact_ranks.append(
            1 + int(np.sum(history_logits > history_logits[exact_target]))
        )

        candidate_logits = history_logits[:candidate_count]
        predicted_index = int(np.argmax(candidate_logits))
        directions = np.asarray(part["directions"], dtype=np.float64)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True),
            1e-12,
        )
        history_direction = np.asarray(
            part["history_direction"],
            dtype=np.float64,
        )
        history_direction /= np.linalg.norm(history_direction)
        angle = float(np.degrees(np.arccos(np.clip(
            directions[predicted_index] @ history_direction,
            -1.0,
            1.0,
        ))))
        history_angles.append(angle)
    exact_ranks_array = np.asarray(exact_ranks)
    history_angles_array = np.asarray(history_angles)
    return {
        "loss": float(np.mean(losses)),
        "exact_rank_mean": float(exact_ranks_array.mean()),
        "exact_rank_median": float(np.median(exact_ranks_array)),
        "exact_top1_count": int((exact_ranks_array == 1).sum()),
        "candidate_top1_angle_mean_deg": float(history_angles_array.mean()),
        "candidate_top1_angle_median_deg": float(
            np.median(history_angles_array)
        ),
        "candidate_top1_within_20deg_count": int(
            (history_angles_array <= 20.0).sum()
        ),
        "count": len(parts),
    }


def write_report(
    path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    history = metrics["history"]
    best = metrics["best_validation"]
    path.write_text(
        f"""# 历史摆放偏好 Head

## 设置

- 初始化：当前逐点方向交互面积模型
- 冻结：共享点云/方向编码器和面积 head
- 训练目标：历史朝上方向在额外候选中的 cross entropy
- history temperature：{config["history_temperature"]}
- 候选近邻监督权重：{config["candidate_history_weight"]}
- 目标余弦温度：{config["target_cosine_temperature"]}
- seed：{config["seed"]}
- 训练零件：{metrics["fit_count"]}
- development 零件：{metrics["development_count"]}
- test 零件未参与训练和选模

## 结果

| 指标 | 最佳 development checkpoint |
|---|---:|
| epoch | {best["epoch"]} |
| exact history loss | {best["loss"]:.4f} |
| exact history rank 中位数 | {best["exact_rank_median"]:.1f} |
| 候选 Top-1 与历史轴夹角中位数 | {best["candidate_top1_angle_median_deg"]:.2f}° |
| 候选 Top-1 在历史轴 20° 内 | {best["candidate_top1_within_20deg_count"]}/{best["count"]} |

训练耗时 `{metrics["elapsed_seconds"]:.2f}` 秒，完成 `{len(history)}` 个 epoch。

模型把真实历史方向作为第 257 个候选输入共享评分函数。融合评估需另外运行历史偏好
与面积分数的权重扫描；本文件不预先挑选 test 上的融合权重。
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
        "--init-model",
        type=Path,
        default=Path(
            "runs/pointnet_point_direction_topk_controlled_"
            "4096_922_10_100_2026-11-20/best_model.pt"
        ),
    )
    parser.add_argument(
        "--name",
        default="history_preference_head_2026-11-24",
    )
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--batch-shapes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--history-temperature", type=float, default=0.1)
    parser.add_argument(
        "--candidate-history-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--target-cosine-temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument("--history-hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (
        args.point_count < 16
        or args.batch_shapes < 1
        or args.epochs < 1
        or args.patience < 1
    ):
        parser.error("point count, batch, epochs, and patience must be positive")
    if args.history_temperature <= 0:
        parser.error("history temperature must be positive")
    if (
        args.candidate_history_weight < 0
        or args.target_cosine_temperature <= 0
    ):
        parser.error(
            "candidate history weight must be non-negative and target "
            "temperature must be positive"
        )

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
    split_path = (root / args.split_file).resolve()
    init_model_path = (root / args.init_model).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_train = split["train_records"]
    test_records = split["validation_records"]
    development_project = choose_development_project(source_train)
    fit_records = [
        record
        for record in source_train
        if record["source_project"] != development_project
    ]
    development_records = [
        record
        for record in source_train
        if record["source_project"] == development_project
    ]

    fit_parts = load_part_rows(fit_records, args.point_count)
    development_parts = load_part_rows(
        development_records,
        args.point_count,
    )
    # Load the test split now so a malformed locked test set fails before training.
    load_part_rows(test_records, args.point_count)

    checkpoint = torch.load(
        init_model_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    model = PointDirectionInteractionScorer(
        point_channels=6,
        point_embedding_dim=int(config["interaction_point_dim"]),
        interaction_hidden_dim=int(config["interaction_hidden_dim"]),
        direction_hidden_dim=int(config["direction_hidden_dim"]),
        candidate_chunk_size=int(config["candidate_chunk_size"]),
        dropout=float(config["dropout"]),
        enable_history_head=True,
        history_hidden_dim=args.history_hidden_dim,
    ).to(device)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("history_head.")
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"Area checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("history_head.")
    optimizer = torch.optim.AdamW(
        model.history_head.parameters(),
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

    empty_prior = np.empty(0, dtype=np.float32)
    empty_global = np.empty(0, dtype=np.float32)
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.eval()
        model.history_head.train()
        order = np.random.permutation(len(fit_parts))
        total_loss = 0.0
        total_exact_loss = 0.0
        total_candidate_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_shapes):
            selected = [
                fit_parts[int(index)]
                for index in order[start:start + args.batch_shapes]
            ]
            (
                points,
                candidate_features,
                _,
                targets,
                _,
                history_directions,
            ) = make_model_inputs(
                selected,
                args.point_count,
                empty_prior,
                np.ones(0, dtype=np.float32),
                empty_global,
                np.ones(0, dtype=np.float32),
                device,
                augment=True,
                generator=generator,
            )
            del targets
            _, candidate_history, exact_history = model.forward_dual(
                points,
                candidate_features,
                history_directions,
            )
            history_logits = torch.cat(
                [candidate_history, exact_history[:, None]],
                dim=1,
            )
            labels = torch.full(
                (len(selected),),
                history_logits.shape[1] - 1,
                dtype=torch.long,
                device=device,
            )
            exact_loss = F.cross_entropy(
                history_logits / args.history_temperature,
                labels,
            )
            candidate_directions = F.normalize(
                candidate_features[:, :, :3],
                dim=-1,
            )
            normalized_history = F.normalize(
                history_directions,
                dim=-1,
            )
            candidate_cosine = torch.einsum(
                "bud,bd->bu",
                candidate_directions,
                normalized_history,
            )
            candidate_targets = F.softmax(
                candidate_cosine / args.target_cosine_temperature,
                dim=1,
            )
            candidate_loss = -(
                candidate_targets
                * F.log_softmax(
                    candidate_history / args.history_temperature,
                    dim=1,
                )
            ).sum(dim=1).mean()
            loss = (
                exact_loss
                + args.candidate_history_weight * candidate_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.history_head.parameters(),
                2.0,
            )
            optimizer.step()
            total_loss += float(loss.item())
            total_exact_loss += float(exact_loss.item())
            total_candidate_loss += float(candidate_loss.item())
            batches += 1
        scheduler.step()

        development_predictions = predict_dual(
            model,
            development_parts,
            args.point_count,
            device,
            args.batch_shapes,
        )
        development = history_metrics(
            development_parts,
            development_predictions,
            args.history_temperature,
        )
        row = {
            "epoch": epoch,
            "train_history_loss": total_loss / batches,
            "train_exact_history_loss": total_exact_loss / batches,
            "train_candidate_history_loss": total_candidate_loss / batches,
            "validation_history_loss": development["loss"],
            "validation_exact_rank_median": (
                development["exact_rank_median"]
            ),
            "validation_candidate_top1_angle_median_deg": (
                development["candidate_top1_angle_median_deg"]
            ),
            "validation_candidate_top1_within_20deg": (
                development["candidate_top1_within_20deg_count"]
            ),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        if development["loss"] < best_loss:
            best_loss = float(development["loss"])
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
            f"loss={row['train_history_loss']:.4f} "
            f"val_loss={row['validation_history_loss']:.4f} "
            f"val_rank={row['validation_exact_rank_median']:.1f} "
            f"val_angle={row['validation_candidate_top1_angle_median_deg']:.2f} "
            f"best={best_epoch}",
            flush=True,
        )
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("History head training did not produce a checkpoint")
    model.load_state_dict(best_state)
    development_predictions = predict_dual(
        model,
        development_parts,
        args.point_count,
        device,
        args.batch_shapes,
    )
    best_metrics = history_metrics(
        development_parts,
        development_predictions,
        args.history_temperature,
    )
    best_metrics["epoch"] = best_epoch
    history_head_parameters = sum(
        parameter.numel()
        for parameter in model.history_head.parameters()
    )
    output_config = {
        "name": args.name,
        "dataset": str(dataset_dir),
        "split_file": str(split_path),
        "init_model": str(init_model_path),
        "development_project": development_project,
        "point_count": args.point_count,
        "batch_shapes": args.batch_shapes,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "history_temperature": args.history_temperature,
        "candidate_history_weight": args.candidate_history_weight,
        "target_cosine_temperature": args.target_cosine_temperature,
        "history_hidden_dim": args.history_hidden_dim,
        "history_head_parameters": history_head_parameters,
        "backbone_frozen": True,
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
    }
    metrics = {
        "fit_count": len(fit_parts),
        "development_count": len(development_parts),
        "test_count": len(test_records),
        "best_validation": best_metrics,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "config": output_config,
    }, run_dir / "history_head_model.pt")
    for directory in (run_dir, report_dir):
        (directory / "config.json").write_text(
            json.dumps(output_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(report_dir / "README.md", output_config, metrics)
    print(json.dumps({
        "name": args.name,
        "best_epoch": best_epoch,
        "elapsed_seconds": metrics["elapsed_seconds"],
        "best_validation": best_metrics,
        "report": str(report_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
