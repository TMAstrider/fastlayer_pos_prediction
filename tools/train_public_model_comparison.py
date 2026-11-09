"""Compare PointNet and Point Transformer V3 on the public pyramid benchmark.

All models use the same fixed point clouds, official split, unit 3-D axis target,
cosine direction loss, and angular evaluation.  The PTv3 branch is deliberately
implemented here with gradients enabled; ``feature_extraction.py`` is for the
separate frozen-encoder experiment and cannot be used for this comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "runs/public_pyramids_axis_v2/pointcloud_cache.npz"
DEFAULT_WEIGHTS = ROOT / "checkpoints/utonia_tiny.pth"
DEFAULT_OUTPUT = ROOT / "runs/public_pyramids_model_comparison"


class PointNetAxis(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 3),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.point_mlp(points.transpose(1, 2).contiguous())
        features = features.amax(dim=2)
        return nn.functional.normalize(self.regressor(features), dim=-1, eps=1e-8)


class DirectionHead(nn.Module):
    def __init__(self, feature_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dimension, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.layers(features), dim=-1, eps=1e-8)


def angle_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = predicted / np.clip(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-8, None)
    target = target / np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-8, None)
    angles = np.degrees(np.arccos(np.clip(np.sum(predicted * target, axis=1), -1.0, 1.0)))
    return {
        "count": int(len(angles)),
        "mean_angle_deg": float(angles.mean()),
        "median_angle_deg": float(np.median(angles)),
        "p90_angle_deg": float(np.percentile(angles, 90)),
        "within_1deg_pct": float((angles <= 1).mean() * 100),
        "within_3deg_pct": float((angles <= 3).mean() * 100),
        "within_5deg_pct": float((angles <= 5).mean() * 100),
        "within_10deg_pct": float((angles <= 10).mean() * 100),
    }


def prepare_cloud(points: np.ndarray, grid_size: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    grid = np.floor((points - points.min(axis=0)) / grid_size).astype(np.int32)
    _, first = np.unique(grid, axis=0, return_index=True)
    first.sort()
    return points[first], grid[first]


def make_ptv3( weights: Path, device: torch.device, pretrained: bool, patch_size: int):
    sys.path.insert(0, str(ROOT / "external" / "Utonia"))
    import utonia
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    config.update(
        enable_flash=False,
        enc_patch_size=[patch_size] * len(config["enc_depths"]),
        shuffle_orders=False,
        traceable=True,
        enc_mode=True,
    )
    encoder = utonia.model.PointTransformerV3(**config)
    if pretrained:
        encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    return encoder.to(device), config


def ptv3_forward(encoder: nn.Module, prepared: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> torch.Tensor:
    import torch_scatter

    sizes = [len(item[0]) for item in prepared]
    coords = torch.from_numpy(np.concatenate([item[0] for item in prepared])).to(device)
    grid = torch.from_numpy(np.concatenate([item[1] for item in prepared])).to(device)
    # Utonia's tiny checkpoint was trained with XYZ + color + normal = 9 channels.
    features = torch.cat([coords, torch.zeros_like(coords), torch.zeros_like(coords)], dim=1)
    point = encoder({
        "coord": coords,
        "grid_coord": grid,
        "feat": features,
        "offset": torch.tensor(np.cumsum(sizes), device=device, dtype=torch.long),
    })
    pooled = []
    while True:
        offsets = torch.cat([point.offset.new_zeros(1), point.offset])
        pooled.append(torch_scatter.segment_csr(point.feat.float(), offsets, reduce="mean"))
        pooled.append(torch_scatter.segment_csr(point.feat.float(), offsets, reduce="max"))
        if "pooling_parent" not in point:
            break
        point = point.pooling_parent
    return torch.cat(pooled, dim=1)


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - (prediction * target).sum(dim=-1)).mean()


@torch.no_grad()
def evaluate_ptv3(encoder, head, loader, prepared_clouds, targets, device):
    encoder.eval(); head.eval()
    predictions = []
    losses = []
    for indices in loader:
        batch_prepared = [prepared_clouds[int(i)] for i in indices]
        target = torch.from_numpy(targets[indices]).to(device)
        pred = head(ptv3_forward(encoder, batch_prepared, device))
        losses.append(float(cosine_loss(pred, target).item()) * len(indices))
        predictions.append(pred.cpu().numpy())
    pred = np.concatenate(predictions)
    metrics = angle_metrics(pred, targets)
    metrics["cosine_loss"] = float(sum(losses) / len(targets))
    return metrics, pred


@torch.no_grad()
def evaluate_pointnet(model, loader, device):
    model.eval(); predictions = []; targets = []; total = 0.0
    for points, target in loader:
        pred = model(points.to(device, non_blocking=True))
        total += float(cosine_loss(pred, target.to(device, non_blocking=True)).item()) * len(points)
        predictions.append(pred.cpu().numpy()); targets.append(target.numpy())
    pred = np.concatenate(predictions); target = np.concatenate(targets)
    metrics = angle_metrics(pred, target); metrics["cosine_loss"] = total / len(target)
    return metrics, pred, target


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_pointnet(args, data, device, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_x = torch.from_numpy(data["train_points"]); train_y = torch.from_numpy(data["train_targets"])
    val_x = torch.from_numpy(data["validation_points"]); val_y = torch.from_numpy(data["validation_targets"])
    test_x = torch.from_numpy(data["test_points"]); test_y = torch.from_numpy(data["test_targets"])
    train_loader = DataLoader(TensorDataset(train_x, train_y), args.pointnet_batch, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TensorDataset(val_x, val_y), args.eval_batch, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TensorDataset(test_x, test_y), args.eval_batch, shuffle=False, pin_memory=device.type == "cuda")
    model = PointNetAxis().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pointnet_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pointnet_epochs)
    best = float("inf"); best_epoch = 0; history = []; started = time.monotonic()
    for epoch in range(1, args.pointnet_epochs + 1):
        model.train(); train_loss = 0.0; count = 0; epoch_start = time.monotonic()
        for points, target in train_loader:
            points = points.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            pred = model(points); loss = cosine_loss(pred, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_loss += float(loss.item()) * len(points); count += len(points)
        scheduler.step()
        val_metrics, _, _ = evaluate_pointnet(model, val_loader, device)
        row = {"epoch": epoch, "train_cosine_loss": train_loss / count, "validation": val_metrics,
               "epoch_seconds": time.monotonic() - epoch_start}
        history.append(row)
        if val_metrics["mean_angle_deg"] < best:
            best = val_metrics["mean_angle_deg"]; best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, output / "best_model.pt")
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[pointnet] epoch {epoch:03d}: train={row['train_cosine_loss']:.5f} val={best:.3f} deg ({row['epoch_seconds']:.1f}s)", flush=True)
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_pred, test_target = evaluate_pointnet(model, test_loader, device)
    report = {"model": "pointnet_scratch", "best_epoch": best_epoch, "training_seconds": time.monotonic() - started,
              "parameter_count": sum(p.numel() for p in model.parameters()), "best_validation": history[best_epoch - 1]["validation"],
              "test": test_metrics, "history": history, "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0}
    np.savez_compressed(output / "test_predictions.npz", predicted_axis=test_pred, target_axis=test_target)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_ptv3(args, data, device, output: Path, pretrained: bool, prepared=None):
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    splits = {name: data[f"{name}_points"] for name in ("train", "validation", "test")}
    targets = {name: data[f"{name}_targets"] for name in ("train", "validation", "test")}
    if prepared is None:
        prepared = {}
        for name in splits:
            prepared[name] = []
            for index, cloud in enumerate(splits[name], 1):
                prepared[name].append(prepare_cloud(cloud, args.grid_size))
                if index % 1000 == 0 or index == len(splits[name]):
                    print(f"Prepared {name}: {index}/{len(splits[name])} point clouds", flush=True)
    encoder, config = make_ptv3(args.weights, device, pretrained, args.patch_size)
    feature_dim = sum(2 * int(c) for c in config["enc_channels"])
    head = DirectionHead(feature_dim).to(device)
    if pretrained:
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters(), "lr": args.ptv3_encoder_lr},
            {"params": head.parameters(), "lr": args.ptv3_head_lr},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.ptv3_scratch_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.ptv3_epochs)
    train_indices = np.arange(len(splits["train"])); val_indices = np.arange(len(splits["validation"]))
    test_indices = np.arange(len(splits["test"]))
    best = float("inf"); best_epoch = 0; history = []; started = time.monotonic()
    for epoch in range(1, args.ptv3_epochs + 1):
        encoder.train(); head.train(); np.random.shuffle(train_indices); train_loss = 0.0; count = 0; epoch_start = time.monotonic()
        for start in range(0, len(train_indices), args.ptv3_batch):
            indices = train_indices[start:start + args.ptv3_batch]
            pred = head(ptv3_forward(encoder, [prepared["train"][int(i)] for i in indices], device))
            target = torch.from_numpy(targets["train"][indices]).to(device)
            loss = cosine_loss(pred, target) / args.grad_accumulation
            loss.backward()
            if ((start // args.ptv3_batch + 1) % args.grad_accumulation == 0) or start + len(indices) >= len(train_indices):
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.item()) * args.grad_accumulation * len(indices); count += len(indices)
        scheduler.step()
        # Evaluate in small batches to cap memory.
        val_metrics, val_pred = evaluate_ptv3(encoder, head, range_batches(val_indices, args.eval_batch), prepared["validation"], targets["validation"], device)
        row = {"epoch": epoch, "train_cosine_loss": train_loss / count, "validation": val_metrics,
               "epoch_seconds": time.monotonic() - epoch_start}
        history.append(row)
        if val_metrics["mean_angle_deg"] < best:
            best = val_metrics["mean_angle_deg"]; best_epoch = epoch
            torch.save({"encoder_state_dict": encoder.state_dict(), "head_state_dict": head.state_dict(), "epoch": epoch, "config": config}, output / "best_model.pt")
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[{output.name}] epoch {epoch:03d}: train={row['train_cosine_loss']:.5f} val={best:.3f} deg ({row['epoch_seconds']:.1f}s)", flush=True)
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint["encoder_state_dict"]); head.load_state_dict(checkpoint["head_state_dict"])
    test_metrics, test_pred = evaluate_ptv3(encoder, head, range_batches(test_indices, args.eval_batch), prepared["test"], targets["test"], device)
    report = {"model": "ptv3_pretrained_finetune" if pretrained else "ptv3_scratch", "best_epoch": best_epoch,
              "training_seconds": time.monotonic() - started, "parameter_count": sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters()),
              "best_validation": history[best_epoch - 1]["validation"], "test": test_metrics, "history": history,
              "batch_size": args.ptv3_batch, "gradient_accumulation": args.grad_accumulation,
              "effective_batch_size": args.ptv3_batch * args.grad_accumulation, "patch_size": args.patch_size,
              "grid_size": args.grid_size, "feature_dimension": feature_dim,
              "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0}
    np.savez_compressed(output / "test_predictions.npz", predicted_axis=test_pred, target_axis=targets["test"])
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def range_batches(indices: np.ndarray, batch_size: int):
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=["pointnet", "ptv3_scratch", "ptv3_pretrained"], default=["pointnet", "ptv3_scratch", "ptv3_pretrained"])
    parser.add_argument("--pointnet-epochs", type=int, default=120); parser.add_argument("--pointnet-batch", type=int, default=64); parser.add_argument("--pointnet-lr", type=float, default=1e-3)
    parser.add_argument("--ptv3-epochs", type=int, default=60); parser.add_argument("--ptv3-batch", type=int, default=4); parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--ptv3-scratch-lr", type=float, default=2e-4); parser.add_argument("--ptv3-encoder-lr", type=float, default=3e-5); parser.add_argument("--ptv3-head-lr", type=float, default=1e-3)
    parser.add_argument("--eval-batch", type=int, default=8); parser.add_argument("--grid-size", type=float, default=0.02); parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--log-every", type=int, default=5); parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.cache.is_file(): parser.error(f"cache not found: {args.cache}")
    if any(x <= 0 for x in (args.pointnet_epochs, args.pointnet_batch, args.ptv3_epochs, args.ptv3_batch, args.grad_accumulation, args.eval_batch)):
        parser.error("epoch and batch arguments must be positive")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}; cache: {args.cache}", flush=True)
    with np.load(args.cache) as loaded:
        data = {key: loaded[key] for key in loaded.files if key.endswith("_points") or key.endswith("_targets")}
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    if "pointnet" in args.models:
        reports["pointnet"] = run_pointnet(args, data, device, args.output / "pointnet")
        torch.cuda.empty_cache()
    if "ptv3_scratch" in args.models:
        prepared = {}
        for name in ("train", "validation", "test"):
            prepared[name] = []
            clouds = data[f"{name}_points"]
            for index, cloud in enumerate(clouds, 1):
                prepared[name].append(prepare_cloud(cloud, args.grid_size))
                if index % 1000 == 0 or index == len(clouds):
                    print(f"Prepared {name}: {index}/{len(clouds)} point clouds", flush=True)
        reports["ptv3_scratch"] = run_ptv3(args, data, device, args.output / "ptv3_scratch", False, prepared)
        torch.cuda.empty_cache()
    if "ptv3_pretrained" in args.models:
        if "ptv3_scratch" not in args.models:
            prepared = {}
            for name in ("train", "validation", "test"):
                prepared[name] = []
                clouds = data[f"{name}_points"]
                for index, cloud in enumerate(clouds, 1):
                    prepared[name].append(prepare_cloud(cloud, args.grid_size))
                    if index % 1000 == 0 or index == len(clouds):
                        print(f"Prepared {name}: {index}/{len(clouds)} point clouds", flush=True)
        reports["ptv3_pretrained"] = run_ptv3(args, data, device, args.output / "ptv3_pretrained", True, prepared)
        torch.cuda.empty_cache()
    summary = {"cache": str(args.cache.resolve()), "device": str(device), "seed": args.seed, "experiments": reports}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({name: value["test"] for name, value in reports.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
