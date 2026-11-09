"""Train a from-scratch PointNet orientation regressor on the 193 real parts.

The model predicts the build-up/up axis in the current input coordinate frame.
Each historical pose is treated as one positive reference: for a sampled random
rotation R, points become R p and the target becomes R [0, 0, 1].
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTCLOUDS = ROOT / "outputs/pointclouds"
DEFAULT_SPLIT = ROOT / "runs/utonia_tiny_direction_ablation/with_normals_r64/split.json"
DEFAULT_OUTPUT = ROOT / "runs/pointnet_real_parts_scratch"


class PointNetAxis(nn.Module):
    """Shared point MLP, symmetric max pooling and a unit-axis regression head."""

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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_rotations(count: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Uniform SO(3) samples, returned as column-vector rotation matrices."""
    q = torch.randn((count, 4), generator=generator, device=device)
    q = nn.functional.normalize(q, dim=1)
    x, y, z, w = q.unbind(dim=1)
    matrices = torch.empty((count, 3, 3), dtype=q.dtype, device=device)
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


def rotate_batch(points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.bmm(points, rotation.transpose(1, 2))


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - (prediction * target).sum(dim=-1)).mean()


def angular_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = predicted / np.clip(np.linalg.norm(predicted, axis=-1, keepdims=True), 1e-8, None)
    target = target / np.clip(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8, None)
    angles = np.degrees(np.arccos(np.clip(np.sum(predicted * target, axis=-1), -1, 1)))
    return {
        "count": int(len(angles)),
        "mean_angle_deg": float(angles.mean()),
        "median_angle_deg": float(np.median(angles)),
        "p90_angle_deg": float(np.percentile(angles, 90)),
        "within_3deg_pct": float((angles <= 3).mean() * 100),
        "within_5deg_pct": float((angles <= 5).mean() * 100),
        "within_10deg_pct": float((angles <= 10).mean() * 100),
        "within_30deg_pct": float((angles <= 30).mean() * 100),
    }


def load_clouds(pointcloud_dir: Path, split_path: Path, device: torch.device):
    manifest = json.loads((pointcloud_dir / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if len(split["records"]) != len(records):
        raise ValueError("The saved 173/20 split does not match the current point-cloud manifest")
    for index, (expected, actual) in enumerate(zip(split["records"], records)):
        if expected["pointcloud_file"] != actual["pointcloud_file"]:
            raise ValueError(f"Split and manifest order differ at record {index}")

    point_clouds = []
    hashes = []
    for record in records:
        with np.load(pointcloud_dir / record["pointcloud_file"]) as cloud:
            points = cloud["points"].astype(np.float32)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f"Invalid points in {record['pointcloud_file']}")
            points = points - points.mean(axis=0, keepdims=True)
            radius = float(np.linalg.norm(points, axis=1).max())
            if radius <= 1e-8:
                raise ValueError(f"Zero-extent point cloud: {record['pointcloud_file']}")
            points /= radius
            point_clouds.append(points)
            hashes.append(str(cloud["mesh_hash"].item()))
    lengths = {len(points) for points in point_clouds}
    if len(lengths) != 1:
        raise ValueError(f"Expected a fixed point count, found {sorted(lengths)}")
    return torch.from_numpy(np.stack(point_clouds)).to(device), records, split, hashes


def make_validation(points: torch.Tensor, indices: list[int], rotations_per_part: int, point_count: int, seed: int):
    device = points.device
    generator = torch.Generator(device=device).manual_seed(seed + 73001)
    input_clouds = []
    target_axes = []
    part_ids = []
    rotation_ids = []
    # Include the historical frame once, then fixed random orientations.
    for index in indices:
        full = points[index]
        sample_ids = torch.randperm(len(full), generator=generator, device=device)[:point_count]
        base = full[sample_ids]
        eye = torch.eye(3, device=device, dtype=points.dtype).unsqueeze(0)
        random = random_rotations(rotations_per_part - 1, generator, device) if rotations_per_part > 1 else eye[:0]
        rotations = torch.cat([eye, random], dim=0)
        expanded = base.unsqueeze(0).expand(len(rotations), -1, -1)
        input_clouds.append(rotate_batch(expanded, rotations))
        target_axes.append(rotations[:, :, 2])
        part_ids.extend([index] * len(rotations))
        rotation_ids.extend(range(len(rotations)))
    return (
        torch.cat(input_clouds), torch.cat(target_axes),
        np.asarray(part_ids, dtype=np.int32), np.asarray(rotation_ids, dtype=np.int32),
    )


@torch.no_grad()
def evaluate(model, inputs, targets, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(inputs), batch_size):
        predictions.append(model(inputs[start:start + batch_size]).cpu().numpy())
    predicted = np.concatenate(predictions)
    target = targets.cpu().numpy()
    return angular_metrics(predicted, target), predicted, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointclouds", type=Path, default=DEFAULT_POINTCLOUDS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--train-rotations", type=int, default=32,
                        help="Fresh independent random orientations generated per training part each epoch")
    parser.add_argument("--validation-rotations", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--target-validation-deg", type=float, default=3.0)
    parser.add_argument("--threshold-confirm-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This run requires CUDA")
    if min(args.points, args.train_rotations, args.validation_rotations, args.epochs, args.batch_size, args.patience, args.threshold_confirm_epochs) <= 0:
        parser.error("Point, epoch and batch counts must be positive")
    if args.target_validation_deg <= 0 or args.learning_rate <= 0:
        parser.error("The target angle and learning rate must be positive")

    seed_everything(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    all_points, records, split, mesh_hashes = load_clouds(args.pointclouds, args.split, device)
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(split["validation_indices"], dtype=np.int64)
    if len(train_indices) != 173 or len(validation_indices) != 20:
        print(f"Warning: expected 173/20 split, got {len(train_indices)}/{len(validation_indices)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "PointNet from scratch; shared MLP 3-64-64-64-128-1024, max pool, 1024-512-256-3 unit-axis head",
        "input": "2048 sampled XYZ points; center each part and scale to unit sphere; no normals",
        "target": "historical raw mesh +Z build-up direction, rotated with each SO(3) augmentation",
        "loss": "mean(1 - dot(normalized_prediction, normalized_target))",
        "coordinate_assumption": "Each exported historical mesh coordinate frame represents its user-confirmed printed pose and +Z is its build-up direction; FastLayer round-trip is still pending.",
        "split_path": str(args.split.resolve()),
        "split_method": split["method"],
        "split_limitation": split["limitation"],
        "training_parts": len(train_indices),
        "validation_parts": len(validation_indices),
        "training_part_indices": train_indices.tolist(),
        "validation_part_indices": validation_indices.tolist(),
        "validation_rotations_per_part": args.validation_rotations,
        "training_rotations_per_part_per_epoch": args.train_rotations,
        "point_count": args.points,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "target_validation_deg": args.target_validation_deg,
        "threshold_confirm_epochs": args.threshold_confirm_epochs,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "metric_scope": "Mean signed-axis angle over fixed validation rotations; 20 validation parts only, not a held-out test set; symmetry is not corrected.",
    }
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    # Augmented rotation labels and validation rotations are checked in the same convention.
    test_rotation = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]], device=device)
    test_target = test_rotation[:, :, 2]
    if not torch.allclose(test_target, torch.tensor([[0., 0., 1.]], device=device)):
        raise RuntimeError("Rotation/target convention self-check failed")
    val_inputs, val_targets, val_part_ids, val_rotation_ids = make_validation(
        all_points, validation_indices.tolist(), args.validation_rotations, args.points, args.seed
    )
    baseline = torch.tensor([0., 0., 1.], device=device).expand_as(val_targets)
    baseline_metrics = angular_metrics(baseline.cpu().numpy(), val_targets.cpu().numpy())

    model = PointNetAxis().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    generator = torch.Generator(device=device).manual_seed(args.seed + 92011)
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    below_threshold_streak = 0
    first_threshold_epoch = None
    history = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_ids = torch.as_tensor(train_indices, device=device).repeat_interleave(args.train_rotations)
        permutation = torch.randperm(len(training_ids), generator=generator, device=device)
        total_loss = 0.0
        total_count = 0
        epoch_started = time.monotonic()
        for start in range(0, len(permutation), args.batch_size):
            local_ids = permutation[start:start + args.batch_size]
            record_ids = training_ids[local_ids]
            clouds = all_points[record_ids]
            # Fresh without-replacement surface sample and fresh independent SO(3) per part each epoch.
            sample_ids = torch.rand(clouds.shape[:2], generator=generator, device=device).topk(
                args.points, dim=1, largest=False
            ).indices
            clouds = torch.gather(clouds, 1, sample_ids.unsqueeze(-1).expand(-1, -1, 3))
            rotations = random_rotations(len(clouds), generator, device)
            augmented = rotate_batch(clouds, rotations)
            target = rotations[:, :, 2]
            prediction = model(augmented)
            loss = cosine_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(clouds)
            total_count += len(clouds)
        scheduler.step()

        validation, val_pred, val_target_np = evaluate(model, val_inputs, val_targets, args.eval_batch_size)
        row = {
            "epoch": epoch,
            "train_cosine_loss": total_loss / max(total_count, 1),
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        if validation["mean_angle_deg"] < best_validation:
            best_validation = validation["mean_angle_deg"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation_metrics": validation}, args.output / "best_model.pt")
            np.savez_compressed(args.output / "best_validation_predictions.npz",
                                predicted_axis=val_pred, target_axis=val_target_np,
                                part_indices=val_part_ids, rotation_indices=val_rotation_ids)
        else:
            stale_epochs += 1
        if validation["mean_angle_deg"] < args.target_validation_deg:
            below_threshold_streak += 1
            if first_threshold_epoch is None:
                first_threshold_epoch = epoch
        else:
            below_threshold_streak = 0
        print(
            f"epoch {epoch:03d}: train={row['train_cosine_loss']:.5f} "
            f"val={validation['mean_angle_deg']:.2f}° median={validation['median_angle_deg']:.2f}° "
            f"best={best_validation:.2f}° (epoch {best_epoch}) {row['epoch_seconds']:.1f}s",
            flush=True,
        )
        if below_threshold_streak >= args.threshold_confirm_epochs:
            print(f"Stopping: validation mean stayed below {args.target_validation_deg:.1f}° for {below_threshold_streak} epochs", flush=True)
            break
        if stale_epochs >= args.patience:
            print(f"Early stopping: no validation improvement for {stale_epochs} epochs", flush=True)
            break

    report = {
        "model": "PointNet scratch on real industrial parts",
        "train_parts": len(train_indices),
        "validation_parts": len(validation_indices),
        "validation_views": len(val_targets),
        "first_validation_below_target_epoch": first_threshold_epoch,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_seconds": time.monotonic() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "baseline_constant_up_axis": baseline_metrics,
        "best_validation": min(history, key=lambda item: item["validation"]["mean_angle_deg"])["validation"],
        "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "history": history,
        "limitations": [split["limitation"], config["coordinate_assumption"], config["metric_scope"]],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Training complete:", json.dumps({key: report[key] for key in (
        "first_validation_below_target_epoch", "best_epoch", "epochs_completed", "best_validation",
        "baseline_constant_up_axis", "peak_cuda_memory_allocated_mib",
    )}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
