"""Train a scratch PointNet on historical Magics poses with SO(3) supervision."""

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
DEFAULT_CLOUDS = ROOT / "outputs/magics_pointclouds"
DEFAULT_OUTPUT = ROOT / "runs/magics_pointnet_scratch_2026-11-09"


class PointNetAxis(nn.Module):
    def __init__(self):
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

    def forward(self, points):
        features = self.point_mlp(points.transpose(1, 2).contiguous()).amax(dim=2)
        return nn.functional.normalize(self.regressor(features), dim=-1, eps=1e-8)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_rotations(count, generator, device):
    q = nn.functional.normalize(torch.randn((count, 4), generator=generator, device=device), dim=1)
    x, y, z, w = q.unbind(dim=1)
    matrices = torch.empty((count, 3, 3), dtype=q.dtype, device=device)
    matrices[:, 0, 0] = 1 - 2 * (y*y + z*z)
    matrices[:, 0, 1] = 2 * (x*y - z*w)
    matrices[:, 0, 2] = 2 * (x*z + y*w)
    matrices[:, 1, 0] = 2 * (x*y + z*w)
    matrices[:, 1, 1] = 1 - 2 * (x*x + z*z)
    matrices[:, 1, 2] = 2 * (y*z - x*w)
    matrices[:, 2, 0] = 2 * (x*z - y*w)
    matrices[:, 2, 1] = 2 * (y*z + x*w)
    matrices[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return matrices


def angle_metrics(predicted, target):
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


def load_clouds(cloud_dir, device, seed, validation_fraction, validation_count=None):
    manifest = json.loads((cloud_dir / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    shape_deduplicated = manifest.get("deduplicated_by_shape", False)
    if shape_deduplicated:
        shape_ids = [record["shape_group_id"] for record in records]
        if len(shape_ids) != len(set(shape_ids)):
            raise ValueError("A deduplicated dataset must contain one reference per shape group")
    if len(records) < 2:
        raise ValueError("At least two independent shapes are required")
    clouds = []
    for record in records:
        with np.load(cloud_dir / record["pointcloud_file"]) as data:
            points = data["points"].astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid point cloud: {record['pointcloud_file']}")
        points -= points.mean(axis=0, keepdims=True)
        radius = float(np.linalg.norm(points, axis=1).max())
        if radius <= 1e-8:
            raise ValueError(f"Zero-extent point cloud: {record['pointcloud_file']}")
        points /= radius
        clouds.append(points)
    lengths = {len(points) for points in clouds}
    if len(lengths) != 1:
        raise ValueError(f"Expected a fixed point count, found {sorted(lengths)}")

    # Hashes are unique in the exported point-cloud manifest. Splitting these IDs
    # keeps byte-identical geometry/pose records on a single side of the split.
    rng = np.random.default_rng(seed + 48317)
    permutation = rng.permutation(len(records))
    requested_validation_count = validation_count
    if validation_count is None:
        validation_count = min(len(records)-1, max(1, int(round(len(records) * validation_fraction))))
    if not 1 <= validation_count < len(records):
        raise ValueError(f"Validation count must be between 1 and {len(records)-1}")
    validation_indices = np.sort(permutation[:validation_count]).astype(np.int64)
    train_indices = np.sort(permutation[validation_count:]).astype(np.int64)
    split = {
        "method": "Deterministic random split of unique geometry_sha256 records; exact mesh duplicates are collapsed before splitting.",
        "seed": seed,
        "validation_fraction": validation_count / len(records),
        "validation_count": validation_count,
        "requested_validation_count": requested_validation_count,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "train_geometry_sha256": [records[int(i)]["geometry_sha256"] for i in train_indices],
        "validation_geometry_sha256": [records[int(i)]["geometry_sha256"] for i in validation_indices],
        "train_records": [records[int(i)] for i in train_indices],
        "validation_records": [records[int(i)] for i in validation_indices],
        "limitation": "Exact pose-mesh hashes are isolated; related but transformed/remeshed shapes may cross the split. Projects may be represented on both sides. Symmetry-equivalent axes are not corrected.",
    }
    if set(split["train_geometry_sha256"]) & set(split["validation_geometry_sha256"]):
        raise RuntimeError("Exact geometry hash leakage across split")
    if shape_deduplicated:
        split["method"] = "Deterministic split of one reference per confirmed rigid-shape group; unresolved near-identical candidates quarantined before splitting."
        split["train_shape_group_ids"] = [records[int(i)]["shape_group_id"] for i in train_indices]
        split["validation_shape_group_ids"] = [records[int(i)]["shape_group_id"] for i in validation_indices]
        if set(split["train_shape_group_ids"]) & set(split["validation_shape_group_ids"]):
            raise RuntimeError("Shape group leakage across split")
        split["limitation"] = "Conservative finite-tolerance geometric grouping, not a certified CAD identity audit. Remeshed copies outside candidate thresholds may remain. Part families and projects are not isolated; symmetry-equivalent axes are not corrected."
    return torch.from_numpy(np.stack(clouds)).to(device), split, manifest


def make_validation(clouds, indices, rotations_per_part, point_count, seed):
    device = clouds.device
    generator = torch.Generator(device=device).manual_seed(seed + 73001)
    input_clouds, targets, part_ids, rotation_ids = [], [], [], []
    for index in indices:
        full = clouds[index]
        sample_ids = torch.randperm(len(full), generator=generator, device=device)[:point_count]
        base = full[sample_ids]
        identity = torch.eye(3, device=device, dtype=clouds.dtype).unsqueeze(0)
        random = random_rotations(rotations_per_part - 1, generator, device)
        rotations = torch.cat([identity, random], dim=0)
        input_clouds.append(torch.bmm(base.unsqueeze(0).expand(len(rotations), -1, -1), rotations.transpose(1, 2)))
        targets.append(rotations[:, :, 2])
        part_ids.extend([int(index)] * len(rotations))
        rotation_ids.extend(range(len(rotations)))
    return (torch.cat(input_clouds), torch.cat(targets),
            np.asarray(part_ids, np.int32), np.asarray(rotation_ids, np.int32))


@torch.no_grad()
def evaluate(model, inputs, targets, batch_size):
    model.eval()
    prediction = torch.cat([model(inputs[i:i+batch_size]) for i in range(0, len(inputs), batch_size)])
    prediction = prediction.cpu().numpy()
    target = targets.cpu().numpy()
    return angle_metrics(prediction, target), prediction, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds", type=Path, default=DEFAULT_CLOUDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--train-rotations", type=int, default=16)
    parser.add_argument("--validation-rotations", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-count", type=int, default=None,
                        help="Fixed number of validation shapes; overrides --validation-fraction")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--target-validation-deg", type=float, default=3.0)
    parser.add_argument("--threshold-confirm-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This run requires CUDA")
    if min(args.points, args.train_rotations, args.validation_rotations, args.epochs,
           args.batch_size, args.patience, args.threshold_confirm_epochs) <= 0:
        parser.error("Counts must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("Validation fraction must lie between zero and one")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    clouds, split, cloud_manifest = load_clouds(
        args.clouds.resolve(), device, args.seed, args.validation_fraction, args.validation_count
    )
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(split["validation_indices"], dtype=np.int64)
    if clouds.shape[1] < args.points:
        raise ValueError(f"Stored clouds have {clouds.shape[1]} points; --points is {args.points}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "PointNet from scratch; same architecture as best prior real-part run",
        "architecture": "shared MLP 3-64-64-64-128-1024, global max pool, 1024-512-256-3 unit-axis head",
        "input": f"{args.points} XYZ surface points; each shape centered and scaled to unit sphere; no normals",
        "reference_pose": "Each selected historical reference mesh is used directly in its recorded print pose.",
        "clouds_directory": str(args.clouds.resolve()),
        "dataset": cloud_manifest.get("dataset", "Magics stored meshes"),
        "deduplicated_by_shape": cloud_manifest.get("deduplicated_by_shape", False),
        "target": "R@[0,0,1] for point row vectors transformed as P@R.T",
        "loss": "mean(1 - dot(normalized_prediction, normalized_target))",
        "training_data": cloud_manifest["source_part_records"],
        "unique_training_shapes": len(cloud_manifest["records"]),
        "training_parts": int(len(train_indices)),
        "validation_parts": int(len(validation_indices)),
        "requested_validation_count": args.validation_count,
        "training_rotations_per_part_per_epoch": args.train_rotations,
        "validation_rotations_per_part": args.validation_rotations,
        "point_count": args.points,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "target_validation_deg": args.target_validation_deg,
        "threshold_confirm_epochs": args.threshold_confirm_epochs,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "metric_scope": "Mean signed-axis angle over fixed random rotations of held-out records; see split.json for shape grouping. Not an independent project/family test. Symmetry is not corrected.",
    }
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    val_inputs, val_targets, val_part_ids, val_rotation_ids = make_validation(
        clouds, validation_indices.tolist(), args.validation_rotations, args.points, args.seed
    )
    baseline = torch.tensor([0., 0., 1.], device=device).expand_as(val_targets)
    baseline_metrics = angle_metrics(baseline.cpu().numpy(), val_targets.cpu().numpy())

    model = PointNetAxis().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    generator = torch.Generator(device=device).manual_seed(args.seed + 92011)
    best_validation, best_epoch, stale_epochs = float("inf"), 0, 0
    below_threshold_streak, first_threshold_epoch = 0, None
    history = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_ids = torch.as_tensor(train_indices, device=device).repeat_interleave(args.train_rotations)
        permutation = torch.randperm(len(training_ids), generator=generator, device=device)
        total_loss, total_count = 0.0, 0
        epoch_started = time.monotonic()
        for start in range(0, len(permutation), args.batch_size):
            local_ids = permutation[start:start + args.batch_size]
            record_ids = training_ids[local_ids]
            full_clouds = clouds[record_ids]
            sample_ids = torch.rand(full_clouds.shape[:2], generator=generator, device=device).topk(
                args.points, dim=1, largest=False
            ).indices
            sampled = torch.gather(full_clouds, 1, sample_ids.unsqueeze(-1).expand(-1, -1, 3))
            rotations = random_rotations(len(sampled), generator, device)
            augmented = torch.bmm(sampled, rotations.transpose(1, 2))
            targets = rotations[:, :, 2]
            predictions = model(augmented)
            loss = (1.0 - (predictions * targets).sum(dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(sampled)
            total_count += len(sampled)
        scheduler.step()
        validation, predicted, target = evaluate(model, val_inputs, val_targets, args.eval_batch_size)
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
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation_metrics": validation}, args.output / "best_model.pt")
            np.savez_compressed(args.output / "best_validation_predictions.npz",
                                predicted_axis=predicted, target_axis=target,
                                part_indices=val_part_ids, rotation_indices=val_rotation_ids)
        else:
            stale_epochs += 1
        if validation["mean_angle_deg"] < args.target_validation_deg:
            below_threshold_streak += 1
            if first_threshold_epoch is None:
                first_threshold_epoch = epoch
        else:
            below_threshold_streak = 0
        progress = {"status": "running", "best_epoch": best_epoch,
                    "best_validation_mean_angle_deg": best_validation,
                    "epochs_completed": len(history), "history": history}
        temporary_progress = args.output / "progress.tmp.json"
        temporary_progress.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        temporary_progress.replace(args.output / "progress.json")
        print(f"epoch {epoch:03d}: train={row['train_cosine_loss']:.5f} "
              f"val={validation['mean_angle_deg']:.2f}° median={validation['median_angle_deg']:.2f}° "
              f"best={best_validation:.2f}° (epoch {best_epoch}) {row['epoch_seconds']:.1f}s", flush=True)
        if below_threshold_streak >= args.threshold_confirm_epochs:
            print(f"Stopping: validation mean stayed below {args.target_validation_deg:.1f}°", flush=True)
            break
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without improvement", flush=True)
            break

    best_row = min(history, key=lambda item: item["validation"]["mean_angle_deg"])
    report = {
        "model": "PointNet scratch on selected historical reference meshes",
        "training_parts": int(len(train_indices)),
        "validation_parts": int(len(validation_indices)),
        "validation_views": int(len(val_targets)),
        "first_validation_below_target_epoch": first_threshold_epoch,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_seconds": time.monotonic() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "baseline_constant_up_axis": baseline_metrics,
        "best_validation": best_row["validation"],
        "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "history": history,
        "limitations": [split["limitation"], config["metric_scope"],
                        "Magics member activity and exact semantic part roles are not independently labeled."],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress["status"] = "complete"
    (args.output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print("Training complete:", json.dumps({key: report[key] for key in (
        "best_epoch", "epochs_completed", "best_validation", "baseline_constant_up_axis",
        "peak_cuda_memory_allocated_mib", "training_seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
