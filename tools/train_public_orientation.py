"""Reproduce the paper's z-invariant PointNet orientation experiment.

The paper does not publish its exact network layer widths or training code.
This script uses a small PointNet-style shared-MLP/global-max-pool regressor
with the paper's 3-value z-invariant target and the dataset's official split.
It is an independently implemented reproduction baseline, not an author-code
replica.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import trimesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "datasets/public_pbflbm/pyramids_10000.zip"
DEFAULT_OUTPUT = ROOT / "runs/public_pyramids_axis_v2"
CACHE_VERSION = "pointnet_axis_v2_row_vector_labels"


class PointNetAxis(nn.Module):
    """PointNet-style point encoder with a three-dimensional axis head."""

    def __init__(self) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 3),
            nn.Tanh(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.point_mlp(points.transpose(1, 2).contiguous())
        global_feature = torch.max(features, dim=2).values
        return self.regressor(global_feature)


def _uniform_surface_points(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    valid = np.isfinite(area) & (area > 1e-14)
    triangles = triangles[valid]
    area = area[valid]
    if len(area) == 0 or float(area.sum()) <= 0:
        raise ValueError("Mesh has no non-degenerate surface area")
    face_ids = rng.choice(len(area), size=count, replace=True, p=area / area.sum())
    selected_triangles = triangles[face_ids]
    uniforms = rng.random((count, 2))
    root_uniform = np.sqrt(uniforms[:, 0])
    first_weight = 1.0 - root_uniform
    second_weight = root_uniform * (1.0 - uniforms[:, 1])
    third_weight = root_uniform * uniforms[:, 1]
    return (
        first_weight[:, None] * selected_triangles[:, 0]
        + second_weight[:, None] * selected_triangles[:, 1]
        + third_weight[:, None] * selected_triangles[:, 2]
    ).astype(np.float32)


def _read_mesh_points(data: bytes, count: int, rng: np.random.Generator,
                      target_axis: np.ndarray, reference_height: float | None) -> np.ndarray:
    loaded = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("STL did not contain a triangle mesh")
        loaded = trimesh.util.concatenate(meshes)
    if reference_height is not None:
        heights = np.asarray(loaded.vertices) @ target_axis
        if not np.isclose(np.ptp(heights), reference_height, rtol=2e-5, atol=1e-4):
            raise ValueError("Target axis does not restore the metadata reference height")
    points = _uniform_surface_points(loaded, count, rng)
    points -= points.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(points, axis=1).max())
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("Sampled geometry has zero radius")
    points /= radius
    return points


def _archive_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(archive: Path, cache_path: Path, points: int, seed: int) -> dict[str, np.ndarray]:
    digest = _archive_hash(archive)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                str(cache["archive_sha256"].item()) == digest
                and int(cache["point_count"].item()) == points
                and int(cache["sampling_seed"].item()) == seed
                and str(cache["cache_version"].item()) == CACHE_VERSION
            ):
                return {name: cache[name] for name in cache.files if name not in {
                    "archive_sha256", "point_count", "sampling_seed", "cache_version"
                }}

    split_points: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    split_targets: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    with zipfile.ZipFile(archive) as dataset:
        names = set(dataset.namelist())
        for split in split_points:
            mesh_names = sorted(
                name for name in names
                if name.startswith(f"stl/{split}/") and name.lower().endswith(".stl")
            )
            if not mesh_names:
                raise ValueError(f"No STL files found under stl/{split}/")
            for index, mesh_name in enumerate(mesh_names, 1):
                sample_name = Path(mesh_name).stem
                info_name = f"info/{sample_name}.json"
                if info_name not in names:
                    raise ValueError(f"Missing metadata for {mesh_name}: {info_name}")
                metadata = json.loads(dataset.read(info_name))
                matrix = np.asarray(metadata["target_rotational_matrix"], dtype=np.float64)
                if matrix.shape != (3, 3):
                    raise ValueError(f"Unexpected rotation matrix shape for {sample_name}: {matrix.shape}")
                target_axis = matrix[:, 2].astype(np.float32)
                target_axis /= np.linalg.norm(target_axis)
                sample_seed = seed ^ int(hashlib.sha256(sample_name.encode()).hexdigest()[:8], 16)
                sample_rng = np.random.default_rng(sample_seed)
                split_points[split].append(_read_mesh_points(
                    dataset.read(mesh_name), points, sample_rng, target_axis,
                    metadata.get("height") if sample_name.startswith("pyramids_") else None,
                ))
                split_targets[split].append(target_axis)
                if index % 1000 == 0 or index == len(mesh_names):
                    print(f"Prepared {split}: {index}/{len(mesh_names)}", flush=True)

    result: dict[str, np.ndarray] = {}
    for split in split_points:
        result[f"{split}_points"] = np.stack(split_points[split]).astype(np.float32)
        result[f"{split}_targets"] = np.stack(split_targets[split]).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **result,
        archive_sha256=np.asarray(digest),
        point_count=np.asarray(points),
        sampling_seed=np.asarray(seed),
        cache_version=np.asarray(CACHE_VERSION),
    )
    return result


def angle_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = predicted / np.clip(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-8, None)
    target = target / np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-8, None)
    dots = np.clip(np.sum(predicted * target, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return {
        "count": int(len(angles)),
        "mean_angle_deg": float(np.mean(angles)),
        "median_angle_deg": float(np.median(angles)),
        "p90_angle_deg": float(np.percentile(angles, 90)),
        "within_1deg_pct": float(np.mean(angles <= 1.0) * 100.0),
        "within_3deg_pct": float(np.mean(angles <= 3.0) * 100.0),
        "within_10deg_pct": float(np.mean(angles <= 10.0) * 100.0),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    for points, target in loader:
        points = points.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(points)
        loss_sum += float(nn.functional.mse_loss(output, target).item()) * len(points)
        count += len(points)
        predictions.append(output.cpu().numpy())
        targets.append(target.cpu().numpy())
    predicted_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    metrics = angle_metrics(predicted_array, target_array)
    metrics["mse"] = loss_sum / count
    return metrics, predicted_array, target_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"Dataset archive not found: {args.archive}")
    if min(args.points, args.batch_size, args.epochs, args.patience) <= 0:
        parser.error("points, batch size, epochs, and patience must be positive")
    if args.workers < 0 or args.learning_rate <= 0:
        parser.error("workers must be nonnegative and learning rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    args.output.mkdir(parents=True, exist_ok=True)
    dataset = prepare_dataset(args.archive, args.output / "pointcloud_cache.npz", args.points, args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}; archive: {args.archive}; points/cloud: {args.points}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        points = torch.from_numpy(dataset[f"{split}_points"])
        targets = torch.from_numpy(dataset[f"{split}_targets"])
        loaders[split] = DataLoader(
            TensorDataset(points, targets),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            drop_last=False,
        )
        print(f"{split}: {len(points)} meshes", flush=True)

    model = PointNetAxis().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        epoch_started = time.monotonic()
        for points, target in loaders["train"]:
            points = points.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(points)
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(points)
            total_loss += float(loss.item()) * count
            total_items += count

        validation, _, _ = evaluate(model, loaders["validation"], device)
        scheduler.step(validation["mse"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_mse": total_loss / max(1, total_items),
            "validation_mse": validation["mse"],
            "validation_mean_angle_deg": validation["mean_angle_deg"],
            "validation_median_angle_deg": validation["median_angle_deg"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        improved = validation["mean_angle_deg"] < best_validation
        if improved:
            best_validation = validation["mean_angle_deg"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "seed": args.seed,
                    "points": args.points,
                    "archive_sha256": _archive_hash(args.archive),
                },
                args.output / "best_model.pt",
            )
        else:
            stale_epochs += 1
        print(
            f"epoch {epoch:03d}: train_mse={row['train_mse']:.6f} "
            f"val={validation['mean_angle_deg']:.3f}° "
            f"median={validation['median_angle_deg']:.3f}° "
            f"best={best_validation:.3f}° (epoch {best_epoch}), "
            f"{row['epoch_seconds']:.1f}s",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement", flush=True)
            break

    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_metrics, test_prediction, test_target = evaluate(model, loaders["test"], device)
    np.savez_compressed(
        args.output / "test_predictions.npz",
        predicted_axis=test_prediction,
        target_axis=test_target,
    )
    report = {
        "dataset": "sebius/pbflbm-part-orientation / pyramids_10000.zip",
        "archive": str(args.archive.resolve()),
        "archive_sha256": _archive_hash(args.archive),
        "split_counts": {split: int(len(dataset[f"{split}_targets"])) for split in ("train", "validation", "test")},
        "network": "PointNet-style shared MLP + global max pooling + 3D tanh regressor",
        "target": "third column of archive target_rotational_matrix (row-vector convention), equivalent to third row of column-vector correction matrix; component MSE with tanh output",
        "points_per_mesh": args.points,
        "normalization": "center sampled points by sample mean and scale to unit-sphere radius",
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "patience": args.patience,
        "seed": args.seed,
        "device": str(device),
        "training_seconds": time.monotonic() - started,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "held_out_test_metrics": test_metrics,
        "history": history,
        "reproduction_note": "The paper does not provide its exact model layer widths or training code; this is a transparent PointNet-style reproduction of the data/target/evaluation design, not an exact author-code replication.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Held-out test:", json.dumps(test_metrics, sort_keys=True), flush=True)
    print(f"Saved report to {args.output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
