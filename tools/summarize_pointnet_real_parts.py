"""Evaluate the real-part PointNet checkpoint on the historical PTv3 validation views."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from orientation_model import direction_metrics
from train_pointnet_real_parts import PointNetAxis


def main() -> None:
    run_dir = ROOT / "runs/pointnet_real_parts_scratch"
    ptv3_dir = ROOT / "runs/utonia_tiny_direction_ablation/with_normals_r64"
    cloud_dir = ROOT / "outputs/pointclouds"
    result_dir = run_dir / "paired_ptv3_validation"
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((cloud_dir / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    ptv3_cfg = json.loads((ptv3_dir / "config.json").read_text(encoding="utf-8"))
    ptv3_saved = np.load(ptv3_dir / "validation_predictions.npz")
    ptv3_pred = ptv3_saved["predicted"]
    ptv3_target = ptv3_saved["target"]
    ptv3_record_ids = ptv3_saved["record_ids"]

    clouds = {}
    for record_id in split["validation_indices"]:
        record = records[record_id]
        with np.load(cloud_dir / record["pointcloud_file"]) as cloud:
            points = cloud["points"].astype(np.float32)
            points -= points.mean(axis=0, keepdims=True)
            points /= np.linalg.norm(points, axis=1).max()
            clouds[record_id] = points

    tasks = [
        (record_id, rotation_index, 1000000007)
        for record_id in split["validation_indices"]
        for rotation_index in range(ptv3_cfg["validation_rotations"])
    ]
    pointnet_inputs = []
    target_rows = []
    for record_id, rotation_index, phase_seed in tasks:
        sample_seed = ptv3_cfg["seed"] + phase_seed + record_id * 1000003 + rotation_index
        rotation_rng = np.random.default_rng(sample_seed)
        rotation = Rotation.random(random_state=rotation_rng).as_matrix()
        sample_rng = np.random.default_rng(sample_seed + 2000000011)
        indices = sample_rng.choice(len(clouds[record_id]), ptv3_cfg["points"], replace=False)
        pointnet_inputs.append(clouds[record_id][indices] @ rotation.T)
        target_rows.append(rotation[:, 2].astype(np.float32))
    pointnet_inputs = np.stack(pointnet_inputs)
    targets = np.stack(target_rows)
    record_ids = np.asarray([task[0] for task in tasks], dtype=np.int64)
    if not np.array_equal(record_ids, ptv3_record_ids):
        raise RuntimeError("Recreated view order does not match the saved PTv3 predictions")
    if not np.allclose(targets, ptv3_target, rtol=0, atol=2e-7):
        raise RuntimeError("Recreated target axes do not match the saved PTv3 targets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = PointNetAxis().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pointnet_pred = []
    with torch.inference_mode():
        for start in range(0, len(pointnet_inputs), 64):
            sample = torch.from_numpy(pointnet_inputs[start:start + 64]).to(device=device, dtype=torch.float32)
            pointnet_pred.append(model(sample).cpu().numpy())
    pointnet_pred = np.concatenate(pointnet_pred).astype(np.float32)

    metrics = {
        "PointNet scratch, matched 160 PTv3 views": direction_metrics(pointnet_pred, targets),
        "Frozen PTv3 + XYZ and normals, saved matched 160 views": direction_metrics(ptv3_pred, ptv3_target),
    }
    metrics["evaluation_note"] = (
        "Both predictions use the same 160 rotation matrices, target axes, and per-view sampled point indices. "
        "Each model uses its own training-time coordinate preprocessing: PointNet centers/scales the full cloud; "
        "PTv3 recenters/rescales each sampled view and voxel-deduplicates it. PTv3 also receives normals."
    )
    pointnet_angles = np.degrees(np.arccos(np.clip(np.sum(pointnet_pred * targets, axis=1), -1, 1)))
    ptv3_angles = np.degrees(np.arccos(np.clip(np.sum(ptv3_pred * ptv3_target, axis=1), -1, 1)))
    metrics["paired_view_delta_PointNet_minus_PTv3_mean_deg"] = float(np.mean(pointnet_angles - ptv3_angles))
    (result_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        result_dir / "predictions.npz", pointnet_pred=pointnet_pred,
        ptv3_pred=ptv3_pred, target=targets, record_ids=record_ids,
        pointnet_angles_deg=pointnet_angles, ptv3_angles_deg=ptv3_angles,
    )

    part_names = []
    for record_id in record_ids:
        record = records[int(record_id)]
        part_names.append(record.get("source_part_name", record.get("source_part_id", str(record_id))))
    with (result_dir / "per_part.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "part_name", "views", "pointnet_mean_deg", "ptv3_mean_deg", "delta_deg"])
        for record_id in split["validation_indices"]:
            mask = record_ids == record_id
            name = records[record_id].get("source_part_name", records[record_id].get("source_part_id", str(record_id)))
            pn_mean = float(pointnet_angles[mask].mean())
            pt_mean = float(ptv3_angles[mask].mean())
            writer.writerow([record_id, name, int(mask.sum()), f"{pn_mean:.4f}", f"{pt_mean:.4f}", f"{pn_mean-pt_mean:.4f}"])

    history = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(epochs, [row["train_cosine_loss"] for row in history], color="#3572A5")
    axes[0].set(xlabel="Epoch", ylabel="Training cosine loss", title="PointNet training loss")
    axes[0].grid(alpha=.25)
    axes[1].plot(epochs, [row["validation"]["mean_angle_deg"] for row in history], color="#D17C22", label="PointNet 640-view validation")
    axes[1].axhline(metrics["Frozen PTv3 + XYZ and normals, saved matched 160 views"]["mean_angle_deg"], color="#36864A", linestyle="--", label="PTv3 matched 160-view mean")
    axes[1].axhline(metrics["PointNet scratch, matched 160 PTv3 views"]["mean_angle_deg"], color="#3572A5", linestyle=":", label="PointNet matched 160-view mean")
    axes[1].set(xlabel="Epoch", ylabel="Mean signed-axis error (degrees)", title="Validation trend")
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "validation_trend.png", dpi=170)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
