"""Compare the old and expanded PointNet runs on identical fixed validation views."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_pointnet_real_parts import load_clouds, make_validation


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("runs/pointnet_real_parts_scratch"))
    parser.add_argument("--expanded", type=Path, default=Path("runs/pointnet_real_parts_expanded_pjf"))
    parser.add_argument("--old-pointclouds", type=Path, default=Path("outputs/pointclouds"))
    parser.add_argument("--expanded-pointclouds", type=Path, default=Path("outputs/expanded_real_pointclouds"))
    parser.add_argument("--output", type=Path, default=Path("reports/expanded_data_pointnet_pilot"))
    args = parser.parse_args()
    old, new = [load_json(path / "metrics.json") for path in (args.old, args.expanded)]
    old_cfg, new_cfg = [load_json(path / "config.json") for path in (args.old, args.expanded)]
    matched_keys = (
        "point_count", "batch_size", "learning_rate", "weight_decay", "seed",
        "epochs_requested", "training_rotations_per_part_per_epoch", "validation_rotations_per_part",
    )
    for key in matched_keys:
        if old_cfg[key] != new_cfg[key]:
            raise ValueError(f"Training settings differ: {key}")
    old_pred, new_pred = [np.load(path / "best_validation_predictions.npz") for path in (args.old, args.expanded)]
    for key in ("part_indices", "rotation_indices", "target_axis"):
        if not np.array_equal(old_pred[key], new_pred[key]):
            raise ValueError(f"Validation mismatch: {key}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is needed to reproduce the original validation random generator")
    validation_batches = []
    for cfg, run, cloud_dir in ((old_cfg, args.old, args.old_pointclouds),
                                (new_cfg, args.expanded, args.expanded_pointclouds)):
        split_path = run / "split.json"
        points, _, split, _ = load_clouds(cloud_dir, split_path, torch.device("cuda"))
        validation_batches.append(make_validation(
            points, split["validation_indices"], cfg["validation_rotations_per_part"],
            cfg["point_count"], cfg["seed"],
        ))
    if not torch.equal(validation_batches[0][0], validation_batches[1][0]):
        raise ValueError("Validation input point clouds differ")

    summary = {
        "old_run": str(args.old), "expanded_run": str(args.expanded),
        "validation_targets_and_view_ids_exactly_equal": True,
        "validation_input_point_clouds_exactly_equal": True,
        "old_train_parts": old["train_parts"], "expanded_train_parts": new["train_parts"],
        "validation_parts": old["validation_parts"], "validation_views": old["validation_views"],
        "old_best_validation": old["best_validation"], "expanded_best_validation": new["best_validation"],
        "old_best_epoch": old["best_epoch"], "expanded_best_epoch": new["best_epoch"],
        "mean_angle_delta_expanded_minus_old_deg": new["best_validation"]["mean_angle_deg"]-old["best_validation"]["mean_angle_deg"],
        "limitations": [
            "Only the extra PJF project contributes added training data; none of the 326 Magics projects is trained on.",
            "The 30 added part objects come from one project and only six distinct P-codes in their names, not 30 certified unique shapes.",
            "One seed, one small historical validation split; validation selects the checkpoint and is not an independent test set.",
            "Fixed per-part views and epochs imply the expanded run has more updates per epoch; update count is not controlled.",
            "Raw historical +Z is assumed build-up; FastLayer round-trip and symmetry/family audit are still pending.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename in ("metrics.json", "config.json", "split.json"):
        shutil.copyfile(args.expanded / filename, args.output / filename)

    def angles(predictions):
        p, t = predictions["predicted_axis"], predictions["target_axis"]
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip(np.sum(p*t, axis=1), -1, 1)))

    old_angles, new_angles = angles(old_pred), angles(new_pred)
    with (args.output / "per_part.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_id", "old_mean_deg", "expanded_mean_deg", "delta_deg"])
        for index in np.unique(old_pred["part_indices"]):
            mask = old_pred["part_indices"] == index
            before, after = float(old_angles[mask].mean()), float(new_angles[mask].mean())
            writer.writerow([int(index), before, after, after-before])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, label in [(old, f'Old: {old["train_parts"]} training objects'),
                       (new, f'Expanded: {new["train_parts"]} training objects')]:
        epochs = [row["epoch"] for row in run["history"]]
        axes[0].plot(epochs, [row["train_cosine_loss"] for row in run["history"]], label=label)
        axes[1].plot(epochs, [row["validation"]["mean_angle_deg"] for row in run["history"]], label=label)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    axes[0].set(title="PointNet training loss", ylabel="Cosine loss")
    axes[1].set(title="Identical 640 validation views", ylabel="Mean direction error (degrees)")
    figure.tight_layout()
    figure.savefig(args.output / "curves.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
