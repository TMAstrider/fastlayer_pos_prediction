"""Build a concise report and plot from the three public benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/public_pyramids_model_comparison/final"
DEFAULT_REPORT = ROOT / "reports/public_pyramids_model_comparison_2026-11-08.md"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(label: str, data: dict) -> dict:
    best_validation = data["best_validation"]
    test = data["test"]
    history = data.get("history", [])
    first_under_3 = next(
        (entry["epoch"] for entry in history
         if entry.get("validation", {}).get("mean_angle_deg", float("inf")) < 3.0),
        None,
    )
    return {
        "model": label,
        "best_epoch": data["best_epoch"],
        "first_validation_under_3deg_epoch": first_under_3,
        "val_mean": best_validation["mean_angle_deg"],
        "test_mean": test["mean_angle_deg"],
        "test_median": test["median_angle_deg"],
        "test_p90": test["p90_angle_deg"],
        "within_1": test["within_1deg_pct"],
        "within_3": test["within_3deg_pct"],
        "within_5": test["within_5deg_pct"],
        "within_10": test["within_10deg_pct"],
        "seconds": data.get("training_seconds", data.get("training_seconds_estimate", 0)),
        "params": data["parameter_count"],
        "memory": data.get("peak_cuda_memory_allocated_mib", float("nan")),
        "history": history,
        "run": data,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    pointnet_path = args.runs / "pointnet_full_160/pointnet/metrics.json"
    if not pointnet_path.is_file():
        pointnet_path = args.runs / "pointnet/metrics.json"
    rows = [
        metric_row("PointNet（从头训练）", read_json(pointnet_path)),
        metric_row("PTv3（从头训练）", read_json(args.runs / "ptv3_scratch/metrics.json")),
        metric_row("PTv3（预训练微调）", read_json(args.runs / "ptv3_pretrained/metrics.json")),
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.runs / "summary.json"
    summary_path.write_text(json.dumps({
        "dataset": "sebius/pbflbm-part-orientation pyramids_10000",
        "split_counts": {"train": 8100, "validation": 900, "test": 1000},
        "target": "Unit 3-D build-up axis from the official target_rotational_matrix metadata",
        "input": "2048 sampled XYZ surface points, centered and scaled to unit-sphere radius",
        "loss": "1 - dot(normalized_prediction, normalized_target)",
        "experiments": {row["model"]: {k: v for k, v in row.items() if k not in ("history", "run")} | {"run": row["run"]} for row in rows},
    }, indent=2), encoding="utf-8")

    figure, (axis, zoom_axis) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1.15]})
    colors = ["#2a6fbb", "#e07a28", "#4a9b68"]
    labels = ["PointNet scratch", "PTv3 scratch", "PTv3 pretrained fine-tune"]
    for row, color, label in zip(rows, colors, labels):
        history = row["history"]
        if not history:
            axis.scatter([row["best_epoch"]], [row["val_mean"]], label=label, color=color)
            continue
        epochs = [entry["epoch"] for entry in history]
        means = [entry["validation"]["mean_angle_deg"] for entry in history]
        axis.plot(epochs, means, label=label, color=color, linewidth=1.8)
        axis.scatter([row["best_epoch"]], [row["val_mean"]], color=color, s=28, zorder=3)
        zoom_count = min(12, len(epochs))
        zoom_axis.plot(epochs[:zoom_count], means[:zoom_count], label=label, color=color, linewidth=2)
    axis.set(title="Full validation curves", xlabel="Epoch", ylabel="Mean angle error (degrees)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    axis.axhline(3, color="#777777", linestyle="--", linewidth=1)
    zoom_axis.set(title="Early training trend (first 12 epochs)", xlabel="Epoch", ylabel="Mean angle error (degrees)", xticks=range(1, 13))
    zoom_axis.set_ylim(bottom=0)
    zoom_axis.grid(True, alpha=0.25)
    zoom_axis.axhline(3, color="#777777", linestyle="--", linewidth=1, label="3 degree threshold")
    zoom_axis.legend(frameon=False)
    figure.tight_layout()
    plot_path = args.runs / "validation_curves.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    lines = [
        "# Public pyramid orientation model comparison",
        "",
        "## Setup",
        "",
        "All three experiments use the same 10,000-sample public pyramid dataset and official split (8,100 train / 900 validation / 1,000 held-out test). Each sample is represented by 2,048 surface XYZ points centered and scaled to the unit sphere. The target is the metadata build-up axis, and all models use normalized 3-D outputs with cosine direction loss `1 - dot(prediction, target)`. Yaw is not predicted.",
        "",
        "PTv3 uses the Utonia Tiny encoder in encoder-only mode, multi-stage mean/max pooling (972-D feature), and a 972→256→64→3 direction head. The pretrained run fine-tunes the full encoder and head with separate learning rates. PTv3 uses a 64-point attention patch and batches of 32 with 2-step gradient accumulation (effective batch 64).",
        "",
        "## Held-out test results",
        "",
        "| Model | First validation epoch <3° | Best epoch | Validation mean | Test mean | Test median | Test P90 | ≤1° | ≤3° | ≤5° | Train time | Peak GPU memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        time_minutes = row["seconds"] / 60
        memory = f"{row['memory'] / 1024:.2f} GiB" if row["memory"] == row["memory"] else "not recorded"
        lines.append(
            f"| {row['model']} | {row['first_validation_under_3deg_epoch'] if row['first_validation_under_3deg_epoch'] is not None else '—'} | {row['best_epoch']} | {row['val_mean']:.2f}° | {row['test_mean']:.2f}° | {row['test_median']:.2f}° | {row['test_p90']:.2f}° | {row['within_1']:.1f}% | {row['within_3']:.1f}% | {row['within_5']:.1f}% | {time_minutes:.1f} min | {memory} |"
        )
    lines += [
        "",
        f"Validation curves: [`{plot_path.name}`](../runs/public_pyramids_model_comparison/final/{plot_path.name}).",
        "",
        "## Interpretation",
        "",
        f"On this controlled geometry task, the lowest test angular error came from **{min(rows, key=lambda row: row['test_mean'])['model']}**. All three validation curves first went below 3°: PointNet at epoch 10, scratch PTv3 at epoch 6, and pretrained PTv3 at epoch 6. Scratch PTv3 briefly rose above 3° at epoch 7, then remained below from epoch 8 onward. PTv3 continued improving gradually through epoch 12, while PointNet continued to improve with its longer run. This supports using 3° as a stopping target for future runs when that precision is sufficient, with a two-epoch confirmation to avoid stopping on a one-epoch dip.",
        "",
        "This run evaluates one target axis against the dataset's single reference axis. It does not assess equivalent/symmetric orientations, support amount, manufacturability constraints, yaw, or transfer to the 193 real parts. PointNet ran for 160 epochs and PTv3 for 12, so use these results as an initial comparison; a validation-converged follow-up on the real-part split is needed before drawing strong architectural conclusions.",
        "",
        "## Artifacts",
        "",
        "- Checkpoints and per-run metrics: `runs/public_pyramids_model_comparison/final/`.",
        "- Test predictions: one `test_predictions.npz` per run.",
        "- Combined metrics: `runs/public_pyramids_model_comparison/final/summary.json`.",
        "- Training script: `tools/train_public_model_comparison.py`.",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.report}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
