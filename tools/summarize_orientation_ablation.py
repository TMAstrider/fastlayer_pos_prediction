import argparse
import json
import math
from pathlib import Path

import numpy as np

from orientation_model import ROOT, direction_metrics


RUNS = [
    ("no_normals_r16", "Baseline: XYZ, 16 rotations"),
    ("no_normals_r64", "More rotations: XYZ, 64"),
    ("with_normals_r16", "Normals: XYZ+normal, 16"),
    ("with_normals_r64", "Both: XYZ+normal, 64"),
]


def main():
    parser = argparse.ArgumentParser(description="Verify matched ablation inputs and summarize all four completed experiments")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/utonia_tiny_direction_ablation")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ptv3_normals_rotation_ablation")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    split_reference = None
    target_reference = None
    batch_size_reference = None
    pipeline_version_reference = None
    summaries = []
    histories = []
    caches = []
    for normal_mode in ["no_normals", "with_normals"]:
        cache_path = arguments.runs / "features" / f"{normal_mode}_r64_p2048.npz"
        with np.load(cache_path) as cache:
            caches.append({
                key: cache[key] for key in (
                    "train_targets", "train_record_ids", "validation_targets", "validation_record_ids"
                )
            })
    for key in caches[0]:
        np.testing.assert_array_equal(caches[0][key], caches[1][key])
    for name, label in RUNS:
        directory = arguments.runs / name
        configuration = json.loads((directory / "config.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        split = json.loads((directory / "split.json").read_text())
        if split_reference is None:
            split_reference = split
        else:
            assert split == split_reference, f"Different split in {name}"
        if batch_size_reference is None:
            batch_size_reference = configuration["batch_size"]
            pipeline_version_reference = configuration["feature_pipeline_version"]
        else:
            assert configuration["batch_size"] == batch_size_reference, f"Different batch size in {name}"
            assert configuration["feature_pipeline_version"] == pipeline_version_reference, f"Different feature pipeline in {name}"
        history = metrics["history"]
        assert len(history) == configuration["epochs"]
        best = min(history, key=lambda record: record["validation"]["mean_angle_deg"])
        with np.load(directory / "validation_predictions.npz") as predictions:
            predicted, target = predictions["predicted"], predictions["target"]
            if target_reference is None:
                target_reference = target.copy()
            else:
                np.testing.assert_array_equal(target, target_reference)
            measured = direction_metrics(predicted, target)
            for key, value in measured.items():
                np.testing.assert_allclose(value, best["validation"][key], atol=1e-5)
            cosine_loss = float((1 - np.sum(predicted * target, axis=-1)).mean())
        summaries.append({
            "run": name,
            "label": label,
            "normals": configuration["use_normals"],
            "rotations_per_training_part": configuration["head_rotations"],
            "training_examples": metrics["training_examples"],
            "epochs": configuration["epochs"],
            "batch_size": configuration["batch_size"],
            "optimizer_steps": math.ceil(metrics["training_examples"] / configuration["batch_size"]) * configuration["epochs"],
            "best_epoch": best["epoch"],
            "best_validation": measured,
            "best_validation_cosine_loss": cosine_loss,
            "train_angle_at_best_epoch": best["training"]["mean_angle_deg"],
            "initial_train_loss": history[0]["train_loss"],
            "final_train_loss": history[-1]["train_loss"],
            "final_train_angle": history[-1]["training"]["mean_angle_deg"],
            "training_seconds": metrics.get("training_seconds"),
        })
        histories.append(history)
    summary = {
        "matched_inputs_verified": True,
        "training_parts": len(split_reference["train_indices"]),
        "validation_parts": len(split_reference["validation_indices"]),
        "validation_examples": len(target_reference),
        "batch_size": batch_size_reference,
        "feature_pipeline_version": pipeline_version_reference,
        "scope": "Frozen Utonia Tiny plus direction head, one seed, one project-held-out validation split; historical +Z label convention pending FastLayer verification.",
        "compute_caveat": f"Same batch size {batch_size_reference} and 120 epochs; 64 rotations entail more examples and optimizer updates than 16 rotations. Effects do not isolate angle coverage from extra optimization.",
        "normal_audit": "Source winding is preserved; finite unit normals alone do not imply a watertight outward-oriented mesh.",
        "runs": summaries,
    }
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# PTv3 法向与随机朝向消融", "",
        "使用相同 173/20 工程划分、配对随机旋转与点采样、同一初始化种子、相同验证输入。编码器冻结，仅训练方向 Head。16 朝向是每个零件 64 朝向的前缀子集。", "",
        "| 配置 | 最佳 epoch | 验证平均角误差 | 验证中位角误差 | 30° 内 | 最佳验证余弦 loss | 训练 loss 首轮 → 末轮 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        validation = row["best_validation"]
        lines.append(
            f"| {row['run']} | {row['best_epoch']} | {validation['mean_angle_deg']:.2f}° | "
            f"{validation['median_angle_deg']:.2f}° | {validation['within_30_deg_fraction']:.2%} | "
            f"{row['best_validation_cosine_loss']:.4f} | {row['initial_train_loss']:.4f} → {row['final_train_loss']:.4f} |"
        )
    baseline = summaries[0]["best_validation"]["mean_angle_deg"]
    lines.extend(["", "## 对照结果", ""])
    for row in summaries[1:]:
        difference = row["best_validation"]["mean_angle_deg"] - baseline
        lines.append(f"- `{row['run']}` 相对本次重新训练基线的平均角误差变化：{difference:+.2f}°（负值为改善）。")
    lines.extend([
        "", "## 解释范围", "",
        "- 同一数据划分与旋转标签在四组间经过程序核验。首次实验与本次使用不同的随机流，因此请以本次重跑的 16 朝向/零法向组作基线。",
        "- 单个随机种子、20 个验证形状；随机旋转不会增加独立形状数，不能据此声称稳定统计增益。",
        f"- 所有组均训练 120 轮、batch size {batch_size_reference}；样本单位是一个零件的一种朝向。每轮先全局打乱训练集的零件-朝向实例，因此 batch 通常同时含多个零件和朝向，不保证按零件均衡。64 朝向组的每轮样本和梯度更新更多，结果同时包含角度覆盖和训练量效应。",
        "- 最佳 epoch 按验证平均角误差选取；表中的验证 loss 是该 checkpoint 的 loss，并非独立按 loss 选出的最优值。",
        "- 历史 +Z 标签、对称等价方向和零件族尚需复核。角度误差不等于打印风险或可打印率。",
        "- 原始网格法向按三角面绕序保留；没有为本次消融修复或重建模型。",
        "", "## 曲线", "", "![Training curves](curves.png)", "",
    ])
    (arguments.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(1, 3, figsize=(16, 4.8))
    for (_, label), history in zip(RUNS, histories):
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=label)
        axes[1].plot(epochs, [row["validation_loss"] for row in history])
        axes[2].plot(epochs, [row["validation"]["mean_angle_deg"] for row in history])
    for axis, title, ylabel in zip(axes,
                                  ["Training loss", "Validation loss", "Validation mean direction error"],
                                  ["Cosine loss", "Cosine loss", "Degrees"]):
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[0].set_yscale("log")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(arguments.output / "curves.png", dpi=160)
    pyplot.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
