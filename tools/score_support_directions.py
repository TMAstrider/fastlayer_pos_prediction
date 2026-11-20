#!/usr/bin/env python3
"""Score a single part over approximate sphere directions by projected support area."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.generate_simplified_supports import (
        DEFAULT_RULES,
        process_config,
        read_mesh,
    )
    from tools.recognize_part_features import recognize_features
    from tools.support_area_scoring import (
        dense_area_ranks,
        evaluate_support_projection,
        normalise_projected_areas,
    )
except ModuleNotFoundError:
    from generate_simplified_supports import DEFAULT_RULES, process_config, read_mesh
    from recognize_part_features import recognize_features
    from support_area_scoring import (
        dense_area_ranks,
        evaluate_support_projection,
        normalise_projected_areas,
    )


def fibonacci_directions(count: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("direction count must be positive")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * index
        vector = np.array(
            [radius * math.cos(theta), radius * math.sin(theta), z],
            dtype=np.float64,
        )
        directions.append({
            "label": f"S{index + 1:04d}",
            "direction": vector.tolist(),
        })
    return directions


def write_visualisations(
    output_dir: Path,
    records: list[dict[str, Any]],
    best_result: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directions = np.asarray([row["build_up_input"] for row in records], dtype=np.float64)
    areas = np.asarray(
        [row["support_projection_area_mm2"] for row in records],
        dtype=np.float64,
    )
    order = np.argsort(areas)

    fig = plt.figure(figsize=(8.5, 7.5))
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        c=areas,
        cmap="viridis_r",
        s=35,
        depthshade=False,
    )
    top = min(5, len(order))
    axis.scatter(
        directions[order[:top], 0],
        directions[order[:top], 1],
        directions[order[:top], 2],
        marker="*",
        s=160,
        color="#e24a33",
        edgecolor="white",
        linewidth=0.5,
        label="lowest projected area",
    )
    axis.set_xlabel("input X")
    axis.set_ylabel("input Y")
    axis.set_zlabel("input Z")
    axis.set_title("Support projection area over build directions")
    axis.legend(loc="upper right")
    fig.colorbar(scatter, ax=axis, shrink=0.72, pad=0.1, label="projected area (mm2)")
    fig.tight_layout()
    fig.savefig(output_dir / "projected_area_sphere.png", dpi=180, facecolor="#f5f7fa")
    plt.close(fig)

    scores = np.asarray([row["score"] for row in records], dtype=np.float64)[order]
    ranked_areas = areas[order]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(np.arange(len(order)), ranked_areas, color="#315f78", linewidth=1.8)
    axes[0].set_ylabel("projected area (mm2)")
    axes[0].grid(alpha=0.2)
    axes[1].plot(np.arange(len(order)), scores, color="#2c8a57", linewidth=1.8)
    axes[1].set_xlabel("direction rank (lowest area first)")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Proxy support score across the sphere")
    fig.tight_layout()
    fig.savefig(output_dir / "projected_area_ranking.png", dpi=180, facecolor="#f5f7fa")
    plt.close(fig)

    mask = best_result["projected_mask"].occupied
    fig, axis = plt.subplots(figsize=(7, 6))
    if mask.size:
        axis.imshow(mask.T, origin="lower", cmap="Greys", interpolation="nearest")
    axis.set_title("Projected support-required regions: best direction")
    axis.set_xlabel("projection cell X")
    axis.set_ylabel("projection cell Y")
    axis.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "best_projected_mask.png", dpi=180, facecolor="#f5f7fa")
    plt.close(fig)


def write_readme(
    path: Path,
    mesh: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    best = records[0]
    worst = records[-1]
    top_rows = "\n".join(
        "| {rank} | `{label}` | {area:.3f} | {surface:.3f} | {score:.4f} |".format(
            rank=row["rank"],
            label=row["direction_label"],
            area=row["support_projection_area_mm2"],
            surface=row["retained_overhang_surface_mm2"],
            score=row["score"],
        )
        for row in records[:10]
    )
    text = f"""# 单零件代理支撑投影面积评分

## 目的

在一个零件的约均匀球面方向上，按固定角度阈值和 `<6 mm` 水平孔豁免规则计算支撑投影面积，并把它转换成可从高到低排序的评分。用户 query 不参与本实验。

## 量化定义

`A_proj` 是保留悬垂面在构建平面上的栅格化并集面积。评分使用本次候选集的 min-max 归一化：

```text
score = 1 - (A_proj - min(A_proj)) / (max(A_proj) - min(A_proj))
```

最低面积为 `1.0`，最高面积为 `0.0`。相同面积得到相同分数和相同排名。

## 运行配置

| 项目 | 值 |
|---|---|
| 零件 | `{mesh}` |
| 材料 | `{arguments.material}` |
| 模式 | `{arguments.mode}` |
| 角度阈值 | `{arguments.resolved_angle:.4f}°` |
| 水平孔阈值 | `< {arguments.resolved_hole:.4f} mm` |
| 球面方向数 | `{len(records)}` |
| 投影栅格 | `{arguments.projection_grid_mm:.4f} mm` |

## 结果

| 指标 | 数值 |
|---|---:|
| 最小投影面积 | `{summary["minimum_area_mm2"]:.3f} mm²` |
| 最大投影面积 | `{summary["maximum_area_mm2"]:.3f} mm²` |
| 中位投影面积 | `{summary["median_area_mm2"]:.3f} mm²` |
| 最小面积方向 | `{best["direction_label"]}` |
| 最大面积方向 | `{worst["direction_label"]}` |
| 不同面积值数量 | `{summary["distinct_area_count"]}` |
| 投影面积与保留悬垂面积 Spearman | `{summary["correlations"]["retained_surface_vs_projected_union_spearman"]:.4f}` |

前十个最低投影面积方向：

| 排名 | 方向 | 投影并集面积 mm² | 保留悬垂面积 mm² | 评分 |
|---:|---|---:|---:|---:|
{top_rows}

## 产物

- `direction_scores.csv`：全部方向、面积、评分和排名；
- `direction_scores.json`：完整记录和汇总；
- `projected_area_sphere.png`：方向球上的面积分布；
- `projected_area_ranking.png`：按面积排序后的面积和评分曲线；
- `best_projected_mask.png`：最佳方向在构建平面上的支撑区域。

## 解释边界

该结果验证的是“固定几何过滤规则下的投影面积评分”，还不是 Magics 支撑量，也不是训练后的 MLP 预测。下一步可用全部零件的这些记录训练候选评分器，并检查网络排序与这里 oracle 排名的一致性。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--material", choices=["316L", "AlSi10Mg", "TC4"], default="AlSi10Mg")
    parser.add_argument("--mode", choices=["standard", "am_build"], default="standard")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--directions", type=int, default=512)
    parser.add_argument("--angle-deg", type=float)
    parser.add_argument("--hole-diameter-mm", type=float)
    parser.add_argument("--projection-grid-mm", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.directions < 1:
        parser.error("--directions must be positive")
    if args.projection_grid_mm <= 0:
        parser.error("--projection-grid-mm must be positive")

    vertices, faces = read_mesh(args.mesh)
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    limits = process_config(rules, args.material, args.mode)
    if args.angle_deg is not None:
        limits["self_support_angle_threshold_deg"] = float(args.angle_deg)
    if args.hole_diameter_mm is not None:
        limits["horizontal_hole_diameter_threshold_mm"] = float(args.hole_diameter_mm)
    args.resolved_angle = limits["self_support_angle_threshold_deg"]
    args.resolved_hole = limits["horizontal_hole_diameter_threshold_mm"]
    annotations = (
        json.loads(args.annotations.read_text(encoding="utf-8"))
        if args.annotations else {}
    )

    print("recognizing orientation-invariant feature geometry...", flush=True)
    recognized = recognize_features(
        vertices,
        faces,
        build_up=np.array([0.0, 0.0, 1.0]),
        small_hole_diameter_mm=limits["horizontal_hole_diameter_threshold_mm"],
    )
    directions = fibonacci_directions(args.directions)
    records: list[dict[str, Any]] = []
    results_by_label: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(directions, start=1):
        result = evaluate_support_projection(
            vertices,
            faces,
            limits,
            annotations,
            np.asarray(item["direction"], dtype=np.float64),
            recognized,
            args.projection_grid_mm,
        )
        report = result["report"]
        records.append({
            "direction_label": item["label"],
            "build_up_input": item["direction"],
            "support_projection_area_mm2": report["areas_mm2"]["projected_union"],
            "projected_triangle_sum_mm2": report["areas_mm2"]["projected_triangle_sum"],
            "retained_overhang_surface_mm2": report["areas_mm2"]["retained_overhang_surface"],
            "downward_candidate_surface_mm2": report["areas_mm2"]["downward_angle_candidates"],
            "feature_exempt_surface_mm2": report["areas_mm2"]["feature_exempt"],
            "support_candidate_face_count": report["face_counts"]["support_candidate_after_exemptions"],
            "small_horizontal_exemptions": report["recognized_feature_summary"]["small_horizontal_exemptions"],
        })
        results_by_label[item["label"]] = result
        if index % 32 == 0 or index == len(directions):
            print(f"scored {index}/{len(directions)} directions", flush=True)

    areas = np.asarray(
        [row["support_projection_area_mm2"] for row in records],
        dtype=np.float64,
    )
    scores, minimum, maximum = normalise_projected_areas(areas)
    ranks = dense_area_ranks(areas)
    for row, score, rank in zip(records, scores, ranks):
        row["score"] = float(score)
        row["rank"] = int(rank)
    records.sort(key=lambda row: (row["support_projection_area_mm2"], row["direction_label"]))

    surface = np.asarray(
        [row["retained_overhang_surface_mm2"] for row in records],
        dtype=np.float64,
    )
    projected_sum = np.asarray(
        [row["projected_triangle_sum_mm2"] for row in records],
        dtype=np.float64,
    )
    ranked_areas = np.asarray(
        [row["support_projection_area_mm2"] for row in records],
        dtype=np.float64,
    )
    from scipy.stats import spearmanr

    correlations = {
        "retained_surface_vs_projected_union_spearman": float(
            spearmanr(surface, ranked_areas).statistic
        ),
        "projected_sum_vs_projected_union_spearman": float(
            spearmanr(projected_sum, ranked_areas).statistic
        ),
        "score_vs_projected_union_spearman": float(
            spearmanr(scores[np.argsort(ranks)], ranked_areas).statistic
        ),
    }
    best_record = records[0]
    best_result = evaluate_support_projection(
        vertices,
        faces,
        limits,
        annotations,
        np.asarray(best_record["build_up_input"], dtype=np.float64),
        recognized,
        args.projection_grid_mm,
    )

    summary = {
        "mesh": str(args.mesh.resolve()),
        "material": args.material,
        "mode": args.mode,
        "angle_threshold_deg": float(args.resolved_angle),
        "horizontal_hole_diameter_threshold_mm": float(args.resolved_hole),
        "projection_grid_mm": float(args.projection_grid_mm),
        "direction_count": len(records),
        "minimum_area_mm2": minimum,
        "maximum_area_mm2": maximum,
        "mean_area_mm2": float(areas.mean()),
        "median_area_mm2": float(np.median(areas)),
        "standard_deviation_area_mm2": float(areas.std()),
        "distinct_area_count": int(len(np.unique(areas))),
        "best_direction": records[0],
        "worst_direction": records[-1],
        "correlations": correlations,
        "score_definition": (
            "score = 1 - (A_proj - min(A_proj)) / (max(A_proj) - min(A_proj)); "
            "higher score means lower projected support area"
        ),
        "recognized_feature_summary": recognized["summary"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "direction_scores.json").write_text(
        json.dumps(
            {"summary": summary, "records": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fields = [
        "rank",
        "direction_label",
        "build_up_x",
        "build_up_y",
        "build_up_z",
        "support_projection_area_mm2",
        "projected_triangle_sum_mm2",
        "retained_overhang_surface_mm2",
        "downward_candidate_surface_mm2",
        "feature_exempt_surface_mm2",
        "support_candidate_face_count",
        "small_horizontal_exemptions",
        "score",
    ]
    with (args.output_dir / "direction_scores.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                **{field: row[field] for field in fields if field != "build_up_x"
                   and field != "build_up_y" and field != "build_up_z"},
                "build_up_x": row["build_up_input"][0],
                "build_up_y": row["build_up_input"][1],
                "build_up_z": row["build_up_input"][2],
            })
    write_visualisations(args.output_dir, records, best_result)
    write_readme(
        args.output_dir / "README.md",
        args.mesh,
        records,
        summary,
        args,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
