#!/usr/bin/env python3
"""Evaluate the projection-score model on 100 project-isolated historical parts."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.stats import rankdata

try:
    from tools.build_projection_score_dataset import (
        FEATURE_NAMES,
        score_one_part,
    )
    from tools.generate_simplified_supports import process_config
    from tools.score_support_directions import fibonacci_directions
    from tools.train_projection_score_mlp import (
        CandidateMLP,
        aggregate_metrics,
        evaluate_baseline,
        evaluate_part,
        load_part_rows,
        predict_parts,
        rank_fusion,
        score_from_lower_is_better,
    )
except ModuleNotFoundError:
    from build_projection_score_dataset import FEATURE_NAMES, score_one_part
    from generate_simplified_supports import process_config
    from score_support_directions import fibonacci_directions
    from train_projection_score_mlp import (
        CandidateMLP,
        aggregate_metrics,
        evaluate_baseline,
        evaluate_part,
        load_part_rows,
        predict_parts,
        rank_fusion,
        score_from_lower_is_better,
    )


def select_external_jobs(
    root: Path,
    current_manifest_path: Path,
    dedup_manifest_path: Path,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    deduplicated = json.loads(dedup_manifest_path.read_text(encoding="utf-8"))
    current_hashes = {
        row["mesh_hash_sha256"]
        for row in current["records"]
    }

    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlap_count = 0
    missing_count = 0
    for record in deduplicated["records"]:
        geometry_hash = record["geometry_sha256"]
        if geometry_hash in current_hashes:
            overlap_count += 1
            continue
        source_stl = (root / record["source_stl_file"]).resolve()
        if not source_stl.is_file():
            missing_count += 1
            continue
        by_project[record["source_project"]].append({
            **record,
            "mesh_path": str(source_stl),
        })

    projects = sorted(by_project)
    if len(projects) < count:
        raise ValueError(
            f"Only {len(projects)} eligible projects for requested {count} parts"
        )
    rng = np.random.default_rng(seed)
    selected_projects = rng.choice(
        np.asarray(projects, dtype=object),
        size=count,
        replace=False,
    )
    selected_projects = sorted(str(project) for project in selected_projects)

    jobs = []
    for index, project in enumerate(selected_projects):
        candidates = sorted(
            by_project[project],
            key=lambda row: row["geometry_sha256"],
        )
        record = candidates[int(rng.integers(0, len(candidates)))]
        geometry_hash = record["geometry_sha256"]
        jobs.append({
            "index": index,
            "job_id": f"external100_{index:03d}__{geometry_hash[:12]}",
            "source_project": project,
            "source_part_id": 100000 + index,
            "source_part_name": record.get("source_stl_file", ""),
            "mesh_hash_sha256": geometry_hash,
            "shape_group_id": record["shape_group_id"],
            "source_stl_file": record["source_stl_file"],
            "source_record_id": record["source_record_id"],
            "mesh_path": record["mesh_path"],
            "triangle_count": int(record["triangle_count"]),
        })

    selected_hashes = [job["mesh_hash_sha256"] for job in jobs]
    if len(set(selected_hashes)) != len(jobs):
        raise ValueError("External selection contains duplicate geometry hashes")
    if set(selected_hashes) & current_hashes:
        raise ValueError("External selection overlaps current projection dataset")

    audit = {
        "selection_method": (
            "Seeded sample of distinct source projects; one geometry per source "
            "project; exact geometry-hash exclusion against the current 193 parts."
        ),
        "seed": seed,
        "requested_count": count,
        "source_deduplicated_count": len(deduplicated["records"]),
        "source_project_count": len(by_project),
        "excluded_current_geometry_count": overlap_count,
        "missing_stl_count": missing_count,
        "eligible_geometry_count": sum(len(rows) for rows in by_project.values()),
        "selected_project_count": len(selected_projects),
    }
    return jobs, audit


def build_external_labels(
    jobs: list[dict[str, Any]],
    directions: np.ndarray,
    limits: dict[str, float],
    projection_grid_mm: float,
    labels_dir: Path,
    workers: int,
    force: bool,
) -> list[dict[str, Any]]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    ordered_jobs = sorted(
        jobs,
        key=lambda row: (-row["triangle_count"], row["job_id"]),
    )
    completed_by_index: dict[int, dict[str, Any]] = {}
    pending = []
    for job in ordered_jobs:
        label_path = labels_dir / f"{job['job_id']}.npz"
        if label_path.is_file() and not force:
            try:
                with np.load(label_path) as data:
                    valid = (
                        len(data["directions"]) == len(directions)
                        and data["features"].shape
                        == (len(directions), len(FEATURE_NAMES))
                    )
                if valid:
                    completed_by_index[job["index"]] = {
                        **job,
                        "label_file": str(label_path),
                        "reused": True,
                    }
                    continue
            except Exception:
                pass
        pending.append((job, label_path))

    print(
        f"{len(jobs)} external parts: {len(completed_by_index)} cached, "
        f"{len(pending)} pending, {workers} workers, "
        f"{len(directions)} directions",
        flush=True,
    )
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    score_one_part,
                    job,
                    directions,
                    limits,
                    projection_grid_mm,
                    label_path,
                ): job["index"]
                for job, label_path in pending
            }
            for finished, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=len(completed_by_index) + 1,
            ):
                job_index = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"External part index {job_index} failed"
                    ) from error
                completed_by_index[job_index] = result
                print(
                    f"[{finished}/{len(jobs)}] {result['job_id']} "
                    f"{result['source_project']}: "
                    f"{result['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    return [completed_by_index[index] for index in sorted(completed_by_index)]


def enriched_per_part_rows(
    predictions: list[np.ndarray],
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for prediction, part in zip(predictions, parts):
        predicted_index = int(np.argmax(prediction))
        oracle_index = int(np.argmax(part["targets"]))
        predicted_area = float(part["areas"][predicted_index])
        oracle_area = float(part["areas"][oracle_index])
        top5 = np.argsort(-prediction)[:5]
        angle = float(np.degrees(np.arccos(np.clip(
            np.dot(
                part["directions"][predicted_index],
                part["directions"][oracle_index],
            ),
            -1.0,
            1.0,
        ))))
        evaluation = evaluate_part(
            prediction,
            part["targets"],
            part["areas"],
            part["directions"],
        )
        # Relative area errors are undefined when the geometric oracle is zero.
        # Preserve those parts for ranking metrics, but report zero-area hits
        # separately from percentage-error quantiles.
        evaluation.pop("top1_area_ratio")
        positive_oracle = oracle_area > 1e-9
        rows.append({
            "source_project": part["record"]["source_project"],
            "source_part_id": part["record"]["source_part_id"],
            "mesh_hash_sha256": part["record"]["mesh_hash_sha256"],
            "triangle_count": part["record"].get("triangle_count", 0),
            **evaluation,
            "predicted_area_mm2": predicted_area,
            "oracle_area_mm2": oracle_area,
            "top1_area_ratio": (
                predicted_area / oracle_area if positive_oracle else None
            ),
            "area_excess_abs_mm2": max(predicted_area - oracle_area, 0.0),
            "area_excess_pct": (
                (predicted_area / oracle_area - 1.0) * 100.0
                if positive_oracle
                else None
            ),
            "oracle_area_is_zero": int(not positive_oracle),
            "predicted_area_rank": int(rankdata(part["areas"], method="dense")[predicted_index]),
            "top1_exact": int(predicted_index == oracle_index),
            "top5_exact": int(oracle_index in top5),
            "top1_angle_to_oracle_deg": angle,
        })
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_metrics([
        {
            key: value
            for key, value in row.items()
            if key not in {"top1_area_ratio", "area_excess_pct"}
        }
        for row in rows
    ])
    positive_oracle_rows = [
        row for row in rows if row["oracle_area_mm2"] > 1e-9
    ]
    zero_oracle_rows = [
        row for row in rows if row["oracle_area_mm2"] <= 1e-9
    ]
    excess = np.asarray([
        row["area_excess_pct"] for row in positive_oracle_rows
    ], dtype=np.float64)
    absolute_excess = np.asarray([
        row["area_excess_abs_mm2"] for row in rows
    ], dtype=np.float64)
    area_ratio = np.asarray([
        row["top1_area_ratio"] for row in positive_oracle_rows
    ], dtype=np.float64)
    positive_count = len(positive_oracle_rows)
    zero_count = len(zero_oracle_rows)
    zero_exact_count = int(sum(
        float(row["predicted_area_mm2"]) <= 1e-9
        for row in zero_oracle_rows
    ))
    return {
        **aggregate,
        "count": len(rows),
        "top1_exact_count": int(sum(row["top1_exact"] for row in rows)),
        "top1_exact_rate": float(np.mean([row["top1_exact"] for row in rows])),
        "top5_exact_count": int(sum(row["top5_exact"] for row in rows)),
        "top5_exact_rate": float(np.mean([row["top5_exact"] for row in rows])),
        "within_1pct_count": int(np.sum(excess <= 1.0)),
        "within_5pct_count": int(np.sum(excess <= 5.0)),
        "within_10pct_count": int(np.sum(excess <= 10.0)),
        "within_15pct_count": int(np.sum(excess <= 15.0)),
        "within_1pct_rate_positive_oracle": float(np.mean(excess <= 1.0)),
        "within_5pct_rate_positive_oracle": float(np.mean(excess <= 5.0)),
        "within_10pct_rate_positive_oracle": float(np.mean(excess <= 10.0)),
        "within_15pct_rate_positive_oracle": float(np.mean(excess <= 15.0)),
        "positive_oracle_count": positive_count,
        "zero_oracle_count": zero_count,
        "zero_oracle_exact_count": zero_exact_count,
        "zero_oracle_exact_rate": (
            zero_exact_count / zero_count if zero_count else 1.0
        ),
        "area_excess_pct_p50": float(np.percentile(excess, 50)),
        "area_excess_pct_p75": float(np.percentile(excess, 75)),
        "area_excess_pct_p90": float(np.percentile(excess, 90)),
        "area_excess_pct_p95": float(np.percentile(excess, 95)),
        "area_excess_pct_max": float(excess.max()),
        "area_excess_abs_mm2_p50": float(np.percentile(absolute_excess, 50)),
        "area_excess_abs_mm2_p90": float(np.percentile(absolute_excess, 90)),
        "area_excess_abs_mm2_max": float(absolute_excess.max()),
        "top1_area_ratio_mean": float(np.mean(area_ratio)),
        "top1_area_ratio_median": float(np.median(area_ratio)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spearman = np.sort(np.asarray([row["spearman"] for row in rows]))
    excess = np.sort(np.asarray([
        row["area_excess_pct"]
        for row in rows
        if row["area_excess_pct"] is not None
    ], dtype=np.float64))
    thresholds = np.linspace(0.0, max(30.0, float(np.ceil(excess.max()))), 301)
    cumulative = np.asarray([np.mean(excess <= value) for value in thresholds])
    figure, axes = plt.subplots(3, 1, figsize=(11, 10))
    axes[0].plot(np.arange(1, len(spearman) + 1), spearman, color="#247b65")
    axes[0].set(
        title="External 100 parts: Spearman by part",
        xlabel="part (sorted)",
        ylabel="Spearman",
        ylim=(min(0.0, float(spearman.min()) - 0.02), 1.01),
    )
    axes[1].plot(np.arange(1, len(excess) + 1), excess, color="#b65b22")
    axes[1].set(
        title="External 100 parts: Top-1 projected-area excess",
        xlabel="part (sorted)",
        ylabel="excess over oracle (%)",
    )
    axes[2].plot(thresholds, cumulative * 100.0, color="#3e63a8")
    axes[2].set(
        title="External 100 parts: cumulative Top-1 area error",
        xlabel="allowed excess over oracle (%)",
        ylabel="parts within tolerance (%)",
        ylim=(0, 101),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor="#f5f7fa")
    plt.close(figure)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    hybrid = metrics["metrics"]["hybrid"]
    model = metrics["metrics"]["model"]
    baseline = metrics["metrics"]["baseline"]
    audit = metrics["selection"]
    comparison = metrics["comparison"]
    comparison_rows = []
    for label, summary in (
        ("训练集", comparison["train"]),
        ("Development", comparison["development"]),
        ("Test", comparison["test"]),
        ("External 100", hybrid),
    ):
        comparison_rows.append(
            f"| {label} | {summary['count']} | "
            f"{summary['spearman_median']:.4f} | "
            f"{summary['kendall_median']:.4f} | "
            f"{summary['ndcg_at_5_median']:.4f} | "
            f"{summary['top1_exact_count']}/{summary['count']} | "
            f"{summary['within_5pct_count']}/"
            f"{summary['positive_oracle_count']} | "
            f"{summary['area_excess_pct_p50']:.2f}% / "
            f"{summary['area_excess_pct_p90']:.2f}% |"
        )
    comparison_table = "\n".join(comparison_rows)
    path.write_text(
        f"""# 投影面积评分器：100 件外部历史零件

## 数据隔离

- 历史记录：1644 条。
- 形状去重：1032 件。
- 与当前 193 件按几何 SHA256 重合：{audit["excluded_current_geometry_count"]} 件，全部排除。
- 可评估几何：{audit["eligible_geometry_count"]} 件，来自 {audit["source_project_count"]} 个项目。
- 最终按项目无放回抽取：{audit["selected_project_count"]} 个项目，每项目 1 件。
- 抽样种子：{audit["seed"]}。

这 100 件不参与训练、development 或早停，并与当前数据集没有完全相同的几何。
形状哈希只能排除完全重复，不能证明不存在近似零件族或设计版本关联。

## 评估配置

- 模型：锚点 CandidateMLP，最佳 epoch {metrics["model"]["best_epoch"]}。
- 每件方向：{metrics["evaluation"]["direction_count"]}。
- 投影栅格：{metrics["evaluation"]["projection_grid_mm"]} mm。
- 规则：AlSi10Mg standard，45° 悬垂阈值，严格 `<6 mm` 水平孔豁免。
- development 选出的几何排名融合权重：{metrics["model"]["fusion_anchor_weight"]}。
- 权重：`{checkpoint_path}`。

## 总体结果

| 指标 | 融合后 | MLP | 几何 fallback |
|---|---:|---:|---:|
| Spearman mean | {hybrid["spearman_mean"]:.4f} | {model["spearman_mean"]:.4f} | {baseline["spearman_mean"]:.4f} |
| Spearman median | {hybrid["spearman_median"]:.4f} | {model["spearman_median"]:.4f} | {baseline["spearman_median"]:.4f} |
| Kendall mean | {hybrid["kendall_mean"]:.4f} | {model["kendall_mean"]:.4f} | {baseline["kendall_mean"]:.4f} |
| Kendall median | {hybrid["kendall_median"]:.4f} | {model["kendall_median"]:.4f} | {baseline["kendall_median"]:.4f} |
| NDCG@5 mean | {hybrid["ndcg_at_5_mean"]:.4f} | {model["ndcg_at_5_mean"]:.4f} | {baseline["ndcg_at_5_mean"]:.4f} |
| NDCG@5 median | {hybrid["ndcg_at_5_median"]:.4f} | {model["ndcg_at_5_median"]:.4f} | {baseline["ndcg_at_5_median"]:.4f} |
| NDCG@10 mean | {hybrid["ndcg_at_10_mean"]:.4f} | {model["ndcg_at_10_mean"]:.4f} | {baseline["ndcg_at_10_mean"]:.4f} |
| NDCG@10 median | {hybrid["ndcg_at_10_median"]:.4f} | {model["ndcg_at_10_median"]:.4f} | {baseline["ndcg_at_10_median"]:.4f} |
| Top-1 归一化面积损失中位数 | {hybrid["top1_regret_median"]:.6f} | {model["top1_regret_median"]:.6f} | {baseline["top1_regret_median"]:.6f} |
| Top-1 精确命中 | {hybrid["top1_exact_count"]}/{hybrid["count"]} | {model["top1_exact_count"]}/{model["count"]} | {baseline["top1_exact_count"]}/{baseline["count"]} |
| Top-5 包含 oracle | {hybrid["top5_exact_count"]}/{hybrid["count"]} | {model["top5_exact_count"]}/{model["count"]} | {baseline["top5_exact_count"]}/{baseline["count"]} |
| Top-1 面积 / oracle 中位数 | {hybrid["top1_area_ratio_median"]:.4f} | {model["top1_area_ratio_median"]:.4f} | {baseline["top1_area_ratio_median"]:.4f} |
| Top-1 方向误差中位数 | {hybrid["top1_angle_to_oracle_deg_median"]:.2f}° | {model["top1_angle_to_oracle_deg_median"]:.2f}° | {baseline["top1_angle_to_oracle_deg_median"]:.2f}° |

融合方法的 Top-1 面积误差：

- 精确命中：{hybrid["top1_exact_count"]}/{hybrid["count"]}。
- Top-5 包含 oracle：{hybrid["top5_exact_count"]}/{hybrid["count"]}。
- oracle 面积为 0：{hybrid["zero_oracle_count"]} 件；其中预测面积也为 0：
  {hybrid["zero_oracle_exact_count"]}/{hybrid["zero_oracle_count"]}。
- 有正面积 oracle 的 {hybrid["positive_oracle_count"]} 件中，
  误差不超过 1%：{hybrid["within_1pct_count"]}/{hybrid["positive_oracle_count"]}；
  5%：{hybrid["within_5pct_count"]}/{hybrid["positive_oracle_count"]}；
  10%：{hybrid["within_10pct_count"]}/{hybrid["positive_oracle_count"]}；
  15%：{hybrid["within_15pct_count"]}/{hybrid["positive_oracle_count"]}。
- 面积误差 P50 / P90 / P95：{hybrid["area_excess_pct_p50"]:.2f}% /
  {hybrid["area_excess_pct_p90"]:.2f}% / {hybrid["area_excess_pct_p95"]:.2f}%。
- 最大误差：{hybrid["area_excess_pct_max"]:.2f}%。

## 与现有数据划分对照

下表统一使用融合模型和同一套 256 方向代理标签口径。

| 数据划分 | 零件数 | Spearman 中位数 | Kendall 中位数 | NDCG@5 中位数 | Top-1 精确命中 | Top-1 ≤5% | 面积误差 P50 / P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
{comparison_table}

外部集的零面积 oracle 不进入相对百分比误差分位数和 ≤5% 分母，
但保留在 Spearman、Kendall、NDCG 与 Top-1 精确命中统计中。

## 产物

- `metrics.json`
- `per_part.csv`
- `selection.json`
- `external_100_metrics.png`
""",
        encoding="utf-8",
    )


def evaluate(
    records: list[dict[str, Any]],
    checkpoint_path: Path,
    run_config_path: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CandidateMLP(config["feature_dim"], config["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    parts = load_part_rows(records)
    predictions = predict_parts(
        model,
        parts,
        np.asarray(config["feature_mean"], dtype=np.float32),
        np.asarray(config["feature_std"], dtype=np.float32),
        torch.device("cpu"),
        float(config.get("anchor_weight", 0.0)),
        float(config.get("residual_scale", 1.0)),
    )
    fusion_weight = float(config["fusion_anchor_weight"])
    hybrid_predictions = [
        rank_fusion(prediction, part["anchor_scores"], fusion_weight)
        for prediction, part in zip(predictions, parts)
    ]
    model_rows = enriched_per_part_rows(predictions, parts)
    hybrid_rows = enriched_per_part_rows(hybrid_predictions, parts)
    baseline_predictions = [
        score_from_lower_is_better(part["retained_surface"])
        for part in parts
    ]
    baseline_rows = enriched_per_part_rows(baseline_predictions, parts)
    for row, part in zip(hybrid_rows, parts):
        row["minimum_area_mm2"] = float(part["areas"].min())
        row["maximum_area_mm2"] = float(part["areas"].max())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "per_part.csv", hybrid_rows)
    write_plot(report_dir / "external_100_metrics.png", hybrid_rows)
    metrics = {
        "metrics": {
            "hybrid": summarize_rows(hybrid_rows),
            "model": summarize_rows(model_rows),
            "baseline": summarize_rows(baseline_rows),
        },
    }
    return metrics


def evaluate_current_splits(
    root: Path,
    checkpoint_path: Path,
    run_config_path: Path,
    reference_report_dir: Path,
) -> dict[str, dict[str, Any]]:
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    dataset_dir = Path(config["dataset"]).resolve()
    split = json.loads((dataset_dir / "split.json").read_text(encoding="utf-8"))
    reference_metrics = json.loads(
        (reference_report_dir / "metrics.json").read_text(encoding="utf-8")
    )
    development_project = reference_metrics["dataset"]["development_projects"][0]
    record_sets = {
        "train": [
            record
            for record in split["train_records"]
            if record["source_project"] != development_project
        ],
        "development": [
            record
            for record in split["train_records"]
            if record["source_project"] == development_project
        ],
        "test": split["validation_records"],
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CandidateMLP(config["feature_dim"], config["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    results = {}
    for name, records in record_sets.items():
        parts = load_part_rows(records)
        predictions = predict_parts(
            model,
            parts,
            np.asarray(config["feature_mean"], dtype=np.float32),
            np.asarray(config["feature_std"], dtype=np.float32),
            torch.device("cpu"),
            float(config.get("anchor_weight", 0.0)),
            float(config.get("residual_scale", 1.0)),
        )
        hybrid_predictions = [
            rank_fusion(
                prediction,
                part["anchor_scores"],
                float(config["fusion_anchor_weight"]),
            )
            for prediction, part in zip(predictions, parts)
        ]
        results[name] = summarize_rows(
            enriched_per_part_rows(hybrid_predictions, parts)
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--dedup-manifest",
        type=Path,
        default=Path("outputs/shape_deduplicated_pointclouds/manifest.json"),
    )
    parser.add_argument(
        "--current-manifest",
        type=Path,
        default=Path("labels/projection_support_area_v0_256/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/projection_score_external_100_2026-11-20"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/projection_score_external_100_2026-11-20"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/projection_score_mlp_anchored_2026-11-20/best_model.pt"),
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("reports/projection_score_mlp_anchored_2026-11-20/config.json"),
    )
    parser.add_argument(
        "--reference-report-dir",
        type=Path,
        default=Path("reports/projection_score_mlp_anchored_2026-11-20"),
    )
    parser.add_argument("--rules", type=Path, default=Path("configs/support_rules_v0.json"))
    parser.add_argument("--material", choices=["316L", "AlSi10Mg", "TC4"], default="AlSi10Mg")
    parser.add_argument("--mode", choices=["standard", "am_build"], default="standard")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--directions", type=int, default=256)
    parser.add_argument("--projection-grid-mm", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.directions < 2 or args.workers < 1:
        parser.error("count, directions, and workers must be positive")

    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    report_dir = (root / args.report_dir).resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    run_config_path = (root / args.run_config).resolve()
    reference_report_dir = (root / args.reference_report_dir).resolve()
    rules_path = (root / args.rules).resolve()
    jobs, audit = select_external_jobs(
        root,
        (root / args.current_manifest).resolve(),
        (root / args.dedup_manifest).resolve(),
        args.count,
        args.seed,
    )
    directions = np.asarray(
        [
            row["direction"]
            for row in fibonacci_directions(args.directions)
        ],
        dtype=np.float64,
    )
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    limits = process_config(rules, args.material, args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "audit": audit,
        "rules_file": str(rules_path),
        "limits": limits,
        "material": args.material,
        "mode": args.mode,
        "jobs": jobs,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    started = time.monotonic()
    records = build_external_labels(
        jobs,
        directions,
        limits,
        args.projection_grid_mm,
        output_dir / "labels",
        args.workers,
        args.force,
    )
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection": audit,
        "material": args.material,
        "mode": args.mode,
        "limits": limits,
        "direction_count": args.directions,
        "projection_grid_mm": args.projection_grid_mm,
        "feature_names": FEATURE_NAMES,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = evaluate(
        records,
        checkpoint_path,
        run_config_path,
        output_dir,
        report_dir,
    )
    metrics.update({
        "selection": audit,
        "evaluation": {
            "part_count": len(records),
            "direction_count": args.directions,
            "projection_grid_mm": args.projection_grid_mm,
            "material": args.material,
            "mode": args.mode,
            "label_build_elapsed_seconds": elapsed,
            "label_workers": args.workers,
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "best_epoch": int(
                json.loads(run_config_path.read_text(encoding="utf-8")).get(
                    "best_epoch",
                    torch.load(checkpoint_path, map_location="cpu", weights_only=False).get(
                        "best_epoch",
                        -1,
                    ),
                )
            ),
            "fusion_anchor_weight": float(
                json.loads(run_config_path.read_text(encoding="utf-8"))[
                    "fusion_anchor_weight"
                ]
            ),
        },
    })
    metrics["comparison"] = evaluate_current_splits(
        root,
        checkpoint_path,
        run_config_path,
        reference_report_dir,
    )
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(report_dir / "README.md", metrics, checkpoint_path)
    print(json.dumps({
        "part_count": len(records),
        "elapsed_seconds": round(elapsed, 2),
        "hybrid": metrics["metrics"]["hybrid"],
        "report": str(report_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
