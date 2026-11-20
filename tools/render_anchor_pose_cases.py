#!/usr/bin/env python3
"""Render historical-GT and anchored-model-prediction pose comparisons."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
    from tools.generate_simplified_supports import process_config, read_mesh
    from tools.recognize_part_features import recognize_features
    from tools.support_area_scoring import evaluate_support_projection
    from tools.train_pointnet_direction_scorer import (
        PointNetDirectionScorer,
        choose_development_project,
        load_geometry_feature_records,
        load_part_rows,
        predict_parts,
        stacked_feature_standardisation,
    )
except ModuleNotFoundError:
    from generate_simplified_supports import process_config, read_mesh
    from recognize_part_features import recognize_features
    from support_area_scoring import evaluate_support_projection
    from train_pointnet_direction_scorer import (
        PointNetDirectionScorer,
        choose_development_project,
        load_geometry_feature_records,
        load_part_rows,
        predict_parts,
        stacked_feature_standardisation,
    )


DEFAULT_JOB_IDS = [
    "shape_0285__cbd917bdc45e",
    "shape_0822__22e02cc4e047",
    "shape_0130__fb0a2edc872a",
    "shape_0348__d909c4b3ea5d",
]


def yaw_align_prediction(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_center = predicted.mean(axis=0)
    reference_center = reference.mean(axis=0)
    predicted_xy = predicted[:, :2] - predicted_center[:2]
    reference_xy = reference[:, :2] - reference_center[:2]
    cosine = float(np.sum(predicted_xy * reference_xy))
    sine = float(np.sum(
        predicted_xy[:, 0] * reference_xy[:, 1]
        - predicted_xy[:, 1] * reference_xy[:, 0]
    ))
    angle = np.arctan2(sine, cosine)
    c, s = np.cos(angle), np.sin(angle)
    yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    aligned = (predicted - predicted_center) @ yaw.T + reference_center
    return aligned, yaw


def exact_gt_area_mm2(
    mesh_path: Path,
    limits: dict[str, Any],
    projection_grid_mm: float,
) -> float:
    vertices, faces = read_mesh(mesh_path)
    recognized = recognize_features(
        vertices,
        faces,
        build_up=np.array([0.0, 0.0, 1.0]),
        small_hole_diameter_mm=limits[
            "horizontal_hole_diameter_threshold_mm"
        ],
    )
    result = evaluate_support_projection(
        vertices,
        faces,
        limits,
        up=np.array([0.0, 0.0, 1.0]),
        recognized_features=recognized,
        projection_grid_mm=projection_grid_mm,
    )
    return float(result["report"]["areas_mm2"]["projected_union"])


def make_jobs(
    arguments: argparse.Namespace,
) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = (root / arguments.dataset).resolve()
    split_path = (root / arguments.split_file).resolve()
    geometry_dir = (root / arguments.geometry_features).resolve()
    run_dir = (root / arguments.run).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development_project = choose_development_project(
        split["train_records"]
    )
    fit_records = [
        record for record in split["train_records"]
        if record["source_project"] != development_project
    ]
    geometry_records, _, _ = load_geometry_feature_records(geometry_dir)
    fit_parts = load_part_rows(
        fit_records,
        arguments.point_count,
        geometry_records,
    )
    test_parts = load_part_rows(
        split["validation_records"],
        arguments.point_count,
        geometry_records,
    )
    by_job = {
        part["record"]["job_id"]: part for part in test_parts
    }
    missing = [job_id for job_id in arguments.job_ids if job_id not in by_job]
    if missing:
        raise ValueError(f"Requested jobs are not in test split: {missing}")

    prior_mean, prior_std = stacked_feature_standardisation(
        fit_parts,
        "candidate_prior",
    )
    global_mean, global_std = stacked_feature_standardisation(
        fit_parts,
        "global_features",
    )
    device = torch.device(
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if arguments.device == "auto"
        else arguments.device
    )
    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    model = PointNetDirectionScorer(
        point_channels=6,
        point_embedding_dim=config["point_embedding_dim"],
        shape_hidden_dim=config["shape_hidden_dim"],
        direction_hidden_dim=config["direction_hidden_dim"],
        candidate_feature_dim=len(prior_mean),
        global_feature_dim=len(global_mean),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    selected_parts = [by_job[job_id] for job_id in arguments.job_ids]
    predictions = predict_parts(
        model,
        selected_parts,
        arguments.point_count,
        prior_mean,
        prior_std,
        global_mean,
        global_std,
        device,
        anchor_weight=float(config.get("anchor_weight", 1.0)),
        residual_scale=float(config.get("residual_scale", 0.5)),
    )

    rules = json.loads((root / arguments.rules).read_text(encoding="utf-8"))
    limits = process_config(rules, arguments.material, arguments.mode)
    jobs = []
    for number, (part, prediction) in enumerate(
        zip(selected_parts, predictions),
        start=1,
    ):
        record = part["record"]
        direction = np.asarray(part["directions"][int(np.argmax(prediction))])
        direction /= np.linalg.norm(direction)
        oracle_index = int(np.argmax(part["targets"]))
        oracle_area = float(part["areas"][oracle_index])
        predicted_area = float(
            part["areas"][int(np.argmax(prediction))]
        )
        top5_indices = np.argsort(-prediction)[:5]
        top5_area = float(part["areas"][top5_indices].min())
        direct30_index = int(np.argmax(part["anchor_scores"]))
        direct30_area = float(part["areas"][direct30_index])
        gt_angle = float(np.degrees(np.arccos(np.clip(
            direction @ np.array([0.0, 0.0, 1.0]),
            -1.0,
            1.0,
        ))))
        correction = Rotation.align_vectors(
            [[0.0, 0.0, 1.0]],
            direction[None, :],
        )[0].as_matrix()
        np.testing.assert_allclose(
            correction @ direction,
            [0.0, 0.0, 1.0],
            atol=1e-6,
        )
        mesh_path = Path(record["mesh_path"])
        gt_area = exact_gt_area_mm2(
            mesh_path,
            limits,
            arguments.projection_grid_mm,
        )
        filename = f"{number:02d}_part_{record['source_part_id']}.png"
        jobs.append({
            "number": number,
            "job_id": record["job_id"],
            "source_part_id": record["source_part_id"],
            "source_project": record["source_project"],
            "source_part_name": record["source_part_name"],
            "mesh_path": str(mesh_path),
            "image_path": str(arguments.output / "images" / filename),
            "image_file": f"images/{filename}",
            "prediction_rotation": correction.tolist(),
            "pred_direction": direction.tolist(),
            "pred_angle_gt_deg": gt_angle,
            "oracle_area_mm2": oracle_area,
            "gt_area_mm2": gt_area,
            "pred_area_mm2": predicted_area,
            "pred_over_oracle": predicted_area / max(oracle_area, 1e-9),
            "gt_over_oracle": gt_area / max(oracle_area, 1e-9),
            "top5_min_area_mm2": top5_area,
            "top5_over_oracle": top5_area / max(oracle_area, 1e-9),
            "top5_contains_oracle": bool(
                oracle_index in top5_indices
            ),
            "direct30_area_mm2": direct30_area,
            "direct30_over_oracle": (
                direct30_area / max(oracle_area, 1e-9)
            ),
        })
    return jobs


def render_job(job: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if not font_path.exists():
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = (
        FontProperties(fname=str(font_path))
        if font_path.exists()
        else FontProperties()
    )
    mesh = trimesh.load(
        job["mesh_path"],
        force="mesh",
        process=False,
    )
    original = np.asarray(mesh.vertices, dtype=np.float64)
    original_center_xy = (
        original[:, :2].min(axis=0)
        + original[:, :2].max(axis=0)
    ) / 2
    original[:, :2] -= original_center_xy
    original[:, 2] -= original[:, 2].min()
    prediction_rotation = np.asarray(job["prediction_rotation"])
    predicted = original @ prediction_rotation.T
    predicted, display_yaw = yaw_align_prediction(predicted, original)
    predicted_center_xy = (
        predicted[:, :2].min(axis=0)
        + predicted[:, :2].max(axis=0)
    ) / 2
    predicted[:, :2] -= predicted_center_xy
    predicted[:, 2] -= predicted[:, 2].min()
    poses = [original, original, predicted]
    original_z = np.array([0.0, 0.0, 1.0])
    marker_directions = [
        original_z,
        original_z,
        display_yaw @ prediction_rotation @ original_z,
    ]
    faces = np.asarray(mesh.faces)
    side = max(float(np.ptp(pose, axis=0).max()) for pose in poses) * 1.18
    if min(float(pose[:, 2].min()) for pose in poses) < -1e-10:
        raise ValueError("Rendered pose extends below the platform")

    figure = plt.figure(figsize=(16, 6.1), facecolor="white")
    titles = [
        "Raw part / input coordinates",
        "GT / historical engineering pose",
        "Pred / anchor-model Top-1 pose",
    ]
    title_colors = ["#325a91", "#27704c", "#9b531b"]
    subtitles = [
        "No platform; red arrow = original part +Z",
        (
            f"Historical-pose support {job['gt_area_mm2']:.1f} mm²"
            f" ({job['gt_over_oracle']:.2f}× oracle)"
        ),
        (
            f"Angle to GT {job['pred_angle_gt_deg']:.1f}°; "
            f"support {job['pred_area_mm2']:.1f} mm²"
            f" ({job['pred_over_oracle']:.2f}× oracle)"
        ),
    ]
    light = np.array([-0.4, -0.6, 0.8])
    light /= np.linalg.norm(light)
    for column, (pose, title, title_color, subtitle) in enumerate(zip(
        poses,
        titles,
        title_colors,
        subtitles,
    )):
        axis = figure.add_subplot(
            1,
            3,
            column + 1,
            projection="3d",
            computed_zorder=False,
        )
        triangles = pose[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normals /= np.maximum(
            np.linalg.norm(normals, axis=1, keepdims=True),
            1e-12,
        )
        brightness = 0.43 + 0.57 * np.clip(normals @ light, 0, 1)
        colors = (
            brightness[:, None]
            * np.array([0.75, 0.81, 0.89])[None, :]
        )
        if column > 0:
            plane = np.array([
                [-side / 2, -side / 2, 0.0],
                [side / 2, -side / 2, 0.0],
                [side / 2, side / 2, 0.0],
                [-side / 2, side / 2, 0.0],
            ])
            axis.add_collection3d(Poly3DCollection(
                [plane],
                facecolor="#e6eaf0",
                edgecolor="#9da8b7",
                linewidth=0.7,
                alpha=0.60,
                zorder=0,
            ))
            for tick in np.linspace(-side / 2, side / 2, 7):
                axis.plot(
                    [-side / 2, side / 2],
                    [tick, tick],
                    [0.0, 0.0],
                    color="#aeb8c5",
                    linewidth=0.4,
                    alpha=0.55,
                    zorder=1,
                )
                axis.plot(
                    [tick, tick],
                    [-side / 2, side / 2],
                    [0.0, 0.0],
                    color="#aeb8c5",
                    linewidth=0.4,
                    alpha=0.55,
                    zorder=1,
                )
        axis.add_collection3d(Poly3DCollection(
            triangles,
            facecolors=colors,
            edgecolors="none",
            linewidths=0,
            antialiased=False,
            zsort="average",
            zorder=2,
        ))
        marker = marker_directions[column]
        marker /= np.linalg.norm(marker)
        arrow_origin = pose.mean(axis=0)
        arrow_length = side * 0.32
        arrow_end = arrow_origin + marker * arrow_length
        for color, width in (("white", 5.5), ("#f04432", 3.0)):
            axis.quiver(
                *arrow_origin,
                *(marker * arrow_length),
                color=color,
                linewidth=width,
                arrow_length_ratio=0.24,
                normalize=False,
                zorder=5,
            )
        axis.text(
            *arrow_end,
            "original +Z",
            fontproperties=font,
            fontsize=9,
            color="#c52e22",
            weight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "#f04432",
                "alpha": 0.9,
                "pad": 2.5,
            },
            zorder=7,
        )
        axis.set(
            xlim=(-side / 2, side / 2),
            ylim=(-side / 2, side / 2),
            zlim=(-side * 0.04, side),
        )
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=38, azim=-58)
        axis.set_axis_off()
        axis.set_title(
            title,
            fontproperties=font,
            color=title_color,
            fontsize=13,
            pad=-8,
        )
        axis.text2D(
            0.5,
            0.015,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            fontproperties=font,
            fontsize=10,
            color="#444444",
        )
    figure.text(
        0.025,
        0.965,
        f"Case {job['number']} · part {job['source_part_id']}",
        fontproperties=font,
        fontsize=14,
        weight="bold",
        va="top",
        color="#202b3b",
    )
    figure.text(
        0.025,
        0.914,
        "\n".join(textwrap.wrap(job["source_part_name"], 112)),
        fontproperties=font,
        fontsize=9,
        va="top",
        color="#555555",
    )
    figure.text(
        0.5,
        0.045,
        (
            "Red arrow = original part +Z and rotates with the pose; "
            "it is not the predicted build direction. Pred is display-only "
            "yaw-aligned about platform Z."
        ),
        ha="center",
        fontproperties=font,
        fontsize=10,
        color="#c52e22",
    )
    figure.text(
        0.5,
        0.012,
        (
            "GT is the historical engineering pose. Oracle is the minimum "
            "proxy support area over 256 sampled directions."
        ),
        ha="center",
        fontproperties=font,
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.08,
        top=0.85,
        wspace=-0.04,
    )
    Path(job["image_path"]).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        job["image_path"],
        dpi=140,
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path(
            "runs/pointnet_30deg_anchor_listwise_topk_4096_"
            "project_isolated_922_10_100_2026-11-20"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256"
        ),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256/"
            "split_project_isolated_100_seed30.json"
        ),
    )
    parser.add_argument(
        "--geometry-features",
        type=Path,
        default=Path(
            "labels/geometry_prior_30deg_expanded_932_100_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/anchor_pose_comparison_2026-11-20"),
    )
    parser.add_argument(
        "--job-ids",
        nargs="+",
        default=DEFAULT_JOB_IDS,
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("configs/support_rules_v0.json"),
    )
    parser.add_argument(
        "--material",
        choices=["316L", "AlSi10Mg", "TC4"],
        default="AlSi10Mg",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "am_build"],
        default="standard",
    )
    parser.add_argument("--projection-grid-mm", type=float, default=0.5)
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.dataset = (root / args.dataset).resolve()
    args.split_file = (root / args.split_file).resolve()
    args.geometry_features = (
        root / args.geometry_features
    ).resolve()
    args.run = (root / args.run).resolve()
    args.rules = (root / args.rules).resolve()
    args.output = (root / args.output).resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(args)
    for job in jobs:
        render_job(job)
        print(
            f"Rendered {job['number']}/{len(jobs)}: "
            f"{job['image_path']}",
            flush=True,
        )
    (args.output / "manifest.json").write_text(
        json.dumps({
            "run": str(args.run),
            "selection": (
                "Four test examples: GT-aligned success, large-angle area "
                "improvement, 30-degree-anchor correction, and Top-1 failure "
                "with oracle in Top-5."
            ),
            "gt": "Historical engineering print pose; build-up is world +Z.",
            "oracle": "Minimum 45-degree proxy area over 256 sampled directions.",
            "yaw": (
                "Prediction is rotated about world +Z only for display; "
                "GT/pred angle is not altered."
            ),
            "samples": jobs,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cards = "".join(
        (
            f"<article><h2>part {job['source_part_id']} · "
            f"与GT夹角 {job['pred_angle_gt_deg']:.1f}° · "
            f"预测/oracle {job['pred_over_oracle']:.2f}×</h2>"
            f"<a href=\"{job['image_file']}\"><img "
            f"src=\"{job['image_file']}\"></a></article>"
        )
        for job in jobs
    )
    (args.output / "index.html").write_text(
        (
            "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
            "<title>GT与anchor模型预测姿态</title>"
            "<style>body{margin:24px;background:#eef1f5;color:#233044;"
            "font:16px system-ui}main{max-width:1600px;margin:auto}"
            "article{background:white;margin:20px 0;box-shadow:"
            "0 2px 10px #0001}h2{padding:12px}img{width:100%;"
            "display:block}</style><main>"
            "<h1>历史 GT 与 anchor 模型预测姿态</h1>"
            "<p>红箭头跟随零件原始 +Z；GT 和 Pred 使用透明平台。"
            "Pred 仅为显示绕平台 Z 做 yaw 对齐，夹角指标只比较打印方向。</p>"
            f"{cards}</main></html>"
        ),
        encoding="utf-8",
    )
    from PIL import Image, ImageOps

    thumbnails = []
    for job in jobs:
        with Image.open(job["image_path"]) as image:
            thumbnails.append(
                ImageOps.contain(
                    image.convert("RGB"),
                    (960, 366),
                )
            )
    rows = (len(jobs) + 1) // 2
    overview = Image.new("RGB", (1920, 366 * rows), "#e9edf2")
    for index, thumbnail in enumerate(thumbnails):
        overview.paste(
            thumbnail,
            ((index % 2) * 960, (index // 2) * 366),
        )
    overview.save(args.output / "overview.jpg", quality=92)
    print(f"Complete: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
