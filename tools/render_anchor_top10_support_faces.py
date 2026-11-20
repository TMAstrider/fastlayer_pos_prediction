#!/usr/bin/env python3
"""Render synthetic GT and model Top-10 poses with support faces highlighted."""
from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

try:
    from tools.generate_simplified_supports import (
        process_config,
        read_mesh,
        rotation_align_to_z,
    )
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
    from generate_simplified_supports import (
        process_config,
        read_mesh,
        rotation_align_to_z,
    )
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


def yaw_align_to_reference(
    vertices: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align only free display yaw about build Z."""
    vertex_center = vertices.mean(axis=0)
    reference_center = reference.mean(axis=0)
    vertex_xy = vertices[:, :2] - vertex_center[:2]
    reference_xy = reference[:, :2] - reference_center[:2]
    cosine = float(np.sum(vertex_xy * reference_xy))
    sine = float(np.sum(
        vertex_xy[:, 0] * reference_xy[:, 1]
        - vertex_xy[:, 1] * reference_xy[:, 0]
    ))
    angle = np.arctan2(sine, cosine)
    c, s = np.cos(angle), np.sin(angle)
    yaw = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    aligned = (
        (vertices - vertex_center)
        @ yaw.T
        + reference_center
    )
    return aligned, yaw


def ground_vertices(vertices: np.ndarray) -> np.ndarray:
    grounded = np.asarray(vertices, dtype=np.float64).copy()
    xy_center = (
        grounded[:, :2].min(axis=0)
        + grounded[:, :2].max(axis=0)
    ) / 2
    grounded[:, :2] -= xy_center
    grounded[:, 2] -= grounded[:, 2].min()
    return grounded


def make_part_jobs(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    split_path = (root / arguments.split_file).resolve()
    geometry_dir = (root / arguments.geometry_features).resolve()
    run_dir = (root / arguments.run).resolve()
    rules_path = (root / arguments.rules).resolve()
    output_dir = (root / arguments.output).resolve()

    split = json.loads(split_path.read_text(encoding="utf-8"))
    development_project = choose_development_project(
        split["train_records"]
    )
    fit_records = [
        record
        for record in split["train_records"]
        if record["source_project"] != development_project
    ]
    geometry_records, _, _ = load_geometry_feature_records(geometry_dir)
    fit_parts = load_part_rows(
        fit_records,
        arguments.point_count,
        geometry_records,
    )
    validation_parts = load_part_rows(
        split["validation_records"],
        arguments.point_count,
        geometry_records,
    )
    by_job = {
        part["record"]["job_id"]: part
        for part in validation_parts
    }
    missing = [
        job_id
        for job_id in arguments.job_ids
        if job_id not in by_job
    ]
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

    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    limits = process_config(
        rules,
        arguments.material,
        arguments.mode,
    )
    part_jobs = []
    for number, (part, prediction) in enumerate(
        zip(selected_parts, predictions),
        start=1,
    ):
        record = part["record"]
        gt_index = int(np.argmin(part["areas"]))
        gt_direction = np.asarray(part["directions"][gt_index])
        gt_direction /= np.linalg.norm(gt_direction)
        gt_area = float(part["areas"][gt_index])
        top_indices = np.argsort(-prediction, kind="stable")[
            :arguments.top_k
        ]
        dense_rank = np.searchsorted(
            np.unique(part["areas"]),
            part["areas"],
        ) + 1
        cases = []
        for model_rank, candidate_index in enumerate(
            top_indices,
            start=1,
        ):
            direction = np.asarray(
                part["directions"][int(candidate_index)]
            )
            direction /= np.linalg.norm(direction)
            area = float(part["areas"][int(candidate_index)])
            angle = float(np.degrees(np.arccos(np.clip(
                direction @ gt_direction,
                -1.0,
                1.0,
            ))))
            filename = (
                f"part_{record['source_part_id']}/"
                f"rank_{model_rank:02d}_candidate_"
                f"{int(candidate_index):03d}.png"
            )
            cases.append({
                "case_kind": "model",
                "model_rank": model_rank,
                "candidate_index": int(candidate_index),
                "direction": direction.tolist(),
                "model_score": float(prediction[candidate_index]),
                "area_mm2": area,
                "area_over_gt": area / max(gt_area, 1e-9),
                "angle_to_gt_deg": angle,
                "area_rank": int(dense_rank[candidate_index]),
                "image_path": str(output_dir / "images" / filename),
                "image_file": f"images/{filename}",
            })
        gt_filename = (
            f"part_{record['source_part_id']}/"
            f"gt_minimum_area_candidate_{gt_index:03d}.png"
        )
        gt_case = {
            "case_kind": "gt",
            "model_rank": None,
            "candidate_index": gt_index,
            "direction": gt_direction.tolist(),
            "model_score": None,
            "area_mm2": gt_area,
            "area_over_gt": 1.0,
            "angle_to_gt_deg": 0.0,
            "area_rank": 1,
            "image_path": str(output_dir / "images" / gt_filename),
            "image_file": f"images/{gt_filename}",
        }
        part_jobs.append({
            "number": number,
            "job_id": record["job_id"],
            "source_part_id": int(record["source_part_id"]),
            "source_project": record["source_project"],
            "source_part_name": record["source_part_name"],
            "mesh_path": record["mesh_path"],
            "gt_index": gt_index,
            "gt_direction": gt_direction.tolist(),
            "gt_rotation": rotation_align_to_z(gt_direction).tolist(),
            "gt_area_mm2": gt_area,
            "limits": limits,
            "projection_grid_mm": arguments.projection_grid_mm,
            "gt_case": gt_case,
            "cases": cases,
            "output_dir": str(output_dir),
        })
    return part_jobs


def render_pose(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    candidate_faces: np.ndarray,
    case: dict[str, Any],
    part: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = vertices[faces]
    selected_stride = max(1, len(faces) // 28_000)
    selected = np.arange(0, len(faces), selected_stride)
    selected = np.unique(np.concatenate([
        selected,
        np.flatnonzero(candidate_faces),
    ]))
    display_triangles = triangles[selected]
    display_candidate = candidate_faces[selected]
    display_normals = np.cross(
        display_triangles[:, 1] - display_triangles[:, 0],
        display_triangles[:, 2] - display_triangles[:, 0],
    )
    display_normals /= np.maximum(
        np.linalg.norm(display_normals, axis=1, keepdims=True),
        1e-12,
    )
    light = np.array([-0.35, -0.55, 0.78])
    light /= np.linalg.norm(light)
    brightness = 0.48 + 0.52 * np.clip(
        display_normals @ light,
        0.0,
        1.0,
    )
    colors = (
        brightness[:, None]
        * np.array([0.56, 0.68, 0.78])[None, :]
    )
    orange = np.array([1.00, 0.24, 0.08])
    colors[display_candidate] = (
        (0.70 + 0.30 * brightness[display_candidate, None])
        * orange[None, :]
    )

    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    span = max(float(np.ptp(vertices, axis=0).max()), 1e-3)
    center_xy = (low[:2] + high[:2]) / 2
    platform_size = span * 1.18

    figure = plt.figure(figsize=(11.2, 5.2), facecolor="white")
    axes = [
        figure.add_subplot(1, 2, index + 1, projection="3d")
        for index in range(2)
    ]
    for axis, azimuth, view_name in zip(
        axes,
        (-55, 125),
        ("View A", "View B"),
    ):
        plane = np.array([
            [
                center_xy[0] - platform_size / 2,
                center_xy[1] - platform_size / 2,
                0.0,
            ],
            [
                center_xy[0] + platform_size / 2,
                center_xy[1] - platform_size / 2,
                0.0,
            ],
            [
                center_xy[0] + platform_size / 2,
                center_xy[1] + platform_size / 2,
                0.0,
            ],
            [
                center_xy[0] - platform_size / 2,
                center_xy[1] + platform_size / 2,
                0.0,
            ],
        ])
        axis.add_collection3d(Poly3DCollection(
            [plane],
            facecolor="#e5eaf0",
            edgecolor="#a7b2bf",
            linewidth=0.7,
            alpha=0.68,
            zorder=0,
        ))
        for tick in np.linspace(
            -platform_size / 2,
            platform_size / 2,
            7,
        ):
            x = center_xy[0] + tick
            y = center_xy[1] + tick
            axis.plot(
                [
                    center_xy[0] - platform_size / 2,
                    center_xy[0] + platform_size / 2,
                ],
                [y, y],
                [0.0, 0.0],
                color="#aeb8c5",
                linewidth=0.45,
                alpha=0.55,
                zorder=1,
            )
            axis.plot(
                [x, x],
                [
                    center_xy[1] - platform_size / 2,
                    center_xy[1] + platform_size / 2,
                ],
                [0.0, 0.0],
                color="#aeb8c5",
                linewidth=0.45,
                alpha=0.55,
                zorder=1,
            )
        axis.add_collection3d(Poly3DCollection(
            display_triangles,
            facecolors=colors,
            edgecolors="none",
            linewidths=0,
            antialiased=False,
            zsort="average",
            zorder=2,
        ))
        candidate_ids = np.flatnonzero(candidate_faces)
        if len(candidate_ids):
            point_stride = max(1, len(candidate_ids) // 6000)
            centers = (
                triangles[candidate_ids[::point_stride]]
                .mean(axis=1)
            )
            axis.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                color="#ff2f12",
                s=2.0,
                alpha=0.86,
                depthshade=False,
                linewidths=0,
                zorder=5,
            )
        arrow_base = np.array([
            center_xy[0] - platform_size * 0.44,
            center_xy[1] - platform_size * 0.44,
            0.0,
        ])
        arrow_length = platform_size * 0.16
        axis.quiver(
            *arrow_base,
            0.0,
            0.0,
            arrow_length,
            color="#c8382d",
            linewidth=1.8,
            arrow_length_ratio=0.18,
            zorder=6,
        )
        axis.text(
            *(arrow_base + np.array([0.0, 0.0, arrow_length * 1.12])),
            "+build Z",
            color="#a52d25",
            fontsize=8,
            weight="bold",
            zorder=7,
        )
        half_span = platform_size / 2
        axis.set_xlim(
            center_xy[0] - half_span,
            center_xy[0] + half_span,
        )
        axis.set_ylim(
            center_xy[1] - half_span,
            center_xy[1] + half_span,
        )
        axis.set_zlim(
            -platform_size * 0.04,
            max(high[2], platform_size * 0.80),
        )
        axis.set_box_aspect((
            1.0,
            1.0,
            max(float(high[2] - low[2]) / platform_size, 0.16),
        ))
        axis.set_proj_type("ortho")
        axis.view_init(elev=28, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(
            view_name,
            fontsize=10,
            color="#38485c",
            pad=-8,
        )

    if case["case_kind"] == "gt":
        header = (
            f"Part {part['source_part_id']} · synthetic GT · "
            f"minimum-area candidate {case['candidate_index']}"
        )
        details = (
            f"Support-candidate area {case['area_mm2']:.2f} mm² · "
            "minimum of 256 sampled directions\n"
            "area rank 1 · model score n/a"
        )
        header_color = "#196b46"
    else:
        header = (
            f"Part {part['source_part_id']} · model rank "
            f"{case['model_rank']} / {len(part['cases'])} · "
            f"candidate {case['candidate_index']}"
        )
        details = (
            f"Support-candidate area {case['area_mm2']:.2f} mm² · "
            f"{case['area_over_gt']:.2f}× synthetic GT · "
            f"angle to GT {case['angle_to_gt_deg']:.1f}°\n"
            f"model score {case['model_score']:.4f} · "
            f"area rank {case['area_rank']}"
        )
        header_color = "#202b3b"
    figure.text(
        0.025,
        0.955,
        header,
        fontsize=13,
        weight="bold",
        color=header_color,
        va="top",
    )
    figure.text(
        0.025,
        0.902,
        details,
        fontsize=8.8,
        color="#555f6d",
        va="top",
        linespacing=1.3,
    )
    figure.text(
        0.5,
        0.026,
        (
            "Orange faces are the retained 45° downward overhangs that "
            "the simplified proxy scores as requiring support. "
            "Blue-gray faces are not retained support candidates."
        ),
        ha="center",
        fontsize=9,
        color="#934126",
    )
    figure.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.08,
        top=0.84,
        wspace=-0.02,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)


def render_part(part: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    from PIL import Image, ImageOps

    vertices, faces = read_mesh(Path(part["mesh_path"]))
    recognized = recognize_features(
        vertices,
        faces,
        build_up=np.array([0.0, 0.0, 1.0]),
        small_hole_diameter_mm=part["limits"][
            "horizontal_hole_diameter_threshold_mm"
        ],
    )
    gt_rotation = np.asarray(part["gt_rotation"])
    reference_pose = vertices @ gt_rotation.T

    rendered_gt_case = None
    rendered_cases = []
    for case in [part["gt_case"], *part["cases"]]:
        direction = np.asarray(case["direction"], dtype=np.float64)
        result = evaluate_support_projection(
            vertices,
            faces,
            part["limits"],
            up=direction,
            recognized_features=recognized,
            projection_grid_mm=part["projection_grid_mm"],
        )
        report = result["report"]
        recomputed_area = float(
            report["areas_mm2"]["projected_union"]
        )
        candidate_count = int(
            report["face_counts"][
                "support_candidate_after_exemptions"
            ]
        )
        np.testing.assert_allclose(
            recomputed_area,
            case["area_mm2"],
            atol=0.251,
        )
        display_vertices, _ = yaw_align_to_reference(
            result["aligned_vertices"],
            reference_pose,
        )
        display_vertices = ground_vertices(display_vertices)
        render_pose(
            Path(case["image_path"]),
            display_vertices,
            faces,
            result["candidate_faces"],
            case,
            part,
        )
        rendered_case = {
            **case,
            "recomputed_area_mm2": recomputed_area,
            "support_candidate_face_count": candidate_count,
            "small_horizontal_exemptions": int(
                report["recognized_feature_summary"][
                    "small_horizontal_exemptions"
                ]
            ),
        }
        if case["case_kind"] == "gt":
            rendered_gt_case = rendered_case
        else:
            rendered_cases.append(rendered_case)
        print(
            f"Part {part['source_part_id']} "
            f"{case['case_kind']} "
            f"{case['model_rank'] or ''}: "
            f"{candidate_count} support-candidate faces",
            flush=True,
        )

    if rendered_gt_case is None:
        raise RuntimeError("Synthetic GT pose was not rendered")
    thumbnails = []
    for case in [rendered_gt_case, *rendered_cases]:
        with Image.open(case["image_path"]) as image:
            thumbnails.append(
                ImageOps.contain(
                    image.convert("RGB"),
                    (600, 370),
                )
            )
    sheet = Image.new("RGB", (3600, 740), "#e9edf2")
    for index, thumbnail in enumerate(thumbnails):
        x = (index % 6) * 600
        y = (index // 6) * 370
        sheet.paste(thumbnail, (x, y))
    sheet_path = (
        Path(part["output_dir"])
        / f"part_{part['source_part_id']}_gt_and_top10.jpg"
    )
    sheet.save(sheet_path, quality=92)
    return {
        **part,
        "gt_case": rendered_gt_case,
        "cases": rendered_cases,
    }, sheet_path


def write_html_report(
    output_dir: Path,
    parts: list[dict[str, Any]],
    sheets: dict[int, Path],
    run: Path,
    rules_path: Path,
    material: str,
    mode: str,
) -> None:
    sections = []
    for part in parts:
        gt_case = part["gt_case"]
        cards = "".join(
            (
                f'<article><a href="{case["image_file"]}">'
                f'<img loading="lazy" src="{case["image_file"]}" '
                f'alt="part {part["source_part_id"]} rank '
                f'{case["model_rank"]}"></a></article>'
            )
            for case in part["cases"]
        )
        sections.append(
            (
                f'<section id="part-{part["source_part_id"]}">'
                f'<h2>Part {part["source_part_id"]}</h2>'
                f'<p>Synthetic GT support area: '
                f'{part["gt_area_mm2"]:.2f} mm². '
                f'Top-10 uses the model ranking; the displayed area is '
                f'the retained 45° overhang projection area.</p>'
                f'<h3>45° synthetic GT / minimum-area reference</h3>'
                f'<article class="reference"><a href="'
                f'{gt_case["image_file"]}"><img loading="lazy" src="'
                f'{gt_case["image_file"]}" alt="part '
                f'{part["source_part_id"]} synthetic GT"></a></article>'
                f'<h3>Model Top-10</h3>'
                f'<a href="{sheets[int(part["source_part_id"])].name}">'
                f'<img class="sheet" src="'
                f'{sheets[int(part["source_part_id"])].name}" '
                f'alt="part {part["source_part_id"]} top 10"></a>'
                f'<div class="grid">{cards}</div></section>'
            )
        )
    output_dir.joinpath("index.html").write_text(
        (
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<title>模型 Top-10 位姿与支撑面高亮</title>'
            '<style>body{margin:24px;background:#eef1f5;color:#233044;'
            'font:16px system-ui}main{max-width:1800px;margin:auto}'
            'section{background:white;margin:24px 0;padding:14px;'
            'box-shadow:0 2px 10px #0001}h2{margin-top:4px}'
            '.sheet{width:100%;display:block}.grid{display:grid;'
            'grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;'
            'margin-top:12px}.grid img{width:100%;display:block}'
            '.reference{max-width:920px;margin:8px 0 16px}'
            '.reference img{width:100%;display:block}'
            '@media(max-width:900px){.grid{grid-template-columns:1fr}}'
            '</style><main>'
            '<h1>模型 Top-10 位姿与支撑面高亮</h1>'
            f'<p>模型：<code>{run}</code>。规则：'
            f'{material}/{mode}，角度定义与过滤合成 GT 一致。'
            '橙色面为保留下来的向下悬垂候选面，也就是简化代理中需要'
            '支撑的位置；蓝灰面不会被该代理计入支撑投影面积。</p>'
            '<p>每张卡片的两个视角共享同一姿态和比例。所有候选方向'
            '都只做绕平台 Z 的显示 yaw 对齐，不改变构建方向、支撑面'
            '判定或面积。</p>'
            '<p>每个零件先给出 256 个方向中投影面积最小的 45° 合成 GT，'
            '再给出模型 Top-10。即使合成 GT 不在模型 Top-10 里，也会'
            '单独保留。</p>'
            f'<p>规则文件：<code>{rules_path}</code></p>'
            + "".join(sections)
            + '</main></html>'
        ),
        encoding="utf-8",
    )


def write_ranking_csv(
    path: Path,
    parts: list[dict[str, Any]],
) -> None:
    fields = [
        "source_part_id",
        "gt_area_mm2",
        "case_kind",
        "model_rank",
        "candidate_index",
        "model_score",
        "area_mm2",
        "area_over_gt",
        "angle_to_gt_deg",
        "area_rank",
        "support_candidate_face_count",
        "small_horizontal_exemptions",
        "image_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for part in parts:
            for case in [part["gt_case"], *part["cases"]]:
                writer.writerow({
                    "source_part_id": part["source_part_id"],
                    "gt_area_mm2": part["gt_area_mm2"],
                    "case_kind": case["case_kind"],
                    "model_rank": case["model_rank"] or "",
                    "candidate_index": case["candidate_index"],
                    "model_score": (
                        ""
                        if case["model_score"] is None
                        else case["model_score"]
                    ),
                    "area_mm2": case["area_mm2"],
                    "area_over_gt": case["area_over_gt"],
                    "angle_to_gt_deg": case["angle_to_gt_deg"],
                    "area_rank": case["area_rank"],
                    "support_candidate_face_count": case[
                        "support_candidate_face_count"
                    ],
                    "small_horizontal_exemptions": case[
                        "small_horizontal_exemptions"
                    ],
                    "image_file": case["image_file"],
                })


def make_overview(
    output_dir: Path,
    sheets: list[Path],
) -> None:
    from PIL import Image, ImageOps

    thumbnails = []
    for path in sheets:
        with Image.open(path) as image:
            thumbnails.append(
                ImageOps.contain(
                    image.convert("RGB"),
                    (960, 230),
                )
            )
    rows = (len(thumbnails) + 1) // 2
    overview = Image.new("RGB", (1920, rows * 230), "#e9edf2")
    for index, thumbnail in enumerate(thumbnails):
        overview.paste(
            thumbnail,
            ((index % 2) * 960, (index // 2) * 230),
        )
    overview.save(output_dir / "overview.jpg", quality=92)


def make_zip(output_dir: Path) -> Path:
    archive = output_dir / "anchor_top10_support_faces.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zipped:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or path == archive:
                continue
            zipped.write(path, path.relative_to(output_dir))
    return archive


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
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--job-ids",
        nargs="+",
        default=DEFAULT_JOB_IDS,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/anchor_top10_support_highlight_2026-11-20"
        ),
    )
    args = parser.parse_args()
    if args.top_k < 1 or args.workers < 1:
        parser.error("--top-k and --workers must be positive")

    root = Path(__file__).resolve().parents[1]
    args.run = (root / args.run).resolve()
    args.dataset = (root / args.dataset).resolve()
    args.split_file = (root / args.split_file).resolve()
    args.geometry_features = (
        root / args.geometry_features
    ).resolve()
    args.rules = (root / args.rules).resolve()
    args.output = (root / args.output).resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    part_jobs = make_part_jobs(args)
    rendered_parts = []
    sheets: dict[int, Path] = {}
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(part_jobs))
    ) as executor:
        futures = {
            executor.submit(render_part, part): part
            for part in part_jobs
        }
        for future in as_completed(futures):
            part, sheet = future.result()
            rendered_parts.append(part)
            sheets[int(part["source_part_id"])] = sheet
            print(
                f"Rendered part {part['source_part_id']}: {sheet.name}",
                flush=True,
            )
    rendered_parts.sort(key=lambda item: item["number"])
    ordered_sheets = [
        sheets[int(part["source_part_id"])]
        for part in rendered_parts
    ]

    manifest = {
        "run": str(args.run),
        "dataset": str(args.dataset),
        "split_file": str(args.split_file),
        "geometry_features": str(args.geometry_features),
        "rules_file": str(args.rules),
        "material": args.material,
        "mode": args.mode,
        "projection_grid_mm": args.projection_grid_mm,
        "top_k": args.top_k,
        "ranking": "Model score descending; displayed areas are targets.",
        "gt": (
            "Minimum 45-degree proxy projected support area among the "
            "256 sampled directions."
        ),
        "highlight": (
            "Orange mesh faces are retained downward overhang candidates "
            "after automatic small-feature exemptions."
        ),
        "display_yaw": (
            "Free yaw about build Z is aligned to the synthetic GT pose "
            "for visualization only."
        ),
        "parts": rendered_parts,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_ranking_csv(args.output / "ranking.csv", rendered_parts)
    write_html_report(
        args.output,
        rendered_parts,
        sheets,
        args.run,
        args.rules,
        args.material,
        args.mode,
    )
    make_overview(args.output, ordered_sheets)
    archive = make_zip(args.output)
    print(
        f"Complete: {args.output / 'index.html'}",
        flush=True,
    )
    print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
