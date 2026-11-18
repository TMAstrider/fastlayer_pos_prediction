#!/usr/bin/env python3
"""Render the requested three-way support comparison for one part."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.generate_simplified_supports import read_mesh, rotation_align_to_z
except ModuleNotFoundError:
    from generate_simplified_supports import read_mesh, rotation_align_to_z


@dataclass
class Panel:
    title: str
    subtitle: str
    triangles: np.ndarray
    feature_faces: set[int]
    platform_z: float
    reference_triangles: np.ndarray | None = None
    support_segments: np.ndarray | None = None
    extent: tuple[np.ndarray, np.ndarray] | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def support_segments(report: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        [item["landing_point_aligned_mm"], item["source_point_aligned_mm"]]
        for item in report["supports"]
    ], dtype=np.float64)


def shifted_triangles(triangles: np.ndarray, z_shift: float) -> np.ndarray:
    shifted = np.asarray(triangles, dtype=np.float64).copy()
    shifted[..., 2] += z_shift
    return shifted


def shifted_segments(segments: np.ndarray, z_shift: float) -> np.ndarray:
    shifted = np.asarray(segments, dtype=np.float64).copy()
    shifted[..., 2] += z_shift
    return shifted


def panel_extent(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    if panel.extent is not None:
        return panel.extent
    points = panel.triangles.reshape(-1, 3)
    if panel.reference_triangles is not None:
        points = np.concatenate([
            points,
            np.asarray(panel.reference_triangles).reshape(-1, 3),
        ], axis=0)
    if panel.support_segments is not None and len(panel.support_segments):
        points = np.concatenate([
            points,
            np.asarray(panel.support_segments).reshape(-1, 3),
        ], axis=0)
    return points.min(axis=0), points.max(axis=0)


def draw_panel(axis: Any, panel: Panel) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    low, high = panel_extent(panel)
    span = np.maximum(high - low, 1.0)
    center = (low + high) / 2.0
    max_span = max(float(span.max()), 1.0)
    padding = max(max_span * 0.08, 2.0)
    platform = np.array([
        [low[0] - padding, low[1] - padding, panel.platform_z],
        [high[0] + padding, low[1] - padding, panel.platform_z],
        [high[0] + padding, high[1] + padding, panel.platform_z],
        [low[0] - padding, high[1] + padding, panel.platform_z],
    ])
    axis.add_collection3d(Poly3DCollection(
        [platform],
        facecolors="#9da8b2",
        edgecolors="#52606b",
        linewidths=0.8,
        alpha=0.30,
    ))

    stride = max(1, len(panel.triangles) // 24_000)
    displayed_faces = np.arange(0, len(panel.triangles), stride)
    colors = np.tile(
        np.array([[0.58, 0.67, 0.75, 0.48]]),
        (len(displayed_faces), 1),
    )
    for index, face_id in enumerate(displayed_faces):
        if int(face_id) in panel.feature_faces:
            colors[index] = [0.12, 0.42, 0.86, 0.92]
    axis.add_collection3d(Poly3DCollection(
        panel.triangles[displayed_faces],
        facecolors=colors,
        edgecolors="none",
    ))

    if panel.reference_triangles is not None:
        reference_stride = max(1, len(panel.reference_triangles) // 20_000)
        axis.add_collection3d(Poly3DCollection(
            panel.reference_triangles[::reference_stride],
            facecolors=[0.02, 0.62, 0.72, 0.72],
            edgecolors="none",
        ))

    if panel.support_segments is not None and len(panel.support_segments):
        for bottom, top in panel.support_segments:
            axis.plot(
                [bottom[0], top[0]],
                [bottom[1], top[1]],
                [bottom[2], top[2]],
                color="#129044",
                linewidth=1.15,
                alpha=0.90,
            )

    arrow_base = np.array([
        low[0] - 0.055 * max_span,
        low[1] - 0.055 * max_span,
        panel.platform_z,
    ])
    arrow_length = max(max_span * 0.14, 4.0)
    axis.quiver(
        *arrow_base,
        0.0,
        0.0,
        arrow_length,
        color="#d54a3c",
        linewidth=2.0,
        arrow_length_ratio=0.17,
    )
    axis.text(
        *(arrow_base + np.array([0.0, 0.0, arrow_length * 1.10])),
        "+build Z",
        color="#ad3027",
        fontsize=9,
    )
    half_span = max_span * 0.58
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(
        panel.platform_z - 0.06 * max_span,
        high[2] + 0.07 * max_span,
    )
    axis.set_box_aspect((1.0, 1.0, max(float(span[2]) / max_span, 0.18)))
    axis.set_xlabel("Build X (mm)", fontsize=9)
    axis.set_ylabel("Build Y (mm)", fontsize=9)
    axis.set_zlabel("Build Z (mm)", fontsize=9)
    axis.view_init(elev=24, azim=-54)
    axis.grid(alpha=0.20)
    axis.set_title(f"{panel.title}\n{panel.subtitle}", fontsize=13, pad=15)


def save_single(path: Path, panel: Panel) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10, 8.2), facecolor="#f4f6f8")
    axis = figure.add_subplot(111, projection="3d")
    draw_panel(axis, panel)
    figure.text(
        0.5,
        0.025,
        "Blue: recognized internal holes | teal: stored Magics support | "
        "green: simplified support columns",
        ha="center",
        fontsize=9,
        color="#48525c",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_combined(path: Path, panels: list[Panel]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(22, 8.2), facecolor="#f4f6f8")
    for index, panel in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        draw_panel(axis, panel)
    figure.suptitle(
        "Part 10013: three-way support visualization",
        fontsize=17,
    )
    figure.text(
        0.5,
        0.025,
        "Blue: recognized internal holes | teal: stored Magics support | "
        "green: simplified support columns | all comparisons are shown in build view",
        ha="center",
        fontsize=10,
        color="#48525c",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part_mesh", type=Path)
    parser.add_argument("reference_support_mesh", type=Path)
    parser.add_argument("historical_support_report", type=Path)
    parser.add_argument("best_support_report", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    part_vertices, part_faces = read_mesh(args.part_mesh)
    reference_vertices, reference_faces = read_mesh(args.reference_support_mesh)
    historical = load_json(args.historical_support_report)
    best = load_json(args.best_support_report)
    features = load_json(args.features)["features"]
    feature_faces = {
        int(face_id)
        for feature in features
        for face_id in feature["face_indices"]
    }

    historical_rotation = rotation_align_to_z(
        np.asarray(historical["build_up_input"], dtype=np.float64)
    )
    best_rotation = rotation_align_to_z(
        np.asarray(best["build_up_input"], dtype=np.float64)
    )
    historical_triangles = (part_vertices @ historical_rotation.T)[part_faces]
    best_triangles = (part_vertices @ best_rotation.T)[part_faces]
    reference_triangles = (
        reference_vertices @ historical_rotation.T
    )[reference_faces]

    historical_platform = float(historical["platform_height_mm"])
    historical_triangles_shifted = shifted_triangles(
        historical_triangles, -historical_platform
    )
    best_platform = float(best["platform_height_mm"])
    best_triangles_shifted = shifted_triangles(best_triangles, -best_platform)
    reference_triangles_shifted = shifted_triangles(
        reference_triangles, -historical_platform
    )
    historical_segments = shifted_segments(
        support_segments(historical), -historical_platform
    )
    best_segments = shifted_segments(support_segments(best), -best_platform)

    historical_extent_points = np.concatenate([
        reference_triangles_shifted.reshape(-1, 3),
        historical_triangles_shifted.reshape(-1, 3),
    ], axis=0)
    historical_extent = (
        historical_extent_points.min(axis=0),
        historical_extent_points.max(axis=0),
    )
    panels = [
        Panel(
            title="1. Original dataset pose + actual Magics support",
            subtitle=(
                f"Stored placement, build-up +Z | "
                f"{len(reference_faces):,} support triangles"
            ),
            triangles=historical_triangles_shifted,
            feature_faces=feature_faces,
            platform_z=0.0,
            reference_triangles=reference_triangles_shifted,
            extent=historical_extent,
        ),
        Panel(
            title="2. Same original pose + our support algorithm",
            subtitle=(
                f"Stored placement unchanged, platform Z=0 | "
                f"{historical['support_summary']['count']} columns"
            ),
            triangles=historical_triangles_shifted,
            feature_faces=feature_faces,
            platform_z=0.0,
            support_segments=historical_segments,
            extent=historical_extent,
        ),
        Panel(
            title="3. Best searched pose + our support algorithm",
            subtitle=(
                f"S047, 51.10 deg from original | "
                f"{best['support_summary']['count']} columns"
            ),
            triangles=best_triangles_shifted,
            feature_faces=feature_faces,
            platform_z=0.0,
            support_segments=best_segments,
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filenames = [
        "01_original_pose_magics_support.png",
        "02_original_pose_our_support.png",
        "03_best_pose_our_support.png",
    ]
    for filename, panel in zip(filenames, panels):
        save_single(args.output_dir / filename, panel)
    combined = args.output_dir / "00_three_panel_comparison.png"
    save_combined(combined, panels)
    print(json.dumps({
        "combined": str(combined.resolve()),
        "individual_images": [
            str((args.output_dir / filename).resolve())
            for filename in filenames
        ],
        "historical_columns": historical["support_summary"]["count"],
        "best_columns": best["support_summary"]["count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
