#!/usr/bin/env python3
"""Render and summarize a stored Magics pose against a simplified support search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.generate_simplified_supports import read_mesh, rotation_align_to_z
except ModuleNotFoundError:
    from generate_simplified_supports import read_mesh, rotation_align_to_z


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_pose(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "build_up_input": report["build_up_input"],
        "support_column_count": report["support_summary"]["count"],
        "support_volume_proxy_mm3": report["support_summary"]["column_volume_proxy_mm3"],
        "build_plate_landings": report["support_summary"]["build_plate_landings"],
        "part_surface_landings": report["support_summary"]["part_surface_landings"],
        "mean_support_height_mm": report["support_summary"]["mean_height_mm"],
        "max_support_height_mm": report["support_summary"]["max_height_mm"],
        "candidate_overhang_area_mm2": report["areas_mm2"]["support_candidate_after_exemptions"],
        "platform_height_mm": report["platform_height_mm"],
        "minimum_part_height_along_build_axis_mm": report["minimum_part_height_along_build_axis_mm"],
        "hard_constraint_violation_count": len(report["hard_constraint_violations"]),
    }


def points_inside_cylinder(
    points: np.ndarray,
    feature: dict[str, Any],
    tolerance_mm: float = 1e-6,
) -> np.ndarray:
    axis = np.asarray(feature["axis_input"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    start = np.asarray(feature["axis_start_input_mm"], dtype=np.float64)
    relative = np.asarray(points, dtype=np.float64) - start
    axial = relative @ axis
    radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
    return (
        (axial >= -tolerance_mm)
        & (axial <= float(feature["axis_length_mm"]) + tolerance_mm)
        & (radial <= float(feature["radius_mm"]) + tolerance_mm)
    )


def support_mesh_statistics(
    support_path: Path,
    features: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise RuntimeError("trimesh is required for reference-support statistics") from exc

    raw = trimesh.load(support_path, force="mesh", process=False)
    processed = trimesh.load(support_path, force="mesh", process=True)
    components = processed.split(only_watertight=False)
    feature_intersections = []
    for feature in features:
        vertex_hits = points_inside_cylinder(raw.vertices, feature)
        centroid_hits = points_inside_cylinder(raw.triangles_center, feature)
        feature_intersections.append({
            "feature_id": feature["feature_id"],
            "diameter_mm": float(feature["diameter_mm"]),
            "axis_length_mm": float(feature["axis_length_mm"]),
            "support_vertices_inside_cylinder": int(vertex_hits.sum()),
            "support_triangle_centroids_inside_cylinder": int(centroid_hits.sum()),
        })
    return {
        "mesh": str(support_path.resolve()),
        "raw_vertex_count": int(len(raw.vertices)),
        "triangle_count": int(len(raw.faces)),
        "triangle_area_mm2": float(raw.area),
        "watertight": bool(processed.is_watertight),
        "winding_consistent": bool(processed.is_winding_consistent),
        "closed_volume_mm3": float(processed.volume) if processed.is_watertight else None,
        "connected_components": int(len(components)),
        "bounds_mm": np.asarray(raw.bounds, dtype=np.float64).tolist(),
        "recognized_hole_intersections": feature_intersections,
    }


def support_segments(report: dict[str, Any]) -> np.ndarray:
    segments = [
        [
            item["landing_point_aligned_mm"],
            item["source_point_aligned_mm"],
        ]
        for item in report["supports"]
    ]
    return np.asarray(segments, dtype=np.float64)


def render_comparison(
    path: Path,
    part_vertices: np.ndarray,
    part_faces: np.ndarray,
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    historical_report: dict[str, Any],
    best_report: dict[str, Any],
    feature_face_indices: set[int],
    metrics: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    historical_up = np.asarray(historical_report["build_up_input"], dtype=np.float64)
    best_up = np.asarray(best_report["build_up_input"], dtype=np.float64)
    historical_rotation = rotation_align_to_z(historical_up)
    best_rotation = rotation_align_to_z(best_up)
    historical_vertices = part_vertices @ historical_rotation.T
    best_vertices = part_vertices @ best_rotation.T
    historical_triangles = historical_vertices[part_faces]
    best_triangles = best_vertices[part_faces]
    reference_triangles = (reference_vertices @ historical_rotation.T)[reference_faces]

    figure = plt.figure(figsize=(18, 7.2), facecolor="#f4f6f8")
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    panels = [
        {
            "triangles": historical_triangles,
            "segments": None,
            "reference": reference_triangles,
            "platform_z": 0.0,
            "title": (
                "Historical Magics pose\n"
                f"{metrics['reference_support']['triangle_count']:,} support triangles"
            ),
        },
        {
            "triangles": historical_triangles,
            "segments": support_segments(historical_report),
            "reference": None,
            "platform_z": float(historical_report["platform_height_mm"]),
            "title": (
                "Historical +Z / simplified estimate\n"
                f"{metrics['historical_pose']['support_column_count']} columns, "
                f"{metrics['historical_pose']['support_volume_proxy_mm3']:.0f} mm3 proxy"
            ),
        },
        {
            "triangles": best_triangles,
            "segments": support_segments(best_report),
            "reference": None,
            "platform_z": float(best_report["platform_height_mm"]),
            "title": (
                f"Searched best {metrics['best_pose']['direction_label']} / simplified estimate\n"
                f"{metrics['best_pose']['support_column_count']} columns, "
                f"{metrics['best_pose']['support_volume_proxy_mm3']:.0f} mm3 proxy"
            ),
        },
    ]

    for axis, panel in zip(axes, panels):
        triangles = panel["triangles"]
        extent_vertices = triangles.reshape(-1, 3)
        if panel["reference"] is not None:
            extent_vertices = np.concatenate(
                [extent_vertices, np.asarray(panel["reference"]).reshape(-1, 3)], axis=0
            )
        low = extent_vertices.min(axis=0)
        high = extent_vertices.max(axis=0)
        span = np.maximum(high - low, 1.0)
        center = (low + high) / 2.0
        padding = max(float(span.max()) * 0.07, 2.0)
        platform_z = float(panel["platform_z"])
        platform = np.array([
            [low[0] - padding, low[1] - padding, platform_z],
            [high[0] + padding, low[1] - padding, platform_z],
            [high[0] + padding, high[1] + padding, platform_z],
            [low[0] - padding, high[1] + padding, platform_z],
        ])
        axis.add_collection3d(Poly3DCollection(
            [platform], facecolors="#98a2ad", edgecolors="#58636e", alpha=0.34
        ))

        stride = max(1, len(triangles) // 36_000)
        displayed_faces = np.arange(0, len(triangles), stride)
        colors = np.tile(np.array([[0.62, 0.71, 0.79, 0.60]]), (len(displayed_faces), 1))
        feature_mask = np.array(
            [int(face_id) in feature_face_indices for face_id in displayed_faces],
            dtype=bool,
        )
        colors[feature_mask] = [0.17, 0.48, 0.82, 0.90]
        axis.add_collection3d(Poly3DCollection(
            triangles[displayed_faces], facecolors=colors, edgecolors="none"
        ))

        if panel["reference"] is not None:
            reference = np.asarray(panel["reference"])
            reference_stride = max(1, len(reference) // 20_000)
            axis.add_collection3d(Poly3DCollection(
                reference[::reference_stride],
                facecolors=[0.03, 0.55, 0.62, 0.56],
                edgecolors="none",
            ))

        if panel["segments"] is not None and len(panel["segments"]):
            segments = np.asarray(panel["segments"])
            segment_stride = max(1, len(segments) // 500)
            for bottom, top in segments[::segment_stride]:
                axis.plot(
                    [bottom[0], top[0]],
                    [bottom[1], top[1]],
                    [bottom[2], top[2]],
                    color="#148242",
                    linewidth=1.0,
                    alpha=0.88,
                )

        arrow_base = np.array([
            low[0] - 0.045 * span.max(),
            low[1] - 0.045 * span.max(),
            platform_z,
        ])
        arrow_length = max(float(span.max()) * 0.14, 4.0)
        axis.quiver(
            *arrow_base,
            0.0,
            0.0,
            arrow_length,
            color="#d04b40",
            linewidth=2.0,
            arrow_length_ratio=0.17,
        )
        axis.text(
            *(arrow_base + np.array([0.0, 0.0, arrow_length * 1.12])),
            "+build Z",
            color="#9f2d26",
            fontsize=8,
        )
        half_span = float(span.max()) * 0.56
        axis.set_xlim(center[0] - half_span, center[0] + half_span)
        axis.set_ylim(center[1] - half_span, center[1] + half_span)
        axis.set_zlim(platform_z - 0.06 * span.max(), high[2] + 0.06 * span.max())
        axis.set_box_aspect((1.0, 1.0, max(float(span[2]) / float(span.max()), 0.18)))
        axis.set_xlabel("Build X (mm)", fontsize=8)
        axis.set_ylabel("Build Y (mm)", fontsize=8)
        axis.set_zlabel("Build Z (mm)", fontsize=8)
        axis.set_title(panel["title"], fontsize=11)
        axis.view_init(elev=24, azim=-54)
        axis.grid(alpha=0.20)

    figure.suptitle(
        "Part 10013: stored Magics pose and actual support vs simplified support search",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.025,
        "Blue: recognized internal-cylinder faces. Green: simplified vertical-column proxies. "
        "The Magics mesh is shown as saved; it is not a simplified support.",
        ha="center",
        fontsize=9,
        color="#48525c",
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def render_orientation_search(
    path: Path,
    search: dict[str, Any],
    all_holes_search: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = search["ranking"]
    directions = np.asarray([row["build_up_input"] for row in rows], dtype=np.float64)
    volumes = np.asarray([row["support_volume_mm3"] for row in rows], dtype=np.float64)
    current = np.asarray(search["current_direction"]["build_up_input"], dtype=np.float64)
    best = np.asarray(search["best_direction"]["build_up_input"], dtype=np.float64)

    figure = plt.figure(figsize=(14, 6.3), facecolor="#f4f6f8")
    sphere_axis = figure.add_subplot(1, 2, 1, projection="3d")
    scatter = sphere_axis.scatter(
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        c=volumes,
        cmap="viridis_r",
        s=32,
        alpha=0.88,
        depthshade=False,
    )
    sphere_axis.scatter(
        [current[0]],
        [current[1]],
        [current[2]],
        marker="X",
        s=150,
        c="#e45a45",
        edgecolors="white",
        linewidths=1.2,
        depthshade=False,
    )
    sphere_axis.scatter(
        [best[0]],
        [best[1]],
        [best[2]],
        marker="*",
        s=220,
        c="#1f9a58",
        edgecolors="white",
        linewidths=1.0,
        depthshade=False,
    )
    sphere_axis.text(
        current[0] * 1.08,
        current[1] * 1.08,
        current[2] * 1.08,
        " Historical +Z",
        color="#b73c2b",
        fontsize=9,
    )
    sphere_axis.text(
        best[0] * 1.08,
        best[1] * 1.08,
        best[2] * 1.08,
        f" Best {search['best_direction']['direction_label']}",
        color="#147a41",
        fontsize=9,
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 160)
    for z in (-0.75, 0.0, 0.75):
        radius = np.sqrt(1.0 - z * z)
        sphere_axis.plot(
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full_like(angles, z),
            color="#71808d",
            linewidth=0.55,
            alpha=0.30,
        )
    for azimuth in (0.0, np.pi / 2.0):
        sphere_axis.plot(
            np.cos(angles) * np.sin(np.linspace(0.0, np.pi, 160)),
            np.sin(angles) * np.sin(np.linspace(0.0, np.pi, 160)),
            np.cos(np.linspace(0.0, np.pi, 160)),
            color="#71808d",
            linewidth=0.55,
            alpha=0.24,
        )
    sphere_axis.set_xlim(-1.05, 1.05)
    sphere_axis.set_ylim(-1.05, 1.05)
    sphere_axis.set_zlim(-1.05, 1.05)
    sphere_axis.set_box_aspect((1, 1, 1))
    sphere_axis.set_xlabel("Build-up X")
    sphere_axis.set_ylabel("Build-up Y")
    sphere_axis.set_zlabel("Build-up Z")
    sphere_axis.view_init(elev=20, azim=-56)
    sphere_axis.set_title(
        f"264 sampled directions; {search['feasible_count']} feasible\n"
        "colour = simplified support-volume proxy",
        fontsize=11,
    )
    colorbar = figure.colorbar(scatter, ax=sphere_axis, shrink=0.62, pad=0.08)
    colorbar.set_label("Support-volume proxy (mm3)")

    distribution_axis = figure.add_subplot(1, 2, 2)
    distribution_axis.hist(
        volumes,
        bins=28,
        color="#7aa7c7",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.90,
    )
    current_volume = float(search["current_direction"]["support_volume_mm3"])
    best_volume = float(search["best_direction"]["support_volume_mm3"])
    distribution_axis.axvline(
        best_volume,
        color="#1f9a58",
        linewidth=2.0,
        label=(
            f"Best {search['best_direction']['direction_label']} "
            f"({best_volume:.0f} mm3)"
        ),
    )
    distribution_axis.axvline(
        current_volume,
        color="#e45a45",
        linewidth=2.0,
        label=f"Historical +Z ({current_volume:.0f} mm3)",
    )
    distribution_axis.set_xlabel("Simplified support-volume proxy (mm3)")
    distribution_axis.set_ylabel("Number of directions")
    distribution_axis.set_title("Support-volume distribution", fontsize=11)
    distribution_axis.grid(axis="y", alpha=0.20)
    distribution_axis.legend(loc="upper right", fontsize=9)
    current_rank = int(search["current_direction"]["rank"])
    distribution_axis.text(
        0.02,
        0.97,
        (
            f"Historical rank: {current_rank}/{search['candidate_count']}\n"
            f"All 6 holes forbidden: {all_holes_search['feasible_count']}/"
            f"{all_holes_search['candidate_count']} feasible\n"
            f"Nearest all-hole case: {all_holes_search['best_direction']['violation_count']} "
            "overhang violations"
        ),
        transform=distribution_axis.transAxes,
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#ffffff",
            "edgecolor": "#b9c3cc",
            "alpha": 0.94,
        },
    )
    figure.suptitle(
        "Part 10013: query-free build-direction search on the orientation sphere",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part_mesh", type=Path)
    parser.add_argument("reference_support_mesh", type=Path)
    parser.add_argument("historical_support_report", type=Path)
    parser.add_argument("best_support_report", type=Path)
    parser.add_argument("orientation_search", type=Path)
    parser.add_argument("all_holes_orientation_search", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    part_vertices, part_faces = read_mesh(args.part_mesh)
    reference_vertices, reference_faces = read_mesh(args.reference_support_mesh)
    historical_report = load_json(args.historical_support_report)
    best_report = load_json(args.best_support_report)
    search = load_json(args.orientation_search)
    all_holes_search = load_json(args.all_holes_orientation_search)
    feature_data = load_json(args.features)
    features = feature_data["features"]

    historical = compact_pose(historical_report)
    best = compact_pose(best_report)
    best_search_row = search["best_direction"]
    best["direction_label"] = best_search_row["direction_label"]
    best["rank"] = best_search_row["rank"]
    current_search_row = search["current_direction"]
    historical["direction_label"] = current_search_row["direction_label"]
    historical["rank"] = current_search_row["rank"]
    angle = float(np.degrees(np.arccos(np.clip(
        np.dot(
            np.asarray(historical["build_up_input"]),
            np.asarray(best["build_up_input"]),
        ),
        -1.0,
        1.0,
    ))))
    reference_statistics = support_mesh_statistics(
        args.reference_support_mesh,
        features,
    )
    all_hole_best = all_holes_search["best_direction"]
    metrics = {
        "part_mesh": str(args.part_mesh.resolve()),
        "reference_support": reference_statistics,
        "recognized_features": {
            "count": len(features),
            "horizontal_to_historical_build_axis": sum(
                bool(feature["is_horizontal_to_build"]) for feature in features
            ),
            "small_horizontal_exemptions": sum(
                bool(feature["small_horizontal_exempt"]) for feature in features
            ),
            "items": [
                {
                    "feature_id": feature["feature_id"],
                    "diameter_mm": float(feature["diameter_mm"]),
                    "axis_length_mm": float(feature["axis_length_mm"]),
                    "confidence": float(feature["confidence"]),
                }
                for feature in features
            ],
        },
        "historical_pose": historical,
        "best_pose": best,
        "best_to_historical_angle_deg": angle,
        "support_search": {
            "candidate_count": int(search["candidate_count"]),
            "feasible_count": int(search["feasible_count"]),
        },
        "all_holes_forbidden": {
            "candidate_count": int(all_holes_search["candidate_count"]),
            "feasible_count": int(all_holes_search["feasible_count"]),
            "best_near_feasible_direction_label": all_hole_best["direction_label"],
            "best_near_feasible_build_up_input": all_hole_best["build_up_input"],
            "best_near_feasible_violation_count": all_hole_best["violation_count"],
            "best_near_feasible_violation_types": all_hole_best["violation_types"],
            "historical_pose_violation_count": all_holes_search["current_direction"]["violation_count"],
            "historical_pose_violation_types": all_holes_search["current_direction"]["violation_types"],
        },
        "reductions_from_historical_to_best": {
            "support_volume_proxy_fraction": (
                1.0
                - best["support_volume_proxy_mm3"]
                / historical["support_volume_proxy_mm3"]
            ),
            "support_column_fraction": (
                1.0
                - best["support_column_count"]
                / historical["support_column_count"]
            ),
            "candidate_overhang_area_fraction": (
                1.0
                - best["candidate_overhang_area_mm2"]
                / historical["candidate_overhang_area_mm2"]
            ),
        },
        "limitations": [
            "The searched supports are 0.4 mm-radius vertical-column proxies, not Magics SG+ or production supports.",
            "The stored Magics support mesh is open and disconnected, so its closed volume is unavailable.",
            "Geometric points inside a recognized hole do not establish whether support in that hole is prohibited.",
            "The all-hole-forbidden case is an explicit user/proxy constraint and is not inferred from feature recognition.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "comparison_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison_path = args.output_dir / "part_10013_comparison.png"
    render_comparison(
        comparison_path,
        part_vertices,
        part_faces,
        reference_vertices,
        reference_faces,
        historical_report,
        best_report,
        {
            int(face_id)
            for feature in features
            for face_id in feature["face_indices"]
        },
        metrics,
    )
    orientation_search_path = args.output_dir / "part_10013_orientation_sphere.png"
    render_orientation_search(
        orientation_search_path,
        search,
        all_holes_search,
    )
    print(json.dumps({
        "comparison": str(comparison_path.resolve()),
        "orientation_search": str(orientation_search_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "historical_pose": historical,
        "best_pose": best,
        "best_to_historical_angle_deg": angle,
        "all_holes_forbidden": metrics["all_holes_forbidden"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
