#!/usr/bin/env python3
"""Export a part, its simplified supports, and a matching build plate scene."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np

from generate_simplified_supports import read_mesh, rotation_align_to_z


def platform_box_aligned(vertices: np.ndarray, up: np.ndarray, top_z: float, margin: float, thickness: float) -> tuple[np.ndarray, np.ndarray]:
    rotation = rotation_align_to_z(up)
    aligned = vertices @ rotation.T
    low, high = aligned.min(axis=0), aligned.max(axis=0)
    pad = max(float((high - low).max()) * margin, 1.0)
    x0, x1 = low[0] - pad, high[0] + pad
    y0, y1 = low[1] - pad, high[1] + pad
    z0, z1 = top_z - thickness, top_z
    box = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    # Quads, outward winding, in bottom/top/front/back/left/right order.
    faces = np.array([
        [0, 3, 2, 1], [4, 5, 6, 7],
        [0, 1, 5, 4], [1, 2, 6, 5],
        [2, 3, 7, 6], [3, 0, 4, 7],
    ], dtype=np.int64)
    return box @ rotation, faces


def support_triangles(supports: list[dict[str, Any]], radius: float, sides: int = 8) -> np.ndarray:
    triangles: list[np.ndarray] = []
    angles = np.arange(sides, dtype=np.float64) * (2.0 * np.pi / sides)
    for support in supports:
        top = np.asarray(support["source_point_input_mm"], dtype=np.float64)
        bottom = np.asarray(support["landing_point_input_mm"], dtype=np.float64)
        axis = top - bottom
        length = np.linalg.norm(axis)
        if length < 1e-8:
            continue
        axis /= length
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ring = radius * (np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v[None, :])
        lower = bottom + ring
        upper = top + ring
        for i in range(sides):
            j = (i + 1) % sides
            triangles.append(np.array([lower[i], lower[j], upper[j]]))
            triangles.append(np.array([lower[i], upper[j], upper[i]]))
            triangles.append(np.array([bottom, lower[j], lower[i]]))
            triangles.append(np.array([top, upper[i], upper[j]]))
    return np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)


def triangulate_quads(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray([[vertices[face[0]], vertices[face[i]], vertices[face[i + 1]]]
                       for face in faces for i in range(1, len(face) - 1)], dtype=np.float64)


def write_scene_obj(
    path: Path,
    material_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    platform_vertices: np.ndarray,
    platform_faces: np.ndarray,
    support_mesh: np.ndarray | None,
    support_object_name: str = "SimplifiedSupportColumns",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    material_path.write_text(
        "newmtl part_material\nKd 0.64 0.73 0.82\nKa 0.10 0.12 0.14\nillum 2\n\n"
        "newmtl support_material\nKd 0.18 0.58 0.32\nKa 0.05 0.12 0.06\nillum 2\n\n"
        "newmtl platform_material\nKd 0.56 0.60 0.65\nKa 0.12 0.13 0.15\nillum 2\n",
        encoding="utf-8",
    )
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"mtllib {material_path.name}\n")
        next_index = 1
        stream.write("o Part\ng part\nusemtl part_material\n")
        for point in vertices:
            stream.write(f"v {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
        for face in faces:
            stream.write("f " + " ".join(str(int(i) + next_index) for i in face) + "\n")
        next_index += len(vertices)
        if support_mesh is not None and len(support_mesh):
            stream.write(f"\no {support_object_name}\ng supports\nusemtl support_material\n")
            for triangle in support_mesh:
                for point in triangle:
                    stream.write(f"v {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
            for face_id in range(len(support_mesh)):
                base = next_index + face_id * 3
                stream.write(f"f {base} {base + 1} {base + 2}\n")
            next_index += len(support_mesh) * 3
        stream.write("\no BuildPlate\ng platform\nusemtl platform_material\n")
        for point in platform_vertices:
            stream.write(f"v {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
        for face in platform_faces:
            stream.write("f " + " ".join(str(int(i) + next_index) for i in face) + "\n")


def write_binary_stl(path: Path, triangles: np.ndarray, header: str) -> None:
    records = np.zeros(len(triangles), dtype=np.dtype([
        ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")
    ]))
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    np.divide(cross, norm[:, None], out=records["normal"], where=norm[:, None] > 1e-12)
    records["vertices"] = triangles
    with path.open("wb") as stream:
        stream.write(header.encode("ascii", errors="replace")[:80].ljust(80, b" "))
        stream.write(struct.pack("<I", len(triangles)))
        stream.write(records.tobytes())


def render_pair(path: Path, part_triangles: np.ndarray, supports: list[dict[str, Any]],
                platform_vertices: np.ndarray, platform_faces: np.ndarray, up: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rotation = rotation_align_to_z(up)
    part_aligned = part_triangles @ rotation.T
    support_rows = supports
    platform_aligned = platform_vertices @ rotation.T
    platform_polygons = [platform_aligned[face] for face in platform_faces]
    low = part_aligned.reshape(-1, 3).min(axis=0)
    high = part_aligned.reshape(-1, 3).max(axis=0)
    span = np.maximum(high - low, 1e-3)
    center = (low + high) / 2

    fig = plt.figure(figsize=(13, 6.4), facecolor="#f4f6f8")
    # Use ASCII labels so the preview remains readable on systems without CJK fonts.
    titles = ["Original part + build plate", "Part + simplified supports + plate"]
    for panel, title in enumerate(titles):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(platform_polygons, facecolors="#8d969f", edgecolors="#58616a", alpha=0.65))
        ax.add_collection3d(Poly3DCollection(part_aligned, facecolors="#7298b5", edgecolors="none", alpha=0.82))
        if panel == 1:
            stride = max(1, len(support_rows) // 700)
            for row in support_rows[::stride]:
                bottom = np.asarray(row["landing_point_input_mm"]) @ rotation.T
                top = np.asarray(row["source_point_input_mm"]) @ rotation.T
                ax.plot([bottom[0], top[0]], [bottom[1], top[1]], [bottom[2], top[2]],
                        color="#21884b", linewidth=0.9, alpha=0.85)
        arrow_base = np.array([low[0] - 0.03 * span.max(), low[1] - 0.03 * span.max(), 0.0])
        arrow_len = max(float(span.max()) * 0.14, 3.0)
        ax.quiver(*arrow_base, 0.0, 0.0, arrow_len, color="#d1473d", linewidth=1.8, arrow_length_ratio=0.16)
        ax.text(*(arrow_base + [0.0, 0.0, arrow_len * 1.12]), "+build Z", color="#9f2d26", fontsize=8)
        ax.set_xlim(center[0] - span.max() * 0.58, center[0] + span.max() * 0.58)
        ax.set_ylim(center[1] - span.max() * 0.58, center[1] + span.max() * 0.58)
        ax.set_zlim(-0.06 * span.max(), high[2] + 0.05 * span.max())
        ax.set_box_aspect((1, 1, max(span[2] / span.max(), 0.15)))
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Build Z (mm)")
        ax.set_title(title, fontsize=12)
        ax.view_init(elev=24, azim=-54)
    fig.suptitle("Same part and plate; green lines show simplified support columns", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="part STL/OBJ/NPZ used by the support calculation")
    parser.add_argument("report", type=Path, help="support JSON emitted by generate_simplified_supports.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform-margin", type=float, default=0.08,
                        help="plate margin as a fraction of the maximum part span")
    parser.add_argument("--platform-thickness-mm", type=float, default=2.0)
    parser.add_argument(
        "--align-to-build-axis",
        action="store_true",
        help="write the part, supports, and plate in a frame whose Z axis is the build direction",
    )
    args = parser.parse_args()
    if args.platform_margin < 0 or args.platform_thickness_mm <= 0:
        parser.error("platform margin must be nonnegative and thickness must be positive")

    vertices, faces = read_mesh(args.mesh)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    up = np.asarray(report["build_up_input"], dtype=np.float64)
    if args.align_to_build_axis:
        rotation = rotation_align_to_z(up)
        output_vertices = vertices @ rotation.T
        output_up = np.array([0.0, 0.0, 1.0])
        supports = [
            {
                **item,
                "source_point_input_mm": item["source_point_aligned_mm"],
                "landing_point_input_mm": item["landing_point_aligned_mm"],
            }
            for item in report["supports"]
        ]
    else:
        output_vertices = vertices
        output_up = up
        supports = report["supports"]
    platform_top = float(report["platform_height_mm"])
    platform_vertices, platform_faces = platform_box_aligned(
        output_vertices,
        output_up,
        platform_top,
        args.platform_margin,
        args.platform_thickness_mm,
    )
    radius = float(report["support_summary"]["column_radius_assumption_mm"])
    support_mesh = support_triangles(supports, radius)
    part_triangles = output_vertices[faces]
    platform_triangles = triangulate_quads(platform_vertices, platform_faces)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_stl = args.output_dir / "part_original.stl"
    if args.align_to_build_axis:
        write_binary_stl(
            raw_stl,
            part_triangles,
            "Part aligned so +Z is the selected build direction; mm assumed",
        )
    else:
        shutil.copy2(args.mesh, raw_stl)
    original_obj = args.output_dir / "part_original_with_platform.obj"
    original_mtl = args.output_dir / "part_original_with_platform.mtl"
    supported_obj = args.output_dir / "part_with_simplified_supports_and_platform.obj"
    supported_mtl = args.output_dir / "part_with_simplified_supports_and_platform.mtl"
    supported_only_stl = args.output_dir / "part_with_simplified_supports.stl"
    platform_stl = args.output_dir / "build_plate.stl"
    combined_stl = args.output_dir / "part_with_simplified_supports_and_platform.stl"
    preview = args.output_dir / "original_vs_supported.png"

    write_scene_obj(
        original_obj,
        original_mtl,
        output_vertices,
        faces,
        platform_vertices,
        platform_faces,
        None,
    )
    write_scene_obj(
        supported_obj,
        supported_mtl,
        output_vertices,
        faces,
        platform_vertices,
        platform_faces,
        support_mesh,
    )
    write_binary_stl(supported_only_stl, np.concatenate([part_triangles, support_mesh], axis=0),
                     "Part and simplified support proxies; mm assumed; no build plate")
    write_binary_stl(platform_stl, platform_triangles, "Build plate; top plane at build Z=0; mm assumed")
    combined_triangles = np.concatenate([part_triangles, support_mesh, platform_triangles], axis=0)
    write_binary_stl(combined_stl, combined_triangles, "Part, simplified support proxy, and build plate; mm assumed")
    render_pair(
        preview,
        part_triangles,
        supports,
        platform_vertices,
        platform_faces,
        output_up,
    )

    artifacts = {
        "source_mesh": str(args.mesh.resolve()),
        "support_report": str(args.report.resolve()),
        "units_assumed": "mm; source PJF units remain unconfirmed",
        "output_coordinate_frame": (
            "aligned_to_build_axis"
            if args.align_to_build_axis
            else "source_input_frame"
        ),
        "platform": {
            "top_plane_coordinate_along_build_axis": platform_top,
            "thickness_mm": args.platform_thickness_mm,
            "margin_fraction": args.platform_margin,
            "dimensions_xyz_input_frame": (platform_vertices.max(axis=0) - platform_vertices.min(axis=0)).tolist(),
        },
        "simple_support_column_count": len(supports),
        "support_radius_mm": radius,
        "files": {
            "original_stl": raw_stl.name,
            "original_with_platform_obj": original_obj.name,
            "supported_part_stl_without_platform": supported_only_stl.name,
            "build_plate_stl": platform_stl.name,
            "supported_with_platform_obj": supported_obj.name,
            "supported_with_platform_stl": combined_stl.name,
            "comparison_preview": preview.name,
        },
        "note": "STL/OBJ supported exports contain geometric vertical-cylinder proxies from the simplified algorithm; they are not Materialise Magics support output or validated production supports.",
    }
    manifest = args.output_dir / "artifact_manifest.json"
    manifest.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
