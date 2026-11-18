#!/usr/bin/env python3
"""Export a part and an existing support mesh with a matching build plate."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

try:
    from tools.export_support_scene import (
        platform_box_aligned,
        triangulate_quads,
        write_binary_stl,
        write_scene_obj,
    )
    from tools.generate_simplified_supports import read_mesh
except ModuleNotFoundError:
    from export_support_scene import (
        platform_box_aligned,
        triangulate_quads,
        write_binary_stl,
        write_scene_obj,
    )
    from generate_simplified_supports import read_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part_mesh", type=Path)
    parser.add_argument("support_mesh", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--build-up",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="build-up direction in the input mesh frame; default +Z",
    )
    parser.add_argument("--platform-top-mm", type=float, default=0.0)
    parser.add_argument("--platform-margin", type=float, default=0.08)
    parser.add_argument("--platform-thickness-mm", type=float, default=2.0)
    args = parser.parse_args()
    if args.platform_margin < 0 or args.platform_thickness_mm <= 0:
        parser.error("platform margin must be nonnegative and thickness must be positive")

    part_vertices, part_faces = read_mesh(args.part_mesh)
    support_vertices, support_faces = read_mesh(args.support_mesh)
    support_triangles = support_vertices[support_faces]
    up = np.asarray(args.build_up, dtype=np.float64)
    up /= np.linalg.norm(up)
    platform_vertices, platform_faces = platform_box_aligned(
        part_vertices,
        up,
        args.platform_top_mm,
        args.platform_margin,
        args.platform_thickness_mm,
    )
    platform_triangles = triangulate_quads(platform_vertices, platform_faces)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_stl = args.output_dir / "part_original.stl"
    support_stl = args.output_dir / "reference_support.stl"
    plate_stl = args.output_dir / "build_plate.stl"
    combined_stl = args.output_dir / "part_with_reference_support_and_platform.stl"
    scene_obj = args.output_dir / "part_with_reference_support_and_platform.obj"
    scene_mtl = args.output_dir / "part_with_reference_support_and_platform.mtl"

    shutil.copy2(args.part_mesh, original_stl)
    shutil.copy2(args.support_mesh, support_stl)
    write_binary_stl(
        plate_stl,
        platform_triangles,
        "Build plate for the stored reference support; top at configured build-Z",
    )
    write_binary_stl(
        combined_stl,
        np.concatenate([
            part_vertices[part_faces],
            support_triangles,
            platform_triangles,
        ], axis=0),
        "Part, stored reference support, and build plate; mm assumed",
    )
    write_scene_obj(
        scene_obj,
        scene_mtl,
        part_vertices,
        part_faces,
        platform_vertices,
        platform_faces,
        support_triangles,
        support_object_name="ReferenceSupport",
    )

    manifest = {
        "source_part_mesh": str(args.part_mesh.resolve()),
        "source_support_mesh": str(args.support_mesh.resolve()),
        "units_assumed": "mm; source PJF units remain unconfirmed",
        "build_up_input": up.tolist(),
        "platform_top_coordinate_along_build_axis_mm": args.platform_top_mm,
        "platform_thickness_mm": args.platform_thickness_mm,
        "platform_margin_fraction": args.platform_margin,
        "files": {
            "original_stl": original_stl.name,
            "reference_support_stl": support_stl.name,
            "build_plate_stl": plate_stl.name,
            "combined_stl": combined_stl.name,
            "editable_scene_obj": scene_obj.name,
            "editable_scene_mtl": scene_mtl.name,
        },
    }
    manifest_path = args.output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path.resolve()),
        **manifest["files"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
