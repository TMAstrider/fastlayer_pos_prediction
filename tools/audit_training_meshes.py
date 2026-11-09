import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import trimesh

from audit_pjf import decode_mesh


def main():
    parser = argparse.ArgumentParser(description="Audit closed surfaces and sampled normals without repairing source meshes")
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--pointclouds", type=Path, default=Path("outputs/pointclouds"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training_mesh_audit.json"))
    arguments = parser.parse_args()
    manifest = json.loads((arguments.pointclouds / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for index, record in enumerate(manifest["records"]):
        project = arguments.dataset / record["source_project"]
        with sqlite3.connect(project.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            blob, = connection.execute(
                "SELECT data FROM MeshV220218 WHERE id = ?", (record["source_mesh_id"],)
            ).fetchone()
        vertices, faces, _ = decode_mesh(blob)
        with np.load(arguments.pointclouds / record["pointcloud_file"]) as cloud:
            triangles = vertices[faces[cloud["sampled_face_ids"]]].astype(np.float64)
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            expected = cross / np.linalg.norm(cross, axis=1, keepdims=True)
            normal_error = float(np.abs(expected - cloud["normals"]).max())
        unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        welded_faces = inverse[faces]
        mesh = trimesh.Trimesh(unique_vertices, welded_faces, process=False)
        records.append({
            "source_project": record["source_project"],
            "source_part_id": record["source_part_id"],
            "source_part_name": record["source_part_name"],
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "signed_volume_raw_units3": float(mesh.volume),
            "degenerate_face_count": int((~mesh.nondegenerate_faces()).sum()),
            "sampled_normal_max_component_error": normal_error,
        })
        if (index + 1) % 25 == 0 or index + 1 == len(manifest["records"]):
            print(f"Audited {index + 1}/{len(manifest['records'])} meshes", flush=True)
    report = {
        "method": "Read original mesh triangles; weld exactly identical vertex coordinates for topology checks. No source files or training inputs are modified.",
        "limitations": "Watertightness and consistent winding do not independently establish physical outward direction for all components or cavities. No self-intersection test is performed.",
        "summary": {
            "parts": len(records),
            "watertight": sum(record["watertight"] for record in records),
            "winding_consistent": sum(record["winding_consistent"] for record in records),
            "negative_signed_volume": sum(record["signed_volume_raw_units3"] < 0 for record in records),
            "parts_with_degenerate_faces": sum(record["degenerate_face_count"] > 0 for record in records),
            "normal_match_within_1e_6": sum(record["sampled_normal_max_component_error"] <= 1e-6 for record in records),
        },
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
