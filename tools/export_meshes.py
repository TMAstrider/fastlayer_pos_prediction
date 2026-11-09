import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import struct

import numpy as np

from audit_pjf import decode_mesh, decode_name, parse_config


STL_RECORD = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])
CHUNK_SIZE = 100000


def safe_name(value):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")[:90] or "unnamed"


def write_stl(path, vertices, faces):
    header = b"PJF raw mesh export; source units; object transforms not applied"
    with path.open("wb") as handle:
        handle.write(header.ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(faces)))
        for start in range(0, len(faces), CHUNK_SIZE):
            triangles = vertices[faces[start:start + CHUNK_SIZE]]
            triangle_values = triangles.astype(np.float64)
            normals = np.cross(
                triangle_values[:, 1] - triangle_values[:, 0],
                triangle_values[:, 2] - triangle_values[:, 0],
            )
            lengths = np.linalg.norm(normals, axis=1)
            np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)
            records = np.zeros(len(triangles), dtype=STL_RECORD)
            records["normal"] = normals
            records["vertices"] = triangles
            handle.write(records.tobytes())


def verify_stl(path, vertices, faces):
    if path.stat().st_size != 84 + 50 * len(faces):
        raise ValueError(f"STL byte count mismatch: {path}")
    with path.open("rb") as handle:
        handle.seek(80)
        if struct.unpack("<I", handle.read(4))[0] != len(faces):
            raise ValueError(f"STL triangle count mismatch: {path}")
        for start in range(0, len(faces), CHUNK_SIZE):
            expected = vertices[faces[start:start + CHUNK_SIZE]]
            records = np.frombuffer(handle.read(50 * len(expected)), dtype=STL_RECORD)
            if not np.array_equal(records["vertices"], expected):
                raise ValueError(f"STL coordinates mismatch: {path}")
            if not np.isfinite(records["normal"]).all():
                raise ValueError(f"STL contains nonfinite normals: {path}")


def export_project(project_path, output_directory):
    project_hash = hashlib.sha256(project_path.name.encode("utf-8")).hexdigest()[:10]
    records = []
    with sqlite3.connect(project_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        parts = {
            part_id: parse_config(config)
            for part_id, config in connection.execute("SELECT id, config FROM PartV220224")
        }
        parents = {}
        for part_id, config in parts.items():
            for child in config.get("childs", "").split(","):
                if child:
                    parents.setdefault(int(child), []).append(part_id)
        materials = [
            decode_name(parse_config(config).get("name", ""))
            for config, in connection.execute("SELECT config FROM Material")
        ]
        for part_id, config in parts.items():
            name = decode_name(config.get("name", ""))
            parent_ids = parents.get(part_id, [])
            if name.lower().startswith("box"):
                category = "review"
                role = "box_name_review_required"
            elif name.startswith("s_") and parent_ids:
                category = "supports"
                role = "support_mesh_candidate"
            elif not parent_ids:
                category = "parts"
                role = "historical_part_pose_candidate"
            else:
                category = "review"
                role = "child_object_review_required"
            mesh_id = config.get("mesh")
            mesh_row = connection.execute(
                "SELECT data FROM MeshV220218 WHERE id = ?", (mesh_id,)
            ).fetchone()
            if mesh_row is None:
                raise ValueError(f"Part {part_id} references missing mesh {mesh_id}")
            vertices, faces, bounds = decode_mesh(mesh_row[0])
            filename = f"part_{part_id}__{safe_name(Path(name).stem)}.stl"
            relative_path = Path(category) / f"project_{project_hash}" / filename
            output_path = output_directory / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_stl(output_path, vertices, faces)
            verify_stl(output_path, vertices, faces)
            records.append({
                "source_project": project_path.name,
                "source_part_id": part_id,
                "source_part_name": name,
                "source_mesh_id": mesh_id,
                "source_mesh_sha256": hashlib.sha256(mesh_row[0]).hexdigest(),
                "category": category,
                "role_hint": role,
                "parent_ids": parent_ids,
                "child_ids": [int(child) for child in config.get("childs", "").split(",") if child],
                "source_config_raw": config,
                "materials": materials,
                "vertex_count_source": len(vertices),
                "triangle_count": len(faces),
                "bounds_raw": bounds.tolist(),
                "stl_file": relative_path.as_posix(),
                "stl_bytes": output_path.stat().st_size,
                "verified_against_source_triangles": True,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Export PJF part and support meshes as verified binary STL files")
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/meshes_stl"))
    arguments = parser.parse_args()
    projects = sorted(arguments.dataset.glob("*.pjf"))
    if not projects:
        parser.error(f"No .pjf files in {arguments.dataset}")
    records = []
    errors = []
    for project_path in projects:
        try:
            project_records = export_project(project_path, arguments.output)
            records.extend(project_records)
            print(f"Exported and verified {len(project_records)} objects: {project_path.name}", flush=True)
        except (sqlite3.Error, ValueError, OSError, struct.error) as error:
            errors.append({"project": project_path.name, "error": str(error)})
    summary = {
        "source_projects": len(projects),
        "exported_meshes": len(records),
        "categories": dict(Counter(record["category"] for record in records)),
        "triangles": sum(record["triangle_count"] for record in records),
        "stl_bytes": sum(record["stl_bytes"] for record in records),
        "errors": errors,
    }
    manifest = {
        "dataset": str(arguments.dataset.resolve()),
        "summary": summary,
        "format": "Binary STL; original triangles and float32 vertex coordinates preserved; no simplification.",
        "coordinates": "Raw mesh coordinates; no centering, scaling, or object/parent transforms applied. Units and transform convention require FastLayer verification.",
        "roles": "Inferred from names and parent-child relationships; supports and Box objects require visual confirmation.",
        "normals": "Recomputed from original triangle winding; zero normals for degenerate triangles. Outward orientation is not independently verified.",
        "topology": "STL stores independent triangles, not indexed vertices. It does not preserve source vertex indexing or materials.",
        "records": records,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Output: {arguments.output.resolve()}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
