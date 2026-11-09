import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import struct

import numpy as np


def parse_config(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def decode_name(value):
    try:
        return "".join(chr(int(code)) for code in value.split(",") if code)
    except ValueError:
        return value


def decode_mesh(data):
    if len(data) < 33 or data[0] != 3:
        raise ValueError("Unsupported mesh header")
    bounds = np.frombuffer(data, dtype="<f4", count=6, offset=1)
    vertex_count = struct.unpack_from("<I", data, 25)[0]
    vertex_end = 29 + vertex_count * 12
    if vertex_end + 4 > len(data):
        raise ValueError("Truncated vertex buffer")
    face_count = struct.unpack_from("<I", data, vertex_end)[0]
    if vertex_end + 4 + face_count * 12 != len(data):
        raise ValueError("Unexpected face buffer length")
    vertices = np.frombuffer(data, dtype="<f4", count=vertex_count * 3, offset=29).reshape(-1, 3)
    faces = np.frombuffer(data, dtype="<u4", count=face_count * 3, offset=vertex_end + 4).reshape(-1, 3)
    if not vertex_count or not np.isfinite(vertices).all():
        raise ValueError("Empty mesh or nonfinite vertices")
    if faces.size and int(faces.max()) >= vertex_count:
        raise ValueError("Face index outside vertex buffer")
    actual_bounds = np.concatenate((vertices.min(axis=0), vertices.max(axis=0)))
    if not np.allclose(bounds, actual_bounds, atol=0.01):
        raise ValueError("Mesh bounds do not match vertex buffer")
    return vertices, faces, actual_bounds


def read_configs(connection, table):
    return [parse_config(row[0]) for row in connection.execute(f'SELECT config FROM "{table}"')]


def audit_project(path, mesh_directory):
    project = {"file": path.name, "bytes": path.stat().st_size}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        project["tables"] = {
            name: connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if name.replace("_", "").isalnum()
        }
        required = {"PartV220224", "MeshV220218", "Material", "WorkBench", "EntryIndex", "SlicePlan"}
        if not required.issubset(project["tables"]):
            raise ValueError("Unsupported project schema")
        project["material"] = read_configs(connection, "Material")
        project["workbench"] = read_configs(connection, "WorkBench")
        project["entry"] = read_configs(connection, "EntryIndex")
        project["slice_plan"] = read_configs(connection, "SlicePlan")
        for material in project["material"]:
            material["decoded_name"] = decode_name(material.get("name", ""))
        parts = {
            part_id: parse_config(text)
            for part_id, text in connection.execute("SELECT id, config FROM PartV220224")
        }
        parents = {}
        for part_id, config in parts.items():
            for child in config.get("childs", "").split(","):
                if child:
                    parents.setdefault(int(child), []).append(part_id)
        project["missing_child_ids"] = sorted(set(parents) - set(parts))
        meshes = {}
        for mesh_id, text, data in connection.execute("SELECT id, config, data FROM MeshV220218"):
            vertices, faces, bounds = decode_mesh(data)
            mesh = {
                "id": mesh_id,
                "name": decode_name(parse_config(text).get("name", "")),
                "sha256": hashlib.sha256(data).hexdigest(),
                "vertex_count": len(vertices),
                "face_count": len(faces),
                "bounds_raw": bounds.tolist(),
            }
            if mesh_directory is not None:
                output = mesh_directory / path.stem / f"{mesh_id}.npz"
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, vertices=vertices, faces=faces)
                mesh["export_file"] = str(output.resolve())
            meshes[mesh_id] = mesh
        project["meshes"] = list(meshes.values())
        project["parts"] = []
        for part_id, config in parts.items():
            name = decode_name(config.get("name", ""))
            values = [float(value) for value in config.get("trfm", "").split(",") if value]
            matrix = np.array(values).reshape(4, 4) if len(values) == 16 else None
            if name.lower().startswith("box"):
                role_hint = "box_name_review_required"
            elif name.startswith("s_"):
                role_hint = "support_mesh_candidate"
            elif not parents.get(part_id):
                role_hint = "historical_part_pose_candidate"
            else:
                role_hint = "child_object_review_required"
            project["parts"].append({
                "id": part_id,
                "name": name,
                "role_hint": role_hint,
                "parent_ids": parents.get(part_id, []),
                "mesh_found": config.get("mesh") in meshes,
                "rotation_block_is_identity": (
                    bool(np.allclose(matrix[:3, :3], np.eye(3))) if matrix is not None else None
                ),
                "config_raw": config,
            })
    return project


def main():
    parser = argparse.ArgumentParser(description="Read-only audit of the observed PJF SQLite mesh format")
    parser.add_argument("dataset", nargs="?", type=Path,
                        default=Path("datasets/initial_data") if Path("datasets/initial_data").is_dir() else Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    parser.add_argument("--export-meshes", type=Path)
    arguments = parser.parse_args()
    files = sorted(arguments.dataset.glob("*.pjf"))
    if not files:
        parser.error(f"No .pjf files in {arguments.dataset}")
    projects = []
    errors = []
    for path in files:
        try:
            projects.append(audit_project(path, arguments.export_meshes))
        except (sqlite3.Error, ValueError, OSError, struct.error) as error:
            errors.append({"file": path.name, "error": str(error)})
    parts = [part for project in projects for part in project["parts"]]
    meshes = [mesh for project in projects for mesh in project["meshes"]]
    role_hints = Counter(part["role_hint"] for part in parts)
    summary = {
        "input_files": len(files),
        "parsed_projects": len(projects),
        "total_bytes": sum(path.stat().st_size for path in files),
        "part_objects": len(parts),
        "mesh_objects": len(meshes),
        "unique_raw_mesh_hashes": len({mesh["sha256"] for mesh in meshes}),
        "vertices": sum(mesh["vertex_count"] for mesh in meshes),
        "triangles": sum(mesh["face_count"] for mesh in meshes),
        "root_objects": sum(not part["parent_ids"] for part in parts),
        "historical_part_pose_candidates": role_hints["historical_part_pose_candidate"],
        "support_mesh_candidates": role_hints["support_mesh_candidate"],
        "box_name_review_required": role_hints["box_name_review_required"],
        "role_hints": dict(role_hints),
        "tags_raw": dict(Counter(part["config_raw"].get("tag") for part in parts)),
        "nonidentity_rotation_blocks": sum(part["rotation_block_is_identity"] is False for part in parts),
        "invalid_transform_lengths": sum(part["rotation_block_is_identity"] is None for part in parts),
        "missing_mesh_references": sum(not part["mesh_found"] for part in parts),
        "missing_child_references": sum(len(project["missing_child_ids"]) for project in projects),
        "materials": dict(Counter(
            material["decoded_name"] for project in projects for material in project["material"]
        )),
        "property_rows": sum(
            count for project in projects for table, count in project["tables"].items()
            if table.startswith("Prop")
        ),
        "errors": errors,
    }
    report = {
        "summary": summary,
        "pose_supervision": {
            "status": "user_confirmed_printed_and_validated_historical_poses",
            "scope": "Root part geometries in each saved project, subject to checking the 11 Box-named roots.",
            "training_use": "Positive examples for pose recommendation; randomly rotated input copies can be paired with inverse transforms back to each historical accepted pose.",
            "not_provided": "No bad-pose labels, pose comparisons, or per-part print outcome measurements were identified in the file tables.",
        },
        "limitations": [
            "Binary layout inferred from these files; not a vendor format specification.",
            "Support and Box object roles are inferred from name prefixes and object relationships, not a vendor specification.",
            "Raw coordinates and transforms are preserved; units and transform convention need confirmation.",
            "Identity object rotation does not imply identity historical pose; rotation may already be included in mesh vertex coordinates.",
            "Raw mesh hashes do not deduplicate rotated, translated, mirrored, or remeshed part families.",
            "All-confirmed-good historical poses are positive supervision; they do not by themselves identify which of several poses is best or predict failure probability.",
        ],
        "projects": projects,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Report: {arguments.output.resolve()}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
