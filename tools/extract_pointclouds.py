import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
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


def safe_file_stem(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "project"


def decode_mesh(data):
    if len(data) < 33 or data[0] != 3:
        raise ValueError("Unsupported mesh header")
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
    return vertices.astype(np.float64), faces.astype(np.int64)


def sample_surface(vertices, faces, point_count, rng):
    if not len(faces):
        raise ValueError("Mesh has no triangles")
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_areas = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(doubled_areas) & (doubled_areas > 1e-14)
    if not valid.any():
        raise ValueError("Mesh contains no nondegenerate triangles")
    face_ids = np.flatnonzero(valid)
    weights = doubled_areas[valid] / doubled_areas[valid].sum()
    sampled_ids = rng.choice(face_ids, size=point_count, replace=True, p=weights)
    selected = triangles[sampled_ids]
    random_uv = rng.random((point_count, 2))
    reflected = random_uv.sum(axis=1) > 1.0
    random_uv[reflected] = 1.0 - random_uv[reflected]
    points = (
        selected[:, 0]
        + random_uv[:, :1] * (selected[:, 1] - selected[:, 0])
        + random_uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    )
    normals = cross[sampled_ids] / doubled_areas[sampled_ids, None]
    surface_area = float(doubled_areas.sum() * 0.5)
    return points.astype(np.float32), normals.astype(np.float32), sampled_ids.astype(np.int32), surface_area


def make_features(points, vertices, surface_area, triangle_count):
    center = points.mean(axis=0, dtype=np.float64)
    centered = points.astype(np.float64) - center
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    total_variance = max(float(eigenvalues.sum()), np.finfo(float).eps)
    variance_ratio = eigenvalues / total_variance
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    distances = np.linalg.norm(centered, axis=1)
    return {
        "point_count": len(points),
        "triangle_count": int(triangle_count),
        "surface_area_raw_units2": surface_area,
        "bbox_min_raw_xyz": bbox_min.tolist(),
        "bbox_max_raw_xyz": bbox_max.tolist(),
        "bbox_size_raw_xyz": bbox_size.tolist(),
        "pointcloud_centroid_raw_xyz": center.tolist(),
        "pca_eigenvalues_raw_units2_desc": eigenvalues.tolist(),
        "pca_variance_ratio_desc": variance_ratio.tolist(),
        "radius_mean_raw_units": float(distances.mean()),
        "radius_std_raw_units": float(distances.std()),
        "radius_max_raw_units": float(distances.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Export historical root-part meshes as sampled point clouds")
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pointclouds"))
    parser.add_argument("--points-per-part", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261108)
    arguments = parser.parse_args()
    if arguments.points_per_part <= 0:
        parser.error("--points-per-part must be positive")
    files = sorted(arguments.dataset.glob("*.pjf"))
    if not files:
        parser.error(f"No .pjf files in {arguments.dataset}")

    rng = np.random.default_rng(arguments.seed)
    records = []
    excluded_boxes = []
    errors = []
    for project_path in files:
        try:
            with sqlite3.connect(project_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
                tables = {
                    row[0]
                    for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if not {"PartV220224", "MeshV220218", "Material"}.issubset(tables):
                    raise ValueError("Unsupported project schema")
                parts = {
                    part_id: parse_config(config)
                    for part_id, config in connection.execute("SELECT id, config FROM PartV220224")
                }
                child_ids = {
                    int(child)
                    for config in parts.values()
                    for child in config.get("childs", "").split(",")
                    if child
                }
                mesh_rows = {
                    mesh_id: (parse_config(config), data)
                    for mesh_id, config, data in connection.execute(
                        "SELECT id, config, data FROM MeshV220218"
                    )
                }
                material_rows = [parse_config(row[0]) for row in connection.execute("SELECT config FROM Material")]
                materials = [decode_name(row.get("name", "")) for row in material_rows]
                for part_id, config in parts.items():
                    if part_id in child_ids:
                        continue
                    name = decode_name(config.get("name", ""))
                    if name.lower().startswith("box"):
                        excluded_boxes.append({"project": project_path.name, "part_id": part_id, "name": name})
                        continue
                    mesh_id = config.get("mesh")
                    if mesh_id not in mesh_rows:
                        raise ValueError(f"Part {part_id} points to missing mesh {mesh_id}")
                    _, mesh_blob = mesh_rows[mesh_id]
                    vertices, faces = decode_mesh(mesh_blob)
                    points, normals, sampled_face_ids, area = sample_surface(
                        vertices, faces, arguments.points_per_part, rng
                    )
                    features = make_features(points, vertices, area, len(faces))
                    mesh_hash = hashlib.sha256(mesh_blob).hexdigest()
                    relative_name = (
                        f"{safe_file_stem(project_path.stem)}__part_{part_id}__{mesh_hash[:10]}.npz"
                    )
                    output_path = arguments.output / relative_name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        output_path,
                        points=points,
                        normals=normals,
                        sampled_face_ids=sampled_face_ids,
                        centroid=points.mean(axis=0),
                        mesh_hash=mesh_hash,
                    )
                    values = [float(value) for value in config.get("trfm", "").split(",") if value]
                    records.append({
                        "source_project": project_path.name,
                        "source_part_id": part_id,
                        "source_part_name": name,
                        "source_mesh_id": mesh_id,
                        "mesh_hash_sha256": mesh_hash,
                        "materials": materials,
                        "historical_pose_status": "user_confirmed_printed_and_validated",
                        "source_transform_raw": values,
                        "source_transform_rotation_is_identity": (
                            bool(np.allclose(np.array(values).reshape(4, 4)[:3, :3], np.eye(3)))
                            if len(values) == 16 else None
                        ),
                        "source_coordinate_units": "unconfirmed; raw coordinates preserved",
                        "pointcloud_file": relative_name,
                        "features": features,
                    })
        except (sqlite3.Error, ValueError, OSError, struct.error) as error:
            errors.append({"file": project_path.name, "error": str(error)})

    feature_names = [
        "point_count", "triangle_count", "surface_area_raw_units2",
        "bbox_x_raw_units", "bbox_y_raw_units", "bbox_z_raw_units",
        "pca_eigenvalue_1_raw_units2", "pca_eigenvalue_2_raw_units2", "pca_eigenvalue_3_raw_units2",
        "pca_variance_ratio_1", "pca_variance_ratio_2", "pca_variance_ratio_3",
        "radius_mean_raw_units", "radius_std_raw_units", "radius_max_raw_units",
    ]
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "features.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_project", "source_part_id", "source_part_name", "materials", "mesh_hash_sha256", "pointcloud_file", *feature_names])
        writer.writeheader()
        for record in records:
            features = record["features"]
            writer.writerow({
                "source_project": record["source_project"],
                "source_part_id": record["source_part_id"],
                "source_part_name": record["source_part_name"],
                "materials": ";".join(record["materials"]),
                "mesh_hash_sha256": record["mesh_hash_sha256"],
                "pointcloud_file": record["pointcloud_file"],
                "point_count": features["point_count"],
                "triangle_count": features["triangle_count"],
                "surface_area_raw_units2": features["surface_area_raw_units2"],
                "bbox_x_raw_units": features["bbox_size_raw_xyz"][0],
                "bbox_y_raw_units": features["bbox_size_raw_xyz"][1],
                "bbox_z_raw_units": features["bbox_size_raw_xyz"][2],
                "pca_eigenvalue_1_raw_units2": features["pca_eigenvalues_raw_units2_desc"][0],
                "pca_eigenvalue_2_raw_units2": features["pca_eigenvalues_raw_units2_desc"][1],
                "pca_eigenvalue_3_raw_units2": features["pca_eigenvalues_raw_units2_desc"][2],
                "pca_variance_ratio_1": features["pca_variance_ratio_desc"][0],
                "pca_variance_ratio_2": features["pca_variance_ratio_desc"][1],
                "pca_variance_ratio_3": features["pca_variance_ratio_desc"][2],
                "radius_mean_raw_units": features["radius_mean_raw_units"],
                "radius_std_raw_units": features["radius_std_raw_units"],
                "radius_max_raw_units": features["radius_max_raw_units"],
            })
    manifest = {
        "dataset": str(arguments.dataset.resolve()),
        "sample_count": len(records),
        "points_per_part": arguments.points_per_part,
        "random_seed": arguments.seed,
        "sampling": "Triangle surface area-weighted; deterministic with the configured seed.",
        "normals": "Sampled triangle normals; winding/outward direction not independently verified.",
        "coordinates": "Raw historical mesh frame; units and PJF-to-machine transform still need verification.",
        "training_note": "Keep orientation and scale for pose learning. Center each cloud for translation invariance; apply random 3D rotations together with inverse-pose labels.",
        "excluded_box_roots": excluded_boxes,
        "errors": errors,
        "records": records,
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "sample_count": len(records),
        "points_per_part": arguments.points_per_part,
        "excluded_box_roots": len(excluded_boxes),
        "unique_mesh_hashes": len({record["mesh_hash_sha256"] for record in records}),
        "errors": errors,
        "output": str(arguments.output.resolve()),
    }, ensure_ascii=True, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
