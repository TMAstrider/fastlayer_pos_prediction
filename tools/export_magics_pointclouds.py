"""Surface-sample unique exported Magics meshes for point-cloud training."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


STL_TRIANGLE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])


def sample_stl(root: Path, record: dict, point_count: int, seed: int):
    source = root / record["stl_file"]
    with source.open("rb") as handle:
        handle.seek(80)
        triangle_count_bytes = handle.read(4)
    if len(triangle_count_bytes) != 4:
        raise ValueError(f"Truncated binary STL header: {source}")
    triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
    if triangle_count != record["triangles"] or source.stat().st_size != 84 + 50 * triangle_count:
        raise ValueError(f"STL header/manifest mismatch: {source}")
    triangles_map = np.memmap(
        source, dtype=STL_TRIANGLE, mode="r", offset=84, shape=(triangle_count,)
    )
    triangles = np.asarray(triangles_map["vertices"], dtype=np.float32)
    del triangles_map
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edges_a, edges_b)
    doubled_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(doubled_area) & (doubled_area > 1e-14)
    face_ids = np.flatnonzero(valid)
    if not len(face_ids):
        raise ValueError(f"No nondegenerate triangles: {source}")
    probabilities = doubled_area[valid].astype(np.float64)
    probabilities /= probabilities.sum()
    mesh_seed = int.from_bytes(hashlib.sha256(
        f"{record['geometry_sha256']}:{seed}".encode("ascii")
    ).digest()[:8], "little")
    rng = np.random.default_rng(mesh_seed)
    sampled_ids = rng.choice(face_ids, size=point_count, replace=True, p=probabilities)
    uv = rng.random((point_count, 2), dtype=np.float32)
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    selected = triangles[sampled_ids]
    points = (
        selected[:, 0]
        + uv[:, :1] * (selected[:, 1] - selected[:, 0])
        + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    ).astype(np.float32)
    return points


def export(manifest_path: Path, output: Path, point_count: int, seed: int, workers: int):
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest["status"] != "complete" or source_manifest["errors"]:
        raise ValueError("Magics mesh extraction has not completed cleanly")
    unique_records = {}
    all_records_by_hash = {}
    for record in source_manifest["records"]:
        key = record["geometry_sha256"]
        all_records_by_hash.setdefault(key, []).append(record)
        unique_records.setdefault(key, record)
    output.mkdir(parents=True, exist_ok=True)

    futures = {}
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for mesh_hash, record in unique_records.items():
            futures[pool.submit(sample_stl, manifest_path.parent, record, point_count, seed)] = mesh_hash
        for index, future in enumerate(as_completed(futures), 1):
            mesh_hash = futures[future]
            record = unique_records[mesh_hash]
            points = future.result()
            point_file = f"{mesh_hash}.npz"
            np.savez_compressed(output / point_file, points=points, geometry_sha256=mesh_hash)
            results[mesh_hash] = {
                "geometry_sha256": mesh_hash,
                "pointcloud_file": point_file,
                "source_stl_file": record["stl_file"],
                "source_project": record["project_relative"],
                "source_member": record["member"],
                "point_count": len(points),
                "triangle_count": record["triangles"],
                "duplicate_record_count": len(all_records_by_hash[mesh_hash]),
            }
            if index % 100 == 0 or index == len(unique_records):
                print(f"{index}/{len(unique_records)} unique meshes sampled", flush=True)

    generated = {
        "source_manifest": manifest_path.resolve().as_posix(),
        "sample_count_unique_geometries": len(results),
        "source_part_records": len(source_manifest["records"]),
        "points_per_mesh": point_count,
        "seed": seed,
        "sampling": "Area-weighted sampling from the original binary STL triangles.",
        "coordinates": "Preserve stored Magics mesh axes; training centers and unit-sphere normalizes each cloud.",
        "records": sorted(results.values(), key=lambda item: item["geometry_sha256"]),
        "errors": [],
    }
    (output / "manifest.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: generated[key] for key in (
        "sample_count_unique_geometries", "source_part_records", "points_per_mesh", "errors"
    )}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/magics_parts_stl/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/magics_pointclouds"))
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.points, args.workers) <= 0:
        parser.error("--points and --workers must be positive")
    export(args.manifest.resolve(), args.output.resolve(), args.points, args.seed, args.workers)


if __name__ == "__main__":
    main()
