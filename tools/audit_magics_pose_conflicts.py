"""Find same-topology rigidly matching Magics meshes with changed tilt pose."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_expanded_data import MTArchive


def topology_hash(dataset: Path, project: str, member: str, surface_member: str):
    source = dataset / f"{project}.magics"
    with MTArchive(source) as archive:
        faces = archive.read(surface_member)
    return hashlib.sha256(faces).hexdigest()


def read_vertices(dataset: Path, record: dict):
    source = dataset / f"{record['project_relative']}.magics"
    with MTArchive(source) as archive:
        raw = archive.read(record["member"])
    return np.frombuffer(raw, dtype="<i4").reshape(-1, 3).astype(np.float64) / 10000.0


def fit_rigid(reference: np.ndarray, moved: np.ndarray):
    a = reference - reference.mean(axis=0, keepdims=True)
    b = moved - moved.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = moved.mean(axis=0) - reference.mean(axis=0) @ rotation
    residual = reference @ rotation + translation - moved
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    max_error = float(np.linalg.norm(residual, axis=1).max())
    # For row-vector Q, e_z dot Q e_z is Q[2,2]. A yaw-only change leaves this 1.
    pose_axis_delta = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
    return rms, max_error, pose_axis_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/magics_parts_stl/manifest.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("reports/magics_pose_conflict_audit_2026-11-09.json"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-fit-error-mm", type=float, default=0.02)
    parser.add_argument("--tilt-threshold-deg", type=float, default=2.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    if document["status"] != "complete" or document["errors"]:
        raise ValueError("Magics extraction has not completed cleanly")

    unique = {}
    for record in document["records"]:
        unique.setdefault(record["geometry_sha256"], record)
    records = list(unique.values())
    futures = {}
    topology_groups = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in records:
            future = pool.submit(topology_hash, args.dataset, record["project_relative"],
                                 record["member"], record["surface_member"])
            futures[future] = record
        for index, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            key = (future.result(), record["vertices"], record["triangles"])
            topology_groups[key].append(record)
            if index % 200 == 0 or index == len(records):
                print(f"{index}/{len(records)} unique mesh topologies checked", flush=True)

    repeated_groups = [group for group in topology_groups.values() if len(group) > 1]
    pose_variants = []
    rigid_pair_count = 0
    comparisons = 0
    for group in repeated_groups:
        reference_record = group[0]
        reference = read_vertices(args.dataset, reference_record)
        for record in group[1:]:
            if record["geometry_sha256"] == reference_record["geometry_sha256"]:
                continue
            moved = read_vertices(args.dataset, record)
            comparisons += 1
            rms, max_error, axis_delta = fit_rigid(reference, moved)
            if rms <= args.max_fit_error_mm and max_error <= args.max_fit_error_mm:
                rigid_pair_count += 1
                if axis_delta >= args.tilt_threshold_deg:
                    pose_variants.append({
                        "reference_project": reference_record["project_relative"],
                        "reference_member": reference_record["member"],
                        "reference_geometry_sha256": reference_record["geometry_sha256"],
                        "other_project": record["project_relative"],
                        "other_member": record["member"],
                        "other_geometry_sha256": record["geometry_sha256"],
                        "vertex_count": record["vertices"],
                        "triangle_count": record["triangles"],
                        "rigid_fit_rms_mm": rms,
                        "rigid_fit_max_mm": max_error,
                        "implied_up_axis_difference_deg": axis_delta,
                    })
    summary = {
        "source_part_records": len(document["records"]),
        "unique_exact_geometry_records": len(records),
        "unique_topology_signatures": len(topology_groups),
        "topology_signatures_with_multiple_meshes": len(repeated_groups),
        "rigid_pose_comparisons": comparisons,
        "rigid_matches_within_tolerance": rigid_pair_count,
        "same_topology_rigid_matches_with_different_up_labels": len(pose_variants),
        "fit_error_tolerance_mm": args.max_fit_error_mm,
        "different_up_threshold_deg": args.tilt_threshold_deg,
        "note": "This conservative check requires identical ordered triangle-index buffers and vertex ordering. It can miss re-tessellated/reindexed duplicates; exact mesh records all share the same stored-frame +Z label.",
    }
    output = {"summary": summary, "pose_variants": pose_variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
