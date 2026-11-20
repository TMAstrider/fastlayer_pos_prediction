#!/usr/bin/env python3
"""Build deterministic support-projection score labels for all extracted parts."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

try:
    from tools.generate_simplified_supports import process_config, read_mesh
    from tools.recognize_part_features import recognize_features
    from tools.score_support_directions import fibonacci_directions
    from tools.support_area_scoring import (
        dense_area_ranks,
        evaluate_support_projection,
        normalise_projected_areas,
    )
except ModuleNotFoundError:
    from generate_simplified_supports import process_config, read_mesh
    from recognize_part_features import recognize_features
    from score_support_directions import fibonacci_directions
    from support_area_scoring import (
        dense_area_ranks,
        evaluate_support_projection,
        normalise_projected_areas,
    )


FEATURE_NAMES = [
    "retained_surface_fraction",
    "exempt_surface_fraction",
    "candidate_surface_fraction",
    "build_height_over_diagonal",
    "projected_bbox_area_over_diagonal_sq",
    "projected_width_over_diagonal",
    "projected_height_over_diagonal",
    "retained_height_mean",
    "retained_height_std",
    "retained_height_max",
    "retained_angle_mean_over_threshold",
    "retained_angle_std_over_threshold",
    "all_down_angle_le_threshold_fraction",
    "all_down_angle_gt_threshold_fraction",
    "small_hole_exemption_count_over_10",
    "direction_x",
    "direction_y",
    "direction_z",
]


def weighted_moments(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(values) or float(weights.sum()) <= 1e-12:
        return 0.0, 0.0, 0.0
    probability = weights / weights.sum()
    mean = float(np.sum(values * probability))
    variance = float(np.sum((values - mean) ** 2 * probability))
    return mean, float(np.sqrt(max(variance, 0.0))), float(values.max())


def candidate_features(
    result: dict[str, Any],
    direction: np.ndarray,
    limits: dict[str, float],
) -> np.ndarray:
    """Describe a candidate direction without using its projected-area target."""
    report = result["report"]
    triangles = result["triangles"]
    normals = result["normals"]
    areas = result["areas"]
    candidate = result["candidate_faces"]
    exempt = result["exempt_faces"]
    angles = result["face_angles"]
    vertices = result["aligned_vertices"]

    total_area = max(float(areas.sum()), 1e-12)
    diagonal = max(float(np.linalg.norm(vertices.max(0) - vertices.min(0))), 1e-12)
    build_height = max(float(np.ptp(vertices[:, 2])), 1e-12)
    candidate_ids = np.flatnonzero(candidate)
    candidate_areas = areas[candidate_ids]
    candidate_centers_z = triangles[candidate_ids, :, 2].mean(axis=1)
    normalized_heights = (candidate_centers_z - float(vertices[:, 2].min())) / build_height
    height_mean, height_std, height_max = weighted_moments(
        normalized_heights, candidate_areas
    )
    candidate_angles = angles[candidate_ids]
    angle_mean, angle_std, _ = weighted_moments(candidate_angles, candidate_areas)

    downward = np.isfinite(angles)
    threshold = float(limits["self_support_angle_threshold_deg"])
    below_threshold_area = float(areas[downward & (angles <= threshold)].sum())
    above_threshold_area = float(areas[downward & (angles > threshold)].sum())
    if len(candidate_ids):
        projected_xy = triangles[candidate_ids, :, :2].reshape(-1, 2)
        width = float(np.ptp(projected_xy[:, 0]))
        height = float(np.ptp(projected_xy[:, 1]))
    else:
        width = height = 0.0

    feature_summary = report["recognized_feature_summary"] or {}
    return np.asarray([
        float(report["areas_mm2"]["downward_angle_candidates"]) / total_area,
        float(areas[exempt].sum()) / total_area,
        float(areas[candidate].sum()) / total_area,
        build_height / diagonal,
        width * height / (diagonal * diagonal),
        width / diagonal,
        height / diagonal,
        height_mean,
        height_std,
        height_max,
        angle_mean / max(threshold, 1e-12),
        angle_std / max(threshold, 1e-12),
        below_threshold_area / total_area,
        above_threshold_area / total_area,
        float(feature_summary.get("small_horizontal_exemptions", 0)) / 10.0,
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
    ], dtype=np.float32)


def score_one_part(
    job: dict[str, Any],
    directions: np.ndarray,
    limits: dict[str, float],
    projection_grid_mm: float,
    output_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    vertices, faces = read_mesh(Path(job["mesh_path"]))
    recognized = recognize_features(
        vertices,
        faces,
        build_up=np.array([0.0, 0.0, 1.0]),
        small_hole_diameter_mm=limits["horizontal_hole_diameter_threshold_mm"],
    )
    features = np.empty((len(directions), len(FEATURE_NAMES)), dtype=np.float32)
    areas = np.empty(len(directions), dtype=np.float64)
    retained_surface = np.empty(len(directions), dtype=np.float64)
    exempt_surface = np.empty(len(directions), dtype=np.float64)
    candidate_faces = np.empty(len(directions), dtype=np.int32)
    small_hole_exemptions = np.empty(len(directions), dtype=np.int16)

    for index, direction in enumerate(directions):
        result = evaluate_support_projection(
            vertices,
            faces,
            limits,
            up=direction,
            recognized_features=recognized,
            projection_grid_mm=projection_grid_mm,
        )
        report = result["report"]
        areas[index] = report["areas_mm2"]["projected_union"]
        retained_surface[index] = report["areas_mm2"]["retained_overhang_surface"]
        exempt_surface[index] = report["areas_mm2"]["feature_exempt"]
        candidate_faces[index] = report["face_counts"][
            "support_candidate_after_exemptions"
        ]
        small_hole_exemptions[index] = int(
            report["recognized_feature_summary"]["small_horizontal_exemptions"]
        )
        features[index] = candidate_features(result, direction, limits)

    scores, minimum, maximum = normalise_projected_areas(areas)
    ranks = dense_area_ranks(areas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(job["pointcloud_path"]) as point_data:
        points = np.asarray(point_data["points"], dtype=np.float32)
        normals = np.asarray(point_data["normals"], dtype=np.float32)
    np.savez_compressed(
        output_path,
        directions=directions.astype(np.float32),
        features=features,
        projected_area_mm2=areas.astype(np.float32),
        score=scores.astype(np.float32),
        rank=ranks.astype(np.int16),
        retained_surface_mm2=retained_surface.astype(np.float32),
        exempt_surface_mm2=exempt_surface.astype(np.float32),
        candidate_face_count=candidate_faces,
        small_hole_exemptions=small_hole_exemptions,
        points=points,
        normals=normals,
    )
    return {
        **job,
        "label_file": str(output_path),
        "direction_count": int(len(directions)),
        "minimum_area_mm2": minimum,
        "maximum_area_mm2": maximum,
        "median_area_mm2": float(np.median(areas)),
        "recognized_internal_cylinders": int(
            recognized["summary"]["recognized_internal_cylinders"]
        ),
        "small_horizontal_exemptions": int(
            recognized["summary"]["small_horizontal_exemptions"]
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def build_jobs(root: Path) -> list[dict[str, Any]]:
    mesh_manifest_path = root / "outputs/meshes_stl/manifest.json"
    point_manifest_path = root / "outputs/pointclouds/manifest.json"
    mesh_records = [
        row for row in json.loads(mesh_manifest_path.read_text(encoding="utf-8"))["records"]
        if row["category"] == "parts"
    ]
    point_records = json.loads(point_manifest_path.read_text(encoding="utf-8"))["records"]
    point_by_hash = {row["mesh_hash_sha256"]: row for row in point_records}
    jobs = []
    for index, mesh in enumerate(mesh_records):
        point = point_by_hash.get(mesh["source_mesh_sha256"])
        if point is None:
            raise ValueError(f"No point cloud for mesh {mesh['source_mesh_sha256']}")
        job_id = (
            f"{point['source_project'].replace('.pjf', '')}__"
            f"part_{int(point['source_part_id']):05d}__"
            f"{point['mesh_hash_sha256'][:12]}"
        )
        jobs.append({
            "index": index,
            "job_id": job_id,
            "source_project": point["source_project"],
            "source_part_id": int(point["source_part_id"]),
            "source_part_name": point["source_part_name"],
            "mesh_hash_sha256": point["mesh_hash_sha256"],
            "triangle_count": int(mesh["triangle_count"]),
            "mesh_path": str(
                (root / "outputs/meshes_stl" / mesh["stl_file"]).resolve()
            ),
            "pointcloud_path": str(
                (root / "outputs/pointclouds" / point["pointcloud_file"]).resolve()
            ),
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--material", choices=["316L", "AlSi10Mg", "TC4"], default="AlSi10Mg")
    parser.add_argument("--mode", choices=["standard", "am_build"], default="standard")
    parser.add_argument("--directions", type=int, default=256)
    parser.add_argument("--projection-grid-mm", type=float, default=0.5)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--validation-project", action="append", default=[])
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.directions < 2 or args.jobs < 1:
        parser.error("--directions must be at least 2 and --jobs must be positive")

    root = args.root.resolve()
    rules_path = args.rules or (root / "configs/support_rules_v0.json")
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    limits = process_config(rules, args.material, args.mode)
    directions = np.asarray(
        [row["direction"] for row in fibonacci_directions(args.directions)],
        dtype=np.float64,
    )
    jobs = build_jobs(root)
    validation_rows = (
        [row for row in jobs if row["source_project"] in set(args.validation_project)]
        if args.validation_project
        else []
    )
    if len(validation_rows) != args.validation_count:
        parser.error(
            f"validation selection produced {len(validation_rows)} parts; "
            f"expected {args.validation_count}"
        )
    validation_indices = {row["index"] for row in validation_rows}
    for row in jobs:
        row["split"] = "validation" if row["index"] in validation_indices else "train"

    output_dir = args.output_dir.resolve()
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    jobs.sort(key=lambda row: (-row["triangle_count"], row["job_id"]))
    pending = []
    completed = []
    for job in jobs:
        label_path = labels_dir / f"{job['job_id']}.npz"
        if label_path.exists() and not args.force:
            try:
                with np.load(label_path) as data:
                    if (
                        len(data["directions"]) == len(directions)
                        and data["features"].shape[1] == len(FEATURE_NAMES)
                    ):
                        completed.append({
                            **job,
                            "label_file": str(label_path),
                            "reused": True,
                        })
                        continue
            except Exception:
                pass
        pending.append((job, label_path))

    print(
        f"{len(jobs)} parts: {len(completed)} cached, {len(pending)} pending, "
        f"{args.jobs} workers, {len(directions)} directions",
        flush=True,
    )
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                score_one_part,
                job,
                directions,
                limits,
                args.projection_grid_mm,
                label_path,
            ): job["index"]
            for job, label_path in pending
        }
        for finished, future in enumerate(
            concurrent.futures.as_completed(futures), start=len(completed) + 1
        ):
            job_index = futures[future]
            try:
                result = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"part index {job_index} failed while building labels"
                ) from error
            completed.append(result)
            print(
                f"[{finished}/{len(jobs)}] part {result['source_part_id']} "
                f"{result['source_project']}: {result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    completed.sort(key=lambda row: row["index"])
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "material": args.material,
        "mode": args.mode,
        "rules_file": str(rules_path.resolve()),
        "limits": limits,
        "direction_count": len(directions),
        "projection_grid_mm": args.projection_grid_mm,
        "feature_names": FEATURE_NAMES,
        "target": "min-max normalised projected support-union area; higher score is better",
        "query_used_for_training": False,
        "part_count": len(completed),
        "train_count": sum(row["split"] == "train" for row in completed),
        "validation_count": sum(row["split"] == "validation" for row in completed),
        "records": completed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    split = {
        "method": "held-out source project; no project overlap between train and validation",
        "validation_projects": sorted(set(args.validation_project)),
        "train_part_indices": [row["index"] for row in completed if row["split"] == "train"],
        "validation_part_indices": [
            row["index"] for row in completed if row["split"] == "validation"
        ],
        "train_records": [row for row in completed if row["split"] == "train"],
        "validation_records": [row for row in completed if row["split"] == "validation"],
    }
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "part_count": manifest["part_count"],
        "train_count": manifest["train_count"],
        "validation_count": manifest["validation_count"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
