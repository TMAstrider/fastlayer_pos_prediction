#!/usr/bin/env python3
"""Build leak-resistant 30-degree geometric prior features for direction scoring."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.spatial import ConvexHull, QhullError

try:
    from tools.generate_simplified_supports import (
        classify_overhang_faces,
        read_mesh,
        rotation_align_to_z,
        triangle_geometry,
    )
    from tools.support_area_scoring import projected_triangle_area_sum
except ModuleNotFoundError:
    from generate_simplified_supports import (
        classify_overhang_faces,
        read_mesh,
        rotation_align_to_z,
        triangle_geometry,
    )
    from support_area_scoring import projected_triangle_area_sum


FEATURE_VERSION = "projection_prior_30deg_v1"
DEFAULT_DATASET = Path(
    "labels/projection_support_area_expanded_932_100_v0_256"
)
DEFAULT_OUTPUT = Path("labels/geometry_prior_30deg_expanded_932_100_v1")

CANDIDATE_FEATURE_NAMES = [
    "overhang_area_mm2",
    "overhang_area_fraction",
    "overhang_candidate_face_fraction",
    "overhang_angle_mean_deg",
    "overhang_angle_std_deg",
    "overhang_angle_min_deg",
    "overhang_angle_max_deg",
    "projected_support_area_mm2",
    "projected_support_over_surface",
    "projected_support_over_radius_sq",
    "projected_convex_hull_area_mm2",
    "projected_hull_over_radius_sq",
    "projected_support_over_projected_hull",
    "build_height_over_part_diameter",
]

GLOBAL_FEATURE_NAMES = [
    "log_surface_area_mm2",
    "log_vertex_radius_mm",
    "surface_area_over_radius_sq",
    "pca_axis_fraction_0",
    "pca_axis_fraction_1",
    "pca_axis_fraction_2",
    "unique_vertex_radius_std_over_radius",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_file(directory: Path, record: dict[str, Any]) -> Path:
    return directory / f"{record['job_id']}.npz"


def compute_global_features(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Return scale-aware and rotation-invariant whole-part descriptors."""
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    triangle_areas = 0.5 * np.linalg.norm(cross, axis=1)
    surface_area_mm2 = float(triangle_areas.sum())

    unique_vertices = np.unique(vertices, axis=0)
    centered = unique_vertices - unique_vertices.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    radius_mm = max(float(radii.max()), 1e-9)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    eigenvalue_sum = max(float(eigenvalues.sum()), 1e-12)
    eigenvalue_fractions = eigenvalues / eigenvalue_sum

    return np.asarray([
        math.log1p(surface_area_mm2),
        math.log1p(radius_mm),
        surface_area_mm2 / radius_mm**2,
        *eigenvalue_fractions.tolist(),
        float(radii.std()) / radius_mm,
    ], dtype=np.float32)


def projected_convex_hull_area(points_xy: np.ndarray) -> float:
    """Return a yaw-invariant projected silhouette upper bound."""
    try:
        return float(ConvexHull(points_xy).volume)
    except QhullError:
        return 0.0


def compute_candidate_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    directions: np.ndarray,
    global_features: np.ndarray,
    angle_threshold_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 30-degree-only per-direction features and their raw area.

    The 45-degree target arrays from the base dataset are never read here.
    """
    direction_count = len(directions)
    candidate_features = np.zeros(
        (direction_count, len(CANDIDATE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    projected_area_mm2 = np.zeros(direction_count, dtype=np.float32)
    radius_mm = max(math.expm1(float(global_features[1])), 1e-9)

    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    surface_area_mm2 = float(
        (0.5 * np.linalg.norm(cross, axis=1)).sum()
    )
    unique_vertices = np.unique(vertices, axis=0)
    try:
        hull_vertices = unique_vertices[
            ConvexHull(unique_vertices).vertices
        ]
    except QhullError:
        hull_vertices = unique_vertices

    for index, direction in enumerate(directions):
        rotation = rotation_align_to_z(direction)
        aligned_vertices = vertices @ rotation.T
        aligned_hull_vertices = hull_vertices @ rotation.T
        triangles_z, normals_z, triangle_areas = triangle_geometry(
            aligned_vertices,
            faces,
        )
        _, candidate_faces, face_angles = (
            classify_overhang_faces(
                normals_z,
                triangle_areas,
                angle_threshold_deg,
            )
        )
        candidate_indices = np.flatnonzero(candidate_faces)
        candidate_weights = triangle_areas[candidate_indices]
        candidate_angles = face_angles[candidate_indices]
        overhang_area_mm2 = float(candidate_weights.sum())
        projected_support_area = projected_triangle_area_sum(
            triangles_z,
            candidate_indices,
        )
        projected_area_mm2[index] = projected_support_area

        if len(candidate_indices):
            angle_probability = candidate_weights / max(
                overhang_area_mm2,
                1e-12,
            )
            angle_mean = float(np.sum(candidate_angles * angle_probability))
            angle_variance = float(np.sum(
                (candidate_angles - angle_mean) ** 2 * angle_probability
            ))
            angle_std = math.sqrt(max(angle_variance, 0.0))
            angle_min = float(candidate_angles.min())
            angle_max = float(candidate_angles.max())
        else:
            angle_mean = angle_std = angle_min = angle_max = 0.0

        projected_hull_area = projected_convex_hull_area(
            aligned_hull_vertices[:, :2]
        )
        build_height = float(
            aligned_vertices[:, 2].max() - aligned_vertices[:, 2].min()
        )
        candidate_features[index] = np.asarray([
            overhang_area_mm2,
            overhang_area_mm2 / max(surface_area_mm2, 1e-12),
            len(candidate_indices) / max(len(faces), 1),
            angle_mean,
            angle_std,
            angle_min,
            angle_max,
            projected_support_area,
            projected_support_area / max(surface_area_mm2, 1e-12),
            projected_support_area / radius_mm**2,
            projected_hull_area,
            projected_hull_area / radius_mm**2,
            projected_support_area / max(projected_hull_area, 1e-12),
            build_height / max(2.0 * radius_mm, 1e-12),
        ], dtype=np.float32)

    return candidate_features, projected_area_mm2


def cached_payload(
    path: Path,
    directions: np.ndarray,
    angle_threshold_deg: float,
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            if str(data["feature_version"].item()) != FEATURE_VERSION:
                return None
            if not math.isclose(
                float(data["angle_threshold_deg"].item()),
                angle_threshold_deg,
            ):
                return None
            if not np.allclose(data["directions"], directions, atol=1e-7):
                return None
            if data["candidate_features"].shape != (
                len(directions),
                len(CANDIDATE_FEATURE_NAMES),
            ):
                return None
            if data["global_features"].shape != (
                len(GLOBAL_FEATURE_NAMES),
            ):
                return None
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception:
        return None


def build_one(
    record: dict[str, Any],
    output_path: Path,
    angle_threshold_deg: float,
    force: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    with np.load(record["label_file"]) as label:
        directions = np.asarray(label["directions"], dtype=np.float64)
    cached = None if force else cached_payload(
        output_path,
        directions,
        angle_threshold_deg,
    )
    if cached is not None:
        projected_area = cached["projected_area_mm2_30"]
        return {
            **record,
            "geometry_feature_file": str(output_path),
            "projected_area_30_min_mm2": float(projected_area.min()),
            "projected_area_30_max_mm2": float(projected_area.max()),
            "projected_area_30_median_mm2": float(np.median(projected_area)),
            "elapsed_seconds": 0.0,
            "reused": True,
        }

    vertices, faces = read_mesh(Path(record["mesh_path"]))
    global_features = compute_global_features(vertices, faces)
    candidate_features, projected_area = compute_candidate_features(
        vertices,
        faces,
        directions,
        global_features,
        angle_threshold_deg,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        feature_version=np.asarray(FEATURE_VERSION),
        source_label_file=np.asarray(record["label_file"]),
        source_mesh_sha256=np.asarray(record["mesh_hash_sha256"]),
        angle_threshold_deg=np.asarray(angle_threshold_deg, dtype=np.float32),
        directions=directions.astype(np.float32),
        candidate_features=candidate_features,
        candidate_feature_names=np.asarray(CANDIDATE_FEATURE_NAMES),
        global_features=global_features,
        global_feature_names=np.asarray(GLOBAL_FEATURE_NAMES),
        projected_area_mm2_30=projected_area,
    )
    return {
        **record,
        "geometry_feature_file": str(output_path),
        "projected_area_30_min_mm2": float(projected_area.min()),
        "projected_area_30_max_mm2": float(projected_area.max()),
        "projected_area_30_median_mm2": float(np.median(projected_area)),
        "elapsed_seconds": time.monotonic() - started,
        "reused": False,
    }


def write_manifest(
    output_dir: Path,
    records: list[dict[str, Any]],
    expected_count: int,
    failures: list[dict[str, Any]],
    arguments: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    records.sort(key=lambda record: record["index"])
    train_count = sum(record["split"] == "train" for record in records)
    validation_count = sum(
        record["split"] == "validation" for record in records
    )
    complete = len(records) == expected_count and not failures
    manifest = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete" if complete else "incomplete",
        "source_dataset": str(arguments.dataset.resolve()),
        "angle_threshold_deg": arguments.angle_threshold_deg,
        "small_hole_exemptions_used": False,
        "target_values_used_as_inputs": False,
        "candidate_feature_names": CANDIDATE_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "part_count": len(records),
        "expected_part_count": expected_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "reused_count": sum(bool(record.get("reused")) for record in records),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "failures": failures,
        "records": records,
        "limitations": [
            "The 30-degree prior is computed independently from the 45-degree target.",
            "The 30-degree prior intentionally applies no small-hole or feature exemptions.",
            "Projected area is the rotation-invariant sum of projected candidate triangles, not a support solid or Magics output.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angle-threshold-deg", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=88)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.angle_threshold_deg < 0 or args.angle_threshold_deg >= 90:
        parser.error("--angle-threshold-deg must be in [0, 90)")
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    root = args.root.resolve()
    dataset_dir = (root / args.dataset).resolve()
    output_dir = (root / args.output_dir).resolve()
    split = load_json(dataset_dir / "split.json")
    source_records = [
        *split["train_records"],
        *split["validation_records"],
    ]
    if args.limit:
        source_records = sorted(
            source_records,
            key=lambda record: (
                -int(record["triangle_count"]),
                record["job_id"],
            ),
        )[:args.limit]
    expected_count = (
        len(split["train_records"]) + len(split["validation_records"])
        if not args.limit
        else args.limit
    )
    if len(source_records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} source records, got {len(source_records)}"
        )

    started = time.monotonic()
    failures: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                build_one,
                record,
                feature_file(output_dir, record),
                args.angle_threshold_deg,
                args.force,
            ): record
            for record in source_records
        }
        for finished, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append({
                    "index": record["index"],
                    "job_id": record["job_id"],
                    "source_project": record["source_project"],
                    "error": f"{type(error).__name__}: {error}",
                })
                print(
                    f"[{finished}/{len(source_records)}] FAILED "
                    f"{record['job_id']}: {error}",
                    flush=True,
                )
                continue
            completed.append(result)
            status = (
                "cached"
                if result["reused"]
                else f"{result['elapsed_seconds']:.1f}s"
            )
            print(
                f"[{finished}/{len(source_records)}] "
                f"shape {result['index']} {status}",
                flush=True,
            )

    write_manifest(
        output_dir,
        completed,
        expected_count,
        failures,
        args,
        time.monotonic() - started,
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "part_count": len(completed),
        "expected_part_count": expected_count,
        "failure_count": len(failures),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
