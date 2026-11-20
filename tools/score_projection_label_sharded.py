#!/usr/bin/env python3
"""Score one projection-label part by sharding its candidate directions."""
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
    from tools.build_expanded_projection_score_dataset import build_jobs
    from tools.build_projection_score_dataset import candidate_features
    from tools.generate_simplified_supports import (
        process_config,
        read_mesh,
    )
    from tools.recognize_part_features import recognize_features
    from tools.score_support_directions import fibonacci_directions
    from tools.support_area_scoring import (
        dense_area_ranks,
        evaluate_support_projection,
        normalise_projected_areas,
    )
except ModuleNotFoundError:
    from build_expanded_projection_score_dataset import build_jobs
    from build_projection_score_dataset import candidate_features
    from generate_simplified_supports import process_config, read_mesh
    from recognize_part_features import recognize_features
    from score_support_directions import fibonacci_directions
    from support_area_scoring import (
        dense_area_ranks,
        evaluate_support_projection,
        normalise_projected_areas,
    )


def score_direction_shard(payload: dict[str, Any]) -> dict[str, Any]:
    job = payload["job"]
    directions = payload["directions"]
    indices = payload["indices"]
    limits = payload["limits"]
    projection_grid_mm = payload["projection_grid_mm"]
    angle_only = bool(payload["angle_only"])
    vertices, faces = read_mesh(Path(job["mesh_path"]))
    recognized = None
    if not angle_only:
        recognized = recognize_features(
            vertices,
            faces,
            build_up=np.array([0.0, 0.0, 1.0]),
            small_hole_diameter_mm=limits[
                "horizontal_hole_diameter_threshold_mm"
            ],
        )
    count = len(indices)
    features = None if angle_only else np.empty((count, 18), dtype=np.float32)
    areas = np.empty(count, dtype=np.float64)
    retained_surface = np.empty(count, dtype=np.float64)
    exempt_surface = np.empty(count, dtype=np.float64)
    candidate_faces = np.empty(count, dtype=np.int32)
    small_hole_exemptions = np.empty(count, dtype=np.int16)
    for local, direction_index in enumerate(indices):
        direction = directions[direction_index]
        result = evaluate_support_projection(
            vertices,
            faces,
            limits,
            up=direction,
            recognized_features=recognized,
            projection_grid_mm=projection_grid_mm,
        )
        report = result["report"]
        areas[local] = report["areas_mm2"]["projected_union"]
        retained_surface[local] = report["areas_mm2"][
            "retained_overhang_surface"
        ]
        exempt_surface[local] = report["areas_mm2"]["feature_exempt"]
        candidate_faces[local] = report["face_counts"][
            "support_candidate_after_exemptions"
        ]
        summary = report["recognized_feature_summary"]
        small_hole_exemptions[local] = int(
            summary["small_horizontal_exemptions"] if summary else 0
        )
        if features is not None:
            features[local] = candidate_features(result, direction, limits)
    return {
        "indices": indices,
        "features": features,
        "areas": areas,
        "retained_surface": retained_surface,
        "exempt_surface": exempt_surface,
        "candidate_faces": candidate_faces,
        "small_hole_exemptions": small_hole_exemptions,
        "recognized_summary": (
            recognized["summary"] if recognized is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--shapes",
        type=Path,
        default=Path(
            "outputs/shape_deduplicated_pointclouds/manifest.json"
        ),
    )
    parser.add_argument(
        "--normals-dir",
        type=Path,
        default=Path("outputs/shape_deduplicated_normals"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(
            "runs/pointnet_shape_dedup_2026-11-09_val100/split.json"
        ),
    )
    parser.add_argument("--rules", type=Path)
    parser.add_argument(
        "--material",
        choices=["316L", "AlSi10Mg", "TC4"],
        default="AlSi10Mg",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "am_build"],
        default="standard",
    )
    parser.add_argument("--directions", type=int, default=256)
    parser.add_argument("--projection-grid-mm", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--angle-only", action="store_true")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--job-index", type=int)
    selector.add_argument("--shape-group-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    root = args.root.resolve()
    jobs = build_jobs(
        root,
        (root / args.shapes).resolve(),
        (root / args.normals_dir).resolve(),
        (root / args.split).resolve(),
    )
    if args.job_index is not None:
        selected = [
            job for job in jobs if job["index"] == args.job_index
        ]
    else:
        selected = [
            job
            for job in jobs
            if job["shape_group_id"] == args.shape_group_id
        ]
    if len(selected) != 1:
        raise ValueError(f"Selector matched {len(selected)} parts; expected one")
    job = selected[0]

    output_dir = (root / args.output_dir).resolve()
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_path = labels_dir / f"{job['job_id']}.npz"
    if output_path.exists() and not args.force:
        raise FileExistsError(output_path)

    rules_path = (
        root / (args.rules or Path("configs/support_rules_v0.json"))
    ).resolve()
    limits = process_config(
        json.loads(rules_path.read_text(encoding="utf-8")),
        args.material,
        args.mode,
    )
    directions = np.asarray(
        [row["direction"] for row in fibonacci_directions(args.directions)],
        dtype=np.float64,
    )
    index_chunks = [
        chunk.astype(np.int64)
        for chunk in np.array_split(
            np.arange(len(directions), dtype=np.int64),
            min(args.workers, len(directions)),
        )
        if len(chunk)
    ]
    started = time.monotonic()
    shards = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(index_chunks)
    ) as executor:
        futures = [
            executor.submit(
                score_direction_shard,
                {
                    "job": job,
                    "directions": directions,
                    "indices": indices,
                    "limits": limits,
                    "projection_grid_mm": args.projection_grid_mm,
                    "angle_only": args.angle_only,
                },
            )
            for indices in index_chunks
        ]
        for finished, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            shard = future.result()
            shards.append(shard)
            print(
                f"[{finished}/{len(futures)}] directions "
                f"{int(shard['indices'][0])}-{int(shard['indices'][-1])}",
                flush=True,
            )

    shards.sort(key=lambda shard: int(shard["indices"][0]))
    indices = np.concatenate([shard["indices"] for shard in shards])
    if not np.array_equal(indices, np.arange(len(directions))):
        raise RuntimeError("Direction shards overlap or miss candidates")
    features = (
        None
        if args.angle_only
        else np.concatenate([shard["features"] for shard in shards])
    )
    areas_raw = np.concatenate([shard["areas"] for shard in shards])
    retained_surface = np.concatenate(
        [shard["retained_surface"] for shard in shards]
    )
    exempt_surface = np.concatenate(
        [shard["exempt_surface"] for shard in shards]
    )
    candidate_faces = np.concatenate(
        [shard["candidate_faces"] for shard in shards]
    )
    small_hole_exemptions = np.concatenate(
        [shard["small_hole_exemptions"] for shard in shards]
    )
    scores, minimum, maximum = normalise_projected_areas(areas_raw)
    ranks = dense_area_ranks(areas_raw)
    with np.load(job["pointcloud_path"]) as point_data:
        points = np.asarray(point_data["points"], dtype=np.float32)
        normals = np.asarray(point_data["normals"], dtype=np.float32)
    payload = {
        "directions": directions.astype(np.float32),
        "projected_area_mm2": areas_raw.astype(np.float32),
        "score": scores.astype(np.float32),
        "rank": ranks.astype(np.int16),
        "retained_surface_mm2": retained_surface.astype(np.float32),
        "exempt_surface_mm2": exempt_surface.astype(np.float32),
        "candidate_face_count": candidate_faces.astype(np.int32),
        "small_hole_exemptions": small_hole_exemptions.astype(np.int16),
        "points": points,
        "normals": normals,
    }
    if features is not None:
        payload["features"] = features.astype(np.float32)
    np.savez_compressed(
        output_path,
        **payload,
    )
    print(json.dumps({
        "shape_index": job["index"],
        "shape_group_id": job["shape_group_id"],
        "workers": len(index_chunks),
        "angle_only": args.angle_only,
        "minimum_area_mm2": minimum,
        "maximum_area_mm2": maximum,
        "median_area_mm2": float(np.median(areas_raw)),
        "elapsed_seconds": time.monotonic() - started,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
