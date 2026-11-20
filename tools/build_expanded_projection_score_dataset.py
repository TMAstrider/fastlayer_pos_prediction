#!/usr/bin/env python3
"""Build 256-direction projection labels for the deduplicated 1032-part set."""
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
    from tools.build_projection_score_dataset import score_one_part
    from tools.generate_simplified_supports import process_config
    from tools.score_support_directions import fibonacci_directions
except ModuleNotFoundError:
    from build_projection_score_dataset import score_one_part
    from generate_simplified_supports import process_config
    from score_support_directions import fibonacci_directions


DEFAULT_SHAPES = Path("outputs/shape_deduplicated_pointclouds/manifest.json")
DEFAULT_NORMALS = Path("outputs/shape_deduplicated_normals")
DEFAULT_SPLIT = Path("runs/pointnet_shape_dedup_2026-11-09_val100/split.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_jobs(
    root: Path,
    shapes_path: Path,
    normals_dir: Path,
    split_path: Path,
) -> list[dict[str, Any]]:
    """Join canonical shape records to the existing 932/100 isolated split."""
    shapes = load_json(shapes_path)["records"]
    split = load_json(split_path)
    train_hashes = {
        record["geometry_sha256"] for record in split["train_records"]
    }
    validation_hashes = {
        record["geometry_sha256"] for record in split["validation_records"]
    }
    if train_hashes & validation_hashes:
        raise ValueError("Train and validation geometry hashes overlap")
    expected_hashes = train_hashes | validation_hashes
    shape_hashes = {record["geometry_sha256"] for record in shapes}
    if shape_hashes != expected_hashes:
        raise ValueError(
            "Shape manifest and split hashes differ: "
            f"{len(shape_hashes - expected_hashes)} only in shapes, "
            f"{len(expected_hashes - shape_hashes)} only in split"
        )

    jobs = []
    for index, shape in enumerate(shapes):
        geometry_hash = shape["geometry_sha256"]
        stl_path = root / shape["source_stl_file"]
        pointcloud_path = normals_dir / shape["pointcloud_file"]
        if not stl_path.is_file():
            raise FileNotFoundError(stl_path)
        if not pointcloud_path.is_file():
            raise FileNotFoundError(pointcloud_path)
        job_id = f"shape_{index:04d}__{geometry_hash[:12]}"
        jobs.append({
            "index": index,
            "job_id": job_id,
            "source_project": shape["source_project"],
            "source_part_id": index,
            "source_part_name": Path(shape["source_stl_file"]).name,
            "source_record_id": shape["source_record_id"],
            "shape_group_id": shape["shape_group_id"],
            "mesh_hash_sha256": geometry_hash,
            "triangle_count": int(shape["triangle_count"]),
            "mesh_path": str(stl_path.resolve()),
            "pointcloud_path": str(pointcloud_path.resolve()),
            "split": (
                "train" if geometry_hash in train_hashes else "validation"
            ),
        })
    return jobs


def cached_result(
    job: dict[str, Any],
    label_path: Path,
    direction_count: int,
) -> dict[str, Any] | None:
    if not label_path.is_file():
        return None
    try:
        with np.load(label_path) as data:
            required = {"directions", "projected_area_mm2", "points", "normals"}
            if not required.issubset(data.files):
                return None
            if len(data["directions"]) != direction_count:
                return None
            areas = np.asarray(data["projected_area_mm2"], dtype=np.float64)
            points = data["points"]
            normals = data["normals"]
            if points.shape != (4096, 3) or normals.shape != (4096, 3):
                return None
    except Exception:
        return None
    return {
        **job,
        "label_file": str(label_path),
        "direction_count": direction_count,
        "minimum_area_mm2": float(areas.min()),
        "maximum_area_mm2": float(areas.max()),
        "median_area_mm2": float(np.median(areas)),
        "elapsed_seconds": 0.0,
        "reused": True,
    }


def write_outputs(
    output_dir: Path,
    completed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    arguments: argparse.Namespace,
    split_path: Path,
    rules_path: Path,
    limits: dict[str, float],
    elapsed_seconds: float,
) -> None:
    completed.sort(key=lambda row: row["index"])
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete" if not failures else "incomplete",
        "dataset": "shape-deduplicated expanded industrial parts",
        "shapes_manifest": str(arguments.shapes.resolve()),
        "normals_dir": str(arguments.normals_dir.resolve()),
        "source_split": str(split_path.resolve()),
        "material": arguments.material,
        "mode": arguments.mode,
        "rules_file": str(rules_path),
        "limits": limits,
        "direction_count": arguments.directions,
        "projection_grid_mm": arguments.projection_grid_mm,
        "target": (
            "min-max normalised projected support-union area; "
            "higher score is better"
        ),
        "query_used_for_training": False,
        "part_count": len(completed),
        "train_count": sum(row["split"] == "train" for row in completed),
        "validation_count": sum(
            row["split"] == "validation" for row in completed
        ),
        "cache_reused_count": sum(bool(row.get("reused")) for row in completed),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "failures": failures,
        "records": completed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_split = load_json(split_path)
    split = {
        "method": source_split["method"],
        "seed": source_split.get("seed"),
        "validation_fraction": source_split.get("validation_fraction"),
        "validation_count": manifest["validation_count"],
        "requested_validation_count": source_split.get(
            "requested_validation_count"
        ),
        "train_part_indices": [
            row["index"] for row in completed if row["split"] == "train"
        ],
        "validation_part_indices": [
            row["index"] for row in completed if row["split"] == "validation"
        ],
        "train_records": [
            row for row in completed if row["split"] == "train"
        ],
        "validation_records": [
            row for row in completed if row["split"] == "validation"
        ],
        "source_split": str(split_path.resolve()),
        "limitation": source_split.get("limitation"),
    }
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--normals-dir", type=Path, default=DEFAULT_NORMALS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
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
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.directions < 2 or args.jobs < 1:
        parser.error("--directions must be at least 2 and --jobs must be positive")

    root = args.root.resolve()
    shapes_path = (root / args.shapes).resolve()
    normals_dir = (root / args.normals_dir).resolve()
    split_path = (root / args.split).resolve()
    rules_path = (
        root / (args.rules or Path("configs/support_rules_v0.json"))
    ).resolve()
    output_dir = (root / args.output_dir).resolve()
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    rules = load_json(rules_path)
    limits = process_config(rules, args.material, args.mode)
    directions = np.asarray(
        [row["direction"] for row in fibonacci_directions(args.directions)],
        dtype=np.float64,
    )
    jobs = build_jobs(root, shapes_path, normals_dir, split_path)
    jobs.sort(key=lambda row: (-row["triangle_count"], row["job_id"]))

    completed = []
    pending = []
    for job in jobs:
        label_path = labels_dir / f"{job['job_id']}.npz"
        result = None if args.force else cached_result(
            job,
            label_path,
            args.directions,
        )
        if result is None:
            pending.append((job, label_path))
        else:
            completed.append(result)
    print(
        f"{len(jobs)} parts: {len(completed)} cached, {len(pending)} pending, "
        f"{args.jobs} workers, {len(directions)} directions",
        flush=True,
    )

    started = time.monotonic()
    failures = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.jobs
    ) as executor:
        futures = {
            executor.submit(
                score_one_part,
                job,
                directions,
                limits,
                args.projection_grid_mm,
                label_path,
            ): job
            for job, label_path in pending
        }
        total = len(jobs)
        for finished, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=len(completed) + 1,
        ):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append({
                    "index": job["index"],
                    "shape_group_id": job["shape_group_id"],
                    "source_project": job["source_project"],
                    "error": f"{type(error).__name__}: {error}",
                })
                print(
                    f"[{finished}/{total}] FAILED shape {job['index']} "
                    f"{job['source_project']}: {error}",
                    flush=True,
                )
                continue
            completed.append(result)
            print(
                f"[{finished}/{total}] shape {result['index']} "
                f"{result['source_project']}: "
                f"{result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    elapsed_seconds = time.monotonic() - started
    write_outputs(
        output_dir,
        completed,
        failures,
        args,
        split_path,
        rules_path,
        limits,
        elapsed_seconds,
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "part_count": len(completed),
        "train_count": sum(row["split"] == "train" for row in completed),
        "validation_count": sum(
            row["split"] == "validation" for row in completed
        ),
        "failure_count": len(failures),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
