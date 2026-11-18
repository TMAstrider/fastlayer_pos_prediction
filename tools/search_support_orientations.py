#!/usr/bin/env python3
"""Search build directions using simplified supports and a user no-support query."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.generate_simplified_supports import (
        DEFAULT_RULES,
        evaluate_supports,
        process_config,
        read_mesh,
        render_preview,
        rotation_align_to_z,
    )
    from tools.recognize_part_features import (
        merge_annotations,
        recognize_features,
        reclassify_features_for_build_direction,
        resolve_support_query,
    )
except ModuleNotFoundError:
    from generate_simplified_supports import (
        DEFAULT_RULES,
        evaluate_supports,
        process_config,
        read_mesh,
        render_preview,
        rotation_align_to_z,
    )
    from recognize_part_features import (
        merge_annotations,
        recognize_features,
        reclassify_features_for_build_direction,
        resolve_support_query,
    )


def default_directions() -> list[dict[str, Any]]:
    directions = []
    for axis, label in zip(np.eye(3), ["X", "Y", "Z"]):
        directions.extend([
            {"label": f"+{label}", "direction": axis.tolist(), "group": "axis"},
            {"label": f"-{label}", "direction": (-axis).tolist(), "group": "axis"},
        ])
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                vector = np.asarray([x, y, z], dtype=np.float64)
                vector /= np.linalg.norm(vector)
                directions.append({
                    "label": f"{x:+.0f},{y:+.0f},{z:+.0f}",
                    "direction": vector.tolist(),
                    "group": "cube_diagonal",
                })
    return directions


def fibonacci_sphere(count: int) -> list[dict[str, Any]]:
    directions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * index
        vector = np.array([radius * math.cos(theta), radius * math.sin(theta), z])
        directions.append({
            "label": f"S{index + 1:03d}",
            "direction": vector.tolist(),
            "group": "fibonacci_sphere",
        })
    return directions


def deduplicate_directions(directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen: set[tuple[float, float, float]] = set()
    for item in directions:
        vector = np.asarray(item["direction"], dtype=np.float64)
        vector /= np.linalg.norm(vector)
        key = tuple(np.round(vector, 8).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append({**item, "direction": vector.tolist()})
    return unique


def violation_counts(violations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for violation in violations:
        kind = str(violation["type"])
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def ranking_key(row: dict[str, Any]) -> tuple[bool, int, float, int]:
    """Rank feasible directions first, then minimize hard violations before cost."""
    feasible = bool(row["feasible"])
    return (
        not feasible,
        0 if feasible else int(row["violation_count"]),
        float(row["support_volume_mm3"]),
        int(row["support_column_count"]),
    )


def write_search_chart(path: Path, records: list[dict[str, Any]], current_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(records, key=ranking_key)
    labels = [
        row["direction_label"] + (" [current]" if row["direction_label"] == current_label else "")
        for row in ordered
    ]
    values = [row["support_volume_mm3"] for row in ordered]
    colors = ["#23834c" if row["feasible"] else "#d15445" for row in ordered]
    fig, ax = plt.subplots(figsize=(10, max(5.2, 0.36 * len(records) + 1.8)))
    bars = ax.barh(labels, values, color=colors, alpha=0.86)
    for bar, row in zip(bars, ordered):
        suffix = (
            f"{row['support_column_count']} columns"
            if row["feasible"]
            else f"{row['violation_count']} violations"
        )
        ax.text(
            bar.get_width() + max(max(values) * 0.01, 0.5),
            bar.get_y() + bar.get_height() / 2,
            suffix,
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Support volume proxy (mm3)")
    ax.set_title("Query-aware orientation search; green = feasible, red = rejected")
    ax.grid(axis="x", alpha=0.22)
    ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="#f5f7fa")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--material", required=True, choices=["316L", "AlSi10Mg", "TC4"])
    parser.add_argument("--mode", required=True, choices=["standard", "am_build"])
    parser.add_argument("--support-query", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--directions-json", type=Path,
                        help="optional JSON object with a directions array")
    parser.add_argument("--sphere-samples", type=int, default=0,
                        help="append this many Fibonacci-sphere directions to the 14 axis/diagonal seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lift-mm", type=float, default=4.0)
    parser.add_argument("--sample-spacing-mm", type=float, default=1.5)
    parser.add_argument("--support-radius-mm", type=float, default=0.4)
    parser.add_argument("--protected-clearance-mm", type=float, default=0.2)
    args = parser.parse_args()
    if args.lift_mm < 0:
        parser.error("lift-mm must be nonnegative")

    vertices, faces = read_mesh(args.mesh)
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    limits = process_config(rules, args.material, args.mode)
    annotations = (
        json.loads(args.annotations.read_text(encoding="utf-8"))
        if args.annotations else {}
    )
    support_query = json.loads(args.support_query.read_text(encoding="utf-8"))
    if args.directions_json:
        directions_data = json.loads(args.directions_json.read_text(encoding="utf-8"))
        directions = directions_data["directions"]
    else:
        directions = default_directions()
        if args.sphere_samples < 0:
            parser.error("sphere-samples must be nonnegative")
        if args.sphere_samples:
            directions = deduplicate_directions(
                directions + fibonacci_sphere(args.sphere_samples)
            )

    # Cylinder fits are orientation invariant. Refit only once, then recompute
    # horizontal small-hole exemptions for each candidate build direction.
    base_features = recognize_features(
        vertices,
        faces,
        build_up=np.array([0.0, 0.0, 1.0]),
        min_confidence=0.58,
        horizontal_tolerance_deg=10.0,
        small_hole_diameter_mm=limits["horizontal_hole_diameter_threshold_mm"],
    )
    current_label = "+Z"

    records: list[dict[str, Any]] = []
    results_by_label: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(directions, start=1):
        label = str(item.get("label", f"direction_{index:03d}"))
        up = np.asarray(item["direction"], dtype=np.float64)
        up /= np.linalg.norm(up)
        oriented_features = reclassify_features_for_build_direction(
            base_features,
            up,
            limits["horizontal_hole_diameter_threshold_mm"],
            horizontal_tolerance_deg=10.0,
        )
        merged_annotations = merge_annotations(annotations, oriented_features, len(faces))
        resolved_query = resolve_support_query(oriented_features, support_query, len(faces))
        merged_annotations["no_support_face_indices"] = sorted(
            set(merged_annotations.get("no_support_face_indices", []))
            | set(resolved_query["no_support_face_indices"])
        )
        clearance = (
            resolved_query["clearance_mm"]
            if resolved_query["clearance_mm"] is not None
            else args.protected_clearance_mm
        )
        rotation = rotation_align_to_z(up)
        minimum_input_z = float((vertices @ rotation.T)[:, 2].min())
        platform_z = minimum_input_z - args.lift_mm
        result = evaluate_supports(
            vertices,
            faces,
            limits,
            merged_annotations,
            up,
            args.sample_spacing_mm,
            0.05,
            250_000,
            args.support_radius_mm,
            platform_z,
            oriented_features,
            clearance,
            resolved_query,
        )
        report = result["report"]
        violation_summary = violation_counts(report["hard_constraint_violations"])
        record = {
            "direction_label": label,
            "direction_group": item.get("group"),
            "build_up_input": up.tolist(),
            "feasible": bool(report["is_feasible_under_supplied_constraints"]),
            "violation_count": len(report["hard_constraint_violations"]),
            "violation_types": violation_summary,
            "support_column_count": report["support_summary"]["count"],
            "support_volume_mm3": report["support_summary"]["column_volume_proxy_mm3"],
            "build_plate_landings": report["support_summary"]["build_plate_landings"],
            "part_surface_landings": report["support_summary"]["part_surface_landings"],
            "mean_support_height_mm": report["support_summary"]["mean_height_mm"],
            "max_support_height_mm": report["support_summary"]["max_height_mm"],
            "minimum_part_height_aligned_mm": minimum_input_z,
            "platform_height_aligned_mm": platform_z,
            "lift_mm": args.lift_mm,
            "candidate_overhang_area_mm2": report["areas_mm2"]["support_candidate_after_exemptions"],
            "no_support_face_count": report["face_counts"]["no_support_faces"],
            "small_horizontal_exemptions": oriented_features["summary"]["small_horizontal_exemptions"],
            "selected_query_feature_ids": resolved_query["selected_feature_ids"],
        }
        records.append(record)
        results_by_label[label] = result
        print(
            f"[{index}/{len(directions)}] {label}: "
            f"{'feasible' if record['feasible'] else 'rejected'}, "
            f"{record['support_column_count']} columns, "
            f"{record['violation_count']} violations"
        )

    ranked = sorted(records, key=ranking_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    current = next((row for row in records if row["direction_label"] == current_label), None)
    best = ranked[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "mesh": str(args.mesh.resolve()),
        "material": args.material,
        "mode": args.mode,
        "support_query": support_query,
        "base_feature_summary": base_features["summary"],
        "candidate_count": len(records),
        "feasible_count": sum(row["feasible"] for row in records),
        "current_direction": current,
        "best_direction": best,
        "ranking": ranked,
    }
    report_path = args.output_dir / "orientation_search.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.output_dir / "orientation_search.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [
            "rank", "direction_label", "build_up_x", "build_up_y", "build_up_z",
            "feasible", "violation_count", "support_column_count",
            "support_volume_mm3", "candidate_overhang_area_mm2",
            "small_horizontal_exemptions", "no_support_face_count",
            "platform_height_aligned_mm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({
                "rank": row["rank"],
                "direction_label": row["direction_label"],
                "build_up_x": row["build_up_input"][0],
                "build_up_y": row["build_up_input"][1],
                "build_up_z": row["build_up_input"][2],
                "feasible": row["feasible"],
                "violation_count": row["violation_count"],
                "support_column_count": row["support_column_count"],
                "support_volume_mm3": row["support_volume_mm3"],
                "candidate_overhang_area_mm2": row["candidate_overhang_area_mm2"],
                "small_horizontal_exemptions": row["small_horizontal_exemptions"],
                "no_support_face_count": row["no_support_face_count"],
                "platform_height_aligned_mm": row["platform_height_aligned_mm"],
            })

    write_search_chart(args.output_dir / "orientation_search.png", records, current_label)
    if best["feasible"]:
        render_preview(
            args.output_dir / "best_orientation_support.png",
            results_by_label[best["direction_label"]],
        )
    print(json.dumps({
        "report": str(report_path.resolve()),
        "candidate_count": len(records),
        "feasible_count": report["feasible_count"],
        "current": current,
        "best": best,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
