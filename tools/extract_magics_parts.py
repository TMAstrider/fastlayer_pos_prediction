"""Extract part meshes from Materialise Magics MT project archives.

The project header is encrypted, so this extractor uses the readable indexed
meshes and the known build-plate and support-box geometry shapes. It writes
binary STL in the project's stored build coordinates and records every source
member and exclusion in a manifest.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import zlib

import numpy as np

from audit_expanded_data import MTArchive
from export_meshes import safe_name, verify_stl, write_stl


COORDINATE_SCALE = 10000.0
PLATFORM_SIGNATURES = {
    (589, 1158): np.array([2.0, 293.0, 293.0]),
    (3482, 6944): np.array([0.2, 420.0, 420.0912]),
}
SUPPORT_BOX_SIZE = np.array([1.1, 20.0, 30.0])
PRIMARY_PLATE_XY_MIN_MM = 0.0
PRIMARY_PLATE_XY_MAX_MM = 420.0
PLATE_Z_TOLERANCE_MM = 0.5


def member_guid(member):
    match = re.fullmatch(r"Stl\{([^}]+)\}_vertices", member)
    if match is None:
        raise ValueError(f"Unexpected mesh member name: {member}")
    return match.group(1)


def classify_mesh(vertices, faces):
    dimensions = np.ptp(vertices, axis=0)
    signature = PLATFORM_SIGNATURES.get((len(vertices), len(faces)))
    if signature is not None and np.allclose(np.sort(dimensions), signature, atol=0.25):
        return "platform"
    if np.allclose(np.sort(dimensions), SUPPORT_BOX_SIZE, atol=0.25):
        return "support_box"
    return "part_candidate"


def replace_with_hardlink(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.samefile(source):
            return
        path.unlink()
    os.link(source, path)


def geometry_review_flags(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    xy_inside = bool(
        bbox_min[0] >= PRIMARY_PLATE_XY_MIN_MM
        and bbox_min[1] >= PRIMARY_PLATE_XY_MIN_MM
        and bbox_max[0] <= PRIMARY_PLATE_XY_MAX_MM
        and bbox_max[1] <= PRIMARY_PLATE_XY_MAX_MM
    )
    xy_overlap = bool(
        bbox_max[0] >= PRIMARY_PLATE_XY_MIN_MM
        and bbox_max[1] >= PRIMARY_PLATE_XY_MIN_MM
        and bbox_min[0] <= PRIMARY_PLATE_XY_MAX_MM
        and bbox_min[1] <= PRIMARY_PLATE_XY_MAX_MM
    )
    below_plate = bool(bbox_min[2] < -PLATE_Z_TOLERANCE_MM)
    review_reasons = []
    if not xy_inside:
        review_reasons.append(
            "xy_partially_outside_420mm_plate" if xy_overlap else "xy_outside_420mm_plate"
        )
    if below_plate:
        review_reasons.append("below_build_plate")
    return {
        "xy_inside_420mm_plate_bbox": xy_inside,
        "xy_overlaps_420mm_plate_bbox": xy_overlap,
        "z_min_mm": round(float(bbox_min[2]), 4),
        "z_max_mm": round(float(bbox_max[2]), 4),
        "below_build_plate": below_plate,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def write_manifest(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract(dataset, output, manifest_path, report_path):
    dataset_root = dataset.parent if dataset.is_file() else dataset
    projects = [dataset] if dataset.is_file() else sorted(dataset.rglob("*.magics"))
    if not projects:
        raise ValueError(f"No .magics projects found under {dataset}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    exclusions = []
    errors = []
    seen_pose_hashes = {}
    totals = Counter()
    summary_records = []

    for project_number, project in enumerate(projects, start=1):
        relative_project = project.relative_to(dataset_root).with_suffix("")
        destination = output / relative_project
        try:
            with MTArchive(project) as archive:
                mesh_members = sorted(
                    name for name in archive.entries
                    if name.endswith("_vertices") and name[:-9] + "_surfaces" in archive.entries
                )
                totals["projects"] += 1
                totals["stored_meshes"] += len(mesh_members)
                for member_number, member in enumerate(mesh_members, start=1):
                    surface_member = member[:-9] + "_surfaces"
                    vertex_bytes = archive.read(member)
                    surface_bytes = archive.read(surface_member)
                    if len(vertex_bytes) % 12 or len(surface_bytes) % 12:
                        raise ValueError(f"Unaligned mesh data: {member}")
                    vertices_i32 = np.frombuffer(vertex_bytes, dtype="<i4").reshape(-1, 3)
                    faces = np.frombuffer(surface_bytes, dtype="<u4").reshape(-1, 3)
                    if not len(vertices_i32) or not len(faces):
                        raise ValueError(f"Empty mesh data: {member}")
                    if int(faces.max()) >= len(vertices_i32):
                        raise ValueError(f"Face index outside vertices: {member}")
                    vertices = vertices_i32.astype(np.float32) / np.float32(COORDINATE_SCALE)
                    role = classify_mesh(vertices, faces)
                    geometry_hash = hashlib.sha256(vertex_bytes + surface_bytes).hexdigest()
                    common = {
                        "source_project": project.as_posix(),
                        "project_relative": relative_project.as_posix(),
                        "member": member,
                        "surface_member": surface_member,
                        "geometry_sha256": geometry_hash,
                        "vertices": int(len(vertices)),
                        "triangles": int(len(faces)),
                        "bbox_min_mm": vertices.min(axis=0).astype(float).round(4).tolist(),
                        "bbox_max_mm": vertices.max(axis=0).astype(float).round(4).tolist(),
                        "bbox_size_mm": np.ptp(vertices, axis=0).astype(float).round(4).tolist(),
                    }
                    flags = geometry_review_flags(vertices)
                    if role != "part_candidate":
                        exclusions.append({**common, **flags, "reason": role})
                        totals[role] += 1
                        continue
                    duplicate = seen_pose_hashes.get(geometry_hash)
                    guid = member_guid(member)
                    filename = f"part_{member_number:03d}__{safe_name(guid)}.stl"
                    relative_output = (relative_project / filename).as_posix()
                    stl_path = output / relative_output
                    stl_path.parent.mkdir(parents=True, exist_ok=True)
                    if duplicate is None:
                        temporary = stl_path.with_suffix(".stl.tmp")
                        write_stl(temporary, vertices, faces)
                        if stl_path.exists():
                            stl_path.unlink()
                        temporary.replace(stl_path)
                        verify_stl(stl_path, vertices, faces)
                        totals["unique_part_geometries"] += 1
                    else:
                        duplicate_source = output / duplicate["stl_file"]
                        replace_with_hardlink(stl_path, duplicate_source)
                        if stl_path.stat().st_size != 84 + 50 * len(faces):
                            raise ValueError(f"Hardlinked STL size mismatch: {stl_path}")
                        totals["duplicate_part_records"] += 1
                    record = {
                        **common,
                        **flags,
                        "stl_file": relative_output,
                        "status": "part_candidate",
                        "is_unique_pose_geometry": duplicate is None,
                        "training_geometry_eligible": (
                            duplicate is None
                            and flags["xy_inside_420mm_plate_bbox"]
                            and not flags["below_build_plate"]
                        ),
                        "duplicate_of_project": (
                            duplicate["project_relative"] if duplicate is not None else None
                        ),
                        "duplicate_of_member": (
                            duplicate["member"] if duplicate is not None else None
                        ),
                        "duplicate_of_stl_file": (
                            duplicate["stl_file"] if duplicate is not None else None
                        ),
                        "up_axis_in_stored_mesh_frame": [0.0, 0.0, 1.0],
                        "yaw_unconstrained": True,
                        "pose_frame": "stored Magics build coordinates",
                        "classification_evidence": "non-platform and non-support-box indexed STL member",
                        "storage_mode": "hardlink" if duplicate is not None else "regular_file",
                        "geometry_verified_against_source_triangles": True,
                    }
                    records.append(record)
                    if duplicate is None:
                        seen_pose_hashes[geometry_hash] = record
                    totals["part_candidate_records"] += 1
        except (OSError, ValueError, KeyError, struct.error, zlib.error) as error:
            errors.append({"source_project": project.as_posix(), "error": str(error)})

        if project_number % 10 == 0 or project_number == len(projects):
            summary = {
                "magics_projects_scanned": project_number,
                "magics_projects_total": len(projects),
                "part_candidate_records": totals["part_candidate_records"],
                "unique_part_geometries": totals["unique_part_geometries"],
                "duplicate_part_records": totals["duplicate_part_records"],
                "platforms_excluded": totals["platform"],
                "support_boxes_excluded": totals["support_box"],
                "errors": len(errors),
            }
            write_manifest(manifest_path, {
                "status": "in_progress",
                "dataset": dataset.resolve().as_posix(),
                "output": output.resolve().as_posix(),
                "summary": summary,
                "pose_label": {
                    "up_axis": [0.0, 0.0, 1.0],
                    "yaw_unconstrained": True,
                    "frame": "stored Magics build coordinates",
                },
                "records": records,
                "exclusions": exclusions,
                "errors": errors,
            })
            print(
                f"{project_number}/{len(projects)} projects, "
                f"{totals['part_candidate_records']} part records, "
                f"{totals['unique_part_geometries']} unique geometries, {len(errors)} errors",
                flush=True,
            )

    project_record_counts = Counter(record["project_relative"] for record in records)
    summary_records = [{
        "project_relative": project.relative_to(dataset_root).with_suffix("").as_posix(),
        "part_candidate_records": project_record_counts[
            project.relative_to(dataset_root).with_suffix("").as_posix()
        ],
    } for project in projects]

    summary = {
        "magics_projects_scanned": len(projects),
        "stored_meshes": totals["stored_meshes"],
        "part_candidate_records": totals["part_candidate_records"],
        "unique_part_geometries": totals["unique_part_geometries"],
        "duplicate_part_records": totals["duplicate_part_records"],
        "platforms_excluded": totals["platform"],
        "support_boxes_excluded": totals["support_box"],
        "projects_with_part_records": sum(item["part_candidate_records"] > 0 for item in summary_records),
        "projects_without_part_records": sum(item["part_candidate_records"] == 0 for item in summary_records),
        "records_inside_420mm_plate_bbox": sum(
            record["xy_inside_420mm_plate_bbox"] for record in records
        ),
        "records_needing_review": sum(record["needs_review"] for record in records),
        "errors": len(errors),
    }
    document = {
        "status": "complete_with_errors" if errors else "complete",
        "dataset": dataset.resolve().as_posix(),
        "output": output.resolve().as_posix(),
        "summary": summary,
        "format": "Binary STL; indexed source triangles expanded without simplification.",
        "units": "millimetres",
        "pose_label": {
            "up_axis": [0.0, 0.0, 1.0],
            "yaw_unconstrained": True,
            "frame": "stored Magics build coordinates",
            "evidence": "Known build-plate meshes are horizontal in stored coordinates; part minima are above the plate.",
        },
        "classification": {
            "platform": "Exact 293x293x2 or 420x420.09x0.2 indexed-mesh signatures",
            "support_box": "30x20x1.1 mm box-like geometry",
            "part_candidate": "Readable STL member not matching either auxiliary shape",
            "duplicate": "Identical vertices, triangles and stored translation; retained as a hardlink",
            "review": "Candidate lies outside the 420 mm plate XY bbox or below the build plate",
        },
        "limitations": [
            "The project header is encrypted, so vendor object names are not available.",
            "Part candidates should be spot-checked against Magics before training.",
            "Yaw around the build Z axis is treated as unconstrained.",
            "Training eligibility is a conservative geometry-only filter; hidden and inactive objects still require review.",
        ],
        "projects": summary_records,
        "records": records,
        "exclusions": exclusions,
        "errors": errors,
    }
    write_manifest(manifest_path, document)
    report_lines = [
        "# Magics extraction",
        "",
        f"- Projects: {summary['magics_projects_scanned']}",
        f"- Stored mesh objects: {summary['stored_meshes']}",
        f"- Part candidate records: {summary['part_candidate_records']}",
        f"- Unique pose geometries: {summary['unique_part_geometries']}",
        f"- Duplicate records (hardlinked): {summary['duplicate_part_records']}",
        f"- Excluded platforms: {summary['platforms_excluded']}",
        f"- Excluded support boxes: {summary['support_boxes_excluded']}",
        f"- Projects with part records: {summary['projects_with_part_records']}",
        f"- Records inside the 420 mm plate bbox: {summary['records_inside_420mm_plate_bbox']}",
        f"- Records marked for review: {summary['records_needing_review']}",
        f"- Errors: {summary['errors']}",
        "",
        f"STL output: `{output.resolve().as_posix()}`",
        "",
        f"Manifest: `{manifest_path.resolve().as_posix()}`",
        "",
        "The stored Magics Z axis is used as the upward direction. Yaw around Z is unconstrained.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/magics_parts_stl"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/magics_parts_stl/manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/magics_extraction_2026-11-09.md"),
    )
    args = parser.parse_args()
    extract(args.dataset.resolve(), args.output.resolve(), args.manifest.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
