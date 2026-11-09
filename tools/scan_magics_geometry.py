"""Summarize readable STL-like geometry members inside Materialise MT projects.

This deliberately records candidate meshes only. An encrypted project header is
needed to prove object role and project-level transforms; this scan does not
claim that every non-platform mesh is a labelled training part.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np

from audit_expanded_data import MTArchive


def scan_project(path: Path):
    result = {"project": path.as_posix(), "bytes": path.stat().st_size, "meshes": []}
    seen = {}
    with MTArchive(path) as archive:
        for name in sorted(n for n in archive.entries if n.endswith("_vertices")):
            surface_name = name[:-9] + "_surfaces"
            if surface_name not in archive.entries:
                result["meshes"].append({"member": name, "error": "missing paired surface member"})
                continue
            vertex_bytes = archive.read(name)
            surface_bytes = archive.read(surface_name)
            if len(vertex_bytes) % 12 or len(surface_bytes) % 12:
                result["meshes"].append({"member": name, "error": "unaligned mesh data"})
                continue
            vertices_i = np.frombuffer(vertex_bytes, dtype="<i4").reshape(-1, 3)
            faces = np.frombuffer(surface_bytes, dtype="<u4").reshape(-1, 3)
            if not len(vertices_i) or not len(faces):
                result["meshes"].append({"member": name, "error": "empty mesh"})
                continue
            lo_i = vertices_i.min(axis=0)
            hi_i = vertices_i.max(axis=0)
            dimensions = (hi_i - lo_i).astype(np.float64) / 10000.0
            sorted_dimensions = np.sort(dimensions)
            # The known printer plates are broad, thin XY slabs. This is only a
            # geometric filter, not a substitute for project object metadata.
            plate_candidate = bool(sorted_dimensions[0] <= 2.0 and sorted_dimensions[1] >= 200.0 and sorted_dimensions[2] >= 200.0)
            digest = hashlib.sha256(vertex_bytes + surface_bytes).hexdigest()
            digest_owner = seen.get(digest)
            if digest_owner is None:
                seen[digest] = name
            result["meshes"].append({
                "member": name,
                "surface_member": surface_name,
                "vertices": int(len(vertices_i)),
                "triangles": int(len(faces)),
                "bbox_min_mm": (lo_i.astype(np.float64) / 10000.0).round(4).tolist(),
                "bbox_max_mm": (hi_i.astype(np.float64) / 10000.0).round(4).tolist(),
                "bbox_size_mm": dimensions.round(4).tolist(),
                "xy_inside_420mm_plate_bbox": bool(lo_i[0] >= -10000 and lo_i[1] >= -10000 and hi_i[0] <= 4300000 and hi_i[1] <= 4300000),
                "plate_geometry_candidate": plate_candidate,
                "duplicate_geometry_of": digest_owner,
                "faces_in_range": bool(int(faces.max()) < len(vertices_i)),
                "geometry_sha256": digest,
                "role": "unknown (part, platform, fixture, duplicate or inactive object)",
                "pose_label": "not assigned; project metadata/transform not yet verified",
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("reports/magics_mesh_geometry_audit_2026-11-09.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    files = sorted(args.dataset.rglob("*.magics"))
    if args.workers < 1:
        parser.error("--workers must be positive")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        projects = list(pool.map(scan_project, files))
    meshes = [mesh for project in projects for mesh in project["meshes"]]
    summary = {
        "magics_projects": len(projects),
        "stored_mesh_objects": len(meshes),
        "geometry_errors": sum("error" in mesh for mesh in meshes),
        "plate_geometry_candidates": sum(mesh.get("plate_geometry_candidate", False) for mesh in meshes),
        "exact_duplicate_mesh_instances": sum(mesh.get("duplicate_geometry_of") is not None for mesh in meshes),
        "non_plate_mesh_candidates": sum(not mesh.get("plate_geometry_candidate", False) and "error" not in mesh for mesh in meshes),
        "training_eligible_parts": 0,
        "limitations": [
            "MT vertex and triangle arrays are readable, but header.xml project metadata is not readable XML.",
            "Mesh member names do not identify parts, platforms, inactive objects, duplicates or object transforms.",
            "The thin-plate test is a geometry heuristic and does not assign part roles or GT labels.",
        ],
    }
    output = {"summary": summary, "projects": projects}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
