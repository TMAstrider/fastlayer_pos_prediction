#!/usr/bin/env python3
"""Generate a first-pass, inspectable support-column proxy for one fixed pose.

The tool recognizes internal cylindrical holes/channels, merges that output
with optional face annotations, evaluates downward-facing triangle facets,
ray-casts retained contact samples toward the part or build plate, and checks
support paths against protected hole volumes. It exports support columns plus
a diagnostic report.

Supported inputs are OBJ, binary/ASCII STL, and NPZ (vertices/faces arrays).
Coordinates and all rule dimensions are expected in millimeters.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.recognize_part_features import (
        merge_annotations,
        recognize_features,
        resolve_support_query,
        segment_segment_distance,
        write_feature_preview,
    )
except ModuleNotFoundError:
    from recognize_part_features import (
        merge_annotations,
        recognize_features,
        resolve_support_query,
        segment_segment_distance,
        write_feature_preview,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "configs/support_rules_v0.json"
RULES_VERSION = "simplified_support_rules_v0"


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int64)
    elif suffix == ".obj":
        vertices_list: list[list[float]] = []
        faces_list: list[list[int]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices_list.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f" and len(fields) >= 4:
                indices = [int(token.split("/", 1)[0]) for token in fields[1:]]
                indices = [index - 1 if index > 0 else len(vertices_list) + index for index in indices]
                faces_list.extend([[indices[0], indices[i], indices[i + 1]]
                                   for i in range(1, len(indices) - 1)])
        vertices = np.asarray(vertices_list, dtype=np.float64)
        faces = np.asarray(faces_list, dtype=np.int64)
    elif suffix == ".stl":
        blob = path.read_bytes()
        binary_count = struct.unpack_from("<I", blob, 80)[0] if len(blob) >= 84 else -1
        if len(blob) == 84 + 50 * binary_count:
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            triangles = np.frombuffer(blob, dtype=dtype, count=binary_count, offset=84)["vertices"].astype(np.float64)
        else:
            text = blob.decode("ascii", errors="ignore")
            raw_vertices = re.findall(
                r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text, flags=re.I
            )
            if len(raw_vertices) % 3:
                raise ValueError("ASCII STL must contain a multiple of three vertex records")
            triangles = np.asarray(raw_vertices, dtype=np.float64).reshape(-1, 3, 3)
        vertices = triangles.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    else:
        raise ValueError(f"Unsupported mesh format {suffix!r}; use .stl, .obj or .npz")

    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Mesh has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Mesh has no triangular faces")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh contains nonfinite vertices or invalid face indices")
    return vertices, faces


def rotation_align_to_z(up: np.ndarray) -> np.ndarray:
    """Return a 3x3 column-vector rotation which maps up to +Z."""
    up = np.asarray(up, dtype=np.float64)
    norm = np.linalg.norm(up)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("build-up vector must be finite and nonzero")
    a = up / norm
    b = np.array([0.0, 0.0, 1.0])
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])
    axis = cross / sine
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew * sine + skew @ skew * (1.0 - cosine)


def process_config(rules: dict[str, Any], material: str, mode: str) -> dict[str, float]:
    try:
        material_rules = rules["configs"][material]
    except KeyError as exc:
        raise ValueError(f"No process rules for material {material!r}; available: {', '.join(rules['configs'])}") from exc
    if mode not in material_rules:
        raise ValueError(f"Unknown process mode {mode!r}; use one of: standard, am_build")
    return {
        "self_support_angle_threshold_deg": float(material_rules[mode]["self_support_angle_threshold_deg"]),
        "horizontal_hole_diameter_threshold_mm": float(material_rules["horizontal_hole_diameter_threshold_mm"]),
        "overhang_bridge_span_threshold_mm": float(material_rules["overhang_bridge_span_threshold_mm"]),
        "overhang_boss_size_threshold_mm": float(material_rules["overhang_boss_size_threshold_mm"]),
    }


def annotation_masks(face_count: int, annotations: dict[str, Any], limits: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    no_support = np.zeros(face_count, dtype=bool)
    feature_exempt = np.zeros(face_count, dtype=bool)
    for face_id in annotations.get("no_support_face_indices", []):
        face_id = int(face_id)
        if face_id < 0 or face_id >= face_count:
            raise ValueError(f"no-support face index {face_id} is out of range")
        no_support[face_id] = True

    counts: dict[str, int] = defaultdict(int)
    valid_kinds = {"horizontal_hole", "bridge", "boss"}
    for feature in annotations.get("features", []):
        kind = str(feature.get("kind", ""))
        if kind not in valid_kinds:
            raise ValueError(f"Unknown feature kind {kind!r}; choose horizontal_hole, bridge, or boss")
        face_ids = np.asarray(feature.get("face_indices", []), dtype=np.int64)
        if np.any(face_ids < 0) or np.any(face_ids >= face_count):
            raise ValueError(f"Feature {kind!r} contains an out-of-range face index")
        measure_key, threshold_key = {
            "horizontal_hole": ("diameter_mm", "horizontal_hole_diameter_threshold_mm"),
            "bridge": ("span_mm", "overhang_bridge_span_threshold_mm"),
            "boss": ("size_mm", "overhang_boss_size_threshold_mm"),
        }[kind]
        if measure_key not in feature:
            raise ValueError(f"Feature {kind!r} requires {measure_key}")
        measure = float(feature[measure_key])
        if not math.isfinite(measure) or measure < 0:
            raise ValueError(f"Feature {kind!r} has invalid {measure_key}")
        if measure < limits[threshold_key]:
            feature_exempt[face_ids] = True
            counts[kind] += int(len(np.unique(face_ids)))
    return no_support, feature_exempt, dict(counts)


def triangle_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    good = double_area > 1e-12
    normals[good] = cross[good] / double_area[good, None]
    return triangles, normals, double_area * 0.5


def watertight_edge_test(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Check that each quantized undirected mesh edge has exactly two faces."""
    scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    quantized = np.rint(vertices / (scale * 1e-8)).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    indexed_faces = inverse[faces]
    edges = np.concatenate([
        indexed_faces[:, [0, 1]],
        indexed_faces[:, [1, 2]],
        indexed_faces[:, [2, 0]],
    ])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def classify_overhang_faces(
    normals: np.ndarray,
    areas: np.ndarray,
    threshold_deg: float,
    feature_exempt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify downward facets using angle-from-horizontal convention."""
    downward = normals[:, 2] < -1e-9
    angles = np.full(len(normals), np.nan, dtype=np.float64)
    angles[downward] = np.degrees(np.arccos(np.clip(-normals[downward, 2], 0.0, 1.0)))
    angle_candidates = downward & (angles <= float(threshold_deg) + 1e-10) & (areas > 1e-12)
    exempt = feature_exempt if feature_exempt is not None else np.zeros(len(normals), dtype=bool)
    return angle_candidates, angle_candidates & ~exempt, angles


def uniform_triangle_samples(triangle: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if count == 1:
        return triangle.mean(axis=0, keepdims=True)
    index = np.arange(count, dtype=np.float64)
    u = (index + 0.5) / count
    v = np.mod((index + seed + 0.5) * 0.6180339887498949, 1.0)
    outside = (u + v) > 1.0
    u[outside] = 1.0 - u[outside]
    v[outside] = 1.0 - v[outside]
    return triangle[0] + u[:, None] * (triangle[1] - triangle[0]) + v[:, None] * (triangle[2] - triangle[0])


def sample_candidate_faces(
    triangles: np.ndarray,
    areas: np.ndarray,
    face_ids: np.ndarray,
    spacing: float,
    max_samples: int,
    platform_z: float,
    contact_band: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    valid = face_ids[areas[face_ids] > 1e-12]
    if not len(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64), 0
    desired = max(1, int(math.ceil(float(areas[valid].sum()) / spacing**2)))
    sample_count = min(desired, max_samples)
    weights = areas[valid] / areas[valid].sum()
    allocation = np.floor(weights * sample_count).astype(np.int64)
    remaining = sample_count - int(allocation.sum())
    if remaining:
        fractional = weights * sample_count - allocation
        allocation[np.argsort(fractional)[-remaining:]] += 1
    point_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    for face_id, count in zip(valid, allocation):
        if count <= 0:
            continue
        points = uniform_triangle_samples(triangles[face_id], int(count), int(face_id))
        above = points[:, 2] > platform_z + contact_band
        points = points[above]
        if len(points):
            point_parts.append(points)
            face_parts.append(np.full(len(points), face_id, dtype=np.int64))
    if not point_parts:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64), desired
    points = np.concatenate(point_parts)
    faces = np.concatenate(face_parts)
    # Remove coincident samples from adjacent facets without allowing dense meshes
    # to multiply support columns along shared triangle boundaries.
    _, unique_indices = np.unique(np.rint(points * 1e5).astype(np.int64), axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices], faces[unique_indices], desired


class VerticalRayIndex:
    """Uniform XY index for exact vertical ray/triangle intersections."""

    def __init__(self, triangles: np.ndarray, cell_size: float):
        self.triangles = triangles
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.large: list[int] = []
        self.denominators = np.zeros(len(triangles), dtype=np.float64)
        for face_id, tri in enumerate(triangles):
            a, b, c = tri[:, :2]
            den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            self.denominators[face_id] = den
            if abs(den) < 1e-12:
                continue
            low = np.floor(tri[:, :2].min(axis=0) / self.cell_size).astype(int)
            high = np.floor(tri[:, :2].max(axis=0) / self.cell_size).astype(int)
            count = int((high[0] - low[0] + 1) * (high[1] - low[1] + 1))
            if count > 3000:
                self.large.append(face_id)
                continue
            for ix in range(low[0], high[0] + 1):
                for iy in range(low[1], high[1] + 1):
                    self.cells[(ix, iy)].append(face_id)

    def first_hit(
        self, point: np.ndarray, excluded_face: int, platform_z: float, tolerance: float = 1e-7
    ) -> tuple[float, int] | None:
        key = tuple(np.floor(point[:2] / self.cell_size).astype(int))
        candidates = self.cells.get(key, [])
        if self.large:
            candidates = [*candidates, *self.large]
        if not candidates:
            return None
        best_z = -np.inf
        best_face = -1
        x, y, source_z = point
        for face_id in candidates:
            if face_id == excluded_face:
                continue
            tri = self.triangles[face_id]
            den = self.denominators[face_id]
            if abs(den) < 1e-12:
                continue
            a, b, c = tri
            w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den
            w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den
            w2 = 1.0 - w0 - w1
            if min(w0, w1, w2) < -tolerance:
                continue
            hit_z = w0 * a[2] + w1 * b[2] + w2 * c[2]
            if hit_z >= platform_z - tolerance and hit_z < source_z - tolerance and hit_z > best_z:
                best_z, best_face = float(hit_z), int(face_id)
        return (best_z, best_face) if best_face >= 0 else None


def evaluate_supports(
    vertices: np.ndarray,
    faces: np.ndarray,
    limits: dict[str, float],
    annotations: dict[str, Any] | None = None,
    up: np.ndarray | None = None,
    sample_spacing_mm: float = 2.0,
    contact_band_mm: float = 0.05,
    max_samples: int = 250_000,
    support_radius_mm: float = 0.4,
    platform_z_mm: float = 0.0,
    recognized_features: dict[str, Any] | None = None,
    protected_clearance_mm: float = 0.0,
    support_query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if support_query and support_query.get("clearance_mm") is not None:
        protected_clearance_mm = float(support_query["clearance_mm"])
    if (
        sample_spacing_mm <= 0
        or contact_band_mm < 0
        or max_samples < 1
        or support_radius_mm <= 0
        or protected_clearance_mm < 0
    ):
        raise ValueError(
            "sampling distance and sample cap must be positive; contact band and "
            "protected clearance cannot be negative; support radius must be positive"
        )
    annotations = annotations or {}
    rotation = rotation_align_to_z(np.asarray(up if up is not None else [0.0, 0.0, 1.0]))
    aligned_vertices = np.asarray(vertices, dtype=np.float64) @ rotation.T
    triangles, normals, areas = triangle_geometry(aligned_vertices, faces)
    if recognized_features is not None:
        annotations = merge_annotations(annotations, recognized_features, len(faces))
    no_support, feature_exempt, feature_face_counts = annotation_masks(len(faces), annotations, limits)
    protected_cylinders: list[dict[str, Any]] = []
    if recognized_features is not None:
        endpoint_extension = support_radius_mm + protected_clearance_mm
        selected_feature_ids = set(
            (support_query or {}).get("selected_feature_ids", [])
        )
        for feature in recognized_features.get("features", []):
            if feature.get("kind") != "internal_cylinder":
                continue
            if feature["feature_id"] not in selected_feature_ids:
                continue
            axis = rotation @ np.asarray(feature["axis_input"], dtype=np.float64)
            start = rotation @ np.asarray(feature["axis_start_input_mm"], dtype=np.float64)
            end = rotation @ np.asarray(feature["axis_end_input_mm"], dtype=np.float64)
            protected_cylinders.append({
                "feature_id": str(feature["feature_id"]),
                "axis_start": start - endpoint_extension * axis,
                "axis_end": end + endpoint_extension * axis,
                "clearance_radius_mm": (
                    float(feature["radius_mm"])
                    + support_radius_mm
                    + protected_clearance_mm
                ),
            })

    # For a downward-facing surface, horizontal is 0 deg (normal = -Z) and
    # vertical is 90 deg (normal has zero Z component).
    angle_candidates, candidate_faces, face_angles = classify_overhang_faces(
        normals, areas, limits["self_support_angle_threshold_deg"], feature_exempt
    )
    exempt_faces = angle_candidates & feature_exempt

    ray_index = VerticalRayIndex(triangles, sample_spacing_mm)
    source_points, source_faces_arr, desired_sample_count = sample_candidate_faces(
        triangles, areas, np.flatnonzero(candidate_faces), sample_spacing_mm,
        max_samples, platform_z_mm, contact_band_mm,
    )

    supports: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for sample_id, (point, source_face) in enumerate(zip(source_points, source_faces_arr)):
        if no_support[source_face]:
            violations.append({
                "type": "overhang_on_no_support_face",
                "sample_id": sample_id,
                "source_face_index": int(source_face),
                "point_aligned_mm": point.tolist(),
            })
            continue
        hit = ray_index.first_hit(point, int(source_face), platform_z_mm)
        if hit is None:
            landing_z, landing_face = float(platform_z_mm), None
        else:
            landing_z, landing_face_id = hit
            landing_face = int(landing_face_id)
            if no_support[landing_face_id]:
                violations.append({
                    "type": "support_landing_on_no_support_face",
                    "sample_id": sample_id,
                    "source_face_index": int(source_face),
                    "landing_face_index": int(landing_face_id),
                    "point_aligned_mm": [float(point[0]), float(point[1]), float(landing_z)],
                })
                continue
        if point[2] - landing_z <= contact_band_mm:
            continue
        bottom = np.array([point[0], point[1], landing_z], dtype=np.float64)
        collision = next((
            cylinder for cylinder in protected_cylinders
            if segment_segment_distance(
                point, bottom, cylinder["axis_start"], cylinder["axis_end"]
            ) < cylinder["clearance_radius_mm"]
        ), None)
        if collision is not None:
            violations.append({
                "type": "support_shaft_intersects_recognized_internal_cylinder",
                "sample_id": sample_id,
                "source_face_index": int(source_face),
                "landing_face_index": landing_face,
                "feature_id": collision["feature_id"],
                "required_clearance_radius_mm": collision["clearance_radius_mm"],
                "support_distance_to_axis_mm": segment_segment_distance(
                    point, bottom, collision["axis_start"], collision["axis_end"]
                ),
                "source_point_aligned_mm": point.tolist(),
                "landing_point_aligned_mm": bottom.tolist(),
            })
            continue
        support_kind = "part_surface" if landing_face is not None else "build_plate"
        supports.append({
            "sample_id": sample_id,
            "source_face_index": int(source_face),
            "landing_face_index": landing_face,
            "source_point_aligned_mm": point.tolist(),
            "landing_point_aligned_mm": bottom.tolist(),
            "height_mm": float(point[2] - landing_z),
            "landing_kind": support_kind,
        })

    def unalign(point: list[float]) -> list[float]:
        return (np.asarray(point, dtype=np.float64) @ rotation).tolist()

    for support in supports:
        support["source_point_input_mm"] = unalign(support["source_point_aligned_mm"])
        support["landing_point_input_mm"] = unalign(support["landing_point_aligned_mm"])
    for violation in violations:
        if "point_aligned_mm" in violation:
            violation["point_input_mm"] = unalign(violation["point_aligned_mm"])
        if "source_point_aligned_mm" in violation:
            violation["source_point_input_mm"] = unalign(violation["source_point_aligned_mm"])
        if "landing_point_aligned_mm" in violation:
            violation["landing_point_input_mm"] = unalign(violation["landing_point_aligned_mm"])

    candidate_area = float(areas[angle_candidates].sum())
    exempt_area = float(areas[exempt_faces].sum())
    final_area = float(areas[candidate_faces].sum())
    support_heights = np.asarray([item["height_mm"] for item in supports], dtype=np.float64)
    minimum_part_height = float(aligned_vertices[:, 2].min())
    signed_volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)
    watertight = watertight_edge_test(np.asarray(vertices, dtype=np.float64), faces)
    below_platform = minimum_part_height < platform_z_mm - contact_band_mm
    if watertight and signed_volume < -1e-9:
        violations.append({
            "type": "mesh_winding_appears_inward",
            "signed_volume_mm3": signed_volume,
        })
    if below_platform:
        violations.append({
            "type": "part_below_build_platform",
            "minimum_part_height_mm": minimum_part_height,
            "platform_height_mm": float(platform_z_mm),
        })
    report = {
        "algorithm": RULES_VERSION,
        "units": "mm",
        "build_up_input": (np.asarray(up if up is not None else [0.0, 0.0, 1.0], dtype=float) / np.linalg.norm(up if up is not None else [0.0, 0.0, 1.0])).tolist(),
        "thresholds": limits,
        "platform_height_mm": float(platform_z_mm),
        "minimum_part_height_along_build_axis_mm": minimum_part_height,
        "mesh_orientation_diagnostic": {
            "signed_volume_mm3_if_closed_and_consistently_wound": signed_volume,
            "watertight_edge_test_passed": watertight,
            "negative_volume_warning": bool(watertight and signed_volume < -1e-9),
            "interpretation": "positive volume usually indicates outward winding for a closed consistently wound mesh; near zero can indicate an open mesh",
        },
        "face_counts": {
            "total": int(len(faces)),
            "downward_angle_candidates": int(angle_candidates.sum()),
            "feature_exempt": int(exempt_faces.sum()),
            "support_candidate_after_exemptions": int(candidate_faces.sum()),
            "no_support_faces": int(no_support.sum()),
            "support_candidate_face_indices": np.flatnonzero(candidate_faces).tolist(),
            "feature_exempt_face_indices": np.flatnonzero(exempt_faces).tolist(),
            "no_support_face_indices": np.flatnonzero(no_support).tolist(),
        },
        "areas_mm2": {
            "downward_angle_candidates": candidate_area,
            "feature_exempt": exempt_area,
            "support_candidate_after_exemptions": final_area,
        },
        "sampling": {
            "spacing_mm": float(sample_spacing_mm),
            "contact_band_mm": float(contact_band_mm),
            "max_samples": int(max_samples),
            "samples_generated": int(len(source_points)),
            "samples_requested_before_cap": int(desired_sample_count),
            "sample_cap_reached": bool(desired_sample_count > max_samples),
        },
        "feature_exempt_face_counts_by_kind": feature_face_counts,
        "supports": supports,
        "support_summary": {
            "count": int(len(supports)),
            "build_plate_landings": int(sum(item["landing_kind"] == "build_plate" for item in supports)),
            "part_surface_landings": int(sum(item["landing_kind"] == "part_surface" for item in supports)),
            "mean_height_mm": float(support_heights.mean()) if len(support_heights) else 0.0,
            "max_height_mm": float(support_heights.max()) if len(support_heights) else 0.0,
            "column_volume_proxy_mm3": float(np.pi * support_radius_mm**2 * support_heights.sum()),
            "column_radius_assumption_mm": float(support_radius_mm),
        },
        "recognized_features": recognized_features,
        "support_query": support_query,
        "protected_clearance_mm": float(protected_clearance_mm),
        "hard_constraint_violations": violations,
        "is_feasible_under_supplied_constraints": len(violations) == 0,
        "limitations": [
            "Round holes, bridges and bosses are exempted only when their triangle face indices and measured size are explicitly supplied in the annotation JSON.",
            "The ray-cast vertical cylinders are a geometric proxy; no lattice, branch, contact-tip, thermal, recoater, or removal simulation is performed.",
            "Ray columns terminate at the first mesh intersection above the configured build-plate plane; this is not a collision-free production support design.",
            "No first-layer adhesion, support anchoring strength, or global structural stability check is performed.",
            "Surface normals rely on input triangle winding. Verify outward winding before interpreting down-facing areas.",
            "Material thresholds are provisional values supplied in the task and have not been calibrated against Magics or print outcomes.",
            "Internal-hole recognition is a deterministic cylindrical-region fit and does not make a hole forbidden by itself; non-cylindrical protected surfaces require query or manual annotations.",
        ],
    }
    return {
        "report": report,
        "aligned_vertices": aligned_vertices,
        "triangles": triangles,
        "normals": normals,
        "areas": areas,
        "candidate_faces": candidate_faces,
        "exempt_faces": exempt_faces,
        "no_support_faces": no_support,
        "supports": supports,
    }


def export_support_obj(path: Path, supports: list[dict[str, Any]], radius_mm: float = 0.4, sides: int = 6) -> None:
    if radius_mm <= 0 or sides < 3:
        raise ValueError("support radius must be positive and cylinder sides at least 3")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    angle = np.arange(sides, dtype=np.float64) * (2 * np.pi / sides)
    offsets = np.column_stack([radius_mm * np.cos(angle), radius_mm * np.sin(angle)])
    for support in supports:
        top = np.asarray(support["source_point_input_mm"], dtype=np.float64)
        bottom = np.asarray(support["landing_point_input_mm"], dtype=np.float64)
        axis = top - bottom
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            continue
        axis /= norm
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ring = offsets[:, :1] * u[None, :] + offsets[:, 1:] * v[None, :]
        start = len(vertices)
        vertices.extend((bottom + ring).tolist())
        vertices.extend((top + ring).tolist())
        for i in range(sides):
            j = (i + 1) % sides
            a, b, c, d = start + i + 1, start + j + 1, start + sides + j + 1, start + sides + i + 1
            faces.extend([[a, b, c], [a, c, d]])
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Simplified vertical support-column proxy; dimensions in mm\n")
        for vertex in vertices:
            stream.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            stream.write(f"f {face[0]} {face[1]} {face[2]}\n")


def render_preview(path: Path, result: dict[str, Any], reference_support: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = result["triangles"]
    count = len(triangles)
    stride = max(1, count // 25_000)
    faces = np.arange(0, count, stride)
    candidate = result["candidate_faces"]
    exempt = result["exempt_faces"]
    forbidden = result["no_support_faces"]
    colors = np.full((len(faces), 4), [0.65, 0.72, 0.78, 0.65])
    colors[candidate[faces]] = [0.95, 0.35, 0.18, 0.95]
    colors[exempt[faces]] = [0.24, 0.64, 0.53, 0.95]
    colors[forbidden[faces]] = [0.25, 0.42, 0.85, 0.95]

    low, high = result["aligned_vertices"].min(axis=0), result["aligned_vertices"].max(axis=0)
    pad = 0.06 * max(float((high - low).max()), 1e-3)
    platform_z = float(result["report"]["platform_height_mm"])
    platform = np.array([
        [low[0] - pad, low[1] - pad, platform_z],
        [high[0] + pad, low[1] - pad, platform_z],
        [high[0] + pad, high[1] + pad, platform_z],
        [low[0] - pad, high[1] + pad, platform_z],
    ])
    support_rows = result["supports"]
    vertices = result["aligned_vertices"]
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    span = np.maximum(high - low, 1e-3)
    center = (low + high) / 2
    if reference_support is None:
        fig = plt.figure(figsize=(10, 8), facecolor="#f5f7fa")
        axes = [fig.add_subplot(111, projection="3d")]
        titles = ["Part + simplified supports"]
        panel_modes = ["predicted"]
    else:
        fig = plt.figure(figsize=(17, 6), facecolor="#f5f7fa")
        axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
        titles = ["Part + overhang candidates", "Stored Magics support", "Simplified support estimate"]
        panel_modes = ["input", "reference", "predicted"]

    for ax, title, panel_mode in zip(axes, titles, panel_modes):
        ax.add_collection3d(Poly3DCollection([platform], facecolors="#9aa6b2", edgecolors="#45515d", alpha=0.2))
        if panel_mode == "input":
            input_colors = np.full((len(faces), 4), [0.60, 0.69, 0.77, 0.70])
            input_colors[candidate[faces]] = [0.93, 0.35, 0.18, 0.95]
            input_colors[exempt[faces]] = [0.24, 0.64, 0.53, 0.95]
            input_colors[forbidden[faces]] = [0.25, 0.42, 0.85, 0.95]
            ax.add_collection3d(Poly3DCollection(triangles[faces], facecolors=input_colors, edgecolors="none"))
        else:
            ax.add_collection3d(Poly3DCollection(
                triangles[faces], facecolors=[0.70, 0.76, 0.80, 0.55], edgecolors="none"
            ))
            if panel_mode == "reference" and reference_support is not None:
                stride_reference = max(1, len(reference_support) // 30_000)
                ax.add_collection3d(Poly3DCollection(
                    reference_support[::stride_reference],
                    facecolors=[0.14, 0.52, 0.67, 0.80], edgecolors="none",
                ))
            elif panel_mode == "predicted":
                stride_support = max(1, len(support_rows) // 450)
                for item in support_rows[::stride_support]:
                    a = item["source_point_aligned_mm"]
                    b = item["landing_point_aligned_mm"]
                    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                            color="#248345", linewidth=0.9, alpha=0.78)

        arrow_base = np.array([low[0] - 0.025 * span.max(), low[1] - 0.025 * span.max(), float(result["report"]["platform_height_mm"])])
        arrow_len = max(float(span.max()) * 0.12, 4.0)
        ax.quiver(*arrow_base, 0.0, 0.0, arrow_len, color="#d04b40", linewidth=1.7, arrow_length_ratio=0.18)
        ax.text(*(arrow_base + np.array([0.0, 0.0, arrow_len * 1.12])), "+build Z", color="#a02e27", fontsize=8)
        ax.set_xlim(center[0] - span.max() / 2, center[0] + span.max() / 2)
        ax.set_ylim(center[1] - span.max() / 2, center[1] + span.max() / 2)
        ax.set_zlim(float(result["report"]["platform_height_mm"]) - 0.02 * span.max(), high[2] + 0.03 * span.max())
        ax.set_box_aspect((1, 1, max(span[2] / span.max(), 0.15)))
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_zlabel("Build Z (mm)", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=24, azim=-54)
    if reference_support is None:
        axes[0].set_title("Simplified support preview\nred: candidate · green: exempt · blue: no-support")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="OBJ, STL, or NPZ with vertices/faces arrays")
    parser.add_argument("--material", required=True, choices=["316L", "AlSi10Mg", "TC4"])
    parser.add_argument("--mode", required=True, choices=["standard", "am_build"])
    parser.add_argument("--build-up", nargs=3, type=float, default=[0.0, 0.0, 1.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--platform-offset-mm", type=float, default=0.0,
                        help="build-plate plane coordinate along the build-up axis; default 0")
    parser.add_argument("--annotations", type=Path, help="JSON with face-index no-support regions and feature exemptions")
    parser.add_argument("--support-query", type=Path,
                        help="structured user query selecting recognized feature regions that forbid support")
    parser.add_argument("--auto-features", dest="auto_features", action="store_true", default=True,
                        help="recognize internal cylindrical holes/channels automatically (default)")
    parser.add_argument("--no-auto-features", dest="auto_features", action="store_false",
                        help="disable automatic feature recognition")
    parser.add_argument("--feature-min-confidence", type=float, default=0.58)
    parser.add_argument("--feature-min-angular-bins", type=int, default=8)
    parser.add_argument("--feature-min-axis-length-mm", type=float, default=0.25)
    parser.add_argument("--feature-normal-tolerance-deg", type=float, default=50.0)
    parser.add_argument("--horizontal-feature-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--protected-clearance-mm", type=float, default=0.2,
                        help="minimum support-shaft clearance outside recognized hole radii")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--sample-spacing-mm", type=float, default=2.0)
    parser.add_argument("--contact-band-mm", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=250_000)
    parser.add_argument("--support-radius-mm", type=float, default=0.4)
    parser.add_argument("--output", type=Path, help="output report JSON; default <mesh>.support.json")
    parser.add_argument("--support-obj", type=Path, help="output support-column proxy OBJ")
    parser.add_argument("--feature-output", type=Path, help="output automatic feature-recognition JSON")
    parser.add_argument("--feature-preview", type=Path, help="output automatic feature-recognition PNG")
    parser.add_argument("--reference-support", type=Path,
                        help="optional stored Magics support mesh for a three-panel visual comparison")
    parser.add_argument("--preview", type=Path, help="output PNG diagnostic preview")
    args = parser.parse_args()

    vertices, faces = read_mesh(args.mesh)
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    limits = process_config(rules, args.material, args.mode)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8")) if args.annotations else {}
    support_query_input = (
        json.loads(args.support_query.read_text(encoding="utf-8"))
        if args.support_query else {}
    )
    recognized_features = None
    if args.auto_features:
        recognized_features = recognize_features(
            vertices,
            faces,
            build_up=np.asarray(args.build_up, dtype=np.float64),
            normal_tolerance_deg=args.feature_normal_tolerance_deg,
            min_confidence=args.feature_min_confidence,
            horizontal_tolerance_deg=args.horizontal_feature_tolerance_deg,
            small_hole_diameter_mm=limits["horizontal_hole_diameter_threshold_mm"],
            min_angular_bins=args.feature_min_angular_bins,
            min_axis_length_mm=args.feature_min_axis_length_mm,
        )
        annotations = merge_annotations(annotations, recognized_features, len(faces))
    resolved_support_query = resolve_support_query(
        recognized_features or {"features": []},
        support_query_input,
        len(faces),
    )
    query_no_support = set(resolved_support_query["no_support_face_indices"])
    annotations["no_support_face_indices"] = sorted(
        set(annotations.get("no_support_face_indices", [])) | query_no_support
    )
    effective_clearance = (
        resolved_support_query["clearance_mm"]
        if resolved_support_query["clearance_mm"] is not None
        else args.protected_clearance_mm
    )
    result = evaluate_supports(
        vertices, faces, limits, annotations, np.asarray(args.build_up),
        args.sample_spacing_mm, args.contact_band_mm, args.max_samples,
        args.support_radius_mm,
        args.platform_offset_mm,
        recognized_features,
        effective_clearance,
        resolved_support_query,
    )
    report = result["report"]
    report["input_mesh"] = str(args.mesh.resolve())
    report["material"] = args.material
    report["process_mode"] = args.mode
    report["source_rule_file"] = str(args.rules.resolve())
    report["annotation_file"] = str(args.annotations.resolve()) if args.annotations else None
    report["support_query_file"] = str(args.support_query.resolve()) if args.support_query else None
    report["resolved_support_query"] = resolved_support_query
    report["automatic_feature_recognition_enabled"] = bool(args.auto_features)
    report["source_mesh_counts"] = {"vertices": int(len(vertices)), "faces": int(len(faces))}
    reference_triangles = None
    if args.reference_support:
        reference_vertices, reference_faces = read_mesh(args.reference_support)
        rotation = rotation_align_to_z(np.asarray(args.build_up, dtype=float))
        reference_triangles = (reference_vertices @ rotation.T)[reference_faces]
        report["reference_support_mesh"] = str(args.reference_support.resolve())
        report["reference_support_mesh_counts"] = {
            "vertices": int(len(reference_vertices)), "faces": int(len(reference_faces)),
        }

    output = args.output or args.mesh.with_suffix(".support.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report["support_obj_output"] = str(args.support_obj.resolve()) if args.support_obj else None
    report["preview_output"] = str(args.preview.resolve()) if args.preview else None
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.feature_output and recognized_features is not None:
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        args.feature_output.write_text(
            json.dumps(recognized_features, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.feature_preview and recognized_features is not None:
        write_feature_preview(args.feature_preview, vertices, faces, recognized_features)
    if args.support_obj:
        args.support_obj.parent.mkdir(parents=True, exist_ok=True)
        export_support_obj(args.support_obj, report["supports"], args.support_radius_mm)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        render_preview(args.preview, result, reference_triangles)
    print(json.dumps({
        "output": str(output.resolve()),
        "face_counts": {key: value for key, value in report["face_counts"].items()
                        if key not in {"support_candidate_face_indices", "feature_exempt_face_indices", "no_support_face_indices"}},
        "support_summary": report["support_summary"],
        "hard_constraint_violation_count": len(report["hard_constraint_violations"]),
        "recognized_features": (
            recognized_features["summary"] if recognized_features is not None else None
        ),
        "preview": str(args.preview.resolve()) if args.preview else None,
        "support_obj": str(args.support_obj.resolve()) if args.support_obj else None,
        "reference_support_mesh": str(args.reference_support.resolve()) if args.reference_support else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
