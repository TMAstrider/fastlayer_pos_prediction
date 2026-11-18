#!/usr/bin/env python3
"""Recognize internal cylindrical holes and channels from triangulated meshes.

The detector is deterministic and geometry-based. It fits candidate cylinders
to smooth face regions, keeps inward-facing cylindrical surfaces as hole/channel
candidates, and writes feature metadata plus process exemptions. Recognition
does not make a region forbidden; that requires a separate user query.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


FEATURE_SCHEMA_VERSION = 1


def triangle_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    good = double_area > 1e-12
    normals[good] = cross[good] / double_area[good, None]
    return triangles, normals, double_area * 0.5


def face_adjacency(vertices: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
    """Build undirected face adjacency while merging duplicated STL vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    quantum = scale * 1e-7
    packed = np.rint(vertices / quantum).astype(np.int64)
    _, vertex_ids = np.unique(packed, axis=0, return_inverse=True)
    indexed_faces = vertex_ids[faces]

    edge_faces = np.concatenate([np.arange(len(faces), dtype=np.int64)] * 3)
    edge_vertices = np.empty((len(edge_faces), 2), dtype=np.int64)
    for edge_id in range(3):
        a = indexed_faces[:, edge_id]
        b = indexed_faces[:, (edge_id + 1) % 3]
        edge_vertices[edge_id * len(faces):(edge_id + 1) * len(faces), 0] = np.minimum(a, b)
        edge_vertices[edge_id * len(faces):(edge_id + 1) * len(faces), 1] = np.maximum(a, b)
    order = np.lexsort((edge_vertices[:, 1], edge_vertices[:, 0]))
    sorted_vertices = edge_vertices[order]
    sorted_faces = edge_faces[order]
    same_edge = np.all(sorted_vertices[:-1] == sorted_vertices[1:], axis=1)
    first_faces = sorted_faces[:-1][same_edge]
    second_faces = sorted_faces[1:][same_edge]
    different_faces = first_faces != second_faces
    rows = first_faces[different_faces]
    columns = second_faces[different_faces]
    data = np.ones(len(rows), dtype=np.uint8)
    adjacency = sparse.coo_matrix((data, (rows, columns)), shape=(len(faces), len(faces))).tocsr()
    return adjacency.maximum(adjacency.T).astype(bool).tocsr()


def smooth_face_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    normal_tolerance_deg: float,
) -> np.ndarray:
    """Assign faces to edge-connected regions with bounded normal change."""
    adjacency = face_adjacency(vertices, faces)
    upper = sparse.triu(adjacency, k=1).tocoo()
    dots = np.einsum("ij,ij->i", normals[upper.row], normals[upper.col])
    compatible = dots >= math.cos(math.radians(normal_tolerance_deg)) - 1e-10
    rows = upper.row[compatible]
    columns = upper.col[compatible]
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(faces), len(faces)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64)


def _fit_cylinder_sign(
    points_2d: np.ndarray,
    normal_directions_2d: np.ndarray,
    sign: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    directions = float(sign) * normal_directions_2d
    design = np.column_stack([
        np.ones(len(points_2d)),
        np.zeros(len(points_2d)),
        directions[:, 0],
        np.zeros(len(points_2d)),
        np.ones(len(points_2d)),
        directions[:, 1],
    ]).reshape(-1, 3)
    rhs = points_2d.reshape(-1)
    solution, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    center = solution[:2]
    radius = float(solution[2])
    predicted = center[None, :] + radius * directions
    residual = np.linalg.norm(points_2d - predicted, axis=1)
    return center, radius, residual


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    anchor = int(np.argmax(np.abs(axis)))
    if axis[anchor] < 0:
        axis = -axis
    return axis


def fit_cylinder(
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
) -> dict[str, Any] | None:
    """Fit a cylinder to a face region using normals and radial residuals."""
    weights = np.sqrt(np.maximum(areas, 1e-15))
    normal_matrix = normals * weights[:, None]
    _, singular_values, right_vectors = np.linalg.svd(normal_matrix, full_matrices=False)
    total = float(singular_values[0])
    if total <= 1e-12 or singular_values[1] / total < 0.10:
        return None
    if singular_values[2] / total > 0.25:
        return None

    axis = _canonical_axis(right_vectors[-1])
    axial_normal_error = float(np.median(np.abs(normals @ axis)))
    if axial_normal_error > 0.25:
        return None

    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, helper)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    points_2d = np.column_stack([centroids @ basis_u, centroids @ basis_v])
    normal_2d = np.column_stack([normals @ basis_u, normals @ basis_v])
    normal_2d_norm = np.linalg.norm(normal_2d, axis=1)
    valid = normal_2d_norm > 0.35
    if valid.sum() < 4:
        return None

    candidate_fits: list[dict[str, Any]] = []
    for sign in (-1, 1):
        center_2d, radius, residual = _fit_cylinder_sign(
            points_2d[valid],
            normal_2d[valid] / normal_2d_norm[valid, None],
            sign,
        )
        if radius <= 1e-6 or not math.isfinite(radius):
            continue
        distances = np.linalg.norm(points_2d[valid] - center_2d[None, :], axis=1)
        radius_std = float(np.std(distances))
        radial_rms = float(np.sqrt(np.mean(residual**2)))
        radial_angles = np.arctan2(
            points_2d[valid, 1] - center_2d[1],
            points_2d[valid, 0] - center_2d[0],
        )
        bin_ids = np.floor((radial_angles + np.pi) / (2.0 * np.pi) * 12.0).astype(np.int64)
        occupied_bins = int(len(np.unique(np.clip(bin_ids, 0, 11))))

        radial_3d = np.column_stack([
            points_2d[valid] @ np.array([basis_u[0], basis_v[0]]),
            points_2d[valid] @ np.array([basis_u[1], basis_v[1]]),
            points_2d[valid] @ np.array([basis_u[2], basis_v[2]]),
        ]) - (basis_u * center_2d[0] + basis_v * center_2d[1])[None, :]
        radial_lengths = np.linalg.norm(radial_3d, axis=1)
        predicted_normals = sign * radial_3d / np.maximum(radial_lengths[:, None], 1e-15)
        normal_angles = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", normals[valid], predicted_normals), -1.0, 1.0
        )))
        median_normal_error = float(np.median(normal_angles))
        if (
            median_normal_error > 20.0
            or radial_rms / radius > 0.08
            or radius_std / radius > 0.12
            or occupied_bins < 2
        ):
            continue

        axial_coordinates = centroids[valid] @ axis
        axial_mid = 0.5 * (float(axial_coordinates.min()) + float(axial_coordinates.max()))
        axis_point = (
            basis_u * center_2d[0]
            + basis_v * center_2d[1]
            + axial_mid * axis
        )
        axis_length = float(np.ptp(axial_coordinates))

        axial_score = max(0.0, 1.0 - axial_normal_error / 0.25)
        radial_score = math.exp(-((radial_rms / radius) / 0.06) ** 2)
        stability_score = math.exp(-((radius_std / radius) / 0.10) ** 2)
        normal_score = math.exp(-((median_normal_error / 15.0) ** 2))
        coverage_score = min(1.0, occupied_bins / 6.0)
        confidence = float(np.prod([
            max(axial_score, 1e-6),
            max(radial_score, 1e-6),
            max(stability_score, 1e-6),
            max(normal_score, 1e-6),
            max(coverage_score, 1e-6),
        ]) ** 0.2)
        candidate_fits.append({
            "axis": axis,
            "center_2d": center_2d,
            "radius": radius,
            "axis_point": axis_point,
            "axis_length": axis_length,
            "sign": sign,
            "confidence": confidence,
            "metrics": {
                "normal_axis_median_error": axial_normal_error,
                "normal_direction_median_error_deg": median_normal_error,
                "radial_rms_mm": radial_rms,
                "radius_std_mm": radius_std,
                "angular_bins_occupied_of_12": occupied_bins,
                "singular_value_ratio_axis": float(singular_values[2] / total),
                "singular_value_ratio_second_axis": float(singular_values[1] / total),
            },
        })
    if not candidate_fits:
        return None
    return max(candidate_fits, key=lambda item: item["confidence"])


def recognize_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    build_up: np.ndarray | None = None,
    normal_tolerance_deg: float = 50.0,
    min_confidence: float = 0.58,
    horizontal_tolerance_deg: float = 10.0,
    small_hole_diameter_mm: float = 6.0,
    min_angular_bins: int = 8,
    min_axis_length_mm: float = 0.25,
) -> dict[str, Any]:
    """Recognize internal cylindrical holes/channels and create annotations."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    up = np.asarray(build_up if build_up is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    up /= np.linalg.norm(up)
    triangles, normals, areas = triangle_geometry(vertices, faces)
    centroids = triangles.mean(axis=1)
    labels = smooth_face_regions(vertices, faces, normals, normal_tolerance_deg)
    mesh_span = float(np.ptp(vertices, axis=0).max())

    features: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for label in np.unique(labels):
        face_ids = np.flatnonzero(labels == label)
        # Internal full cylinders generally need at least a handful of faces.
        if len(face_ids) < 4:
            continue
        fit = fit_cylinder(centroids[face_ids], normals[face_ids], areas[face_ids])
        if fit is None:
            rejections["not_a_stable_cylinder"] += 1
            continue
        if fit["confidence"] < min_confidence:
            rejections["low_confidence"] += 1
            continue
        if fit["sign"] >= 0:
            rejections["external_cylinder"] += 1
            continue
        if fit["metrics"]["angular_bins_occupied_of_12"] < min_angular_bins:
            rejections["insufficient_angular_coverage"] += 1
            continue

        axis = np.asarray(fit["axis"], dtype=np.float64)
        center = np.asarray(fit["axis_point"], dtype=np.float64)
        radius = float(fit["radius"])
        region_vertices = triangles[face_ids].reshape(-1, 3)
        vertex_axial = region_vertices @ axis
        axial_mid = 0.5 * (float(vertex_axial.min()) + float(vertex_axial.max()))
        center += axis * (axial_mid - float(center @ axis))
        axis_length = float(np.ptp(vertex_axial))
        if axis_length < min_axis_length_mm:
            rejections["short_axis"] += 1
            continue
        if 2.0 * radius > mesh_span * 1.02:
            rejections["diameter_exceeds_part_span"] += 1
            continue
        half_length = 0.5 * axis_length
        axis_to_build = math.degrees(math.acos(float(np.clip(abs(axis @ up), 0.0, 1.0))))
        axis_to_build_plane = 90.0 - axis_to_build
        is_horizontal = axis_to_build_plane <= horizontal_tolerance_deg + 1e-10
        diameter = 2.0 * radius
        small_exempt = bool(is_horizontal and diameter < small_hole_diameter_mm)
        feature_id = f"internal_cylinder_{len(features) + 1:04d}"
        features.append({
            "feature_id": feature_id,
            "kind": "internal_cylinder",
            "functional_role": "hole_or_channel",
            "face_indices": face_ids.tolist(),
            "face_count": int(len(face_ids)),
            "surface_area_mm2": float(areas[face_ids].sum()),
            "axis_input": axis.tolist(),
            "centerline_point_input_mm": center.tolist(),
            "axis_start_input_mm": (center - half_length * axis).tolist(),
            "axis_end_input_mm": (center + half_length * axis).tolist(),
            "axis_length_mm": axis_length,
            "diameter_mm": diameter,
            "radius_mm": radius,
            "axis_to_build_axis_deg": axis_to_build,
            "axis_to_build_plane_deg": axis_to_build_plane,
            "is_horizontal_to_build": bool(is_horizontal),
            "small_horizontal_exempt": small_exempt,
            "support_policy": (
                "size_exempt"
                if small_exempt else "support_allowed"
            ),
            "confidence": float(fit["confidence"]),
            "fit_metrics": fit["metrics"],
        })

    exemption_features = [{
        "kind": "horizontal_hole",
        "name": feature["feature_id"],
        "face_indices": feature["face_indices"],
        "diameter_mm": feature["diameter_mm"],
        "source": "automatic_geometry",
        "confidence": feature["confidence"],
    } for feature in features if feature["is_horizontal_to_build"]]
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "build_up_input": up.tolist(),
        "parameters": {
            "smooth_normal_tolerance_deg": float(normal_tolerance_deg),
            "minimum_confidence": float(min_confidence),
            "horizontal_tolerance_deg": float(horizontal_tolerance_deg),
            "small_hole_diameter_threshold_mm": float(small_hole_diameter_mm),
            "minimum_angular_bins_of_12": int(min_angular_bins),
            "minimum_axis_length_mm": float(min_axis_length_mm),
        },
        "summary": {
            "recognized_internal_cylinders": len(features),
            "recognized_horizontal_internal_cylinders": sum(
                feature["is_horizontal_to_build"] for feature in features
            ),
            "small_horizontal_exemptions": sum(
                feature["small_horizontal_exempt"] for feature in features
            ),
            "recognized_feature_face_count": sum(
                int(feature["face_count"]) for feature in features
            ),
            "recognized_feature_area_mm2": float(
                sum(feature["surface_area_mm2"] for feature in features)
            ),
            "rejected_region_counts": dict(sorted(rejections.items())),
        },
        "features": features,
        "suggested_annotation": {
            # Recognition alone does not make a region forbidden. The caller
            # must resolve a user support query before adding hard no-support faces.
            "no_support_face_indices": [],
            "features": exemption_features,
        },
    }


def merge_annotations(
    manual: dict[str, Any] | None,
    recognized: dict[str, Any],
    face_count: int,
) -> dict[str, Any]:
    """Merge manual face annotations with automatic recognized-hole constraints."""
    manual = manual or {}
    no_support = {
        int(face_id)
        for face_id in manual.get("no_support_face_indices", [])
    }
    no_support.update(
        int(face_id)
        for face_id in recognized.get("suggested_annotation", {}).get("no_support_face_indices", [])
    )
    features = list(manual.get("features", []))
    existing_auto = {
        (str(feature.get("kind", "")), tuple(int(i) for i in feature.get("face_indices", [])))
        for feature in features
        if feature.get("source") == "automatic_geometry"
    }
    for feature in recognized.get("suggested_annotation", {}).get("features", []):
        key = (str(feature["kind"]), tuple(int(i) for i in feature["face_indices"]))
        if key not in existing_auto:
            features.append(feature)
            existing_auto.add(key)
    if any(face_id < 0 or face_id >= face_count for face_id in no_support):
        raise ValueError("merged no-support annotation contains an out-of-range face index")
    return {
        "no_support_face_indices": sorted(no_support),
        "features": features,
    }


def reclassify_features_for_build_direction(
    recognized: dict[str, Any],
    build_up: np.ndarray,
    small_hole_diameter_mm: float,
    horizontal_tolerance_deg: float = 10.0,
) -> dict[str, Any]:
    """Recompute orientation-dependent exemption policy without refitting geometry."""
    up = np.asarray(build_up, dtype=np.float64)
    up /= np.linalg.norm(up)
    features: list[dict[str, Any]] = []
    for source in recognized.get("features", []):
        feature = dict(source)
        axis = np.asarray(feature["axis_input"], dtype=np.float64)
        axis_to_build = math.degrees(math.acos(float(np.clip(abs(axis @ up), 0.0, 1.0))))
        axis_to_build_plane = 90.0 - axis_to_build
        is_horizontal = axis_to_build_plane <= horizontal_tolerance_deg + 1e-10
        small_exempt = bool(is_horizontal and float(feature["diameter_mm"]) < small_hole_diameter_mm)
        feature.update({
            "axis_to_build_axis_deg": axis_to_build,
            "axis_to_build_plane_deg": axis_to_build_plane,
            "is_horizontal_to_build": bool(is_horizontal),
            "small_horizontal_exempt": small_exempt,
            "support_policy": "size_exempt" if small_exempt else "support_allowed",
        })
        features.append(feature)
    exemption_features = [{
        "kind": "horizontal_hole",
        "name": feature["feature_id"],
        "face_indices": feature["face_indices"],
        "diameter_mm": feature["diameter_mm"],
        "source": "automatic_geometry",
        "confidence": feature["confidence"],
    } for feature in features if feature["is_horizontal_to_build"]]
    result = dict(recognized)
    parameters = dict(recognized.get("parameters", {}))
    parameters["horizontal_tolerance_deg"] = float(horizontal_tolerance_deg)
    result["build_up_input"] = up.tolist()
    result["parameters"] = parameters
    result["features"] = features
    result["suggested_annotation"] = {
        "no_support_face_indices": [],
        "features": exemption_features,
    }
    summary = dict(recognized.get("summary", {}))
    summary["recognized_internal_cylinders"] = len(features)
    summary["recognized_horizontal_internal_cylinders"] = sum(
        feature["is_horizontal_to_build"] for feature in features
    )
    summary["small_horizontal_exemptions"] = sum(
        feature["small_horizontal_exempt"] for feature in features
    )
    result["summary"] = summary
    return result


def resolve_support_query(
    recognized: dict[str, Any],
    query: dict[str, Any] | None,
    face_count: int,
) -> dict[str, Any]:
    """Resolve a structured user query into selected protected feature faces.

    Selection criteria are ORed: an explicit feature id or any matching filter
    selects a feature. Face indices provide an additional direct selection path.
    """
    query = query or {}
    block = query.get("no_support", {})
    if not isinstance(block, dict):
        raise ValueError("support-query field 'no_support' must be an object")
    features = recognized.get("features", [])
    feature_by_id = {str(feature["feature_id"]): feature for feature in features}
    requested_ids = [str(feature_id) for feature_id in block.get("feature_ids", [])]
    unknown_ids = sorted(set(requested_ids) - set(feature_by_id))
    if unknown_ids:
        raise ValueError(f"support query references unknown feature ids: {unknown_ids}")

    kinds = {str(kind) for kind in block.get("feature_kinds", [])}
    all_internal_cylinders = bool(block.get("all_internal_cylinders", False))
    horizontal_only = block.get("horizontal_only")
    min_diameter = block.get("min_diameter_mm")
    max_diameter = block.get("max_diameter_mm")
    selected: list[dict[str, Any]] = []
    for feature in features:
        selected_by_id = feature["feature_id"] in requested_ids
        selected_by_kind = bool(kinds and feature["kind"] in kinds)
        selected_by_all = all_internal_cylinders and feature["kind"] == "internal_cylinder"
        selected_by_geometry = bool(
            horizontal_only is not None
            or min_diameter is not None
            or max_diameter is not None
        )
        if horizontal_only is not None:
            selected_by_geometry &= bool(feature["is_horizontal_to_build"]) == bool(horizontal_only)
        if min_diameter is not None:
            selected_by_geometry &= float(feature["diameter_mm"]) >= float(min_diameter)
        if max_diameter is not None:
            selected_by_geometry &= float(feature["diameter_mm"]) <= float(max_diameter)
        if selected_by_id or selected_by_kind or selected_by_all or selected_by_geometry:
            selected.append(feature)

    no_support_faces = {
        int(face_id)
        for face_id in block.get("face_indices", [])
    }
    for feature in selected:
        no_support_faces.update(int(face_id) for face_id in feature["face_indices"])
    if any(face_id < 0 or face_id >= face_count for face_id in no_support_faces):
        raise ValueError("support query contains an out-of-range face index")
    clearance = block.get("clearance_mm")
    if clearance is not None and (not math.isfinite(float(clearance)) or float(clearance) < 0):
        raise ValueError("support-query clearance_mm must be finite and nonnegative")

    return {
        "query_text": query.get("query_text"),
        "selected_feature_ids": [feature["feature_id"] for feature in selected],
        "no_support_face_indices": sorted(no_support_faces),
        "clearance_mm": None if clearance is None else float(clearance),
        "resolved_selection": {
            "explicit_feature_ids": requested_ids,
            "feature_kinds": sorted(kinds),
            "all_internal_cylinders": all_internal_cylinders,
            "horizontal_only": horizontal_only,
            "min_diameter_mm": min_diameter,
            "max_diameter_mm": max_diameter,
            "direct_face_count": len(block.get("face_indices", [])),
        },
    }


def segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Return the shortest distance between two finite 3D line segments."""
    p1 = np.asarray(first_start, dtype=np.float64)
    q1 = np.asarray(first_end, dtype=np.float64)
    p2 = np.asarray(second_start, dtype=np.float64)
    q2 = np.asarray(second_end, dtype=np.float64)
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:
        return float(np.linalg.norm(p1 - p2))
    if a <= epsilon:
        s, t = 0.0, float(np.clip(f / max(e, epsilon), 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= epsilon:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denominator, 0.0, 1.0))
                if denominator > epsilon else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def write_feature_preview(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    recognized: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = np.asarray(vertices)[np.asarray(faces)]
    protected = np.zeros(len(faces), dtype=bool)
    for feature in recognized["features"]:
        protected[np.asarray(feature["face_indices"], dtype=np.int64)] = True
    fig = plt.figure(figsize=(8.5, 7.2), facecolor="#f4f6f8")
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        triangles[~protected], facecolors="#7898b5", edgecolors="none", alpha=0.42
    ))
    ax.add_collection3d(Poly3DCollection(
        triangles[protected], facecolors="#dd3b3b", edgecolors="#7a1414", linewidths=0.15, alpha=0.92
    ))
    for feature in recognized["features"]:
        start = np.asarray(feature["axis_start_input_mm"])
        end = np.asarray(feature["axis_end_input_mm"])
        ax.plot(*np.column_stack([start, end]), color="#ffd23f", linewidth=2.0)
    low, high = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    center, extent = (low + high) / 2, np.maximum(high - low, 1e-3)
    max_span = max(float(extent.max()), 1.0)
    ax.set_xlim(center[0] - max_span * 0.55, center[0] + max_span * 0.55)
    ax.set_ylim(center[1] - max_span * 0.55, center[1] + max_span * 0.55)
    ax.set_zlim(center[2] - max_span * 0.55, center[2] + max_span * 0.55)
    ax.set_box_aspect(tuple(np.maximum(extent / max_span, 0.12)))
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"Recognized internal cylinders: {recognized['summary']['recognized_internal_cylinders']}"
    )
    ax.view_init(elev=25, azim=-55)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def recognize_file(
    path: Path,
    output: Path | None = None,
    preview: Path | None = None,
    build_up: np.ndarray | None = None,
    small_hole_diameter_mm: float = 6.0,
    min_confidence: float = 0.58,
    min_angular_bins: int = 8,
    min_axis_length_mm: float = 0.25,
) -> dict[str, Any]:
    try:
        from tools.generate_simplified_supports import read_mesh
    except ModuleNotFoundError:
        from generate_simplified_supports import read_mesh

    vertices, faces = read_mesh(path)
    result = recognize_features(
        vertices,
        faces,
        build_up=build_up,
        min_confidence=min_confidence,
        small_hole_diameter_mm=small_hole_diameter_mm,
        min_angular_bins=min_angular_bins,
        min_axis_length_mm=min_axis_length_mm,
    )
    result["input_mesh"] = str(path.resolve())
    result["source_mesh_counts"] = {"vertices": int(len(vertices)), "faces": int(len(faces))}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_json"] = str(output.resolve())
    if preview is not None:
        write_feature_preview(preview, vertices, faces, result)
        result["preview"] = str(preview.resolve())
    return result


def _batch_worker(
    payload: tuple[str, str, str, list[float], float, float, int, float],
) -> dict[str, Any]:
    (
        mesh_text, output_text, preview_text, build_up,
        diameter, confidence, min_angular_bins, min_axis_length_mm,
    ) = payload
    preview = Path(preview_text) if preview_text else None
    result = recognize_file(
        Path(mesh_text),
        Path(output_text),
        preview,
        np.asarray(build_up, dtype=np.float64),
        diameter,
        confidence,
        min_angular_bins,
        min_axis_length_mm,
    )
    return {
        "mesh": str(Path(mesh_text).resolve()),
        "output": str(Path(output_text).resolve()),
        "preview": str(preview.resolve()) if preview is not None else None,
        "summary": result["summary"],
        "faces": result["source_mesh_counts"]["faces"],
    }


def write_batch_csv(output_dir: Path, parts: list[dict[str, Any]]) -> tuple[Path, Path]:
    part_path = output_dir / "part_summary.csv"
    with part_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "mesh", "faces", "internal_cylinders", "horizontal_internal_cylinders",
            "small_horizontal_exemptions", "recognized_feature_face_count",
            "recognized_feature_area_mm2",
            "feature_json",
        ])
        writer.writeheader()
        for part in parts:
            summary = part["summary"]
            writer.writerow({
                "mesh": part["mesh"],
                "faces": part.get("faces", ""),
                "internal_cylinders": summary["recognized_internal_cylinders"],
                "horizontal_internal_cylinders": summary["recognized_horizontal_internal_cylinders"],
                "small_horizontal_exemptions": summary["small_horizontal_exemptions"],
                "recognized_feature_face_count": summary["recognized_feature_face_count"],
                "recognized_feature_area_mm2": summary["recognized_feature_area_mm2"],
                "feature_json": part["output"],
            })

    detail_path = output_dir / "feature_summary.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "mesh", "feature_id", "diameter_mm", "axis_length_mm",
            "axis_to_build_plane_deg", "is_horizontal", "small_horizontal_exempt",
            "face_count", "surface_area_mm2", "confidence", "support_policy",
        ])
        writer.writeheader()
        for part in parts:
            data = json.loads(Path(part["output"]).read_text(encoding="utf-8"))
            for feature in data["features"]:
                writer.writerow({
                    "mesh": part["mesh"],
                    "feature_id": feature["feature_id"],
                    "diameter_mm": feature["diameter_mm"],
                    "axis_length_mm": feature["axis_length_mm"],
                    "axis_to_build_plane_deg": feature["axis_to_build_plane_deg"],
                    "is_horizontal": feature["is_horizontal_to_build"],
                    "small_horizontal_exempt": feature["small_horizontal_exempt"],
                    "face_count": feature["face_count"],
                    "surface_area_mm2": feature["surface_area_mm2"],
                    "confidence": feature["confidence"],
                    "support_policy": feature["support_policy"],
                })
    return part_path, detail_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, nargs="?", help="single OBJ, STL, or NPZ mesh")
    parser.add_argument("--input-root", type=Path, help="recursively process all .stl files below this root")
    parser.add_argument("--output", type=Path, help="JSON output for a single mesh")
    parser.add_argument("--output-dir", type=Path, help="JSON output directory for batch mode")
    parser.add_argument("--preview", type=Path, help="preview PNG for a single mesh")
    parser.add_argument("--no-batch-previews", action="store_true", help="omit per-part previews in batch mode")
    parser.add_argument("--build-up", nargs=3, type=float, default=[0.0, 0.0, 1.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--small-hole-diameter-mm", type=float, default=6.0)
    parser.add_argument("--min-confidence", type=float, default=0.58)
    parser.add_argument("--min-angular-bins", type=int, default=8)
    parser.add_argument("--min-axis-length-mm", type=float, default=0.25)
    parser.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 16)))
    args = parser.parse_args()
    if bool(args.mesh) == bool(args.input_root):
        parser.error("provide exactly one of mesh or --input-root")
    if args.small_hole_diameter_mm <= 0:
        parser.error("small-hole-diameter-mm must be positive")
    if not 1 <= args.min_angular_bins <= 12:
        parser.error("min-angular-bins must be between 1 and 12")
    if args.min_axis_length_mm <= 0:
        parser.error("min-axis-length-mm must be positive")

    build_up = np.asarray(args.build_up, dtype=np.float64)
    if args.mesh:
        result = recognize_file(
            args.mesh,
            args.output,
            args.preview,
            build_up,
            args.small_hole_diameter_mm,
            args.min_confidence,
            args.min_angular_bins,
            args.min_axis_length_mm,
        )
        summary = {
            "input_mesh": result["input_mesh"],
            "output_json": result.get("output_json"),
            "preview": result.get("preview"),
            "summary": result["summary"],
            "features": [{
                "feature_id": feature["feature_id"],
                "diameter_mm": feature["diameter_mm"],
                "axis_length_mm": feature["axis_length_mm"],
                "axis_to_build_plane_deg": feature["axis_to_build_plane_deg"],
                "small_horizontal_exempt": feature["small_horizontal_exempt"],
                "confidence": feature["confidence"],
            } for feature in result["features"]],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    meshes = sorted(args.input_root.rglob("*.stl"))
    if not meshes:
        parser.error(f"no .stl files found below {args.input_root}")
    output_dir = args.output_dir or (args.input_root / "recognized_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))
    payloads = []
    for mesh in meshes:
        relative = mesh.relative_to(args.input_root)
        stem = "__".join(relative.with_suffix("").parts)
        output = output_dir / f"{stem}.features.json"
        preview = output_dir / "previews" / f"{stem}.features.png"
        payloads.append((
            str(mesh), str(output), str(preview), build_up.tolist(),
            args.small_hole_diameter_mm, args.min_confidence,
            args.min_angular_bins, args.min_axis_length_mm,
        ))
    if args.no_batch_previews:
        payloads = [
            (mesh, output, "", up, diameter, confidence, bins, min_length)
            for mesh, output, _, up, diameter, confidence, bins, min_length in payloads
        ]

    results: list[dict[str, Any]] = []
    if jobs == 1:
        for payload in payloads:
            if not payload[2]:
                mesh, output, _, up, diameter, confidence, bins, min_length = payload
                result = recognize_file(
                    Path(mesh), Path(output), None, np.asarray(up),
                    diameter, confidence, bins, min_length,
                )
                results.append({
                    "mesh": str(Path(mesh).resolve()),
                    "output": str(Path(output).resolve()),
                    "preview": None,
                    "summary": result["summary"],
                    "faces": result["source_mesh_counts"]["faces"],
                })
            else:
                results.append(_batch_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_batch_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["mesh"])
    batch = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "input_root": str(args.input_root.resolve()),
        "mesh_count": len(meshes),
        "summary": {
            "parts_with_recognized_internal_cylinders": sum(
                row["summary"]["recognized_internal_cylinders"] > 0 for row in results
            ),
            "recognized_internal_cylinders": sum(
                row["summary"]["recognized_internal_cylinders"] for row in results
            ),
            "recognized_horizontal_internal_cylinders": sum(
                row["summary"]["recognized_horizontal_internal_cylinders"] for row in results
            ),
            "small_horizontal_exemptions": sum(
                row["summary"]["small_horizontal_exemptions"] for row in results
            ),
        },
        "parts": results,
    }
    batch_path = output_dir / "batch_summary.json"
    batch_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    part_csv, feature_csv = write_batch_csv(output_dir, results)
    print(json.dumps({
        "batch_summary": str(batch_path.resolve()),
        "part_summary_csv": str(part_csv.resolve()),
        "feature_summary_csv": str(feature_csv.resolve()),
        "mesh_count": len(meshes),
        **batch["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
