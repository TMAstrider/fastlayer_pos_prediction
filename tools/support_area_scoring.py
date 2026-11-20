"""Deterministic support-projection scoring for candidate build directions.

This module intentionally models only the support requirement:

1. rotate the part so a candidate build direction becomes +Z;
2. classify down-facing overhang triangles with the configured angle threshold;
3. apply explicit and automatically recognized small-feature exemptions;
4. project retained overhang triangles onto the build plane;
5. rasterize their union and report the projected area.

It does not ray-cast support columns or evaluate user no-support queries. Those
operations belong to the full support proxy and the inference-time query filter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

try:
    from tools.generate_simplified_supports import (
        annotation_masks,
        classify_overhang_faces,
        rotation_align_to_z,
        triangle_geometry,
    )
    from tools.recognize_part_features import (
        merge_annotations,
        reclassify_features_for_build_direction,
    )
except ModuleNotFoundError:
    from generate_simplified_supports import (
        annotation_masks,
        classify_overhang_faces,
        rotation_align_to_z,
        triangle_geometry,
    )
    from recognize_part_features import (
        merge_annotations,
        reclassify_features_for_build_direction,
    )


@dataclass(frozen=True)
class ProjectedAreaMask:
    occupied: np.ndarray
    area_mm2: float
    cell_size_mm: float
    minimum_cell_x: int
    minimum_cell_y: int


def normalise_projected_areas(
    areas: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Map projected areas to scores where the smallest area receives 1.0."""
    values = np.asarray(areas, dtype=np.float64)
    minimum = float(values.min()) if len(values) else 0.0
    maximum = float(values.max()) if len(values) else 0.0
    if maximum - minimum <= 1e-12:
        return np.ones_like(values), minimum, maximum
    return 1.0 - (values - minimum) / (maximum - minimum), minimum, maximum


def dense_area_ranks(areas: np.ndarray) -> np.ndarray:
    """Return one-based dense ranks, assigning equal areas the same rank."""
    unique = np.unique(np.asarray(areas, dtype=np.float64))
    return np.searchsorted(unique, np.asarray(areas, dtype=np.float64)) + 1


def projected_triangle_area_sum(
    triangles: np.ndarray,
    face_indices: np.ndarray,
) -> float:
    """Return the sum of triangle areas projected onto the XY build plane."""
    indices = np.asarray(face_indices, dtype=np.int64)
    if not len(indices):
        return 0.0
    projected = np.asarray(triangles, dtype=np.float64)[indices, :, :2]
    cross = (
        (projected[:, 1, 0] - projected[:, 0, 0])
        * (projected[:, 2, 1] - projected[:, 0, 1])
        - (projected[:, 1, 1] - projected[:, 0, 1])
        * (projected[:, 2, 0] - projected[:, 0, 0])
    )
    return float(np.abs(cross).sum() * 0.5)


def rasterize_projected_union(
    triangles: np.ndarray,
    face_indices: np.ndarray,
    cell_size_mm: float = 0.5,
    max_cells: int = 10_000_000,
) -> ProjectedAreaMask:
    """Rasterize the union of projected triangles on a build-plane grid.

    Pixel centers are used as occupancy samples. The grid is anchored at the
    coordinate origin, so translating the input mesh does not change the area
    estimate. Overlapping triangles at different build heights are counted once.
    """
    if not np.isfinite(cell_size_mm) or cell_size_mm <= 0:
        raise ValueError("projection cell size must be finite and positive")
    if max_cells <= 0:
        raise ValueError("max projected cells must be positive")

    indices = np.asarray(face_indices, dtype=np.int64)
    if not len(indices):
        return ProjectedAreaMask(
            occupied=np.zeros((0, 0), dtype=bool),
            area_mm2=0.0,
            cell_size_mm=float(cell_size_mm),
            minimum_cell_x=0,
            minimum_cell_y=0,
        )

    projected = np.asarray(triangles, dtype=np.float64)[indices, :, :2]
    cell_indices = np.floor(projected / float(cell_size_mm)).astype(np.int64)
    minimum = cell_indices.reshape(-1, 2).min(axis=0)
    maximum = cell_indices.reshape(-1, 2).max(axis=0)
    width, height = (maximum - minimum + 1).tolist()
    if width * height > max_cells:
        raise ValueError(
            f"projected raster would contain {width * height} cells, "
            f"above the configured limit {max_cells}; increase cell size"
        )

    # Build one or more x intervals for each raster row from every projected
    # triangle, then merge the intervals per row. This samples cell centers and
    # avoids Pillow's boundary expansion when a mesh contains thousands of very
    # small adjacent triangles.
    intervals_by_row: list[list[tuple[int, int]]] = [[] for _ in range(int(height))]
    tolerance = 1e-10
    for triangle in projected:
        cross = (
            (triangle[1, 0] - triangle[0, 0])
            * (triangle[2, 1] - triangle[0, 1])
            - (triangle[1, 1] - triangle[0, 1])
            * (triangle[2, 0] - triangle[0, 0])
        )
        if abs(float(cross)) <= 1e-12:
            continue
        y_min = float(triangle[:, 1].min())
        y_max = float(triangle[:, 1].max())
        row_start = int(math.ceil(y_min / float(cell_size_mm) - 0.5 - tolerance))
        row_end = int(math.floor(y_max / float(cell_size_mm) - 0.5 + tolerance))
        row_start = max(row_start, int(minimum[1]))
        row_end = min(row_end, int(maximum[1]))
        if row_start > row_end:
            continue
        row_indices = np.arange(row_start, row_end + 1, dtype=np.int64)
        y_centers = (row_indices + 0.5) * float(cell_size_mm)
        x_lower = np.full(len(row_indices), np.inf, dtype=np.float64)
        x_upper = np.full(len(row_indices), -np.inf, dtype=np.float64)
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            delta_y = float(second[1] - first[1])
            if abs(delta_y) <= tolerance:
                continue
            edge_low = min(float(first[1]), float(second[1]))
            edge_high = max(float(first[1]), float(second[1]))
            active = (
                (y_centers >= edge_low - tolerance)
                & (y_centers <= edge_high + tolerance)
            )
            if np.any(active):
                ratio = (y_centers[active] - float(first[1])) / delta_y
                x_values = float(first[0]) + ratio * float(second[0] - first[0])
                active_indices = np.flatnonzero(active)
                x_lower[active_indices] = np.minimum(
                    x_lower[active_indices], x_values
                )
                x_upper[active_indices] = np.maximum(
                    x_upper[active_indices], x_values
                )
        valid_rows = np.isfinite(x_lower) & np.isfinite(x_upper)
        for offset in np.flatnonzero(valid_rows):
            column_start = int(math.ceil(
                x_lower[offset] / float(cell_size_mm) - 0.5 - tolerance
            ))
            column_end = int(math.floor(
                x_upper[offset] / float(cell_size_mm) - 0.5 + tolerance
            ))
            column_start = max(column_start, int(minimum[0]))
            column_end = min(column_end, int(maximum[0]))
            if column_start > column_end:
                continue
            row_index = int(row_indices[offset])
            intervals_by_row[row_index - int(minimum[1])].append(
                (column_start, column_end)
            )

    covered_cells = 0
    for intervals in intervals_by_row:
        if not intervals:
            continue
        intervals.sort()
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end + 1:
                end = max(end, next_end)
            else:
                covered_cells += end - start + 1
                start, end = next_start, next_end
        covered_cells += end - start + 1
    occupied = np.zeros((int(width), int(height)), dtype=bool)
    # The mask itself is used only for the selected best direction's diagnostic
    # image. Reusing the merged intervals avoids a second triangle sweep.
    for row_offset, intervals in enumerate(intervals_by_row):
        if not intervals:
            continue
        intervals.sort()
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end + 1:
                end = max(end, next_end)
            else:
                occupied[
                    start - int(minimum[0]):end - int(minimum[0]) + 1,
                    row_offset,
                ] = True
                start, end = next_start, next_end
        occupied[
            start - int(minimum[0]):end - int(minimum[0]) + 1,
            row_offset,
        ] = True
    return ProjectedAreaMask(
        occupied=occupied,
        area_mm2=float(covered_cells * float(cell_size_mm) ** 2),
        cell_size_mm=float(cell_size_mm),
        minimum_cell_x=int(minimum[0]),
        minimum_cell_y=int(minimum[1]),
    )


def evaluate_support_projection(
    vertices: np.ndarray,
    faces: np.ndarray,
    limits: dict[str, float],
    annotations: dict[str, Any] | None = None,
    up: np.ndarray | None = None,
    recognized_features: dict[str, Any] | None = None,
    projection_grid_mm: float = 0.5,
    max_projection_cells: int = 10_000_000,
) -> dict[str, Any]:
    """Evaluate angle-filtered support projection for one build direction."""
    up_vector = np.asarray(up if up is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(up_vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("build-up vector must be finite and nonzero")
    up_vector /= norm

    rotation = rotation_align_to_z(up_vector)
    aligned_vertices = np.asarray(vertices, dtype=np.float64) @ rotation.T
    triangles, normals, areas = triangle_geometry(aligned_vertices, faces)

    oriented_features = None
    merged_annotations = annotations or {}
    if recognized_features is not None:
        oriented_features = reclassify_features_for_build_direction(
            recognized_features,
            up_vector,
            limits["horizontal_hole_diameter_threshold_mm"],
        )
        merged_annotations = merge_annotations(
            annotations,
            oriented_features,
            len(faces),
        )

    no_support, feature_exempt, feature_face_counts = annotation_masks(
        len(faces),
        merged_annotations,
        limits,
    )
    angle_candidates, candidate_faces, face_angles = classify_overhang_faces(
        normals,
        areas,
        limits["self_support_angle_threshold_deg"],
        feature_exempt,
    )
    exempt_faces = angle_candidates & feature_exempt
    candidate_indices = np.flatnonzero(candidate_faces)
    projected_sum = projected_triangle_area_sum(triangles, candidate_indices)
    projected_mask = rasterize_projected_union(
        triangles,
        candidate_indices,
        projection_grid_mm,
        max_projection_cells,
    )

    report = {
        "algorithm": "support_projection_proxy_v0",
        "build_up_input": up_vector.tolist(),
        "thresholds": dict(limits),
        "projection": {
            "cell_size_mm": float(projection_grid_mm),
            "occupied_cells": int(projected_mask.occupied.sum()),
            "union_area_mm2": float(projected_mask.area_mm2),
            "triangle_area_sum_mm2": float(projected_sum),
            "duplicate_projection_overlap_mm2": float(
                max(projected_sum - projected_mask.area_mm2, 0.0)
            ),
        },
        "face_counts": {
            "total": int(len(faces)),
            "downward_angle_candidates": int(angle_candidates.sum()),
            "feature_exempt": int(exempt_faces.sum()),
            "support_candidate_after_exemptions": int(candidate_faces.sum()),
            "no_support_faces": int(no_support.sum()),
            "support_candidate_face_indices": candidate_indices.tolist(),
        },
        "areas_mm2": {
            "downward_angle_candidates": float(areas[angle_candidates].sum()),
            "feature_exempt": float(areas[exempt_faces].sum()),
            "retained_overhang_surface": float(areas[candidate_faces].sum()),
            "projected_triangle_sum": float(projected_sum),
            "projected_union": float(projected_mask.area_mm2),
        },
        "feature_exempt_face_counts_by_kind": feature_face_counts,
        "recognized_feature_summary": (
            oriented_features.get("summary", {}) if oriented_features is not None else None
        ),
        "limitations": [
            "This evaluator applies deterministic angle and small-feature rules only.",
            "Projected area is a rasterized union of retained overhang triangles, not a support solid or Magics output.",
            "User no-support queries are intentionally excluded from this training label.",
        ],
    }
    return {
        "report": report,
        "aligned_vertices": aligned_vertices,
        "triangles": triangles,
        "normals": normals,
        "areas": areas,
        "face_angles": face_angles,
        "candidate_faces": candidate_faces,
        "exempt_faces": exempt_faces,
        "no_support_faces": no_support,
        "projected_mask": projected_mask,
        "oriented_features": oriented_features,
    }
