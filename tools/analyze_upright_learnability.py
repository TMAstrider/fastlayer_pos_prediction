"""Audit whether the stored build-direction label is visible from shape alone.

This script is diagnostic only: it reads the existing deduplicated point clouds,
the fixed 932/100 split, and the saved predictions of the stopped full
equivariant run. It does not train or modify any model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/upright_learnability_analysis_2026-11-09'


def describe_shape(points: np.ndarray, normals: np.ndarray) -> dict:
    points = points.astype(np.float64)
    normals = normals.astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

    extent = points.max(0) - points.min(0)
    centered = points - points.mean(0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(centered))
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    pca_angle = float(np.degrees(np.arccos(np.clip(np.abs(eigenvectors[2]).max(), 0, 1))))

    # Quantized surface-normal clusters approximate planar patches. Point
    # samples are area-uniform, so cluster counts are area weights.
    keys = np.round(normals, 2)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    top = np.argsort(counts)[::-1][:10]
    top_angle = float(np.degrees(np.arccos(np.clip(
        np.abs(unique[top] @ np.array([0., 0., 1.])).max(), 0, 1))))
    largest_direction = unique[top[0]] / max(np.linalg.norm(unique[top[0]]), 1e-9)
    largest_members = np.all(keys == unique[top[0]], axis=1)
    largest_mean = normals[largest_members].mean(0)
    largest_mean /= max(np.linalg.norm(largest_mean), 1e-9)
    largest_signed = float(np.degrees(np.arccos(np.clip(largest_mean[2], -1, 1))))
    largest_unsigned = min(largest_signed, 180. - largest_signed)

    zmin, zmax = points[:, 2].min(), points[:, 2].max()
    height = max(zmax - zmin, 1e-9)
    bottom = points[:, 2] < zmin + 0.02 * height
    bottom_normal_z = float(normals[bottom, 2].mean()) if bottom.any() else float('nan')

    return {
        'extent_xyz': extent.tolist(),
        'pca_angle_deg': pca_angle,
        'z_extent_rank': int(np.argsort(np.argsort(extent))[2]),
        'top10_normal_angle_deg': top_angle,
        'largest_face_fraction': float(counts[top[0]] / len(normals)),
        'largest_face_signed_angle_deg': largest_signed,
        'largest_face_unsigned_angle_deg': largest_unsigned,
        'bottom_2pct_normal_z': bottom_normal_z,
    }


def main() -> None:
    report = REPORT
    report.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'outputs/shape_deduplicated_normals/manifest.json').read_text(encoding='utf8'))
    split = json.loads((ROOT / 'runs/pointnet_shape_dedup_2026-11-09_val100/split.json').read_text(encoding='utf8'))
    validation_ids = set(split['validation_indices'])

    rows = []
    for record in manifest['records']:
        with np.load(ROOT / 'outputs/shape_deduplicated_normals' / record['pointcloud_file']) as data:
            description = describe_shape(data['points'], data['normals'])
        rows.append({'part_index': record['part_index'], **description})

    # Attach the saved best-validation predictions of the stopped full run.
    prediction_path = ROOT / 'runs/equivariant_full_resampled_2026-11-09/best_validation_predictions.npz'
    prediction_summary = {}
    if prediction_path.exists():
        with np.load(prediction_path) as saved:
            predicted = saved['predicted_axis'].astype(np.float64)
            target = saved['target_axis'].astype(np.float64)
            part_ids = saved['part_indices']
        predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-9)
        target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-9)
        cosine = np.clip((predicted * target).sum(1), -1, 1)
        angle = np.degrees(np.arccos(cosine))
        unsigned = np.degrees(np.arccos(np.abs(cosine)))
        per_part = []
        for part in np.unique(part_ids):
            mask = part_ids == part
            per_part.append({
                'part_index': int(part),
                'mean_signed_angle_deg': float(angle[mask].mean()),
                'mean_unsigned_angle_deg': float(unsigned[mask].mean()),
            })
        prediction_summary = {
            'part_count': int(len(np.unique(part_ids))),
            'mean_signed_angle_deg': float(angle.mean()),
            'median_signed_angle_deg': float(np.median(angle)),
            'mean_unsigned_angle_deg': float(unsigned.mean()),
            'median_unsigned_angle_deg': float(np.median(unsigned)),
            'within_10_unsigned_pct': float(100 * np.mean(unsigned < 10)),
            'within_20_unsigned_pct': float(100 * np.mean(unsigned < 20)),
            'within_30_unsigned_pct': float(100 * np.mean(unsigned < 30)),
            'parts_below_20_signed': int(sum(row['mean_signed_angle_deg'] < 20 for row in per_part)),
            'per_part': per_part,
        }

    all_arrays = {key: np.array([row[key] for row in rows]) for key in rows[0] if key != 'part_index'}
    validation_rows = [row for row in rows if row['part_index'] in validation_ids]
    validation_arrays = {key: np.array([row[key] for row in validation_rows]) for key in validation_rows[0] if key != 'part_index'}

    def percentiles(array: np.ndarray) -> dict:
        values = np.asarray(array, dtype=float)
        values = values[np.isfinite(values)]
        return {f'p{q}': float(np.percentile(values, q)) for q in (5, 10, 25, 50, 75, 90, 95)}

    summary = {
        'shape_count': len(rows),
        'validation_shape_count': len(validation_rows),
        'pca_angle_deg': percentiles(all_arrays['pca_angle_deg']),
        'pca_within_10deg_pct': float(100 * np.mean(all_arrays['pca_angle_deg'] < 10)),
        'z_extent_rank_counts': {str(i): int(np.sum(all_arrays['z_extent_rank'] == i)) for i in range(3)},
        'top10_normal_angle_deg': percentiles(all_arrays['top10_normal_angle_deg']),
        'top10_normal_within_10deg_pct': float(100 * np.mean(all_arrays['top10_normal_angle_deg'] < 10)),
        'top10_normal_within_20deg_pct': float(100 * np.mean(all_arrays['top10_normal_angle_deg'] < 20)),
        'largest_face_fraction': percentiles(all_arrays['largest_face_fraction']),
        'largest_face_signed_angle_deg': percentiles(all_arrays['largest_face_signed_angle_deg']),
        'largest_face_signed_within_10deg_pct': float(100 * np.mean(np.abs(all_arrays['largest_face_signed_angle_deg']) < 10)),
        'largest_face_unsigned_within_10deg_pct': float(100 * np.mean(all_arrays['largest_face_unsigned_angle_deg'] < 10)),
        'bottom_2pct_normal_z': percentiles(all_arrays['bottom_2pct_normal_z']),
        'bottom_flat_share_pct': float(100 * np.mean(all_arrays['bottom_2pct_normal_z'] < -0.8)),
        'validation': {
            'pca_within_10deg_pct': float(100 * np.mean(validation_arrays['pca_angle_deg'] < 10)),
            'top10_normal_within_10deg_pct': float(100 * np.mean(validation_arrays['top10_normal_angle_deg'] < 10)),
            'bottom_flat_share_pct': float(100 * np.mean(validation_arrays['bottom_2pct_normal_z'] < -0.8)),
        },
        'saved_prediction': prediction_summary,
    }

    (report / 'evidence.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    with (report / 'shape_geometry.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != 'saved_prediction'}, ensure_ascii=False, indent=2))
    if prediction_summary:
        print(json.dumps({key: value for key, value in prediction_summary.items() if key != 'per_part'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
