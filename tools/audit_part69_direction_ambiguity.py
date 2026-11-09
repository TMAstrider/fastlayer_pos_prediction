"""Measure the half-turn surface symmetry and error contribution of part 69."""

import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/eight_part_diagnostic'
    mesh = trimesh.load(output / 'part_069_GT.stl', force='mesh', process=True)
    rotation = np.diag([-1., 1., -1.])
    center = mesh.bounds.mean(0)
    points, _ = trimesh.sample.sample_surface(mesh, 12000, seed=20261109)
    points = np.vstack([mesh.vertices, points])
    transformed = (points-center) @ rotation.T + center
    _, distances, _ = trimesh.proximity.closest_point(mesh, transformed)
    result = {
        'rotation': rotation.tolist(), 'rotation_center_mm': center.tolist(),
        'target_before': [0, 0, 1], 'target_after': [0, 0, -1],
        'surface_query_count': len(points),
        'half_turn_surface_distance_mm': {
            'mean': float(distances.mean()), 'p95': float(np.percentile(distances, 95)),
            'max': float(distances.max()),
        },
        'scope': 'Original vertices plus 12000 seeded area samples to exact triangle surface; approximate symmetry, not an exhaustive certified Hausdorff bound.',
    }
    with np.load(root / 'runs/pointnet_random_so3_8parts_extended_2026-11-09/independent_predictions.npz') as saved:
        predicted, target, ids = saved['predicted_axis'], saved['target_axis'], saved['part_indices']
    predicted = predicted/np.linalg.norm(predicted, axis=1, keepdims=True)
    target = target/np.linalg.norm(target, axis=1, keepdims=True)
    dots = np.clip((predicted*target).sum(-1), -1, 1)
    angles = np.degrees(np.arccos(dots))
    result['per_part'] = []
    for index in np.unique(ids):
        local = ids == index
        result['per_part'].append({
            'index': int(index), 'cosine_loss': float((1-dots[local]).mean()),
            'signed_mean_angle_deg': float(angles[local].mean()),
            'unsigned_axis_mean_angle_deg': float(np.degrees(np.arccos(np.abs(dots[local]))).mean()),
            'contribution_to_total_loss': float((1-dots[local]).sum()/len(ids)),
        })
    result['overall_loss'] = float((1-dots).mean())
    result['part69_fraction_of_total_loss_pct'] = float((1-dots[ids==69]).sum()/(1-dots).sum()*100)
    (output / 'part69_ambiguity_audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
