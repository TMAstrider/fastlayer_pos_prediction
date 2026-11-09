"""Attach source-mesh face normals to existing points without resampling XYZ."""

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'reports/pointnet_small_fit_2026-11-09/config.json').read_text(encoding='utf8'))
    cloud_dir = root / 'outputs/shape_deduplicated_pointclouds'
    output = root / 'outputs/eight_part_normals'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in zip(config['selected_indices'], config['selected_records']):
        with np.load(cloud_dir / record['pointcloud_file']) as saved:
            points = saved['points'].copy()
        mesh = trimesh.load(root / record['source_stl_file'], force='mesh', process=True)
        if not mesh.is_winding_consistent or not mesh.is_watertight or mesh.volume <= 0:
            raise ValueError(f'Check source orientation before attaching outward normals: {index}')
        _, distances, face_indices = trimesh.proximity.closest_point(mesh, points)
        if distances.max() > .01:
            raise ValueError(f'Points not on source mesh {index}: {distances.max()} mm')
        normals = mesh.face_normals[face_indices].astype(np.float32)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
        destination = output / record['pointcloud_file']
        np.savez_compressed(destination, points=points, normals=normals)
        rows.append({'part_index': index, 'file': record['pointcloud_file'],
                     'point_bytes_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
                     'point_to_mesh_max_distance_mm': float(distances.max()),
                     'point_to_mesh_mean_distance_mm': float(distances.mean()),
                     'mesh_watertight': True, 'mesh_winding_consistent': True,
                     'mesh_positive_volume': True,
                     'normal_method': 'Closest source triangle face normal, not estimated from sparse points.'})
        print(f'Part {index}: {len(points)} unchanged XYZ + normals; max surface residual {distances.max():.6f} mm', flush=True)
    (output / 'manifest.json').write_text(json.dumps({'records': rows,
        'limitations': ['At triangle edges, closest-face tie can choose either incident face normal.',
                        'Watertightness and positive total volume do not certify every physical surface orientation.']}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
