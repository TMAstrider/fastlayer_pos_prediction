"""Attach source triangle normals to the unchanged deduplicated point clouds."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def attach(item):
    index, record = item
    source = ROOT / 'outputs/shape_deduplicated_pointclouds' / record['pointcloud_file']
    destination = ROOT / 'outputs/shape_deduplicated_normals' / record['pointcloud_file']
    audit_path = destination.with_suffix('.json')
    with np.load(source) as data:
        points = data['points'].copy()
    point_hash = hashlib.sha256(points.tobytes()).hexdigest()
    if destination.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf8'))
        if audit['point_bytes_sha256'] == point_hash:
            with np.load(destination) as data:
                np.testing.assert_array_equal(data['points'], points)
                np.testing.assert_allclose(np.linalg.norm(data['normals'], axis=1), 1, atol=1e-5)
            return audit
    started = time.monotonic()
    source_mesh = ROOT / record['source_stl_file']
    mesh = trimesh.load(source_mesh, force='mesh', process=True)
    watertight = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent)
    volume = float(mesh.volume)
    if watertight and winding and volume < 0:
        mesh.invert()
        action = 'Inverted closed, consistently wound negative-volume mesh to outward normals'
    else:
        action = 'Kept source face normals; no geometric repair or arbitrary orientation inference'
    distance_chunks, face_chunks = [], []
    for start in range(0, len(points), 128):
        _, local_distances, local_faces = trimesh.proximity.closest_point(mesh, points[start:start+128])
        distance_chunks.append(local_distances)
        face_chunks.append(local_faces)
    distances, face_ids = np.concatenate(distance_chunks), np.concatenate(face_chunks)
    if not np.isfinite(distances).all() or distances.max() > .01:
        raise ValueError(f'Part {index}: source surface residual {distances.max()} mm')
    normals = mesh.face_normals[face_ids].astype(np.float32)
    if not np.isfinite(normals).all():
        raise ValueError(f'Part {index}: invalid normals')
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
    np.savez_compressed(destination, points=points, normals=normals)
    audit = {
        'part_index': index, 'pointcloud_file': record['pointcloud_file'],
        'shape_group_id': record['shape_group_id'], 'source_stl_file': record['source_stl_file'],
        'point_bytes_sha256': point_hash, 'point_count': len(points),
        'source_mesh_sha256': hashlib.sha256(source_mesh.read_bytes()).hexdigest(),
        'point_to_mesh_max_distance_mm': float(distances.max()),
        'point_to_mesh_mean_distance_mm': float(distances.mean()),
        'mesh_watertight': watertight, 'mesh_winding_consistent': winding,
        'source_signed_volume': volume, 'normal_orientation_action': action,
        'outward_orientation_checked': watertight and winding and volume != 0,
        'seconds': time.monotonic()-started,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    output = ROOT / 'outputs/shape_deduplicated_normals'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'outputs/shape_deduplicated_pointclouds/manifest.json').read_text(encoding='utf8'))
    rows, failures = [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(attach, item): item[0] for item in enumerate(manifest['records'])}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append({'part_index': futures[future], 'error': repr(error)})
                print(f'Part {futures[future]} needs retry: {error!r}', flush=True)
            count = len(rows)+len(failures)
            if count == 1 or count % 25 == 0 or count == len(futures):
                print(f'{count}/{len(futures)} normals attached; failures {len(failures)}; {time.monotonic()-started:.1f}s', flush=True)
    initial_failures, failures = failures, []
    for failure in initial_failures:
        index = failure['part_index']
        try:
            rows.append(attach((index, manifest['records'][index])))
            print(f'Retry recovered part {index}', flush=True)
        except Exception as error:
            failures.append({'part_index': index, 'error': repr(error)})
            print(f'Retry failed part {index}: {error!r}', flush=True)
    result = {
        'records': sorted(rows, key=lambda row: row['part_index']), 'failures': failures,
        'workers': args.workers, 'seconds': time.monotonic()-started,
        'initial_parallel_failures': initial_failures,
        'method': 'Unchanged XYZ; closest source triangle face normals; no support geometry added',
        'limitations': ['Open or inconsistently wound meshes retain source face signs; outward normals are not certified.',
                        'Closest-triangle ties at edges can choose either incident face.'],
    }
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    if failures:
        raise RuntimeError(f'{len(failures)} failures; see {output / "manifest.json"}')
    print(json.dumps({'status': 'complete', 'count': len(rows), 'seconds': result['seconds'],
                      'uncertified_outward_count': sum(not row['outward_orientation_checked'] for row in rows)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
