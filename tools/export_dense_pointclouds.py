"""Build a denser point pool for the deduplicated shapes from their source STL files.

The current pool is 4096 area weighted points per shape. This writes 8192 point
pools with triangle normals, in the same layout as the existing tensors, so the
same training and evaluation code can load either.

Normals come from the sampled triangle; the orientation is corrected with the
signed volume recorded for the 4096 pool, so both pools agree on outward facing.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_build_direction_prior as core

ROOT = core.ROOT
SOURCE_POINTS = ROOT / 'outputs/shape_deduplicated_pointclouds'
SOURCE_NORMALS = ROOT / 'outputs/shape_deduplicated_normals'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--points-out', type=Path,
                        default=ROOT/'outputs/shape_deduplicated_pointclouds_8192')
    parser.add_argument('--normals-out', type=Path,
                        default=ROOT/'outputs/shape_deduplicated_normals_8192')
    arguments = parser.parse_args()
    source_manifest = json.loads((SOURCE_POINTS/'manifest.json').read_text(encoding='utf8'))
    audit = {row['part_index']: row
             for row in json.loads((SOURCE_NORMALS/'manifest.json').read_text(encoding='utf8'))['records']}
    records = source_manifest['records'][:arguments.limit] if arguments.limit else source_manifest['records']
    arguments.points_out.mkdir(parents=True, exist_ok=True)
    arguments.normals_out.mkdir(parents=True, exist_ok=True)
    cloud_records, normal_records, failures = [], [], []
    for index, record in enumerate(records):
        try:
            mesh = trimesh.load(ROOT/record['source_stl_file'], process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            points, faces = trimesh.sample.sample_surface(mesh, arguments.points,
                                                          seed=arguments.seed + index)
            if len(points) != arguments.points:
                raise ValueError(f'expected {arguments.points} points, got {len(points)}')
            normals = np.asarray(mesh.face_normals)[faces]
            if (audit[index].get('source_signed_volume') or 0.) < 0:
                normals = -normals
            points = points.astype(np.float32)
            normals = normals.astype(np.float32)
            np.savez_compressed(arguments.points_out/record['pointcloud_file'], points=points)
            np.savez_compressed(arguments.normals_out/record['pointcloud_file'],
                                points=points, normals=normals)
            cloud_records.append({**record, 'point_count': int(len(points))})
            normal_records.append({**audit[index], 'pointcloud_file': record['pointcloud_file'],
                                   'point_count': int(len(points))})
        except Exception as error:
            failures.append({'part_index': index, 'error': repr(error)})
        if (index + 1) % 200 == 0:
            print(f'{index + 1}/{len(records)} shapes; failures {len(failures)}', flush=True)
    document = {
        'purpose': f'{arguments.points} point pool resampled from source STL for resolution checks',
        'points_per_shape': arguments.points,
        'sampling': 'trimesh sample_surface, area weighted, deterministic seed per shape',
        'source_manifest': str(SOURCE_POINTS/'manifest.json'),
        'records': cloud_records,
        'errors': failures,
        'limitations': [
            'Normals are sampled triangle normals; outward orientation is corrected only by signed volume.',
            'Sampling is independent per shape and not identical to the original 4096 sampler.',
        ],
    }
    (arguments.points_out/'manifest.json').write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding='utf8')
    normal_document = {**document, 'records': normal_records}
    (arguments.normals_out/'manifest.json').write_text(
        json.dumps(normal_document, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'wrote {len(cloud_records)} shapes, failures {len(failures)}')


if __name__ == '__main__':
    main()
