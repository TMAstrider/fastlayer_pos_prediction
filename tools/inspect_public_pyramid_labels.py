"""Inspect the public pyramid dataset: what the single label actually is.

Reads a few samples straight from the archive, restores the reference orientation
(V_reference = V_input @ M), and reports which end of the pyramid points up, the
overhang area under the reference axis versus the flipped axis, and whether the
label is the print cheap orientation or an as-designed convention.
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_build_direction_prior as core

ROOT = core.ROOT
ARCHIVE = ROOT / 'datasets/public_pbflbm/pyramids_10000.zip'
REPORT = ROOT / 'reports/public_pyramid_label_audit_2026-11-10'


def load_sample(archive, index, split='train'):
    with zipfile.ZipFile(archive) as z:
        info = json.loads(z.read(f'info/pyramids_{index:07d}.json'))
        data = z.read(f'stl/{split}/pyramids_{index:07d}.stl')
    mesh = trimesh.load(io.BytesIO(data), file_type='stl', process=False)
    return info, np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)


def surface_samples(vertices, faces, count, rng):
    edge1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    areas = .5*np.linalg.norm(np.cross(edge1, edge2), axis=1)
    picks = rng.choice(len(faces), count, p=areas/areas.sum())
    u = rng.random((count, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    triangle = vertices[faces[picks]]
    points = triangle[:, 0] + u[:, :1]*(triangle[:, 1]-triangle[:, 0]) + u[:, 1:2]*(triangle[:, 2]-triangle[:, 0])
    normals = np.cross(edge1[picks], edge2[picks])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return points.astype(np.float32), normals.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--points', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20261110)
    arguments = parser.parse_args()
    rng = np.random.default_rng(arguments.seed)
    directions = core.sphere_directions(512)
    rows = []
    for index in range(arguments.samples):
        info, vertices, faces = load_sample(ARCHIVE, index)
        matrix = np.asarray(info['target_rotational_matrix'], float)
        restored = vertices @ matrix
        height = float(restored[:, 2].max() - restored[:, 2].min())
        low = restored[:, 2] <= restored[:, 2].min() + 1e-3*height
        high = restored[:, 2] >= restored[:, 2].max() - 1e-3*height
        points, normals = surface_samples(restored, faces, arguments.points, rng)
        up = np.array([0., 0., 1.], np.float32)
        reference = core.raw_terms(points, normals, up[None], 40.)[0]
        flipped = core.raw_terms(points, normals, -up[None], 40.)[0]
        terms = core.raw_terms(points, normals, directions, 40.)
        centre, scale = terms.mean(0), np.maximum(terms.std(0), 1e-6)
        weight = np.array([1., 1., 0., 1.], np.float32)
        cost = ((terms - centre)/scale) @ weight
        rows.append({
            'index': index,
            'height': info['height'],
            'base_radius': info['radius'],
            'base_sides': info['n_longitude'],
            'vertices_at_bottom': int(low.sum()),
            'vertices_at_top': int(high.sum()),
            'bottom_is_flat_base': bool(low.sum() > high.sum()),
            'top_is_flat_base': bool(high.sum() > low.sum()),
            'overhang_reference': float(reference[0]),
            'overhang_flipped': float(flipped[0]),
            'support_reference': float(reference[1]),
            'support_flipped': float(flipped[1]),
            'reference_cost_percentile': float((cost < ((reference - centre)/scale) @ weight).mean()),
            'flipped_cost_percentile': float((cost < ((flipped - centre)/scale) @ weight).mean()),
        })
    summary = {
        'samples': len(rows),
        'flat_base_at_bottom_fraction': float(np.mean([r['bottom_is_flat_base'] for r in rows])),
        'flat_base_at_top_fraction': float(np.mean([r['top_is_flat_base'] for r in rows])),
        'reference_overhang_median': float(np.median([r['overhang_reference'] for r in rows])),
        'flipped_overhang_median': float(np.median([r['overhang_flipped'] for r in rows])),
        'reference_support_median': float(np.median([r['support_reference'] for r in rows])),
        'flipped_support_median': float(np.median([r['support_flipped'] for r in rows])),
        'reference_cost_percentile_median': float(np.median([r['reference_cost_percentile'] for r in rows])),
        'flipped_cost_percentile_median': float(np.median([r['flipped_cost_percentile'] for r in rows])),
        'reference_is_cheaper_fraction': float(np.mean(
            [r['reference_cost_percentile'] < r['flipped_cost_percentile'] for r in rows])),
        'base_sides_values': sorted(set(r['base_sides'] for r in rows)),
        'height_range': [min(r['height'] for r in rows), max(r['height'] for r in rows)],
        'radius_range': [min(r['base_radius'] for r in rows), max(r['base_radius'] for r in rows)],
        'rows': rows,
        'limitations': [
            'Only the reference orientation and its 180 degree flip are compared; other resting '
            'poses (for example on a side face) are not enumerated.',
            'The support proxy is height weighted overhang area, not a sliced support volume.',
            'The dataset label may be an as-designed convention; this audit only measures agreement '
            'with a print oriented cost.',
        ],
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
