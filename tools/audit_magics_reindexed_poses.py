"""Check equal-size Magics meshes for rigid matches, including reordered vertices.

Only meshes with equal vertex/triangle counts are compared. Rigid registration
must preserve the full triangle connectivity after mapping vertices. This does
not detect remeshed copies or resolve all degenerate/symmetric PCA frames.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, permutations, product
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from audit_expanded_data import MTArchive
from audit_magics_pose_conflicts import fit_rigid


def load_mesh(dataset, record):
    with MTArchive(dataset / (record['project_relative'] + '.magics')) as archive:
        vertices = np.frombuffer(archive.read(record['member']), dtype='<i4').reshape(-1, 3)
        faces = np.frombuffer(archive.read(record['surface_member']), dtype='<u4').reshape(-1, 3)
    return vertices.astype(np.float64) / 10000., faces


def canonical_faces(faces):
    # Triangle winding is irrelevant to shape identity here.
    sorted_faces = np.sort(faces, axis=1)
    return sorted_faces[np.lexsort(sorted_faces.T[::-1])]


def rigid_matches(a, af, b, bf, tolerance):
    ac, bc = a - a.mean(0), b - b.mean(0)
    af_sorted = canonical_faces(af)
    rms, maximum, angle = fit_rigid(a, b)
    if maximum <= tolerance and np.array_equal(af_sorted, canonical_faces(bf)):
        return {'rms_mm': rms, 'max_mm': maximum, 'up_difference_deg': angle,
                'registration': 'ordered_vertices'}

    av, ae = np.linalg.eigh(ac.T @ ac / len(ac))
    bv, be = np.linalg.eigh(bc.T @ bc / len(bc))
    if np.max(np.abs(np.sqrt(np.maximum(av, 0)) - np.sqrt(np.maximum(bv, 0)))) > tolerance:
        return None
    tree = cKDTree(bc)
    best = None
    # Permutations cover axes with nearly equal second moments. Exact spherical
    # or axial degeneracies may still require a continuous registration search.
    for perm in permutations(range(3)):
        if np.max(np.abs(np.sqrt(np.maximum(av, 0)) - np.sqrt(np.maximum(bv[list(perm)], 0)))) > tolerance:
            continue
        for signs in product((-1., 1.), repeat=3):
            q = ae @ np.diag(signs) @ be[:, perm].T
            if np.linalg.det(q) < 0:
                continue
            distance, mapping = tree.query(ac @ q, workers=1)
            if distance.max() > tolerance or len(np.unique(mapping)) != len(a):
                continue
            # Mapping is a-index -> b-index, so map a's triangles into b space.
            if not np.array_equal(canonical_faces(mapping[af]), canonical_faces(bf)):
                continue
            rms, maximum, angle = fit_rigid(a, b[mapping])
            if maximum > tolerance:
                continue
            candidate = {'rms_mm': rms, 'max_mm': maximum, 'up_difference_deg': angle,
                         'registration': 'pca_vertex_mapping_and_connectivity'}
            if best is None or angle < best['up_difference_deg']:
                best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('datasets'))
    parser.add_argument('--manifest', type=Path, default=Path('outputs/magics_parts_stl/manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/magics_reindexed_pose_audit_2026-11-09.json'))
    parser.add_argument('--tolerance-mm', type=float, default=.02)
    parser.add_argument('--up-threshold-deg', type=float, default=2.)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    unique = {r['geometry_sha256']: r for r in manifest['records']}
    groups = defaultdict(list)
    for record in unique.values():
        groups[(record['vertices'], record['triangles'])].append(record)
    repeated = [group for group in groups.values() if len(group) > 1]

    def check_group(group):
        meshes = [load_mesh(args.dataset, record) for record in group]
        results = []
        for i, j in combinations(range(len(group)), 2):
            result = rigid_matches(*meshes[i], *meshes[j], args.tolerance_mm)
            if result:
                result['reference'] = {k: group[i][k] for k in ('geometry_sha256', 'project_relative', 'member', 'stl_file')}
                result['other'] = {k: group[j][k] for k in ('geometry_sha256', 'project_relative', 'member', 'stl_file')}
                results.append(result)
        return results

    matches = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, results in enumerate(pool.map(check_group, repeated), 1):
            matches.extend(results)
            if index % 10 == 0 or index == len(repeated):
                print(f'groups {index}/{len(repeated)}, rigid matches {len(matches)}', flush=True)
    conflicts = [row for row in matches if row['up_difference_deg'] >= args.up_threshold_deg]
    summary = {
        'unique_stored_meshes': len(unique),
        'equal_vertex_triangle_count_groups': len(repeated),
        'candidate_pairs': sum(len(g) * (len(g)-1) // 2 for g in repeated),
        'confirmed_rigid_same_mesh_pairs': len(matches),
        'pairs_with_up_difference_at_least_threshold': len(conflicts),
        'up_threshold_deg': args.up_threshold_deg,
        'max_vertex_error_tolerance_mm': args.tolerance_mm,
        'max_matched_up_difference_deg': max((r['up_difference_deg'] for r in matches), default=None),
        'limitations': 'Requires equal vertex/triangle counts and matching connectivity. PCA search may miss degenerate frames. Remeshed copies are not covered. Ordered-vertex fits use the existing vertex correspondence; symmetry-equivalent direction labels are not adjudicated.',
    }
    args.output.write_text(json.dumps({'summary': summary, 'conflicts': conflicts, 'matches': matches}, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
