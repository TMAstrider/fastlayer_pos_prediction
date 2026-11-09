"""Build a conservative, rigid-shape-disjoint industrial orientation dataset.

Exact rigid mesh matches are merged. Unresolved near-identical surface-moment
candidates are quarantined rather than claimed to be different parts. Scale is
preserved when comparing shapes; reference printing coordinates are untouched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from audit_pjf import decode_mesh
from audit_magics_reindexed_poses import load_mesh, rigid_matches
from export_meshes import write_stl, verify_stl

ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def inventory(root):
    magics = json.loads((root / 'outputs/magics_parts_stl/manifest.json').read_text(encoding='utf8'))
    magics_clouds = json.loads((root / 'outputs/magics_pointclouds/manifest.json').read_text(encoding='utf8'))
    cloud_lookup = {r['geometry_sha256']: r for r in magics_clouds['records']}
    by_hash = defaultdict(list)
    for r in magics['records']:
        by_hash[r['geometry_sha256']].append(r)
    records = []
    for key, copies in sorted(by_hash.items()):
        chosen = min(copies, key=lambda r: (r['needs_review'], max(0., r['z_min_mm']-10), r['project_relative'], r['member']))
        cloud = cloud_lookup[key]
        records.append({
            'id': 'magics:' + key, 'source_format': 'magics', 'geometry_sha256': key,
            'mesh_record': chosen, 'source_project': chosen['project_relative'],
            'source_stl_file': str((root / 'outputs/magics_parts_stl' / chosen['stl_file']).relative_to(root)),
            'source_cloud': str((root / 'outputs/magics_pointclouds' / cloud['pointcloud_file']).relative_to(root)),
            'needs_review': chosen['needs_review'], 'z_min': chosen['z_min_mm'],
            'occurrences': [{'project': r['project_relative'], 'member': r['member'], 'stl_file': r['stl_file'], 'z_min_mm': r['z_min_mm']} for r in copies],
        })
    projects = {p.name: p for p in (root / 'datasets').rglob('*.pjf')}
    pjf_clouds = json.loads((root / 'outputs/expanded_real_pointclouds/manifest.json').read_text(encoding='utf8'))
    by_hash = defaultdict(list)
    for r in pjf_clouds['records']:
        by_hash[r['mesh_hash_sha256']].append(r)
    stl_manifest = json.loads((root / 'outputs/meshes_stl/manifest.json').read_text(encoding='utf8'))
    stl_lookup = {(r['source_project'], r['source_part_id']): r['stl_file'] for r in stl_manifest['records'] if r['category'] == 'parts'}
    for key, copies in sorted(by_hash.items()):
        chosen = min(copies, key=lambda r: (r['source_project'], r['source_part_id']))
        source = projects[chosen['source_project']]
        stl = stl_lookup.get((chosen['source_project'], chosen['source_part_id']))
        records.append({
            'id': 'pjf:' + key, 'source_format': 'pjf', 'geometry_sha256': key,
            'mesh_record': chosen, 'project_path': str(source.relative_to(root)),
            'source_project': chosen['source_project'],
            'source_stl_file': str(Path('outputs/meshes_stl') / stl) if stl else None,
            'source_cloud': str((root / 'outputs/expanded_real_pointclouds' / chosen['pointcloud_file']).resolve().relative_to(root)),
            'needs_review': False, 'z_min': chosen['features']['bbox_min_raw_xyz'][2],
            'occurrences': [{'project': r['source_project'], 'part_id': r['source_part_id'], 'name': r['source_part_name']} for r in copies],
        })
    return records


def read_mesh(root, record):
    if record['source_format'] == 'magics':
        return load_mesh(root / 'datasets', record['mesh_record'])
    with sqlite3.connect((root / record['project_path']).resolve().as_uri() + '?mode=ro', uri=True) as connection:
        row = connection.execute('SELECT data FROM MeshV220218 WHERE id=?', (record['mesh_record']['source_mesh_id'],)).fetchone()
    vertices, faces, _ = decode_mesh(row[0])
    return vertices.astype(np.float64), faces


def surface_descriptor(vertices, faces):
    # Exact triangle integrals, independent of vertex density and triangulation
    # of coplanar faces. Origin shift prevents loss of precision for off-plate
    # coordinates. Covariance is over area, not the list of mesh vertices.
    origin = vertices.mean(0)
    v = vertices - origin
    area_sum, first, second = 0., np.zeros(3), np.zeros((3, 3))
    for start in range(0, len(faces), 50000):
        t = v[faces[start:start+50000]]
        area = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)*.5
        sums = t.sum(1)
        area_sum += float(area.sum())
        first += (sums * (area/3)[:, None]).sum(0)
        second += (np.einsum('nki,nkj,n->ij', t, t, area) + np.einsum('ni,nj,n->ij', sums, sums, area))/12
    if area_sum <= 0 or not np.isfinite(area_sum):
        raise ValueError('Invalid surface area')
    centroid = first/area_sum
    covariance = second/area_sum - np.outer(centroid, centroid)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(0)
    return {'area_mm2': area_sum, 'surface_radii_mm': np.sqrt(eigenvalues).tolist(),
            'surface_centroid_mm': (centroid+origin).tolist(), 'vertices': len(vertices), 'triangles': len(faces)}


def representative_rank(record):
    # Prefer named PJF root parts over heuristic Magics object candidates, then
    # ordinary near-platform records. Never choose based on model accuracy.
    return (record['needs_review'], max(0., record['z_min']-10),
            record['source_format'] != 'pjf', abs(record['z_min']-4), record['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/shape_deduplicated_pointclouds')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--descriptor-tolerance-mm', type=float, default=.02)
    parser.add_argument('--area-relative-tolerance', type=float, default=.001)
    parser.add_argument('--fit-tolerance-mm', type=float, default=.002)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    threadpool_limits(limits=1)
    records = inventory(ROOT)
    ids = [r['id'] for r in records]
    lookup = {r['id']: r for r in records}
    descriptor_path = args.output / 'surface_descriptors.json'
    cached = json.loads(descriptor_path.read_text(encoding='utf8')) if descriptor_path.exists() else {}
    descriptors = {key: value for key, value in cached.items() if key in lookup}
    missing = [r for r in records if r['id'] not in descriptors]
    def describe(record):
        return record['id'], surface_descriptor(*read_mesh(ROOT, record))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (key, value) in enumerate(pool.map(describe, missing), 1):
            descriptors[key] = value
            if index % 100 == 0 or index == len(missing):
                print(f'descriptors {index}/{len(missing)}', flush=True)
                descriptor_path.write_text(json.dumps(descriptors), encoding='utf8')
    matrix = np.array([descriptors[key]['surface_radii_mm'] for key in ids])
    tree = cKDTree(matrix)
    candidates = []
    for i, j in sorted(tree.query_pairs(args.descriptor_tolerance_mm, p=np.inf)):
        a, b = descriptors[ids[i]], descriptors[ids[j]]
        if abs(a['area_mm2']-b['area_mm2']) <= max(.01, args.area_relative_tolerance*max(a['area_mm2'], b['area_mm2'])):
            candidates.append((ids[i], ids[j]))
    print(json.dumps({'input_stored_meshes': len(records), 'input_occurrences': sum(len(r['occurrences']) for r in records), 'descriptor_candidate_pairs': len(candidates)}, indent=2), flush=True)
    (args.output / 'inventory.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf8')
    if args.prepare_only:
        return

    groups = UnionFind(ids)
    evidence = []
    prior_evidence = args.output / 'rigid_evidence.json'
    if prior_evidence.exists():
        for row in json.loads(prior_evidence.read_text(encoding='utf8')):
            if row['a'] in lookup and row['b'] in lookup and row['max_mm'] <= args.fit_tolerance_mm:
                groups.join(row['a'], row['b'])
                evidence.append(row)
    previous = ROOT / 'reports/magics_reindexed_pose_audit_2026-11-09.json'
    if previous.exists():
        for row in json.loads(previous.read_text(encoding='utf8'))['matches']:
            a, b = ('magics:'+row[k]['geometry_sha256'] for k in ('reference', 'other'))
            if a in lookup and b in lookup and groups.find(a) != groups.find(b) and row['max_mm'] <= args.fit_tolerance_mm:
                groups.join(a, b)
                evidence.append({'a': a, 'b': b, **{k:v for k,v in row.items() if k not in ('reference', 'other')}})
    by_size = defaultdict(list)
    for record in records:
        d = descriptors[record['id']]
        by_size[(d['vertices'], d['triangles'])].append(record['id'])
    # Equal-count fits also cover degenerate moment frames / different moments
    # from numerical integration, without relying on candidate filtering.
    verification_pairs = set(candidates)
    for members in by_size.values():
        verification_pairs.update(tuple(sorted(pair)) for pair in combinations(members, 2))
    verification_pairs = sorted(verification_pairs)
    pending = []
    for a, b in verification_pairs:
        if groups.find(a) != groups.find(b):
            da, db = descriptors[a], descriptors[b]
            if da['vertices'] == db['vertices'] and da['triangles'] == db['triangles']:
                pending.append((a, b))
    def verify(pair):
        a, b = pair
        result = rigid_matches(*read_mesh(ROOT, lookup[a]), *read_mesh(ROOT, lookup[b]), args.fit_tolerance_mm)
        return a, b, result
    with ThreadPoolExecutor(max_workers=min(args.workers, 2)) as pool:
        for index, (a, b, result) in enumerate(pool.map(verify, pending), 1):
            if result:
                groups.join(a, b)
                evidence.append({'a': a, 'b': b, **result})
            if index % 20 == 0 or index == len(pending):
                print(f'rigid fits {index}/{len(pending)}, matches {len(evidence)}', flush=True)
                (args.output / 'rigid_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')
    # A descriptor collision is not proof of identity. Do not merge different
    # parts on descriptors alone. Quarantine unresolved pairs, including copies
    # with different tessellation, so neither side leaks into this clean split.
    unresolved = [(a,b) for a,b in candidates if groups.find(a) != groups.find(b)]
    quarantine_roots = {groups.find(key) for pair in unresolved for key in pair}
    members = defaultdict(list)
    for key in ids:
        members[groups.find(key)].append(key)
    exported, group_rows = [], []
    for members_ids in sorted(members.values(), key=lambda row: min(row)):
        group_id = 'shape_' + hashlib.sha256('\n'.join(sorted(members_ids)).encode()).hexdigest()[:20]
        chosen = min((lookup[key] for key in members_ids), key=representative_rank)
        quarantined = groups.find(chosen['id']) in quarantine_roots
        group_rows.append({'shape_group_id': group_id, 'representative_id': chosen['id'], 'member_ids': sorted(members_ids),
                           'quarantined': quarantined, 'reason': 'unresolved_surface_descriptor_collision' if quarantined else None,
                           'source_occurrences': sum(len(lookup[key]['occurrences']) for key in members_ids)})
        if quarantined:
            continue
        stl_path = chosen['source_stl_file']
        if stl_path is None:
            stl_output = args.output / 'reference_stl' / (group_id + '.stl')
            stl_output.parent.mkdir(exist_ok=True)
            vertices, faces = read_mesh(ROOT, chosen)
            write_stl(stl_output, vertices, faces)
            verify_stl(stl_output, vertices, faces)
            stl_path = str(stl_output.relative_to(ROOT))
        filename = group_id + '.npz'
        # Copy the stored-pose cloud; do not rotate into PCA coordinates.
        with np.load(ROOT / chosen['source_cloud']) as original:
            points = original['points'].astype(np.float32)
        np.savez_compressed(args.output / filename, points=points, geometry_sha256=chosen['geometry_sha256'], shape_group_id=group_id)
        exported.append({'geometry_sha256': chosen['geometry_sha256'], 'shape_group_id': group_id,
                         'pointcloud_file': filename, 'source_stl_file': stl_path,
                         'source_project': chosen['source_project'], 'source_format': chosen['source_format'],
                         'source_record_id': chosen['id'], 'point_count': len(points),
                         'triangle_count': descriptors[chosen['id']]['triangles'], 'gt_up_axis': [0,0,1],
                         'shape_member_count': len(members_ids), 'duplicate_record_count': group_rows[-1]['source_occurrences']})
    summary = {
        'source_occurrences': sum(len(r['occurrences']) for r in records), 'input_stored_meshes': len(records),
        'source_format_stored_meshes': {fmt: sum(r['source_format']==fmt for r in records) for fmt in ('magics','pjf')},
        'confirmed_shape_groups': len(members), 'merged_stored_meshes': len(records)-len(members),
        'confirmed_rigid_pair_evidence': len(evidence), 'descriptor_candidate_pairs': len(candidates),
        'unresolved_candidate_pairs': len(unresolved), 'quarantined_shape_groups': len(quarantine_roots),
        'clean_training_shapes': len(exported), 'clean_source_occurrences': sum(r['duplicate_record_count'] for r in exported),
        'descriptor_tolerance_mm': args.descriptor_tolerance_mm, 'fit_tolerance_mm': args.fit_tolerance_mm,
        'area_relative_tolerance': args.area_relative_tolerance,
        'representative_selection': 'non-review, near-platform, named PJF root preferred, then stable ID; no accuracy-based selection',
        'limitations': ['Conservative finite-tolerance geometry deduplication, not a certified CAD identity index.',
                       'Unresolved descriptor collisions are excluded, not merged; may exclude distinct similar parts.',
                       'Non-identical remeshes outside descriptor thresholds may remain undetected.',
                       'Source object activity and geometry semantics are not independently verified for every Magics candidate.',
                       'Different physical sizes are not merged. Part families are not identified.'],
    }
    audit = {'summary': summary, 'groups': group_rows, 'unresolved_pairs': unresolved, 'rigid_evidence': evidence}
    (args.output / 'deduplication.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    manifest = {'dataset': 'Initial 193 PJF + 30 additional PJF + Magics, conservative rigid-shape deduplication',
                'deduplicated_by_shape': True, 'one_reference_pose_per_shape': True,
                'deduplication_report': 'deduplication.json', 'source_part_records': summary['source_occurrences'],
                'sample_count_unique_geometries': len(exported), 'points_per_mesh': 4096,
                'coordinates': 'Stored accepted reference printing axes preserved; center/scale and rotate labels during training.',
                'records': sorted(exported, key=lambda r:r['shape_group_id']), 'errors': [], 'limitations': summary['limitations']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    report_dir = ROOT / 'reports/shape_deduplication_2026-11-09'
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output / 'deduplication.json', report_dir / 'deduplication.json')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
