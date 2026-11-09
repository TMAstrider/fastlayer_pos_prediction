"""Independently resample source meshes for a frozen-weight point-sensitivity audit."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/equivariant_full_fresh_mesh_samples_2026-11-09'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def streamed_surface(path, count, rng, invert):
    with path.open('rb') as handle:
        header = handle.read(84)
    face_count = struct.unpack_from('<I',header,80)[0]
    if path.stat().st_size != 84+50*face_count:
        raise ValueError('Expected exported binary STL')
    dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    block_size = 32768
    starts, totals = [], []
    def read_triangles(handle, length):
        records = np.frombuffer(handle.read(length*50),dtype=dtype)
        return records['vertices'].astype(np.float64)
    def doubled_area(triangles):
        return np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=-1)
    with path.open('rb') as handle:
        handle.seek(84)
        for start in range(0,face_count,block_size):
            triangles = read_triangles(handle,min(block_size,face_count-start))
            area = doubled_area(triangles)
            area[~np.isfinite(area) | (area<=1e-14)] = 0
            starts.append(start)
            totals.append(float(area.sum()))
    totals = np.asarray(totals)
    if totals.sum()<=0:
        raise ValueError('No valid triangle area')
    blocks = rng.choice(len(starts),count,p=totals/totals.sum())
    points, normals, faces = np.empty((count,3),np.float32),np.empty((count,3),np.float32),np.empty(count,np.int32)
    with path.open('rb') as handle:
        for block in np.unique(blocks):
            start = starts[block]
            handle.seek(84+50*start)
            triangles = read_triangles(handle,min(block_size,face_count-start))
            area = doubled_area(triangles)
            area[~np.isfinite(area) | (area<=1e-14)] = 0
            slots = np.flatnonzero(blocks==block)
            selected = rng.choice(len(area),len(slots),p=area/area.sum())
            chosen = triangles[selected]
            uv = rng.random((len(slots),2))
            reflected = uv.sum(-1)>1
            uv[reflected] = 1-uv[reflected]
            points[slots] = chosen[:,0]+uv[:,:1]*(chosen[:,1]-chosen[:,0])+uv[:,1:]*(chosen[:,2]-chosen[:,0])
            normals[slots] = np.cross(chosen[:,1]-chosen[:,0],chosen[:,2]-chosen[:,0])/area[selected,None]*(-1 if invert else 1)
            faces[slots] = start+selected
    return points,normals,faces,float(totals.sum()/2)


def prepare(item):
    index, record, previous_audit = item
    started = time.monotonic()
    source = ROOT/record['source_stl_file'].replace('\\', '/')
    audit_path = (OUTPUT/record['pointcloud_file']).with_suffix('.json')
    if audit_path.exists():
        saved = json.loads(audit_path.read_text(encoding='utf8'))
        if saved['source_mesh_sha256']==previous_audit['source_mesh_sha256'] and saved.get('sampler')=='streamed_binary_stl_v1':
            return saved
    mesh_hash = file_hash(source)
    if mesh_hash != previous_audit['source_mesh_sha256']:
        raise ValueError(f'Source mesh changed: {index}')
    with np.load(ROOT/'outputs/shape_deduplicated_pointclouds'/record['pointcloud_file']) as data:
        original = data['points'].astype(np.float32)
    assert hashlib.sha256(original.tobytes()).hexdigest() == previous_audit['point_bytes_sha256']
    center = original.mean(0, keepdims=True)
    radius = float(np.linalg.norm(original-center, axis=-1).max())
    rng = np.random.default_rng(20261110+510001+index*1009)
    inverted = previous_audit['normal_orientation_action'].startswith('Inverted')
    points, normals, face_ids, area = streamed_surface(source,3*2048,rng,inverted)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-5)
    if not np.isfinite(points).all():
        raise ValueError(f'Nonfinite sample: {index}')
    distances, _ = cKDTree(original).query(points, workers=1)
    features = np.concatenate([(points-center)/radius, normals], axis=-1).reshape(3,2048,6)
    np.savez_compressed(OUTPUT/record['pointcloud_file'], features=features, points=points.reshape(3,2048,3),
                        face_indices=face_ids.reshape(3,2048), reference_center=center, reference_radius=radius)
    result = {'part_index':index, 'shape_group_id':record['shape_group_id'],'sampler':'streamed_binary_stl_v1',
            'source_stl_file':record['source_stl_file'], 'source_mesh_sha256':mesh_hash,
            'sampling_seed':20261110+510001+index*1009, 'pointcloud_file':record['pointcloud_file'],
            'features_sha256':hashlib.sha256(features.tobytes()).hexdigest(),
            'fresh_point_bytes_sha256':hashlib.sha256(points.tobytes()).hexdigest(),
            'old_point_bytes_sha256':previous_audit['point_bytes_sha256'],
            'outward_orientation_checked':previous_audit['outward_orientation_checked'],
            'normal_orientation_action':previous_audit['normal_orientation_action'],
            'reference_center':center[0].tolist(), 'reference_radius':radius, 'surface_area':area,
            'fresh_nearest_old_min_distance_mm':float(distances.min()),
            'fresh_nearest_old_mean_distance_mm':float(distances.mean()),
            'fresh_within_1e_6_mm_of_old_fraction':float((distances<=1e-6).mean()),
            'seconds':time.monotonic()-started}
    audit_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not Path('/proc/self/cgroup').exists():
        parser.error('Run this audit inside WSL using tools/run_fresh_points_audit.sh')
    if args.workers < 1 or (OUTPUT.exists() and not args.resume):
        parser.error('Require positive workers and a new output directory')
    manifest = json.loads((ROOT/'outputs/shape_deduplicated_pointclouds/manifest.json').read_text(encoding='utf8'))
    normal_audit = json.loads((ROOT/'outputs/shape_deduplicated_normals/manifest.json').read_text(encoding='utf8'))
    prior = {row['part_index']:row for row in normal_audit['records']}
    OUTPUT.mkdir(exist_ok=args.resume)
    started = time.monotonic()
    records, failures = [], []
    items = [(index,record,prior[index]) for index,record in enumerate(manifest['records'])]
    if args.workers==1:
        for item in items:
            try:
                records.append(prepare(item))
            except Exception as error:
                failures.append({'part_index':item[0],'error':repr(error)})
                print(f'Failed {item[0]}: {error!r}',flush=True)
            count = len(records)+len(failures)
            if count==1 or count%50==0 or count==len(items):
                print(f'{count}/{len(items)} fresh meshes; failures {len(failures)}; {time.monotonic()-started:.1f}s',flush=True)
    else:
        raise ValueError('This audit uses one process to bound host memory')
    result = {'status':'complete' if not failures else 'failed','records':sorted(records,key=lambda row:row['part_index']),
              'failures':failures,'workers':args.workers,'seconds':time.monotonic()-started,
              'variants':3,'points_per_variant':2048,'sampling':'Independent area-weighted triangle sampling plus uniform barycentric sampling',
              'normalization':'Unchanged original 4096-point centroid and maximum radius, not the fresh sample centroid',
              'normals':'Exact sampled source face; same closed-negative-volume inversion policy as original normals',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'memory_policy':'One process; sequential 32768-triangle file blocks, no full mesh load; no whole-file hashing buffer'}
    (OUTPUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    if failures:
        raise RuntimeError(f'{len(failures)} failed source meshes')
    print(json.dumps({'status':result['status'],'count':len(records),'seconds':result['seconds']},indent=2),flush=True)


if __name__=='__main__':
    main()
