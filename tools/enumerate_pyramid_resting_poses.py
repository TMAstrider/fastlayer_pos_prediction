"""Enumerate physically acceptable resting poses for the public pyramids and ask
where the single dataset label sits among them.

For every face of a pyramid, the pose that rests on that face is tested:
  overhang  area of down facing surfaces more than 2 mm above the bed (>1% rejected)
  contact   area fraction within 1 mm of the bed
  stability centre of mass projected inside the contact polygon

The dataset stores one reference orientation per sample. This reports how many
poses pass, and whether the stored label is one of them, which decides whether the
single label is "the only correct orientation" or one element of an acceptable set.
"""

import argparse
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
OVERHANG_DEG = 40.
BED_BAND_MM = 2.
CONTACT_BAND_MM = 1.
OVERHANG_LIMIT = .01


def load(index, split='train'):
    with zipfile.ZipFile(ARCHIVE) as z:
        info = json.loads(z.read(f'info/pyramids_{index:07d}.json'))
        data = z.read(f'stl/{split}/pyramids_{index:07d}.stl')
    mesh = trimesh.load(__import__('io').BytesIO(data), file_type='stl', process=False)
    return info, mesh


def face_frames(mesh):
    normals = np.asarray(mesh.face_normals, float)
    areas = np.asarray(mesh.area_faces, float)
    centroids = np.asarray(mesh.triangles_center, float)
    order = np.argsort(-areas)
    unique, seen = [], []
    for index in order:
        n = normals[index]
        if all(float(n @ s) < .999 for s in seen):
            unique.append(index)
            seen.append(n)
    return normals, areas, centroids, unique


def rotate_to(mesh, up):
    """Rotate a copy of the mesh so that `up` becomes +z."""
    vertices = np.asarray(mesh.vertices, float)
    rotation = trimesh.geometry.align_vectors(up, [0, 0, 1])[:3, :3]
    rotated = vertices @ rotation.T
    return rotated, np.asarray(mesh.faces), rotation


def pose_metrics(mesh, rotation):
    vertices = np.asarray(mesh.vertices, float) @ rotation.T
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals, float) @ rotation.T
    areas = np.asarray(mesh.area_faces, float)
    centroids = np.asarray(mesh.triangles_center, float) @ rotation.T
    low = vertices[:, 2].min()
    high = vertices[:, 2].max()
    above = centroids[:, 2] > low + BED_BAND_MM
    down = normals[:, 2] < -np.cos(np.deg2rad(OVERHANG_DEG))
    overhang = float(areas[down & above].sum() / max(areas.sum(), 1e-12))
    contact_vertices = vertices[vertices[:, 2] <= low + CONTACT_BAND_MM]
    contact = float(areas[(centroids[:, 2] <= low + CONTACT_BAND_MM) & down].sum() / max(areas.sum(), 1e-12))
    stable = False
    if len(contact_vertices) >= 3:
        plane = contact_vertices[:, :2]
        centre = plane.mean(0)
        angles = np.arctan2(plane[:, 1]-centre[1], plane[:, 0]-centre[0])
        hull = plane[np.argsort(angles)]
        mass = np.asarray(mesh.center_mass, float) @ rotation.T
        point = mass[:2]
        inside = True
        for index in range(len(hull)):
            a, b = hull[index], hull[(index+1) % len(hull)]
            if np.cross(b-a, point-a) < -1e-6:
                inside = False
                break
        stable = bool(inside and contact > .001)
    return {'overhang': overhang, 'contact': contact, 'stable': stable,
            'height': float(high-low)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=24)
    arguments = parser.parse_args()
    rows = []
    for index in range(arguments.samples):
        info, mesh = load(index)
        normals, areas, centroids, faces = face_frames(mesh)
        label = np.asarray(info['target_rotational_matrix'], float)[:, 2]
        poses = []
        for face in faces:
            up = -normals[face]
            rotation = trimesh.geometry.align_vectors(up, [0, 0, 1])[:3, :3]
            metrics = pose_metrics(mesh, rotation)
            metrics['up'] = up.tolist()
            metrics['area_fraction'] = float(areas[face]/areas.sum())
            metrics['accepted'] = bool(metrics['overhang'] <= OVERHANG_LIMIT and metrics['stable']
                                       and metrics['contact'] > .001)
            metrics['label_angle'] = float(np.degrees(np.arccos(np.clip(up @ label, -1, 1))))
            poses.append(metrics)
        accepted = [pose for pose in poses if pose['accepted']]
        label_rotation = trimesh.geometry.align_vectors(label, [0, 0, 1])[:3, :3]
        label_pose = pose_metrics(mesh, label_rotation)
        nearest = min([pose['label_angle'] for pose in accepted], default=180.)
        rows.append({
            'index': index,
            'faces': len(faces),
            'accepted_poses': len(accepted),
            'accepted_angles': sorted(round(pose['label_angle'], 2) for pose in accepted),
            'label_angle_to_nearest_accepted': round(nearest, 2),
            'label_is_accepted': bool(label_pose['overhang'] <= OVERHANG_LIMIT
                                      and label_pose['stable'] and label_pose['contact'] > .001),
            'label_overhang': round(label_pose['overhang'], 4),
            'label_contact': round(label_pose['contact'], 4),
            'label_stable': label_pose['stable'],
            'label_height': round(label_pose['height'], 2),
        })
    summary = {
        'samples': len(rows),
        'mean_accepted_poses': float(np.mean([r['accepted_poses'] for r in rows])),
        'median_accepted_poses': float(np.median([r['accepted_poses'] for r in rows])),
        'min_accepted_poses': int(min(r['accepted_poses'] for r in rows)),
        'max_accepted_poses': int(max(r['accepted_poses'] for r in rows)),
        'label_is_accepted_fraction': float(np.mean([r['label_is_accepted'] for r in rows])),
        'label_within_10_deg_of_accepted_fraction': float(np.mean(
            [r['label_angle_to_nearest_accepted'] <= 10 for r in rows])),
        'label_within_20_deg_of_accepted_fraction': float(np.mean(
            [r['label_angle_to_nearest_accepted'] <= 20 for r in rows])),
        'label_stable_fraction': float(np.mean([r['label_stable'] for r in rows])),
        'label_overhang_median': float(np.median([r['label_overhang'] for r in rows])),
        'label_contact_median': float(np.median([r['label_contact'] for r in rows])),
        'criteria': {'overhang_limit': OVERHANG_LIMIT, 'overhang_deg': OVERHANG_DEG,
                     'bed_band_mm': BED_BAND_MM, 'contact_band_mm': CONTACT_BAND_MM},
        'rows': rows,
        'limitations': [
            'Overhang is area based and uses the 40 deg rule; no sliced support volume.',
            'Stability is a static centre of mass check against the contact polygon, no tipping '
            'or residual stress analysis.',
            'Faces are the STL triangles grouped by normal, so tessellation does not create '
            'duplicate poses.',
        ],
    }
    (REPORT/'resting_poses.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))
    for row in rows[:12]:
        print(row)


if __name__ == '__main__':
    main()
