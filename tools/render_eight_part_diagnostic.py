"""Render the eight diagnostic meshes and audit the difficult part's geometry."""

import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import cKDTree
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def draw(axis, mesh, elevation=28, azimuth=-55, platform=True):
    vertices = mesh.vertices.copy()
    vertices -= (vertices.min(0) + vertices.max(0)) / 2
    vertices[:, 2] -= vertices[:, 2].min()
    extent = float(np.ptp(vertices, axis=0).max())
    faces = mesh.faces
    axis.computed_zorder = False
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    brightness = .45 + .55*np.abs(normals @ np.array([.3, -.4, .866]))
    axis.add_collection3d(Poly3DCollection(
        triangles, facecolors=brightness[:, None]*np.array([.65, .79, .94]),
        edgecolors='none', rasterized=True, zorder=2))
    half = extent*.64
    if platform:
        plane = [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]]
        axis.add_collection3d(Poly3DCollection([plane], facecolor='#abbac9', alpha=.6,
                                             edgecolor='#7b8895', linewidth=.5, zorder=0))
    origin = [-extent*.49, -extent*.49, 0]
    axis.quiver(*origin, 0, 0, extent*.3, color='#d63d32', linewidth=2, zorder=4)
    axis.text(origin[0], origin[1], extent*.35, '+Z (GT)', color='#bd3027', fontsize=8, zorder=4)
    axis.set(xlim=(-half, half), ylim=(-half, half), zlim=(0, extent*1.05))
    axis.set_box_aspect((1.28, 1.28, 1.05))
    axis.set_proj_type('ortho')
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()


def main():
    source = ROOT / 'reports/pointnet_random_so3_8parts_extended_2026-11-09'
    output = ROOT / 'outputs/eight_part_diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'config.json').read_text(encoding='utf-8-sig'))
    metrics = json.loads((source / 'metrics.json').read_text(encoding='utf8'))
    errors = {row['part_index']: row['metrics'] for row in metrics['per_part_independent_rotations']}
    figure = plt.figure(figsize=(16, 9), layout='constrained')
    records = []
    difficult = None
    for position, (index, record) in enumerate(zip(config['selected_indices'], config['selected_records'])):
        path = ROOT / record['source_stl_file']
        mesh = trimesh.load(path, force='mesh', process=True)
        target = output / f'part_{index:03d}_GT.stl'
        shutil.copyfile(path, target)
        axis = figure.add_subplot(2, 4, position+1, projection='3d')
        draw(axis, mesh)
        dimensions = ' x '.join(f'{value:.1f}' for value in mesh.extents)
        axis.set_title(f'Index {index} | error {errors[index]["mean_angle_deg"]:.2f} deg\n{dimensions} mm',
                       fontsize=11, color='#c02d26' if index == 69 else '#243b53')
        detail = plt.figure(figsize=(10, 5), layout='constrained')
        for column, azimuth in enumerate((-55, 125)):
            view = detail.add_subplot(1, 2, column+1, projection='3d')
            draw(view, mesh, azimuth=azimuth)
            view.set_title(f'Index {index} | GT | view {column+1}')
        detail.savefig(output / f'part_{index:03d}.png', dpi=140)
        plt.close(detail)
        row = {'index': index, 'source_stl': str(path), 'preview_stl': str(target),
               'source_project': record['source_project'], 'extents_mm': mesh.extents.tolist(),
               'watertight': bool(mesh.is_watertight), 'metrics': errors[index]}
        records.append(row)
        if index == 69:
            difficult = mesh
    figure.suptitle('Eight training shapes in stored GT printing orientation | each panel has its own scale', fontsize=15)
    figure.savefig(output / 'overview.png', dpi=150)
    plt.close(figure)
    figure = plt.figure(figsize=(12, 9), layout='constrained')
    for position, (elevation, azimuth, label) in enumerate((
            (25, -55, 'GT oblique'), (25, 125, 'GT reverse oblique'),
            (90, -90, 'Top (+Z side)'), (-90, -90, 'Underside (-Z side)'))):
        axis = figure.add_subplot(2, 2, position+1, projection='3d')
        draw(axis, difficult, elevation, azimuth, platform=position < 2)
        axis.set_title(f'Index 69 | {label}')
    figure.savefig(output / 'part_069_detail.png', dpi=150)
    plt.close(figure)
    vertices = difficult.vertices.copy()
    center = difficult.center_mass
    covariance = difficult.moment_inertia
    _, basis = np.linalg.eigh(covariance)
    tree = cKDTree(vertices)
    symmetry = []
    for position in range(3):
        axis = basis[:, position]
        rotation = 2*np.outer(axis, axis)-np.eye(3)
        transformed = (vertices-center) @ rotation.T+center
        distances = tree.query(transformed)[0]
        symmetry.append({'axis': axis.tolist(),
                         'gt_axis_difference_deg': float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
                         'vertex_nn_mean_mm': float(distances.mean()),
                         'vertex_nn_p95_mm': float(np.percentile(distances, 95)),
                         'vertex_nn_max_mm': float(distances.max())})
    audit = {'parts': records, 'part_69_half_turns_about_inertia_axes': symmetry,
             'symmetry_measure_scope': 'Vertex-nearest-neighbor screening, not a certified surface symmetry test.',
             'render_scope': 'Full stored GT meshes translated to z=0; no predicted pose shown; separate scale per part.'}
    (output / 'manifest.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf8')
    body = ['<meta charset="utf-8"><title>8 parts GT preview</title>',
            '<h1>Eight parts in stored GT printing orientation</h1>',
            '<p>Each part uses its own display scale. Red arrow is printer +Z. Platform is 40% transparent.</p>',
            '<img src="overview.png" style="width:100%;max-width:1500px">']
    for row in records:
        index = row['index']
        body.append(f'<h2>Index {index} | mean error {row["metrics"]["mean_angle_deg"]:.2f}°</h2>'
                    f'<p>{html.escape(row["source_project"])}</p>'
                    f'<p><a href="part_{index:03d}_GT.stl">Open GT STL</a></p>'
                    f'<pre>{html.escape(row["source_stl"])}</pre>'
                    f'<img src="part_{index:03d}.png" style="width:100%;max-width:1000px">')
    body.append('<h2>Index 69 top / underside</h2><img src="part_069_detail.png" style="width:100%;max-width:1200px">')
    (output / 'index.html').write_text('\n'.join(body), encoding='utf8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
