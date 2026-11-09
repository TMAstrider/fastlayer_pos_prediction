"""Render selected stored-pose pairs found by the Magics rigid-shape audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from audit_magics_reindexed_poses import load_mesh


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, nargs='+', default=[14, 2, 5, 1], help='One-based indices in the conflicts list')
    parser.add_argument('--output', type=Path, default=root / 'reports/magics_pose_conflicts_preview_2026-11-09.png')
    args = parser.parse_args()
    report = json.loads((root / 'reports/magics_reindexed_pose_audit_2026-11-09.json').read_text(encoding='utf8'))
    manifest = json.loads((root / 'outputs/magics_parts_stl/manifest.json').read_text(encoding='utf8'))
    records = {r['geometry_sha256']: r for r in manifest['records']}
    examples = [report['conflicts'][i-1] for i in args.examples]
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    fig = plt.figure(figsize=(12, 4.25*len(examples)), layout='constrained')
    for row_id, pair in enumerate(examples):
        loaded = []
        for label in ('reference', 'other'):
            record = records[pair[label]['geometry_sha256']]
            vertices, faces = load_mesh(root / 'datasets', record)
            # Translation is irrelevant to up direction. Both views rest on
            # z=0 to compare tilt; the original z is shown in the title.
            vertices -= [vertices[:, 0].mean(), vertices[:, 1].mean(), vertices[:, 2].min()]
            loaded.append((vertices, faces, record))
        extent = max(float(np.ptp(v, axis=0).max()) for v, _, _ in loaded)
        for col, (vertices, faces, record) in enumerate(loaded):
            ax = fig.add_subplot(len(examples), 2, row_id*2+col+1, projection='3d')
            if len(faces) > 16000:
                # Preview only: evenly spaced faces reduce rendering cost.
                faces = faces[np.linspace(0, len(faces)-1, 16000, dtype=int)]
            mesh = Poly3DCollection(vertices[faces], facecolor=('#5ca9db' if col == 0 else '#efb365'), edgecolor='#28323b', linewidth=.05, alpha=1.)
            ax.add_collection3d(mesh)
            half = extent*.65
            plane = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]])
            ax.add_collection3d(Poly3DCollection([plane], facecolor='#8c9caa', alpha=.4, edgecolor='#566574', linewidth=.5))
            ax.quiver(-extent*.5,-extent*.5,0,0,0,extent*.3,color='#269148',arrow_length_ratio=.15,linewidth=2)
            ax.text(-extent*.5,-extent*.5,extent*.34,'+Z',color='#217b3d')
            ax.set_xlim(-half,half); ax.set_ylim(-half,half); ax.set_zlim(0,extent*1.05)
            ax.set_box_aspect((1.3,1.3,1.05)); ax.view_init(elev=22,azim=-55)
            ax.set_axis_off()
            title = f"{'A' if col == 0 else 'B'} · {record['project_relative'].split('/')[-1]} · 朝上轴差 {pair['up_difference_deg']:.2f}°\n{Path(record['stl_file']).name.split('__')[0]} · 原始最低 Z={record['z_min_mm']:.2f} mm"
            ax.set_title(title, fontsize=10, pad=2)
    output = args.output
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
