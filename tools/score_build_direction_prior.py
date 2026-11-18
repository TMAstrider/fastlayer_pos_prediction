"""Score candidate build directions with computable physics terms and measure
how much of the historical (human chosen) printing pose they explain.

No training. Every deduplicated shape is stored in its historical printing pose,
so the human up direction is +Z in the stored frame. Ranking 512 sphere
directions with a cost computed from the stored cloud, then asking where +Z sits,
measures how much of the human decision geometry explains.

Terms (area weighted 4096 point pool, 4 terms per direction):
  overhang  surface fraction whose normal is within theta of the down axis
  support   overhang fraction weighted by height above the lowest point
  height    part height along the direction (build time proxy)
  contact   first layer contact fraction on the build plate

Optional --checkpoint additionally scores the direction predicted by a trained
model on the same shapes, so a hand written cost and a learned regressor can be
compared under one metric.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / 'outputs/shape_deduplicated_pointclouds'
NORMALS = ROOT / 'outputs/shape_deduplicated_normals'
SPLIT = ROOT / 'runs/pointnet_shape_dedup_2026-11-09_val100/split.json'
REPORT = ROOT / 'reports/build_direction_physics_prior_2026-11-10'
HUMAN = np.array([0., 0., 1.], dtype=np.float32)
NAMES = ['overhang', 'support', 'height', 'contact']
CONTACT_DEG, CONTACT_MM = 15., 2.
COVERAGE = (10, 20, 30, 45)
TOP_K = (1, 2, 4, 8, 16, 32, 64, 128)


def sphere_directions(count):
    index = np.arange(count, dtype=np.float64) + .5
    z = 1. - 2.*index/count
    radius = np.sqrt(np.clip(1. - z*z, 0., None))
    angle = np.pi*(1. + 5.**.5)*index
    return np.stack([radius*np.cos(angle), radius*np.sin(angle), z], 1).astype(np.float32)


def raw_terms(points, normals, directions, overhang_deg):
    up = points @ directions.T
    facing = normals @ directions.T
    low, high = up.min(0), up.max(0)
    height = high - low
    diagonal = max(float(np.linalg.norm(points.max(0) - points.min(0))), 1e-6)
    down = facing < -np.cos(np.deg2rad(overhang_deg))
    support = (down * (up - low) / np.maximum(height, 1e-6)).mean(0)
    contact = ((facing < -np.cos(np.deg2rad(CONTACT_DEG))) & (up - low < CONTACT_MM)).mean(0)
    return np.stack([down.mean(0), support, height/diagonal, -contact], 1).astype(np.float32)


def raw_terms_dir(points, normals, direction, overhang_deg):
    return raw_terms(points, normals, direction.astype(np.float32)[None, :], overhang_deg)[0]


def load(limit, points_dir=None, normals_dir=None):
    points_dir = Path(points_dir) if points_dir else POINTS
    normals_dir = Path(normals_dir) if normals_dir else NORMALS
    manifest = json.loads((points_dir / 'manifest.json').read_text(encoding='utf8'))
    audit = {row['part_index']: row
             for row in json.loads((normals_dir / 'manifest.json').read_text(encoding='utf8'))['records']}
    split = json.loads(SPLIT.read_text(encoding='utf8'))
    records = manifest['records'][:limit] if limit else manifest['records']
    shapes, flipped = [], 0
    for index, record in enumerate(records):
        with np.load(points_dir / record['pointcloud_file']) as data:
            points = data['points'].astype(np.float32)
        with np.load(normals_dir / record['pointcloud_file']) as data:
            normals = data['normals'].astype(np.float32)
        if (audit[index].get('source_signed_volume') or 0.) < 0:
            normals = -normals
            flipped += 1
        shapes.append((points, normals))
    return shapes, split, flipped


def summarise(values):
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p10': float(np.percentile(values, 10)), 'p90': float(np.percentile(values, 90)),
            'top1_percent': float((values <= .01).mean()),
            'top5_percent': float((values <= .05).mean()),
            'top10_percent': float((values <= .10).mean()),
            'top25_percent': float((values <= .25).mean()),
            'bottom_half': float((values >= .5).mean())}


def model_directions(checkpoint, count=2048):
    import torch
    sys.path.insert(0, str(ROOT / 'tools'))
    from equivariant_axis import EquivariantAxis
    from train_full_online_so3 import load_bases
    model = EquivariantAxis()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    device = torch.device('cpu')
    bases, _, _ = load_bases(True, 20261110, device, point_count=count)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(bases), 64):
            predictions.append(model(bases[start:start+64]).numpy())
    return np.concatenate(predictions).astype(np.float32), payload['epoch']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directions', type=int, default=512)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--overhang', default='40')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=REPORT / 'report.json')
    arguments = parser.parse_args()
    directions = sphere_directions(arguments.directions)
    shapes, split, flipped = load(arguments.limit)
    train = [i for i in split['train_indices'] if i < len(shapes)]
    validation = [i for i in split['validation_indices'] if i < len(shapes)]
    print(f'{len(shapes)} shapes, {len(train)} train, {len(validation)} validation, '
          f'{flipped} normal sets flipped by signed volume', flush=True)

    grid = [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 1),
            (0, 1, 1, 0), (1, 1, 0, 1), (.5, 1, .5, .5), (1, 1, 1, 1)]
    weights = [np.array(w, dtype=np.float32) for w in grid]
    results = {}
    for overhang_deg in [float(v) for v in arguments.overhang.split(',')]:
        stacks, humans = [], []
        for points, normals in shapes:
            raw = raw_terms(points, normals, directions, overhang_deg)
            centre, scale = raw.mean(0), np.maximum(raw.std(0), 1e-6)
            stacks.append((raw - centre) / scale)
            humans.append((raw_terms_dir(points, normals, HUMAN, overhang_deg) - centre) / scale)
        stacks, humans = np.stack(stacks), np.stack(humans)
        train_costs = stacks[train] @ np.stack(weights).T
        train_human = humans[train] @ np.stack(weights).T
        objective = (train_costs < train_human[:, None, :]).mean((0, 1))
        weight = weights[int(np.argmin(objective))]
        all_costs, all_human = stacks @ weight, humans @ weight
        rank = (all_costs < all_human[:, None]).mean(1)
        order = np.argsort(all_costs, 1)
        below_human = (all_costs < all_human[:, None]).sum(1)
        top_angles = np.degrees(np.arccos(np.clip(directions[order[:, 0]] @ HUMAN, -1, 1)))
        coverage = {}
        for k in TOP_K:
            chosen = directions[order[:, :k]]
            closest = np.degrees(np.arccos(np.clip((chosen @ HUMAN).max(1), -1, 1)))
            coverage[str(k)] = {f'within_{deg}_deg': float((closest <= deg).mean())
                                for deg in COVERAGE}
            coverage[str(k)]['human_rank_below_k'] = float((below_human < k).mean())
            coverage[str(k)]['validation_within_20_deg'] = float((closest[validation] <= 20).mean())
            coverage[str(k)]['validation_within_45_deg'] = float((closest[validation] <= 45).mean())
        single = {}
        for name, unit in zip(NAMES, np.eye(4, dtype=np.float32)):
            single[name] = summarise((stacks @ unit < (humans @ unit)[:, None]).mean(1))
        block = {
            'overhang_deg': overhang_deg,
            'weights': dict(zip(NAMES, [float(v) for v in weight])),
            'train_mean_percentile': float(objective.min()),
            'human_rank_percentile': {'all_shapes': summarise(rank),
                                      'validation': summarise(rank[validation])},
            'physics_top1_angle_deg': {'all_shapes': summarise(top_angles),
                                       'validation': summarise(top_angles[validation])},
            'candidate_coverage': coverage,
            'single_term_human_rank': single,
            'per_shape_human_rank_percentile': rank.tolist(),
            'per_shape_physics_top1_angle_deg': top_angles.tolist(),
        }
        if arguments.checkpoint:
            predicted, epoch = model_directions(arguments.checkpoint)
            predicted = predicted[:len(shapes)]
            cos = np.clip((predicted * HUMAN).sum(1), -1, 1)
            model_angle = np.degrees(np.arccos(cos))
            model_cost = []
            for i, (points, normals) in enumerate(shapes):
                raw = raw_terms(points, normals, directions, overhang_deg)
                centre, scale = raw.mean(0), np.maximum(raw.std(0), 1e-6)
                value = (raw_terms_dir(points, normals, predicted[i], overhang_deg) - centre) / scale
                model_cost.append(float((((raw - centre) / scale) @ weight < value @ weight).mean()))
            model_rank = np.array(model_cost)
            same_shape = model_rank - rank
            block['model'] = {
                'checkpoint': str(arguments.checkpoint),
                'epoch': int(epoch),
                'angle_to_human_deg': {'all_shapes': summarise(model_angle),
                                       'validation': summarise(model_angle[validation])},
                'human_rank_percentile': {'all_shapes': summarise(model_rank),
                                          'validation': summarise(model_rank[validation])},
                'validation_better_than_physics_top1_fraction': float(
                    (model_angle[validation] < top_angles[validation]).mean()),
                'validation_rank_delta_mean': float((model_rank - rank)[validation].mean()),
            }
        results[str(overhang_deg)] = block
        printable = {k: v for k, v in block.items()
                     if not k.startswith('per_shape')}
        print(json.dumps(printable, indent=1), flush=True)

    REPORT.mkdir(parents=True, exist_ok=True)
    document = {
        'purpose': 'How much of the human printing pose is explained by computable physics terms, '
                   'compared against a trained direction regressor when --checkpoint is given.',
        'reference': 'Historical stored pose +Z, same frame as the stored cloud; lower cost is better.',
        'directions': arguments.directions,
        'shapes': len(shapes), 'train_shapes': len(train), 'validation_shapes': len(validation),
        'normals_flipped_by_signed_volume': flipped,
        'terms': NAMES, 'contact_threshold_deg': CONTACT_DEG, 'contact_band_mm': CONTACT_MM,
        'weight_grid': [list(map(float, w)) for w in grid],
        'limitations': [
            'The pool is area weighted but finite (4096 samples); areas are fractions, not mm^2.',
            'Support volume is a height weighted overhang proxy, not a sliced support computation.',
            'Functional and mating surfaces, surface finish and nesting with other parts are not modelled.',
            'The human pose may be machine assisted; this measures agreement, not optimality.',
            f'Sphere grid spacing (~{180/np.sqrt(arguments.directions/4):.1f} deg) bounds the resolution '
            'of any direction taken from the grid.',
            'human_rank_below_k counts grid directions strictly cheaper than the human pose.',
        ],
        'results': results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'wrote {arguments.output}', flush=True)


if __name__ == '__main__':
    main()
