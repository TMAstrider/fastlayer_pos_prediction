"""Evaluate a trained build direction scorer with local refinement and yaw checks.

The primary number is the angle between the refined best direction and the human
printing pose on the held out shapes. Local refinement (hill climbing on the
sphere) removes the ~9 degree grid resolution limit, so the number is comparable
to the 63.95 degree regression baseline and the 78 degree physics argmin.

Also reported: where the physics cost already ranks the human pose in the lowest
10%, versus where it does not, and how much the score changes under a pure yaw
rotation (which must not matter).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_build_direction_prior as core
import train_build_direction_scorer as trainer


def summarise(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p25': float(np.percentile(values, 25)), 'p75': float(np.percentile(values, 75)),
            'p90': float(np.percentile(values, 90))}


def tangent_basis(direction):
    u = direction / max(np.linalg.norm(direction), 1e-9)
    reference = np.array([0., 0., 1.], np.float32)
    if abs(float(u @ reference)) > .99:
        reference = np.array([1., 0., 0.], np.float32)
    e1 = np.cross(reference, u)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def refine(score, direction, steps=(8., 4., 2.), rounds=2, azimuths=8):
    best, best_score = direction.astype(np.float32), float(score(direction[None, :])[0])
    for step in steps:
        for _ in range(rounds):
            e1, e2 = tangent_basis(best)
            radius = np.deg2rad(step)
            angles = np.linspace(0, 2*np.pi, azimuths, endpoint=False)
            candidates = np.stack([
                np.cos(radius)*best + np.sin(radius)*(np.cos(a)*e1 + np.sin(a)*e2)
                for a in angles]).astype(np.float32)
            scores = score(candidates)
            index = int(np.argmax(scores))
            if scores[index] > best_score:
                best, best_score = candidates[index], float(scores[index])
    return best, best_score


def refine_multi(score, starts, steps=(8., 4., 2.), rounds=2, azimuths=8):
    """Hill climb from several starting directions; returns the best one found."""
    best = starts.astype(np.float32).copy()
    best_scores = score(best)
    for step in steps:
        for _ in range(rounds):
            proposals = []
            for start in best:
                e1, e2 = tangent_basis(start)
                radius = np.deg2rad(step)
                angles = np.linspace(0, 2*np.pi, azimuths, endpoint=False)
                proposals.append(np.stack([
                    np.cos(radius)*start + np.sin(radius)*(np.cos(a)*e1 + np.sin(a)*e2)
                    for a in angles]))
            proposals = np.stack(proposals)
            scores = score(proposals.reshape(-1, 3)).reshape(len(best), azimuths)
            winners = scores.argmax(1)
            improved = scores.max(1) > best_scores
            best[improved] = proposals[np.arange(len(best)), winners][improved]
            best_scores[improved] = scores.max(1)[improved]
    return best[int(np.argmax(best_scores))], float(best_scores.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--overhang', type=float, default=40.)
    parser.add_argument('--candidates', type=int, default=512)
    parser.add_argument('--no-physics', action='store_true')
    parser.add_argument('--yaw-average', type=int, default=4)
    parser.add_argument('--restarts', type=int, default=8)
    parser.add_argument('--subset-average', type=int, default=1)
    parser.add_argument('--points', type=int, default=trainer.POINT_COUNT)
    parser.add_argument('--points-dir', type=Path)
    parser.add_argument('--normals-dir', type=Path)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    payload = torch.load(arguments.checkpoint, map_location='cpu', weights_only=False)
    model = trainer.Scorer(0 if arguments.no_physics else 4)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    trainer.SAMPLE_POINTS = arguments.points
    shapes, split, directions, terms, grid_angles, flipped = trainer.load_data(
        arguments.overhang, arguments.points_dir, arguments.normals_dir)
    validation = [i for i in split['validation_indices'] if i < len(shapes)]
    matrices = trainer.canonical_matrices(directions)
    means = np.stack([t.mean(0) for t in terms])
    stds = np.stack([np.maximum(t.std(0), 1e-6) for t in terms])
    weights = np.array([1., 1., 0., 1.], np.float32)

    rows = []
    for index in validation:
        def score(candidates, yaw_average=2):
            total = 0.
            for _ in range(arguments.subset_average):
                total += trainer.score_candidates(
                    model, shapes, index, candidates, terms, means, stds, device, None,
                    arguments.overhang, yaw_average=yaw_average)
            return total / arguments.subset_average

        grid_scores = score(directions, arguments.yaw_average)
        order = np.argsort(-grid_scores)
        refined, _ = refine(score, directions[order[0]])
        refined_angle = float(np.degrees(np.arccos(np.clip(refined @ trainer.HUMAN, -1, 1))))
        multi, _ = refine_multi(score, directions[order[:arguments.restarts]])
        multi_angle = float(np.degrees(np.arccos(np.clip(multi @ trainer.HUMAN, -1, 1))))

        human_scores = score(np.stack([trainer.HUMAN] * 8), arguments.yaw_average)
        cost = trainer.normalise_terms(terms[index], means[index], stds[index]) @ weights
        human_cost = float(trainer.normalise_terms(
            core.raw_terms_dir(shapes[index][0], shapes[index][1], trainer.HUMAN, arguments.overhang),
            means[index], stds[index]) @ weights)

        near = []
        e1, e2 = tangent_basis(trainer.HUMAN)
        for radius_deg in (3., 5., 10.):
            radius = np.deg2rad(radius_deg)
            probes = np.stack([np.cos(radius)*trainer.HUMAN
                               + np.sin(radius)*(np.cos(a)*e1 + np.sin(a)*e2)
                               for a in np.linspace(0, 2*np.pi, 8, endpoint=False)]).astype(np.float32)
            near.append({'radius_deg': radius_deg,
                         'score_below_human_median': float(np.median(human_scores) - np.median(score(probes)))})
        rows.append({
            'index': int(index),
            'grid_top1_angle': float(grid_angles[index][order[0]]),
            'refined_top1_angle': refined_angle,
            'multistart_top1_angle': multi_angle,
            'top5_closest_angle': float(np.min(grid_angles[index][order[:5]])),
            'human_rank': float((grid_scores > np.median(human_scores)).mean()),
            'human_score_yaw_spread': float(human_scores.max() - human_scores.min()),
            'physics_top1_angle': float(grid_angles[index][np.argmin(cost)]),
            'physics_human_rank': float((cost < human_cost).mean()),
            'near_profile': near,
        })
    refined = np.array([r['refined_top1_angle'] for r in rows])
    multi = np.array([r['multistart_top1_angle'] for r in rows])
    grid = np.array([r['grid_top1_angle'] for r in rows])
    rank = np.array([r['human_rank'] for r in rows])
    physics_rank = np.array([r['physics_human_rank'] for r in rows])
    spread = np.array([r['human_score_yaw_spread'] for r in rows])
    difficulty = physics_rank <= .1
    summary = {
        'checkpoint': str(arguments.checkpoint),
        'epoch': int(payload.get('epoch', -1)),
        'candidates': len(directions),
        'validation_shapes': len(rows),
        'yaw_average': arguments.yaw_average,
        'model': {
            'refined_top1_angle': summarise(refined),
            'refined_within_10_deg': float((refined <= 10).mean()),
            'refined_within_20_deg': float((refined <= 20).mean()),
            'refined_within_30_deg': float((refined <= 30).mean()),
            'multistart_top1_angle': summarise(multi),
            'multistart_within_10_deg': float((multi <= 10).mean()),
            'multistart_within_20_deg': float((multi <= 20).mean()),
            'grid_top1_angle': summarise(grid),
            'top5_closest_angle': summarise([r['top5_closest_angle'] for r in rows]),
            'top5_within_20_deg': float((np.array([r['top5_closest_angle'] for r in rows]) <= 20).mean()),
            'human_rank': summarise(rank),
            'human_top10_fraction': float((rank <= .1).mean()),
            'human_top5_fraction': float((rank <= .05).mean()),
            'human_score_yaw_spread_median': float(np.median(spread)),
            'near_profile': {f'{p["radius_deg"]:g}deg': float(np.median(
                [r['near_profile'][i]['score_below_human_median'] for r in rows]))
                for i, p in enumerate(rows[0]['near_profile'])},
        },
        'physics_only': {
            'top1_angle': summarise([r['physics_top1_angle'] for r in rows]),
            'top1_within_20_deg': float((np.array([r['physics_top1_angle'] for r in rows]) <= 20).mean()),
            'human_rank': summarise(physics_rank),
            'human_top10_fraction': float((physics_rank <= .1).mean()),
        },
        'where_physics_already_ranks_human_top10': {
            'shapes': int(difficulty.sum()),
            'refined_top1_median': float(np.median(refined[difficulty])) if difficulty.any() else None,
            'human_rank_median': float(np.median(rank[difficulty])) if difficulty.any() else None,
        },
        'where_physics_fails': {
            'shapes': int((~difficulty).sum()),
            'refined_top1_median': float(np.median(refined[~difficulty])) if (~difficulty).any() else None,
            'human_rank_median': float(np.median(rank[~difficulty])) if (~difficulty).any() else None,
            'physics_human_rank_median': float(np.median(physics_rank[~difficulty])) if (~difficulty).any() else None,
            'human_top10_fraction': float((rank[~difficulty] <= .1).mean()),
            'physics_human_top10_fraction': float((physics_rank[~difficulty] <= .1).mean()),
        },
    }
    output = arguments.output or arguments.checkpoint.parent / 'evaluation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    (output.parent / 'predictions.json').write_text(
        json.dumps([{k: v for k, v in r.items() if k != 'near_profile'} for r in rows],
                   ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
