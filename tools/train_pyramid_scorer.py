"""Control experiment on the public pyramid dataset: can the scoring formulation
reach the precision that plain regression already reaches?

The pyramid dataset has a single well defined build axis per sample, so it isolates
the formulation from the label ambiguity of the real parts. Same scorer family as
`train_build_direction_scorer.py`, but 3 input channels (the public cache has no
normals) and no physics features.

Reported: angle between the refined best candidate and the dataset axis on the
official validation split, next to the published regressors (0.71 deg PointNet,
1.33 deg earlier reproduction).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_build_direction_scorer as trainer

ROOT = trainer.ROOT
CACHE = ROOT / 'runs/public_pyramids_axis_v2/pointcloud_cache.npz'
RUNS = ROOT / 'runs'
REPORTS = ROOT / 'reports'
DIRECTIONS = 512


def sphere_directions(count):
    index = np.arange(count, dtype=np.float64) + .5
    z = 1. - 2.*index/count
    radius = np.sqrt(np.clip(1. - z*z, 0., None))
    angle = np.pi*(1. + 5.**.5)*index
    return np.stack([radius*np.cos(angle), radius*np.sin(angle), z], 1).astype(np.float32)


def perturb(direction, count, low_deg, high_deg):
    u = np.asarray(direction, np.float32)
    reference = np.array([0., 0., 1.], np.float32)
    if abs(float(u @ reference)) > .99:
        reference = np.array([1., 0., 0.], np.float32)
    e1 = np.cross(reference, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    radius = np.deg2rad(np.random.uniform(low_deg, high_deg, count))
    azimuth = np.random.uniform(0, 2*np.pi, count)
    return (np.cos(radius)[:, None]*u + np.sin(radius)[:, None]*(
        np.cos(azimuth)[:, None]*e1 + np.sin(azimuth)[:, None]*e2)).astype(np.float32)


@torch.no_grad()
def score(model, points, candidates, device, yaw_average=4):
    """points: (2048, 3) already centred. candidates: (C, 3). Returns (C,) scores."""
    tensor = torch.from_numpy(points).to(device)
    bases = torch.from_numpy(trainer.canonical_matrices(np.asarray(candidates, np.float32))).to(device)
    total = torch.zeros(len(candidates), device=device)
    for _ in range(yaw_average):
        yaw = torch.from_numpy(trainer.yaw_matrices(len(candidates))).to(device)
        rotated = torch.einsum('nd,cdm->cnm', tensor, bases @ yaw)
        total += model(rotated, None, torch.zeros(len(candidates), 0, device=device))
    return (total / yaw_average).cpu().numpy()


def hill_climb(model, points, start, device, steps=(8., 4., 2., 1.), rounds=2, azimuths=8):
    best = np.asarray(start, np.float32)
    best_score = float(score(model, points, best[None], device, 2)[0])
    for step in steps:
        for _ in range(rounds):
            u = best/max(np.linalg.norm(best), 1e-9)
            reference = np.array([0., 0., 1.], np.float32)
            if abs(float(u @ reference)) > .99:
                reference = np.array([1., 0., 0.], np.float32)
            e1 = np.cross(reference, u)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(u, e1)
            radius = np.deg2rad(step)
            candidates = np.stack([np.cos(radius)*u + np.sin(radius)*(np.cos(a)*e1 + np.sin(a)*e2)
                                   for a in np.linspace(0, 2*np.pi, azimuths, endpoint=False)]).astype(np.float32)
            scores = score(model, points, candidates, device, 2)
            if scores.max() > best_score:
                best, best_score = candidates[int(np.argmax(scores))], float(scores.max())
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='pyramid_scorer_control_2026-11-10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--temperature', type=float, default=.05)
    parser.add_argument('--eval-shapes', type=int, default=60)
    parser.add_argument('--seed', type=int, default=20261110)
    arguments = parser.parse_args()
    run, report = RUNS/arguments.name, REPORTS/arguments.name
    if run.exists():
        parser.error('Existing run; choose a new name')
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = np.load(CACHE, allow_pickle=False)
    train_points, train_targets = cache['train_points'], cache['train_targets']
    validation_points, validation_targets = cache['validation_points'], cache['validation_targets']
    print(f'train {len(train_points)}, validation {len(validation_points)}, device {device}', flush=True)

    model = trainer.Scorer(physics_features=0, in_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs, eta_min=1e-5)
    directions = sphere_directions(DIRECTIONS)
    print(f'parameters {sum(p.numel() for p in model.parameters())}', flush=True)

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_points))
        total, count = 0., 0
        for start in range(0, len(order), arguments.batch):
            chunk = order[start:start+arguments.batch]
            candidates = []
            for index in chunk:
                target = train_targets[index]
                picks = [target]
                picks.extend(perturb(target, 2, 5., 15.))
                picks.extend(perturb(target, 3, 20., 70.))
                picks.extend(directions[np.random.choice(len(directions), 2, replace=False)])
                candidates.append(np.stack(picks))
            candidates = np.stack(candidates)
            b, k = candidates.shape[:2]
            points = np.empty((b, k, train_points.shape[1], 3), np.float32)
            for row, index in enumerate(chunk):
                matrix = trainer.canonical_matrices(candidates[row].astype(np.float32)) @ trainer.yaw_matrices(k)
                points[row] = np.einsum('nd,cdm->cnm', train_points[index], matrix)
            logits = model(torch.from_numpy(points.reshape(b*k, -1, 3)).to(device), None,
                           torch.zeros(b*k, 0, device=device)).reshape(b, k)
            loss = F.cross_entropy(logits/arguments.temperature,
                                   torch.zeros(b, dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.item())*b
            count += b
        scheduler.step()
        epoch_loss = total/max(count, 1)
        message = f'epoch {epoch}: loss {epoch_loss:.4f}'
        if epoch % 4 == 0 or epoch == arguments.epochs:
            model.eval()
            angles = []
            for index in range(min(arguments.eval_shapes, len(validation_points))):
                points_np = validation_points[index]
                target = validation_targets[index]
                scores = score(model, points_np, directions, device)
                best = hill_climb(model, points_np, directions[int(np.argmax(scores))], device)
                angles.append(float(np.degrees(np.arccos(np.clip(best @ target, -1, 1)))))
            message += (f' | val mean {np.mean(angles):.3f} deg'
                        f' median {np.median(angles):.3f} p90 {np.percentile(angles, 90):.3f}'
                        f' within3 {np.mean(np.array(angles) <= 3):.3f}')
            run.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'validation_mean_deg': float(np.mean(angles))}, run/'best_model.pt')
        print(message, flush=True)
    run.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (report/'config.json').write_text(json.dumps({**vars(arguments),
                                                  'model': 'Scorer(physics=0, in_channels=3)',
                                                  'candidates': DIRECTIONS,
                                                  'positives': 1, 'near_negatives': 2,
                                                  'medium_negatives': 3, 'random_negatives': 2},
                                                 ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
