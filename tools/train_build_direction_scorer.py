"""Train a score field over candidate build directions and rank them.

Formulation: s(shape, u) = w . phi(shape, u) + f_theta(shape rotated so u = +z),
where phi are normalised physics terms (overhang, support proxy, height, plate
contact) and f_theta is a point encoder that sees the shape already placed in the
candidate orientation. The human printing pose is the only positive; negatives are
sampled from the physics low cost band (hard) and from the whole sphere (random).

Reported numbers are angles between the top ranked candidate and the human pose on
the held out 100 shapes, plus the rank of the human direction itself, and a
physics-only baseline with the identical candidate set.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_build_direction_prior as core

ROOT = core.ROOT
RUNS = ROOT / 'runs'
REPORTS = ROOT / 'reports'
HUMAN = core.HUMAN
HARD_BAND = .10          # physics cost within the lowest 10% counts as hard
HARD_MIN_ANGLE = 20.      # a hard negative must be at least this far from the human
NEGATIVES_HARD = 3
NEGATIVES_NEAR = 3
NEGATIVES_RANDOM = 2
POINT_COUNT = 2048


def canonical_matrices(directions):
    """Columns (e1, e2, u): p @ M places the candidate direction on +z."""
    out = np.empty((len(directions), 3, 3), dtype=np.float32)
    for index, u in enumerate(directions):
        u = u / max(np.linalg.norm(u), 1e-9)
        reference = np.array([0., 0., 1.], np.float32)
        if abs(float(u @ reference)) > .99:
            reference = np.array([1., 0., 0.], np.float32)
        e2 = np.cross(reference, u)
        e2 /= max(np.linalg.norm(e2), 1e-9)
        e1 = np.cross(e2, u)
        out[index] = np.stack([e1, e2, u], 1)
    return out


class Scorer(nn.Module):
    def __init__(self, physics_features=4, in_channels=6):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(in_channels, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(256*2 + physics_features, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1))

    def forward(self, points, normals, physics):
        features = torch.cat([points, normals], -1) if normals is not None else points
        encoded = self.point(features)
        pooled = torch.cat([encoded.max(1).values, encoded.mean(1)], -1)
        return self.head(torch.cat([pooled, physics], -1))[..., 0]


def load_data(overhang_deg, points_dir=None, normals_dir=None):
    shapes, split, flipped = core.load(0, points_dir, normals_dir)
    directions = core.sphere_directions(512)
    grid_terms, grid_cost, grid_angle = [], [], []
    for points, normals in shapes:
        raw = core.raw_terms(points, normals, directions, overhang_deg)
        grid_terms.append(raw)
        angle = np.degrees(np.arccos(np.clip(directions @ HUMAN, -1, 1)))
        grid_angle.append(angle)
    return shapes, split, directions, grid_terms, np.stack(grid_angle), flipped


def normalise_terms(raw, mean, std):
    return (raw - mean) / np.maximum(std, 1e-6)


SAMPLE_POINTS = POINT_COUNT


def subset_indices(count=None, pool=None):
    """Sample `count` indices (default: the training sample size) from a pool of `pool` points."""
    count = count or SAMPLE_POINTS
    pool = pool if pool is not None else max(count, POINT_COUNT)
    return np.random.choice(pool, min(count, pool), replace=False)


def perturb(direction, count, low_deg, high_deg):
    """Random directions between low_deg and high_deg away from direction."""
    u = np.asarray(direction, dtype=np.float32)
    reference = np.array([0., 0., 1.], np.float32)
    if abs(float(u @ reference)) > .99:
        reference = np.array([1., 0., 0.], np.float32)
    e1 = np.cross(reference, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    radius = np.deg2rad(np.random.uniform(low_deg, high_deg, count))
    azimuth = np.random.uniform(0, 2*np.pi, count)
    return (np.cos(radius)[:, None]*u + np.sin(radius)[:, None]*(np.cos(azimuth)[:, None]*e1
            + np.sin(azimuth)[:, None]*e2)).astype(np.float32)


def centre_scale(points):
    """Remove the stored location and size so only shape and orientation remain."""
    centred = points - points.mean(0, keepdims=True)
    return (centred / max(float(np.linalg.norm(centred, axis=1).max()), 1e-9)).astype(np.float32)


def yaw_matrices(count):
    angles = np.random.uniform(0, 2*np.pi, count).astype(np.float32)
    cosine, sine = np.cos(angles), np.sin(angles)
    out = np.zeros((count, 3, 3), np.float32)
    out[:, 0, 0], out[:, 1, 1], out[:, 2, 2] = cosine, cosine, 1.
    out[:, 0, 1], out[:, 1, 0] = -sine, sine
    return out


def make_batch(shapes, indices, candidates, overhang_deg, means, stds):
    """candidates: (B, K, 3) unit directions. Returns canonical tensors and physics."""
    b, k = candidates.shape[:2]
    points = np.empty((b, k, POINT_COUNT, 3), np.float32)
    normals = np.empty_like(points)
    physics = np.empty((b, k, 4), np.float32)
    for row, shape_index in enumerate(indices):
        pool_points, pool_normals = shapes[shape_index]
        ids = subset_indices()
        selected = centre_scale(pool_points[ids])
        selected_normals = pool_normals[ids]
        matrix = canonical_matrices(candidates[row]) @ yaw_matrices(k)
        points[row] = np.einsum('nd,cdm->cnm', selected, matrix)
        normals[row] = np.einsum('nd,cdm->cnm', selected_normals, matrix)
        physics[row] = normalise_terms(
            np.stack([core.raw_terms_dir(pool_points, pool_normals, u, overhang_deg)
                      for u in candidates[row]]), means[shape_index], stds[shape_index])
    return points, normals, physics


@torch.no_grad()
def score_candidates(model, shapes, index, candidates, terms, means, stds, device, ids,
                     overhang_deg, yaw_average=2, chunk=128, sample_points=None):
    pool_points, pool_normals = shapes[index]
    if ids is None:
        ids = subset_indices(pool=len(pool_points))
    selected = torch.from_numpy(centre_scale(pool_points[ids])).to(device)
    selected_normals = torch.from_numpy(pool_normals[ids]).to(device)
    bases = canonical_matrices(np.asarray(candidates, dtype=np.float32))
    physics_raw = np.stack([core.raw_terms_dir(pool_points, pool_normals, np.asarray(u, np.float32),
                                               overhang_deg) for u in candidates])
    physics_all = normalise_terms(physics_raw, means[index], stds[index])
    scores = np.zeros(len(candidates), dtype=np.float32)
    for _ in range(yaw_average):
        yaw = torch.from_numpy(yaw_matrices(len(candidates))).to(device)
        for start in range(0, len(candidates), chunk):
            matrix = torch.from_numpy(bases[start:start+chunk]).to(device) @ yaw[start:start+chunk]
            points = torch.einsum('nd,cdm->cnm', selected, matrix)
            normals = torch.einsum('nd,cdm->cnm', selected_normals, matrix)
            physics = torch.from_numpy(physics_all[start:start+chunk]).to(device)
            scores[start:start+chunk] += model(points, normals, physics).cpu().numpy()
    return scores / yaw_average


def score_all(model, shapes, index, directions, terms, means, stds, matrices, device, ids,
              chunk=128, yaw_average=2, overhang_deg=40.):
    return score_candidates(model, shapes, index, directions, terms, means, stds, device, ids,
                            overhang_deg, yaw_average=yaw_average, chunk=chunk)


def score_human(model, shapes, index, terms, means, stds, device, ids, overhang_deg, yaw_average=2):
    return float(score_candidates(model, shapes, index, [HUMAN], terms, means, stds, device, ids,
                                  overhang_deg, yaw_average=yaw_average)[0])


class SampleDataset(torch.utils.data.Dataset):
    """One item is the candidate set of a single shape, built on a worker process."""

    def __init__(self, shapes, indices, terms, means, stds, band, directions, overhang_deg, points):
        self.shapes, self.indices = shapes, list(indices)
        self.terms, self.means, self.stds = terms, means, stds
        self.band, self.directions = band, directions
        self.overhang_deg, self.points = overhang_deg, points

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        points, normals = self.shapes[index]
        ids = np.random.choice(len(points), self.points, replace=False)
        selected = centre_scale(points[ids])
        selected_normals = normals[ids]
        hard = self.band[index]
        picks = [HUMAN]
        picks.extend(perturb(HUMAN, NEGATIVES_NEAR, 5., 15.))
        if len(hard):
            picks.extend(self.directions[np.random.choice(
                hard, min(NEGATIVES_HARD, len(hard)), replace=False)])
        picks.extend(self.directions[np.random.choice(
            len(self.directions), NEGATIVES_RANDOM, replace=False)])
        candidates = np.stack(picks).astype(np.float32)
        matrix = canonical_matrices(candidates) @ yaw_matrices(len(candidates))
        canonical = np.einsum('nd,cdm->cnm', selected, matrix)
        canonical_normals = np.einsum('nd,cdm->cnm', selected_normals, matrix)
        physics = normalise_terms(
            np.stack([core.raw_terms_dir(points, normals, u, self.overhang_deg)
                      for u in candidates]), self.means[index], self.stds[index])
        return (torch.from_numpy(canonical), torch.from_numpy(canonical_normals),
                torch.from_numpy(physics))


def angle_to_human(directions):
    return np.degrees(np.arccos(np.clip(directions @ HUMAN, -1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='build_direction_scorer_2026-11-10')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-shapes', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=.05)
    parser.add_argument('--overhang', type=float, default=40.)
    parser.add_argument('--no-physics', action='store_true')
    parser.add_argument('--yaw-average', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--points', type=int, default=POINT_COUNT)
    parser.add_argument('--points-dir', type=Path)
    parser.add_argument('--normals-dir', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261110)
    arguments = parser.parse_args()
    run = RUNS / arguments.name
    report = REPORTS / arguments.name
    if run.exists():
        parser.error('Existing run; choose a new name')
    global SAMPLE_POINTS
    SAMPLE_POINTS = arguments.points
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shapes, split, directions, terms, grid_angles, flipped = load_data(
        arguments.overhang, arguments.points_dir, arguments.normals_dir)
    train = [i for i in split['train_indices'] if i < len(shapes)]
    validation = [i for i in split['validation_indices'] if i < len(shapes)]
    if arguments.limit:
        train, validation = train[:arguments.limit], validation[:max(4, arguments.limit//4)]
    matrices = canonical_matrices(directions)
    means = np.stack([t.mean(0) for t in terms])
    stds = np.stack([np.maximum(t.std(0), 1e-6) for t in terms])
    print(f'{len(shapes)} shapes, {len(train)} train, {len(validation)} validation, device {device}', flush=True)

    # Hard negative pool: physics low cost band, far from the human direction.
    band = {}
    for index in train:
        cost = normalise_terms(terms[index], means[index], stds[index]) @ np.array([1., 1., 0., 1.], np.float32)
        cutoff = np.quantile(cost, HARD_BAND)
        band[index] = np.nonzero((cost <= cutoff) & (grid_angles[index] > HARD_MIN_ANGLE))[0]
    print('hard negative pool sizes: median %d, min %d, empty %d' % (
        int(np.median([len(b) for b in band.values()])), int(min(len(b) for b in band.values())),
        sum(len(b) == 0 for b in band.values())), flush=True)

    physics_features = 0 if arguments.no_physics else 4
    model = Scorer(physics_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs, eta_min=1e-5)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'model parameters {parameters}', flush=True)

    dataset = SampleDataset(shapes, train, terms, means, stds, band, directions,
                            arguments.overhang, arguments.points)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=arguments.batch_shapes, shuffle=True,
        num_workers=arguments.workers, persistent_workers=arguments.workers > 0,
        prefetch_factor=4 if arguments.workers > 0 else None, drop_last=False)
    history = []
    started = time.monotonic()
    best = None
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        total, count = 0., 0
        for points, normals, physics in loader:
            batch, k = points.shape[:2]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(points.reshape(batch*k, -1, 3).to(device, non_blocking=True),
                               normals.reshape(batch*k, -1, 3).to(device, non_blocking=True),
                               physics.reshape(batch*k, 4)[:, :physics_features].to(device, non_blocking=True)
                               ).reshape(batch, k)
            loss = F.cross_entropy(logits.float() / arguments.temperature,
                                   torch.zeros(batch, dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.item()) * batch
            count += batch
        scheduler.step()
        epoch_loss = total / max(count, 1)

        metrics = None
        if epoch % 5 == 0 or epoch == arguments.epochs:
            model.eval()
            top1, ranks, oracle5 = [], [], []
            for index in validation:
                ids = subset_indices()
                scores = score_all(model, shapes, index, directions, terms, means, stds,
                                   matrices, device, ids, yaw_average=arguments.yaw_average,
                                   overhang_deg=arguments.overhang)
                order = np.argsort(-scores)
                top1.append(float(grid_angles[index][order[0]]))
                human_score = score_human(model, shapes, index, terms, means, stds, device, ids,
                                          arguments.overhang, yaw_average=arguments.yaw_average)
                ranks.append(float((scores > human_score).mean()))
                oracle5.append(float(np.min(grid_angles[index][order[:5]]) if len(order) >= 5 else 90.))
            metrics = {'epoch': epoch, 'train_loss': epoch_loss,
                       'top1_angle_median': float(np.median(top1)),
                       'top1_angle_mean': float(np.mean(top1)),
                       'human_rank_median': float(np.median(ranks)),
                       'top5_within_20_fraction': float(np.mean(np.array(oracle5) <= 20)),
                       'seconds': time.monotonic() - started}
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}), flush=True)
            if best is None or metrics['top1_angle_median'] < best['top1_angle_median']:
                best = metrics
                run.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                            'metrics': metrics,
                            'args': {key: (str(value) if isinstance(value, Path) else value)
                                     for key, value in vars(arguments).items()}},
                           run / 'best_model.pt')
        history.append({'epoch': epoch, 'train_loss': epoch_loss})
        if epoch % 10 == 0 or epoch == arguments.epochs:
            print(f'epoch {epoch}: loss {epoch_loss:.4f}', flush=True)

    run.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    config = {**{key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(arguments).items()},
              'parameters': parameters, 'device': str(device),
              'sample_points': arguments.points,
              'train_shapes': len(train), 'validation_shapes': len(validation),
              'candidates': len(directions), 'point_count': POINT_COUNT,
              'hard_band': HARD_BAND, 'hard_min_angle_deg': HARD_MIN_ANGLE,
              'negatives_hard': NEGATIVES_HARD, 'negatives_near': NEGATIVES_NEAR,
              'negatives_random': NEGATIVES_RANDOM,
              'loss': 'listwise softmax over positive and sampled negatives',
              'physics_features': ([] if arguments.no_physics else core.NAMES),
              'human_negatives': 'separate scoring of the exact human direction',
              'limitations': [
                  'The human pose is treated as one acceptable answer, not the unique one.',
                  'Physics terms are proxies: overhang area, height weighted support, height, first layer contact.',
                  'No accessibility or secluded support term yet.',
                  'Validation shapes come from projects seen in training; they are not an independent test set.',
              ]}
    (run / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    (run / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf8')
    (report / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    if best:
        (report / 'best_metrics.json').write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf8')
    print('best', json.dumps(best, indent=1), flush=True)


if __name__ == '__main__':
    main()
