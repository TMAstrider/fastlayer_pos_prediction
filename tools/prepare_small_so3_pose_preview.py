"""Recreate exact held-out SO(3) rotations for saved eight-part predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_magics_pointnet import random_rotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text(encoding='utf-8-sig'))
    with np.load(args.run / 'independent_predictions.npz') as saved:
        targets = saved['target_axis']
        part_indices = saved['part_indices']
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(config['final_test_seed'])
    rotations = []
    for index in config['selected_indices']:
        count = int((part_indices == index).sum())
        if count != config['fresh_views_per_part']:
            raise ValueError(f'Unexpected view count for part {index}: {count}')
        rotations.append(random_rotations(count, generator, device).cpu().numpy())
    rotations = np.concatenate(rotations)
    np.testing.assert_allclose(rotations[:, :, 2], targets, atol=1e-6)
    np.testing.assert_array_equal(part_indices, np.repeat(config['selected_indices'], config['fresh_views_per_part']))
    destination = args.run / 'independent_preview_rotations.npz'
    np.savez_compressed(destination, augmentation_rotations=rotations, part_indices=part_indices)
    print(f'Exported {len(rotations)} rotations; all saved targets match: {destination}')


if __name__ == '__main__':
    main()
