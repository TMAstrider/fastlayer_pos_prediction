"""Check fixed-subset sensitivity of the eight-part equivariant model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from equivariant_axis import EquivariantAxis
from train_eight_equivariant import measured


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='equivariant_eight_parts_2026-11-09')
    args = parser.parse_args()
    run = root/'runs'/args.name
    report = root/'reports'/args.name
    config = json.loads((run/'config.json').read_text(encoding='utf8'))
    split = json.loads((root/'runs/pointnet_shape_dedup_2026-11-09_val100/split.json').read_text(encoding='utf8'))
    rng = np.random.default_rng(config['seed'])
    rng.choice(split['train_indices'],8,replace=False)
    variants = {name:[] for name in ['original','disjoint_remaining_2048','resampled_subset_seed1','resampled_subset_seed2','resampled_subset_seed3']}
    for index,record in zip(config['selected_indices'],config['selected_records']):
        with np.load(root/'outputs/eight_part_normals'/record['pointcloud_file']) as data:
            points = data['points'].astype(np.float32)
            normals = data['normals'].astype(np.float32)
        points -= points.mean(0,keepdims=True)
        points /= np.linalg.norm(points,axis=-1).max()
        original = rng.choice(len(points),2048,replace=False)
        indices = {'original':original,'disjoint_remaining_2048':np.setdiff1d(np.arange(len(points)),original)}
        for seed in (1,2,3):
            indices[f'resampled_subset_seed{seed}'] = np.random.default_rng(90001+index*17+seed).choice(len(points),2048,replace=False)
        for name,ids in indices.items():
            variants[name].append(np.concatenate([points[ids],normals[ids]],axis=-1))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    model = EquivariantAxis().to(device)
    checkpoint = torch.load(run/'best_model.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    with np.load(run/'independent_preview_rotations.npz') as data:
        rotations = torch.from_numpy(data['augmentation_rotations'].reshape(8,64,3,3)).to(device)
    rows = []
    for name,clouds in variants.items():
        metrics,_,_,restored = measured(model,torch.from_numpy(np.stack(clouds)).to(device),rotations)
        per_part = np.degrees(np.arccos(np.clip(restored[:,:,2],-1,1))).mean(1)
        rows.append({'point_subset':name,'metrics':metrics,
                     'per_part_mean_deg':dict(zip(map(str,config['selected_indices']),map(float,per_part)))})
    result = {'checkpoint_epoch':checkpoint['epoch'],'training_performed':False,
              'variants':rows,'normalization':'Unchanged 4096-point reference mean and radius',
              'limitations':['New subsets of original 4096 surface samples, not independent fresh mesh sampling.',
                             'No new-shape evaluation; this only checks sensitivity to the point subset.']}
    (report/'resampling_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    table = '\n'.join(f'| {r["point_subset"]} | {r["metrics"]["mean_angle_deg"]:.3f}° | {r["per_part_mean_deg"]["69"]:.3f}° |' for r in rows)
    text = f'''\n\n## 点子集敏感性补充（不训练）

保持相同权重、中心与尺度和全部 512 个旋转，替换点子集。余下 2048 点与原固定评估子集完全不重叠；另有三个固定种子的重新抽点。仍来自原 4096 点缓存，不是从 mesh 独立重新采样，更不是新件泛化。若训练启用在线抽点，全部 4096 点都是训练采样池，因此这里验证新点组合，而非未见过的表面采样点。

| 点子集 | 八件平均误差 | 69 号平均误差 |
|---|---:|---:|
{table}

完整逐件指标见 resampling_audit.json。该补充不用于重新选择权重或修改本轮训练结果。
'''
    with (report/'README.md').open('a',encoding='utf8') as handle:
        handle.write(text)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
