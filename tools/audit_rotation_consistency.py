"""Measure signed direction accuracy and rotation consistency without training."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnose_pointnet_random_so3 import rotate_features
from train_full_online_so3 import load_bases
from train_magics_pointnet import PointNetAxis, random_rotations


ROOT = Path(__file__).resolve().parents[1]


def angle(vectors, reference):
    return np.degrees(np.arccos(np.clip((vectors*reference).sum(-1), -1, 1)))


def summarize_axes(axes, identity_axis):
    mean = axes.mean(0)
    concentration = float(np.linalg.norm(mean))
    scatter = axes.T @ axes / len(axes)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    axis = eigenvectors[:, -1]
    if axis @ mean < 0:
        axis = -axis
    consensus = mean/max(concentration, 1e-12)
    return {
        'mean_gt_error_deg': float(angle(axes, np.array([0., 0., 1.])).mean()),
        'signed_concentration': concentration,
        'mean_spread_to_consensus_deg': float(angle(axes, consensus).mean()) if concentration > 1e-6 else None,
        'consensus_gt_error_deg': float(angle(consensus, np.array([0., 0., 1.]))) if concentration > 1e-6 else None,
        'mean_error_to_identity_prediction_deg': float(angle(axes, identity_axis).mean()),
        'axial_second_moment_max_eigenvalue': float(eigenvalues[-1]),
        'mean_unsigned_spread_to_axis_deg': float(np.degrees(np.arccos(np.clip(np.abs(axes @ axis), 0, 1))).mean()),
        'minority_axis_sign_fraction': float(min((axes @ axis >= 0).mean(), (axes @ axis < 0).mean())),
    }


def self_check():
    up = np.array([0., 0., 1.])
    stable = summarize_axes(np.tile(up, (32, 1)), up)
    bipolar = summarize_axes(np.tile([up, -up], (16, 1)), up)
    assert stable['signed_concentration'] == 1 and stable['mean_gt_error_deg'] == 0
    assert bipolar['signed_concentration'] == 0 and bipolar['mean_gt_error_deg'] == 90
    assert bipolar['axial_second_moment_max_eigenvalue'] == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', default='pointnet_full_online_so3_normals_2026-11-09')
    parser.add_argument('--name', default='rotation_consistency_full_2026-11-09')
    parser.add_argument('--views', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if min(args.views, args.batch_size) < 2 or Path(args.name).name != args.name:
        parser.error('Require at least two views/batch entries and a single-component name')
    self_check()
    training = ROOT / 'runs' / args.training
    report = ROOT / 'reports' / args.name
    output = ROOT / 'outputs' / args.name
    if report.exists():
        parser.error('Report already exists; select a new name')
    config = json.loads((training / 'config.json').read_text(encoding='utf8'))
    split = json.loads((training / 'split.json').read_text(encoding='utf8'))
    snapshots = []
    for filename in ('best_model.pt', 'last_model.pt'):
        for attempt in range(3):
            try:
                blob = (training / filename).read_bytes()
                checkpoint = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
                break
            except (RuntimeError, EOFError):
                if attempt == 2:
                    raise
        snapshots.append((filename, blob, checkpoint))
    report.mkdir(parents=True)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    device = torch.device('cuda')
    bases, manifest, hashes = load_bases(config['use_normals'], config['seed'], torch.device('cpu'))
    assert hashes == config['point_subset_hashes']
    seed = 20261109+79103
    generator = torch.Generator(device=device).manual_seed(seed)
    rotations = random_rotations(args.views, generator, device)
    up = torch.tensor([0., 0., 1.], device=device)
    restored_gt = torch.bmm(rotations[:, None, :, 2], rotations).squeeze(1)
    torch.testing.assert_close(restored_gt, up.expand_as(restored_gt), atol=2e-6, rtol=2e-6)
    val_ids = set(split['validation_indices'])
    all_rows, summaries = [], []
    started = time.monotonic()
    for filename, blob, checkpoint in snapshots:
        label = f'{Path(filename).stem}_epoch{checkpoint["epoch"]}'
        (output / f'{label}.pt').write_bytes(blob)
        model = PointNetAxis(input_channels=bases.shape[-1]).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        restored_all = []
        with torch.inference_mode():
            for index, base in enumerate(bases):
                base = base.to(device)
                identity_axis = model(base[None])[0].cpu().numpy()
                restored = []
                for start in range(0, args.views, args.batch_size):
                    local = rotations[start:start+args.batch_size]
                    inputs = rotate_features(base[None].expand(len(local), -1, -1), local)
                    predicted = model(inputs)
                    original_axis = torch.bmm(predicted[:, None, :], local).squeeze(1)
                    restored.append(original_axis.cpu().numpy())
                axes = np.concatenate(restored)
                axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
                row = {'checkpoint': label, 'epoch': checkpoint['epoch'], 'part_index': index,
                       'scope': 'validation' if index in val_ids else 'train',
                       'shape_group_id': manifest['records'][index]['shape_group_id'],
                       **summarize_axes(axes, identity_axis)}
                all_rows.append(row)
                restored_all.append(axes)
                if (index+1) % 200 == 0:
                    print(f'{label}: {index+1}/{len(bases)} parts; {time.monotonic()-started:.1f}s', flush=True)
        np.savez_compressed(output / f'{label}_restored_axes.npz', axes=np.stack(restored_all),
                            rotations=rotations.cpu().numpy(), part_indices=np.arange(len(bases)))
        for scope in ('train', 'validation'):
            rows = [row for row in all_rows if row['checkpoint'] == label and row['scope'] == scope]
            summaries.append({
                'checkpoint': label, 'scope': scope, 'parts': len(rows),
                'mean_gt_error_deg': float(np.mean([row['mean_gt_error_deg'] for row in rows])),
                'mean_signed_concentration': float(np.mean([row['signed_concentration'] for row in rows])),
                'mean_spread_to_consensus_deg': float(np.mean([row['mean_spread_to_consensus_deg'] for row in rows])),
                'mean_error_to_identity_prediction_deg': float(np.mean([row['mean_error_to_identity_prediction_deg'] for row in rows])),
                'concentration_ge_095_parts': sum(row['signed_concentration'] >= .95 for row in rows),
                'concentrated_but_gt_over_30deg_parts': sum(row['signed_concentration'] >= .95 and row['consensus_gt_error_deg'] > 30 for row in rows),
                'axial_concentration_ge_09_signed_below_05_parts': sum(row['axial_second_moment_max_eigenvalue'] >= .9 and row['signed_concentration'] < .5 for row in rows),
            })
        del model
    with (report / 'per_part.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    metadata = {'status': 'complete', 'views_per_part': args.views, 'seed': seed,
                'sampling': 'Same deterministic random SO(3) draws for every shape and both checkpoints; fixed training point subsets',
                'restoration': 'Predicted row vector in augmented frame @ R returns history-frame vector',
                'checkpoints': [{'file': name, 'sha256': hashlib.sha256(blob).hexdigest(), 'epoch': ckpt['epoch']}
                                for name, blob, ckpt in snapshots],
                'training_changed': False, 'seconds': time.monotonic()-started, 'summary': summaries,
                'limitations': ['32 rotations are a finite diagnostic, not a proof of equivariance.',
                                'A consistent wrong axis does not prove symmetry, label error or manufacturability.',
                                'High scatter can include competing modes or sign flips, not only arbitrary instability.',
                                'Validation-selected checkpoint; no new shape test set or retraining.',
                                'Different rotation draws from training progress; aggregate angles need not match exactly.']}
    (report / 'metrics.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained', sharex=True, sharey=True)
    for axis, (filename, _, checkpoint) in zip(axes, snapshots):
        label = f'{Path(filename).stem}_epoch{checkpoint["epoch"]}'
        for scope, color in [('train', '#3b82f6'), ('validation', '#ea580c')]:
            rows = [row for row in all_rows if row['checkpoint'] == label and row['scope'] == scope]
            axis.scatter([row['signed_concentration'] for row in rows], [row['mean_gt_error_deg'] for row in rows],
                         s=12, alpha=.45, c=color, label=scope)
        axis.axvline(.95, color='gray', linestyle='--', linewidth=1)
        axis.set(title=label, xlabel='Signed concentration: 1 = consistent across rotations',
                 ylabel='Mean GT angle (degrees)', xlim=(0, 1.01), ylim=(0, 180))
        axis.legend()
    fig.savefig(report / 'consistency_vs_accuracy.png', dpi=160)
    plt.close(fig)
    table = '\n'.join(f'| {row["checkpoint"]} | {row["scope"]} | {row["mean_gt_error_deg"]:.2f}° | '
                      f'{row["mean_signed_concentration"]:.3f} | {row["mean_spread_to_consensus_deg"]:.2f}° | '
                      f'{row["concentration_ge_095_parts"]}/{row["parts"]} |' for row in summaries)
    text = f'''# 旋转一致性诊断（2026-11-09）

没有训练新模型或改变长训。对最佳验证权重和最近完整权重各做 1032 件×{args.views} 个固定新 SO(3) 推理，使用同一组旋转和原训练固定点子集。权重快照和逐件还原预测在 `outputs/{args.name}`，SHA256 见 metrics.json。

输入按 P@R.T 旋转，预测方向按 pred@R 转回历史帧。GT 在该帧始终是 +Z。向量平均的长度（集中度）为 1 表示所有旋转预测一致，接近 0 可能是散乱、正反抵消或多峰；不能直接用集中度作正确率。额外记录轴二阶矩、正反比例、到未旋转输入预测的角差，辅助区分。

| 权重 | 范围 | 平均 GT 角误差 | 平均集中度 | 到本件共识方向平均角差 | 集中度≥0.95 的件数 |
|---|---|---:|---:|---:|---:|
{table}

0.95 仅是本次描述统计阈值，未调参，不用于删件、训练或宣称工艺等价。完整逐件数据在 per_part.csv；少量新旋转的误差与长训 probe 数值不必完全相同。

判读：低集中度表明预测随输入旋转明显变化，优先核查旋转建模、多模式或符号切换；高集中度但偏离 GT 表示稳定偏向另一方向，需要查看形状、GT 与工艺上下文，不能自动判断标签错了；训练件与留出件的集中度差距则提示跨形状的旋转行为差异。几何对称与工艺等价仍需另外审计。

![旋转集中度与历史方向误差](consistency_vs_accuracy.png)
'''
    (report / 'README.md').write_text(text, encoding='utf8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
