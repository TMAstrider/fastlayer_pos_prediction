"""Evaluate the eight-shape diagnostic checkpoint on eight held-out shapes."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from train_magics_pointnet import PointNetAxis, angle_metrics, evaluate, random_rotations


ROOT = Path(__file__).resolve().parents[1]


def main():
    name = 'pointnet_eight_unseen_parts_2026-11-09'
    run = ROOT / 'runs' / name
    report = ROOT / 'reports' / name
    if (run / 'config.json').exists():
        raise FileExistsError('Evaluation already exists; preserve it before choosing a new run name.')
    training = ROOT / 'runs/pointnet_random_so3_8parts_extended_2026-11-09'
    training_config = json.loads((training / 'config.json').read_text(encoding='utf-8-sig'))
    split = json.loads((ROOT / 'runs/pointnet_shape_dedup_2026-11-09_val100/split.json').read_text(encoding='utf8'))
    cloud_dir = ROOT / 'outputs/shape_deduplicated_pointclouds'
    manifest = json.loads((cloud_dir / 'manifest.json').read_text(encoding='utf8'))
    records = manifest['records']
    seed = 20261111
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(split['validation_indices'], 8, replace=False).tolist())
    trained = training_config['selected_indices']
    assert not set(selected) & set(split['train_indices'])
    assert not set(selected) & set(trained)
    assert not {records[i]['shape_group_id'] for i in selected} & {records[i]['shape_group_id'] for i in trained}
    device = torch.device('cuda')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    generator = torch.Generator(device=device).manual_seed(seed+73001)
    inputs, targets, rotations_list = [], [], []
    for index in selected:
        with np.load(cloud_dir / records[index]['pointcloud_file']) as data:
            points = data['points'].astype(np.float32)
        points -= points.mean(0, keepdims=True)
        points /= np.linalg.norm(points, axis=1).max()
        base = torch.from_numpy(points[rng.choice(len(points), 2048, replace=False)]).to(device)
        rotations = random_rotations(64, generator, device)
        inputs.append(torch.bmm(base.unsqueeze(0).expand(64, -1, -1), rotations.transpose(1, 2)))
        targets.append(rotations[:, :, 2])
        rotations_list.append(rotations.cpu().numpy())
    checkpoint_path = training / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = PointNetAxis().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    started = time.monotonic()
    metrics, predicted, target = evaluate(model, torch.cat(inputs), torch.cat(targets), 64)
    metrics['cosine_loss'] = float((1-(predicted*target).sum(-1)).mean())
    elapsed = time.monotonic()-started
    per_part = []
    for position, index in enumerate(selected):
        bounds = slice(position*64, (position+1)*64)
        local = angle_metrics(predicted[bounds], target[bounds])
        local['cosine_loss'] = float((1-(predicted[bounds]*target[bounds]).sum(-1)).mean())
        per_part.append({'part_index': index, 'shape_group_id': records[index]['shape_group_id'], 'metrics': local})
    config = {
        'task': 'Eight previously unseen shapes evaluated with the eight-training-shape PointNet checkpoint',
        'evaluation_scope': 'unseen_parts', 'selected_indices': selected,
        'selected_records': [records[i] for i in selected], 'training_indices': trained,
        'selection': 'Seeded uniform sample without replacement from original 100 held-out shapes, before inference.',
        'seed': seed, 'points': 2048, 'fresh_views_per_part': 64,
        'point_sampling': 'One fixed seeded 2048 subset of stored 4096 points per shape, then 64 new SO(3) rotations.',
        'input': 'XYZ only, normalized using stored 4096-point mean and maximum radius; no normals or supports.',
        'target': 'R[:,2] for P@R.T; signed up direction, no symmetry correction; yaw ignored.',
        'checkpoint': str(checkpoint_path.relative_to(ROOT)),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'checkpoint_additional_epoch': checkpoint['epoch'],
        'training_performed': False, 'checkpoint_selection_on_this_evaluation': False,
        'device': torch.cuda.get_device_name(),
    }
    result = {'status': 'complete', 'unseen_parts_512_rotations': metrics,
              'per_part': per_part, 'inference_seconds': elapsed,
              'limitations': ['Eight held-out shape groups only; no family/project isolation.',
                              'Original 100-part pool was used for earlier different-model validation; not a pristine final test set.',
                              'Current checkpoint trained only on the listed eight training shapes; no fitting or selection on these eight.',
                              'One point subset per shape; no symmetry-equivalent direction correction.']}
    run.mkdir(parents=True)
    report.mkdir(parents=True)
    for folder in (run, report):
        (folder / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
        (folder / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    ids = np.repeat(selected, 64)
    rotations = np.concatenate(rotations_list)
    np.testing.assert_allclose(rotations[:, :, 2], target, atol=1e-6)
    np.savez_compressed(run / 'independent_predictions.npz', predicted_axis=predicted, target_axis=target, part_indices=ids)
    np.savez_compressed(run / 'independent_preview_rotations.npz', augmentation_rotations=rotations, part_indices=ids)
    rows = '\n'.join(f'| {row["part_index"]} | {row["metrics"]["mean_angle_deg"]:.3f}° | '
                     f'{row["metrics"]["median_angle_deg"]:.3f}° | {row["metrics"]["within_10deg_pct"]:.3f}% |'
                     for row in per_part)
    text = f'''# 八个未见零件评测（2026-11-09）

使用八零件延长训练的最佳 PointNet（新增 epoch {checkpoint['epoch']}），没有重新训练或按本次结果选择权重。固定 seed={seed}，从原保留的 100 件里事先随机选择 8 件：{selected}，与当前模型的 8 个训练零件及其形状组均不重叠。

输入 XYZ 2048 点，无法向/支撑；加载、中心化和单位球归一化规则沿用八件训练。每件固定一个点子集，生成 64 个全新均匀 SO(3) 朝向，共 512 视图。loss 和评测仍区分上下，不做对称修正，忽略 yaw。所有方向保存实际旋转矩阵供三联图使用。

总体：平均误差 **{metrics['mean_angle_deg']:.3f}°**，中位 **{metrics['median_angle_deg']:.3f}°**，P90 **{metrics['p90_angle_deg']:.3f}°**，cosine loss **{metrics['cosine_loss']:.6f}**；≤3° **{metrics['within_3deg_pct']:.3f}%**、≤10° **{metrics['within_10deg_pct']:.3f}%**、≤30° **{metrics['within_30deg_pct']:.3f}%**。

| 索引 | 64 朝向平均误差 | 中位误差 | ≤10° |
| --- | ---: | ---: | ---: |
{rows}

对照当前模型在原 8 个训练形状的新朝向上平均 15.164°、中位 3.870°、≤10° 86.328%。两个数回答不同问题：同件朝向泛化与未见形状泛化。此处 8 件结果仅用于检查当前 checkpoint 的跨形状表现，不能当作全工业数据精度。

原 100 件池曾参与另一个 932 件模型的选模；本次 8 件未用于当前八件模型的训练或选模，因此称保留验证子集，不宣称是从未用于任何分析的最终测试集。形状去重有限容差，未按零件族/工程隔离。

三联图入口：`outputs/eight_unseen_part_pose_comparison/index.html`，每件只展示第一个保存朝向，不按误差挑选。图上夹角是单次误差，表中为 64 朝向平均。保留初始方向箭头和 40% 透明平台。

评测脚本 `tools/evaluate_eight_unseen_parts.py`；权重哈希、完整来源与划分见 config.json，逐件指标见 metrics.json，原始预测与旋转见 runs/{name}/。本次未启用 W&B。
'''
    (report / 'README.md').write_text(text, encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
