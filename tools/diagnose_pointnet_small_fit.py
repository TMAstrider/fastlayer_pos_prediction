"""Fit fixed rotations of eight training shapes; test fresh rotations of them."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from train_magics_pointnet import PointNetAxis, angle_metrics, evaluate, random_rotations


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'runs/pointnet_small_fit_2026-11-09'
    report = root / 'reports/pointnet_small_fit_2026-11-09'
    run.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    seed = 20261110
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    source = root / 'runs/pointnet_shape_dedup_2026-11-09_val100'
    split = json.loads((source / 'split.json').read_text(encoding='utf8'))
    cloud_dir = root / 'outputs/shape_deduplicated_pointclouds'
    manifest = json.loads((cloud_dir / 'manifest.json').read_text(encoding='utf8'))
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(split['train_indices'], 8, replace=False).tolist())
    assert not set(selected) & set(split['validation_indices'])
    generator = torch.Generator(device=device).manual_seed(seed+7)
    train_inputs, train_targets, fresh_inputs, fresh_targets = [], [], [], []
    for index in selected:
        record = manifest['records'][index]
        with np.load(cloud_dir / record['pointcloud_file']) as data:
            points = data['points'].astype(np.float32)
        points -= points.mean(0, keepdims=True)
        points /= np.linalg.norm(points, axis=1).max()
        points = torch.from_numpy(points[rng.choice(len(points), 2048, replace=False)]).to(device)
        for count, inputs_list, targets_list in ((16, train_inputs, train_targets), (64, fresh_inputs, fresh_targets)):
            rotations = random_rotations(count, generator, device)
            inputs_list.append(torch.bmm(points.unsqueeze(0).expand(count, -1, -1), rotations.transpose(1, 2)))
            targets_list.append(rotations[:, :, 2])
    train_inputs, train_targets = torch.cat(train_inputs), torch.cat(train_targets)
    fresh_inputs, fresh_targets = torch.cat(fresh_inputs), torch.cat(fresh_targets)
    config = {'task': 'Fixed-example overfit diagnostic, not held-out-shape evaluation',
              'parts': 8, 'fixed_training_views_per_part': 16, 'fresh_views_per_part': 64,
              'points': 2048, 'batch_size': 32, 'max_epochs': 1000, 'seed': seed,
              'model': 'Same PointNet architecture; scratch; dropout disabled for memorization diagnostic',
              'optimizer': 'AdamW lr=0.001 weight_decay=0', 'loss': 'unit-axis cosine',
              'gradient_clip': 1., 'learning_rate_schedule': 'constant',
              'selected_indices': selected, 'selected_records': [manifest['records'][i] for i in selected],
              'fresh_view_scope': 'Same shapes and same base points, unseen fixed random rotations; no new-part test',
              'stop': 'Train eval-mode mean below 3 degrees for 3 consecutive checks, or 1000 epochs'}
    for folder in (run, report):
        (folder / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    model = PointNetAxis().to(device)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    history, best, streak = [], float('inf'), 0
    started = time.monotonic()
    for epoch in range(1, 1001):
        model.train()
        permutation = torch.randperm(len(train_inputs), generator=generator, device=device)
        losses = []
        for start in range(0, len(permutation), 32):
            ids = permutation[start:start+32]
            pred = model(train_inputs[ids])
            loss = (1-(pred*train_targets[ids]).sum(-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0:
            train_metrics, _, _ = evaluate(model, train_inputs, train_targets, 64)
            row = {'epoch': epoch, 'online_train_loss': float(np.mean(losses)), 'train_eval': train_metrics}
            history.append(row)
            current = train_metrics['mean_angle_deg']
            if current < best:
                best = current
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, run / 'best_model.pt')
            streak = streak+1 if current < 3 else 0
            (run / 'progress.json').write_text(json.dumps({'status': 'running', 'history': history}, indent=2), encoding='utf8')
            print(f'epoch {epoch}: fixed train eval {current:.3f} deg, best {best:.3f}, elapsed {time.monotonic()-started:.1f}s', flush=True)
            if streak >= 3:
                break
    checkpoint = torch.load(run / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    train_metrics, _, _ = evaluate(model, train_inputs, train_targets, 64)
    fresh_metrics, _, _ = evaluate(model, fresh_inputs, fresh_targets, 64)
    result = {'status': 'complete', 'best_epoch': checkpoint['epoch'], 'epochs_completed': epoch,
              'seconds': time.monotonic()-started, 'fixed_training_views': train_metrics,
              'same_shapes_fresh_rotations': fresh_metrics, 'history': history,
              'limitations': ['Eight randomly selected training shapes only, one seed; no industrial accuracy claim.',
                             'No dropout or weight decay; deliberately easier than the main training setup.',
                             'Fresh rotation metrics evaluated only after selection by fixed training error.']}
    for folder in (run, report):
        (folder / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    (run / 'progress.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    text = f'''# 小样本固定朝向拟合诊断（2026-11-09）

从主实验 932 个训练零件中固定随机选择 8 件，不使用原 100 件验证零件训练。每件固定 16 个训练旋转、64 个另外的旋转，2048 点、batch32。PointNet 从头训练，全网更新，关闭 dropout 和 weight decay，恒定 lr=0.001，余弦损失与主实验一致。

完成 {epoch} 轮，用时 {result['seconds']:.1f} 秒；按固定训练视图的 eval 模式误差选择最佳 epoch {checkpoint['epoch']}。

| 范围 | 平均误差 | 中位误差 | ≤3° | ≤10° |
| --- | ---: | ---: | ---: | ---: |
| 128 个固定训练视图 | {train_metrics['mean_angle_deg']:.3f}° | {train_metrics['median_angle_deg']:.3f}° | {train_metrics['within_3deg_pct']:.3f}% | {train_metrics['within_10deg_pct']:.3f}% |
| 同 8 件的 512 个新旋转 | {fresh_metrics['mean_angle_deg']:.3f}° | {fresh_metrics['median_angle_deg']:.3f}° | {fresh_metrics['within_3deg_pct']:.3f}% | {fresh_metrics['within_10deg_pct']:.3f}% |

这项诊断刻意简化了拟合难度：固定点、固定训练旋转、无 dropout、无 weight decay。同件新旋转使用相同基础点子集，仅旋转改变。结果只用于分析固定样本拟合和朝向泛化，不是未知零件验证；不能单独证明生产标签正确或已排除全部训练问题。完整配置、选中零件与逐检查点结果见本目录 JSON。权重位于 `runs/pointnet_small_fit_2026-11-09/best_model.pt`。
'''
    (report / 'README.md').write_text(text, encoding='utf8')
    print(json.dumps({key:value for key,value in result.items() if key != 'history'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
