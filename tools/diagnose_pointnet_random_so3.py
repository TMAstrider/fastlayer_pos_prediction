"""Test online SO(3) learning on the same eight shapes as the fixed-view fit."""

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from train_magics_pointnet import PointNetAxis, angle_metrics, evaluate, random_rotations
from wandb_tracking import start_run, log_epoch, finish_run


def measured(model, inputs, targets):
    metrics, predicted, target = evaluate(model, inputs, targets, 64)
    metrics['cosine_loss'] = float((1-(predicted*target).sum(-1)).mean())
    return metrics, predicted, target


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='pointnet_random_so3_8parts_2026-11-09')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--min-learning-rate', type=float, default=1e-5)
    parser.add_argument('--cosine-schedule', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-seed-offset', type=int, default=9703)
    parser.add_argument('--test-seed-offset', type=int, default=83079)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error('Require positive epochs and 0 <= minimum LR <= initial LR')
    if Path(args.name).name != args.name:
        parser.error('Name must be a single directory name')
    run = root / 'runs' / args.name
    report = root / 'reports' / args.name
    if (run / 'config.json').exists():
        parser.error('Run already exists; choose a new --name to preserve prior results')
    run.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    prior = json.loads((root / 'reports/pointnet_small_fit_2026-11-09/config.json').read_text(encoding='utf8'))
    split = json.loads((root / 'runs/pointnet_shape_dedup_2026-11-09_val100/split.json').read_text(encoding='utf8'))
    seed = prior['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    selected = prior['selected_indices']
    assert set(selected) <= set(split['train_indices'])
    assert not set(selected) & set(split['validation_indices'])
    cloud_dir = root / 'outputs/shape_deduplicated_pointclouds'
    manifest = json.loads((cloud_dir / 'manifest.json').read_text(encoding='utf8'))
    rng = np.random.default_rng(seed)
    assert selected == sorted(rng.choice(split['train_indices'], 8, replace=False).tolist())
    reference_generator = torch.Generator(device=device).manual_seed(seed+7)
    clouds, old_inputs, old_targets, probe_inputs, probe_targets = [], [], [], [], []
    for index in selected:
        record = manifest['records'][index]
        with np.load(cloud_dir / record['pointcloud_file']) as data:
            points = data['points'].astype(np.float32)
        points -= points.mean(0, keepdims=True)
        points /= np.linalg.norm(points, axis=1).max()
        base = torch.from_numpy(points[rng.choice(len(points), 2048, replace=False)]).to(device)
        clouds.append(base)
        for count, inputs_list, targets_list in ((16, old_inputs, old_targets), (64, probe_inputs, probe_targets)):
            rotations = random_rotations(count, reference_generator, device)
            inputs_list.append(torch.bmm(base.unsqueeze(0).expand(count, -1, -1), rotations.transpose(1, 2)))
            targets_list.append(rotations[:, :, 2])
    clouds = torch.stack(clouds)
    old_inputs, old_targets = torch.cat(old_inputs), torch.cat(old_targets)
    probe_inputs, probe_targets = torch.cat(probe_inputs), torch.cat(probe_targets)
    generator = torch.Generator(device=device).manual_seed(seed+args.training_seed_offset)
    config = {**prior, 'task': 'Online random SO(3) on the same eight training shapes',
              'training_rotations_per_part_per_epoch': 16, 'training_views_fixed': False,
              'fixed_training_views_per_part': None, 'training_generator_seed': seed+args.training_seed_offset,
              'probe': 'Same 512 fresh-rotation views from prior diagnostic, never backpropagated; selects checkpoint',
              'final_test': '512 independent rotations of same shapes, generated only after selection',
              'point_sampling': 'Same fixed 2048-point subset per shape as prior diagnostic; rotations alone change',
              'stop': f'Probe mean below 3 degrees for three consecutive checks, or {args.epochs} additional epochs',
              'max_epochs': args.epochs, 'initial_learning_rate': args.learning_rate,
              'minimum_learning_rate': args.min_learning_rate if args.cosine_schedule else args.learning_rate,
              'learning_rate_schedule': 'cosine' if args.cosine_schedule else 'constant',
              'optimizer': 'AdamW weight_decay=0; fresh optimizer for this run',
              'model': 'Same PointNet architecture; dropout disabled; checkpoint initialization' if args.checkpoint else prior['model'],
              'initial_checkpoint': str(args.checkpoint) if args.checkpoint else None,
              'final_test_seed': seed+args.test_seed_offset,
              'wandb': 'disabled by default; only enabled explicitly via WANDB_MODE',
              'source_commit': '4575b82'}
    for folder in (run, report):
        (folder / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    tracking = start_run(run, config, tags=['same-shape', 'online-so3', '8-parts'])
    model = PointNetAxis().to(device)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    if args.checkpoint:
        initial = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initial['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_learning_rate) if args.cosine_schedule else None
    training_ids = torch.arange(8, device=device).repeat_interleave(16)
    history, train_history, best, streak = [], [], float('inf'), 0
    started = time.monotonic()
    if args.checkpoint:
        initial_metrics, _, _ = measured(model, probe_inputs, probe_targets)
        best = initial_metrics['mean_angle_deg']
        torch.save({'model_state_dict': model.state_dict(), 'epoch': 0}, run / 'best_model.pt')
        history.append({'epoch': 0, 'fixed_probe': initial_metrics})
        print(f'Initial checkpoint probe: {best:.3f} deg, loss {initial_metrics["cosine_loss"]:.5f}', flush=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        ids = training_ids[torch.randperm(len(training_ids), generator=generator, device=device)]
        losses, last_inputs, last_targets = [], [], []
        for start in range(0, len(ids), 32):
            base = clouds[ids[start:start+32]]
            rotations = random_rotations(len(base), generator, device)
            inputs = torch.bmm(base, rotations.transpose(1, 2))
            targets = rotations[:, :, 2]
            predicted = model(inputs)
            loss = (1-(predicted*targets).sum(-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.item()))
            last_inputs.append(inputs)
            last_targets.append(targets)
        row = {'epoch': epoch, 'online_train_loss': float(np.mean(losses)),
               'learning_rate': optimizer.param_groups[0]['lr']}
        train_history.append(dict(row))
        if scheduler:
            scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            probe_metrics, _, _ = measured(model, probe_inputs, probe_targets)
            recent_metrics, _, _ = measured(model, torch.cat(last_inputs), torch.cat(last_targets))
            row.update({'recent_training_views_eval': recent_metrics, 'fixed_probe': probe_metrics})
            history.append(row)
            current = probe_metrics['mean_angle_deg']
            if current < best:
                best = current
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict()}, run / 'best_model.pt')
            streak = streak+1 if current < 3 else 0
            temporary = run / 'progress.tmp.json'
            temporary.write_text(json.dumps({'status': 'running', 'best_probe_deg': best,
                                             'epochs_completed': epoch, 'history': history,
                                             'train_history': train_history}, indent=2), encoding='utf8')
            temporary.replace(run / 'progress.json')
            print(f'epoch {epoch}: loss {row["online_train_loss"]:.4f}, probe {current:.2f} deg, best {best:.2f}, elapsed {time.monotonic()-started:.1f}s', flush=True)
            if streak >= 3:
                log_epoch(tracking, row)
                break
        log_epoch(tracking, row)
    checkpoint = torch.load(run / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    probe_metrics, _, _ = measured(model, probe_inputs, probe_targets)
    old_metrics, _, _ = measured(model, old_inputs, old_targets)
    test_generator = torch.Generator(device=device).manual_seed(seed+args.test_seed_offset)
    test_inputs, test_targets = [], []
    for base in clouds:
        rotations = random_rotations(64, test_generator, device)
        test_inputs.append(torch.bmm(base.unsqueeze(0).expand(64, -1, -1), rotations.transpose(1, 2)))
        test_targets.append(rotations[:, :, 2])
    test_metrics, predicted, target = measured(model, torch.cat(test_inputs), torch.cat(test_targets))
    per_part = []
    for local, index in enumerate(selected):
        bounds = slice(local*64, (local+1)*64)
        per_part.append({'part_index': index, 'shape_group_id': manifest['records'][index]['shape_group_id'],
                         'metrics': angle_metrics(predicted[bounds], target[bounds])})
    result = {'status': 'complete', 'best_epoch': checkpoint['epoch'], 'epochs_completed': epoch,
              'seconds': time.monotonic()-started, 'old_fixed_128_views': old_metrics,
              'selection_probe_512_views': probe_metrics, 'independent_same_shape_512_rotations': test_metrics,
              'per_part_independent_rotations': per_part, 'history': history, 'train_history': train_history,
              'limitations': ['Same eight shapes and fixed points; not new-part generalization.',
                             f'One seed, dropout and weight decay disabled, {epoch*4} optimizer steps in this run.',
                             'Same-shape probe selects checkpoint; final rotation set evaluated only once.']}
    for folder in (run, report):
        (folder / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    finish_run(tracking, result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    epochs = [row['epoch'] for row in history]
    axes[0].plot([row['epoch'] for row in train_history], [row['online_train_loss'] for row in train_history], label='Online train', alpha=.5)
    axes[0].plot(epochs, [row['fixed_probe']['cosine_loss'] for row in history], label='Fixed probe, eval')
    axes[0].set(xlabel='Epoch (4 updates)', ylabel='Cosine loss')
    axes[0].legend()
    axes[1].plot(epochs, [row['fixed_probe']['mean_angle_deg'] for row in history], label='Fixed probe')
    recent_rows = [row for row in history if 'recent_training_views_eval' in row]
    axes[1].plot([row['epoch'] for row in recent_rows], [row['recent_training_views_eval']['mean_angle_deg'] for row in recent_rows], label='Recent training views, eval')
    axes[1].set(xlabel='Epoch (4 updates)', ylabel='Mean angle (degrees)')
    axes[1].legend()
    fig.savefig(report / 'training_curve.png', dpi=160)
    plt.close(fig)
    (run / 'progress.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    np.savez_compressed(run / 'independent_predictions.npz', predicted_axis=predicted, target_axis=target, part_indices=np.repeat(selected,64))
    prior_metrics = json.loads((root / 'reports/pointnet_small_fit_2026-11-09/metrics.json').read_text(encoding='utf8'))
    text = f'''# 八零件在线 SO(3) 诊断（2026-11-09）

沿用固定旋转实验的 8 个训练零件、相同 2048 点子集、batch32、无 dropout / weight decay、余弦损失。每次更新重新生成 SO(3)，每件每轮 16 个新朝向，一轮 4 次更新。原有 100 个验证零件不参与训练。起始权重：{args.checkpoint or '从头初始化'}；优化器重新初始化，初始 lr={args.learning_rate}，调度为 {config['learning_rate_schedule']}，最低 lr={config['minimum_learning_rate']}。

本轮最多 {args.epochs} 轮，实际 {epoch} 轮，约 {result['seconds']/60:.2f} 分钟。按固定 512 个同件探针旋转选择本轮 epoch {checkpoint['epoch']}。选择之后生成另一组独立种子的 512 个旋转，评估一次。

| 范围 | 平均误差 | 余弦损失 | ≤3° | ≤10° |
| --- | ---: | ---: | ---: | ---: |
| 原实验 128 个固定视图，当前不参与拟合 | {old_metrics['mean_angle_deg']:.3f}° | {old_metrics['cosine_loss']:.5f} | {old_metrics['within_3deg_pct']:.3f}% | {old_metrics['within_10deg_pct']:.3f}% |
| 512 个选模探针旋转 | {probe_metrics['mean_angle_deg']:.3f}° | {probe_metrics['cosine_loss']:.5f} | {probe_metrics['within_3deg_pct']:.3f}% | {probe_metrics['within_10deg_pct']:.3f}% |
| 512 个另外保留的同件旋转 | {test_metrics['mean_angle_deg']:.3f}° | {test_metrics['cosine_loss']:.5f} | {test_metrics['within_3deg_pct']:.3f}% | {test_metrics['within_10deg_pct']:.3f}% |

此前固定训练视图可拟合到 {prior_metrics['fixed_training_views']['mean_angle_deg']:.3f}°，但同件 512 个新旋转误差为 {prior_metrics['same_shapes_fresh_rotations']['mean_angle_deg']:.3f}°。本轮的选模探针就是此前那组新旋转，未用于反向传播；最终旋转集合独立生成。两轮选模依据不同，不能视为严格相同选模预算的比较。

本轮保持固定基础点、模型结构和旋转标签规则，训练预算、学习率与起始权重见配置。它检验已见形状的朝向泛化，不检验新零件泛化，也没有证明所有几何存在唯一可识别的方向。训练 loss、近期训练视图 eval loss、探针指标以及逐件独立旋转指标都在 metrics.json；逐轮在线 loss 另存于 train_history；权重保存在 `runs/{args.name}/best_model.pt`。
'''
    (report / 'README.md').write_text(text, encoding='utf8')
    print(json.dumps({key:value for key,value in result.items() if key not in ('history','train_history','per_part_independent_rotations')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
