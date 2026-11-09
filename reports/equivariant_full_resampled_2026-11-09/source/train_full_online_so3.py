"""Scale the eight-part fixed-point, online SO(3) recipe to the full split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from diagnose_pointnet_random_so3 import rotate_features
from train_magics_pointnet import PointNetAxis, angle_metrics, random_rotations


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def load_bases(use_normals, seed, device, point_count=2048):
    directory = ROOT / 'outputs/shape_deduplicated_pointclouds'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
    normal_directory = ROOT / 'outputs/shape_deduplicated_normals'
    bases, hashes = [], []
    for index, record in enumerate(manifest['records']):
        with np.load(directory / record['pointcloud_file']) as data:
            original = data['points'].copy()
        points = original.astype(np.float32)
        points -= points.mean(0, keepdims=True)
        points /= np.linalg.norm(points, axis=1).max()
        ids = np.random.default_rng(seed+1009*index).choice(len(points), point_count, replace=False)
        base = points[ids]
        if use_normals:
            with np.load(normal_directory / record['pointcloud_file']) as data:
                np.testing.assert_array_equal(data['points'], original)
                normals = data['normals'].astype(np.float32)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
            base = np.concatenate([base, normals[ids]], axis=-1)
        if not np.isfinite(base).all():
            raise ValueError(f'Invalid features: {index}')
        bases.append(base)
        hashes.append(hashlib.sha256(ids.tobytes()).hexdigest())
    return torch.from_numpy(np.stack(bases)).to(device), manifest, hashes


@torch.no_grad()
def assess(model, bases, indices, views, seed, batch_size, keep_rotations=False):
    device = bases.device
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = torch.as_tensor(indices, device=device).repeat_interleave(views)
    rotations = random_rotations(len(ids), generator, device)
    targets = rotations[:, :, 2]
    predictions = []
    model.eval()
    for start in range(0, len(ids), batch_size):
        selected = slice(start, start+batch_size)
        inputs = rotate_features(bases[ids[selected]], rotations[selected])
        predictions.append(model(inputs))
    predicted = torch.cat(predictions).cpu().numpy()
    target = targets.cpu().numpy()
    metrics = angle_metrics(predicted, target)
    metrics['cosine_loss'] = float((1-(predicted*target).sum(-1)).mean())
    per_part = []
    for local, index in enumerate(indices):
        selected = slice(local*views, (local+1)*views)
        part_metrics = angle_metrics(predicted[selected], target[selected])
        part_metrics['cosine_loss'] = float((1-(predicted[selected]*target[selected]).sum(-1)).mean())
        per_part.append({'part_index': int(index), 'metrics': part_metrics})
    arrays = {'predicted_axis': predicted, 'target_axis': target,
              'part_indices': ids.cpu().numpy(), 'rotation_indices': np.tile(np.arange(views), len(indices))}
    if keep_rotations:
        arrays['rotations'] = rotations.cpu().numpy()
    return metrics, per_part, arrays


def plot_curve(report, train_history, history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    axes[0].plot([row['epoch'] for row in train_history],
                 [row['online_train_loss'] for row in train_history], label='Online train', alpha=.65)
    for key, label in [('same_shape_probe', 'Training shapes, new SO(3)'),
                       ('validation', '100 unseen shapes')]:
        epochs = [row['epoch'] for row in history]
        axes[0].plot(epochs, [row[key]['cosine_loss'] for row in history], label=label)
        axes[1].plot(epochs, [row[key]['mean_angle_deg'] for row in history], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Signed cosine loss')
    axes[1].set(xlabel='Epoch', ylabel='Mean direction error (degrees)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(report / 'training_curve.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='pointnet_full_online_so3_normals_2026-11-09')
    parser.add_argument('--use-normals', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--minimum-epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--early-stopping', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error('Name must be a single directory component')
    if min(args.epochs, args.minimum_epochs, args.patience, args.eval_every, args.batch_size) < 1:
        parser.error('Counts must be positive')
    run, report = ROOT / 'runs' / args.name, ROOT / 'reports' / args.name
    if (run / 'config.json').exists() and not args.resume:
        parser.error('Existing run; choose another name')
    previous_config = None
    if args.resume:
        if not (run / 'last_model.pt').exists():
            parser.error('Resume requires last_model.pt')
        previous_config = json.loads((run / 'config.json').read_text(encoding='utf8'))
        for key in ('use_normals', 'seed', 'batch_size', 'eval_every', 'learning_rate'):
            if previous_config[key] != getattr(args, key):
                parser.error(f'Resume must preserve {key}')
    if not torch.cuda.is_available():
        parser.error('CUDA required')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    bases, manifest, subset_hashes = load_bases(args.use_normals, args.seed, device)
    split_file = ROOT / 'runs/pointnet_shape_dedup_2026-11-09_val100/split.json'
    split = json.loads(split_file.read_text(encoding='utf8'))
    train_ids, val_ids = split['train_indices'], split['validation_indices']
    assert len(train_ids) == 932 and len(val_ids) == 100
    assert not set(train_ids) & set(val_ids)
    assert set(train_ids) | set(val_ids) == set(range(len(bases)))
    train_groups = {manifest['records'][i]['shape_group_id'] for i in train_ids}
    val_groups = {manifest['records'][i]['shape_group_id'] for i in val_ids}
    assert not train_groups & val_groups
    assert [manifest['records'][i]['geometry_sha256'] for i in train_ids] == split['train_geometry_sha256']
    assert [manifest['records'][i]['geometry_sha256'] for i in val_ids] == split['validation_geometry_sha256']
    run.mkdir(parents=True, exist_ok=args.resume)
    report.mkdir(parents=True, exist_ok=True)
    source_files = ['tools/train_full_online_so3.py', 'tools/train_magics_pointnet.py',
                    'tools/diagnose_pointnet_random_so3.py', 'tools/prepare_full_part_normals.py']
    config = {
        **vars(args), 'task': 'Online SO(3) learning on 932 historical industrial shapes',
        'training_parts': 932, 'validation_parts': 100, 'point_count': 2048,
        'input_channels': 6 if args.use_normals else 3, 'initialization': 'From scratch; no eight-part checkpoint or synthetic data',
        'model': 'Same PointNetAxis as eight-part experiments, all layers trained',
        'architecture': 'Shared MLP C-64-64-64-128-1024, global max, head 1024-512-256-3, unit normalization',
        'dropout': 0, 'weight_decay': 0, 'optimizer': 'AdamW', 'gradient_clip': 1.,
        'learning_rate_schedule': f'Cosine {args.learning_rate} -> 1e-5 over {args.epochs} epochs',
        'point_processing': 'Same as eight-part recipe: center and max-radius normalize 4096 points, select fixed 2048 once per shape, no resampling per appearance',
        'point_subset_rng': 'numpy.default_rng(seed + 1009 * manifest index)',
        'point_subset_hashes': subset_hashes,
        'training_rotations_per_part_per_epoch': 16, 'training_views_fixed': False,
        'target': 'P_input=P_history@R.T; normals=N_history@R.T; target=R[:,2]=R@[0,0,1]',
        'loss': 'mean(1-dot(normalized_predicted_axis,target_axis)); signed, no symmetry correction',
        'same_shape_probe': 'All 932 training shapes x 4 fixed held-out SO(3) draws; no backpropagation',
        'validation': 'Same 100 validation shapes x 32 fixed SO(3) draws, checkpoint selection',
        'final_evaluation': 'At best validation checkpoint: 932 training shapes x 32 new SO(3), 100 validation shapes x 64 new SO(3)',
        'stop': f'Max {args.epochs} epochs; plateau early stopping {args.early_stopping}; three checks below 3 degrees',
        'changes_from_eight_part': ['932 instead of 8 shapes; initialization from scratch for full-scale experiment.',
                                   'Same fixed-point preprocessing, online rotations, architecture, batch size, no dropout/weight decay.',
                                   '200-epoch default cosine schedule is a full-data budget; eight-part continuation had 4000 epochs x 4 updates and a pretrained eight-part checkpoint.',
                                   'Open or inconsistent meshes retain source normal signs, explicitly audited; no shape exclusions.'],
        'steps_per_epoch': (len(train_ids)*16+args.batch_size-1)//args.batch_size,
        'split_source': str(split_file), 'shape_family_or_project_isolated': False,
        'wandb': 'disabled', 'device': torch.cuda.get_device_name(0), 'torch_version': str(torch.__version__),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_file_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
    }
    if previous_config:
        assert previous_config['point_subset_hashes'] == subset_hashes
        for folder in (run, report):
            if not (folder / 'initial_config.json').exists():
                write_json(folder / 'initial_config.json', previous_config)
        config['initialization'] = previous_config['initialization']
        config['continuations'] = previous_config.get('continuations', [])
    for folder in (run, report):
        write_json(folder / 'config.json', config)
        write_json(folder / 'split.json', split)
    if args.use_normals:
        audit = json.loads((ROOT / 'outputs/shape_deduplicated_normals/manifest.json').read_text(encoding='utf8'))
        assert not audit['failures'] and len(audit['records']) == len(bases)
        write_json(report / 'normal_attachment_audit.json', audit)
    audit_generator = torch.Generator(device=device).manual_seed(args.seed+80)
    rotations = random_rotations(64, audit_generator, device)
    identity = torch.eye(3, device=device).expand(64, -1, -1)
    torch.testing.assert_close(rotations @ rotations.transpose(1, 2), identity, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(64, device=device), atol=2e-6, rtol=2e-6)
    target = rotations[:, :, 2]
    restored = torch.bmm(target[:, None, :], rotations).squeeze(1)
    expected = torch.tensor([0., 0., 1.], device=device).expand_as(restored)
    torch.testing.assert_close(restored, expected, atol=2e-6, rtol=2e-6)
    write_json(report / 'frame_audit.json', {'passed': True, 'rotations_checked': 64,
               'max_recovered_up_error': float((restored-expected).abs().max()),
               'normal_rotation': 'Same R.T as points; no centering or scaling on normals'})
    model = PointNetAxis(input_channels=bases.shape[-1]).to(device)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    generator = torch.Generator(device=device).manual_seed(args.seed+19703)
    training_ids = torch.tensor(train_ids, device=device).repeat_interleave(16)
    history, train_history = [], []
    best_val, best_train, best_epoch, best_train_epoch = float('inf'), float('inf'), 0, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    start_epoch, prior_seconds = 0, 0.
    if args.resume:
        checkpoint = torch.load(run / 'last_model.pt', map_location=device, weights_only=True)
        start_epoch = checkpoint['epoch']
        if args.epochs <= start_epoch:
            parser.error('Requested epochs must exceed saved checkpoint epoch')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-start_epoch, eta_min=1e-5)
        generator.set_state(checkpoint['generator_state'].cpu())
        progress = json.loads((run / 'progress.json').read_text(encoding='utf8'))
        history = [row for row in progress['history'] if row['epoch'] <= start_epoch]
        train_history = [row for row in progress['train_history'] if row['epoch'] <= start_epoch]
        if not history or history[-1]['epoch'] != start_epoch:
            parser.error('Checkpoint does not match recorded evaluation history')
        best_row = min(history[1:], key=lambda row: row['validation']['mean_angle_deg'])
        best_val, best_epoch = best_row['validation']['mean_angle_deg'], best_row['epoch']
        best_row = min(history[1:], key=lambda row: row['same_shape_probe']['mean_angle_deg'])
        best_train, best_train_epoch = best_row['same_shape_probe']['mean_angle_deg'], best_row['epoch']
        prior_seconds = checkpoint.get('elapsed_seconds', progress['seconds'])
        continuation = {'checkpoint_epoch': start_epoch, 'new_max_epochs': args.epochs,
                        'previous_max_epochs': previous_config['epochs'], 'early_stopping': args.early_stopping,
                        'learning_rate_at_resume': optimizer.param_groups[0]['lr'],
                        'schedule': 'Continue at saved optimizer LR, cosine decay to 1e-5 over remaining epochs',
                        'restored': ['model', 'optimizer', 'training_rotation_generator', 'torch_rng', 'cuda_rng'],
                        'discarded_unsaved_epochs': progress['epochs_completed']-start_epoch}
        config['continuations'].append(continuation)
        config['learning_rate_schedule'] = continuation['schedule']
        for folder in (run, report):
            write_json(folder / 'config.json', config)
        print(f'Resuming epoch {start_epoch}, LR {optimizer.param_groups[0]["lr"]:.8f}, max {args.epochs}, plateau stop {args.early_stopping}', flush=True)
    train_metrics, _, _ = assess(model, bases, train_ids, 4, args.seed+73001, 64)
    val_metrics, _, arrays = assess(model, bases, val_ids, 32, args.seed+83001, 64)
    constant_baseline = angle_metrics(np.tile([0., 0., 1.], (len(arrays['target_axis']), 1)), arrays['target_axis'])
    if not args.resume:
        history.append({'epoch': 0, 'same_shape_probe': train_metrics, 'validation': val_metrics})
    else:
        np.testing.assert_allclose(val_metrics['mean_angle_deg'], history[-1]['validation']['mean_angle_deg'], atol=1e-4)
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
    print(f'Initial: same-shape {train_metrics["mean_angle_deg"]:.2f} deg, unseen {val_metrics["mean_angle_deg"]:.2f} deg; {config["steps_per_epoch"]} updates/epoch', flush=True)
    threshold_streak, stop_reason = 0, 'maximum_epochs'
    with (report / 'train_history.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'online_train_loss', 'learning_rate', 'epoch_seconds', 'optimizer_steps'])
        writer.writeheader()
        writer.writerows(train_history)
        handle.flush()
        for epoch in range(start_epoch+1, args.epochs+1):
            epoch_started = time.monotonic()
            model.train()
            ids = training_ids[torch.randperm(len(training_ids), generator=generator, device=device)]
            losses = []
            for start in range(0, len(ids), args.batch_size):
                base = bases[ids[start:start+args.batch_size]]
                rotations = random_rotations(len(base), generator, device)
                inputs = rotate_features(base, rotations)
                targets = rotations[:, :, 2]
                predicted = model(inputs)
                loss = (1-(predicted*targets).sum(-1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.detach()*len(base))
            row = {'epoch': epoch, 'online_train_loss': float(torch.stack(losses).sum()/len(ids)),
                   'learning_rate': optimizer.param_groups[0]['lr'],
                   'epoch_seconds': time.monotonic()-epoch_started,
                   'optimizer_steps': epoch*config['steps_per_epoch']}
            train_history.append(row)
            writer.writerow(row)
            handle.flush()
            scheduler.step()
            if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
                train_metrics, _, _ = assess(model, bases, train_ids, 4, args.seed+73001, 64)
                val_metrics, _, arrays = assess(model, bases, val_ids, 32, args.seed+83001, 64)
                history.append({**row, 'same_shape_probe': train_metrics, 'validation': val_metrics})
                current = val_metrics['mean_angle_deg']
                if current < best_val:
                    best_val, best_epoch = current, epoch
                    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                                'input_channels': bases.shape[-1], 'validation': val_metrics,
                                'same_shape_probe': train_metrics}, run / 'best_model.pt')
                    np.savez_compressed(run / 'best_validation_predictions.npz', **arrays)
                if train_metrics['mean_angle_deg'] < best_train:
                    best_train, best_train_epoch = train_metrics['mean_angle_deg'], epoch
                    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                                'input_channels': bases.shape[-1], 'same_shape_probe': train_metrics}, run / 'best_same_shape_model.pt')
                threshold_streak = threshold_streak+1 if current < 3 else 0
                plot_curve(report, train_history, history)
                write_json(report / 'evaluation_history.json', history)
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                            'generator_state': generator.get_state(), 'rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state(), 'input_channels': bases.shape[-1],
                            'elapsed_seconds': prior_seconds+time.monotonic()-started}, run / 'last_model.pt')
                print(f'Epoch {epoch}: loss {row["online_train_loss"]:.5f}; same-shape {train_metrics["mean_angle_deg"]:.2f} deg, unseen {current:.2f} deg; best unseen {best_val:.2f} @{best_epoch}; {row["epoch_seconds"]:.1f}s/epoch; elapsed {time.monotonic()-started:.1f}s', flush=True)
            elif epoch % 2 == 0:
                print(f'Epoch {epoch}: loss {row["online_train_loss"]:.5f}; {row["epoch_seconds"]:.1f}s/epoch', flush=True)
            write_json(run / 'progress.json', {'status': 'running', 'epochs_completed': epoch,
                       'best_epoch': best_epoch, 'best_validation_mean_angle_deg': best_val,
                       'best_same_shape_probe_mean_angle_deg': best_train, 'seconds': prior_seconds+time.monotonic()-started,
                       'history': history, 'train_history': train_history})
            if threshold_streak >= 3:
                stop_reason = 'validation_below_3deg_three_checks'
                break
            if args.early_stopping and epoch >= args.minimum_epochs and epoch-best_epoch >= args.patience:
                stop_reason = 'validation_plateau'
                break
    checkpoint = torch.load(run / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    train_metrics, train_parts, train_arrays = assess(model, bases, train_ids, 32, args.seed+183079, 64, True)
    val_metrics, val_parts, val_arrays = assess(model, bases, val_ids, 64, args.seed+283079, 64, True)
    np.savez_compressed(run / 'independent_same_shape_predictions.npz', **train_arrays)
    np.savez_compressed(run / 'independent_unseen_predictions.npz', **val_arrays)
    best_row = next(row for row in history if row['epoch'] == best_epoch)
    result = {'status': 'complete', 'epochs_completed': epoch, 'best_epoch': best_epoch,
              'best_same_shape_epoch': best_train_epoch, 'stop_reason': stop_reason,
              'training_parts': len(train_ids), 'validation_parts': len(val_ids),
              'seconds': prior_seconds+time.monotonic()-started, 'optimizer_steps': epoch*config['steps_per_epoch'],
              'best_checkpoint_same_shape_probe': best_row['same_shape_probe'],
              'best_checkpoint_validation': best_row['validation'],
              'independent_same_shape_rotations': train_metrics, 'independent_unseen_shape_rotations': val_metrics,
              'per_part_same_shape': train_parts, 'per_part_unseen_shape': val_parts,
              'baseline_constant_up': constant_baseline,
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'history': history, 'train_history': train_history,
              'limitations': ['Signed historical direction; symmetry-equivalent alternatives are not accounted for.',
                             'One seed, fixed point subsets; evaluates unseen rotations, not new point sampling.',
                             'Validation shapes select checkpoint; new rotations do not make these shapes independent test shapes.',
                             split['limitation']]}
    for folder in (run, report):
        write_json(folder / 'metrics.json', result)
    write_json(run / 'progress.json', result)
    summary = f'''# 全量工业件在线 SO(3) 训练（2026-11-09）

932 个训练件、100 个验证件，沿用之前的去重划分。输入固定 2048 点 {'XYZ+法向' if args.use_normals else 'XYZ'}；原始 4096 点先居中、单位半径归一化，再每件固定取 2048 点，每次出现在线重新生成随机 SO(3)。GT 与点云一起转，法向使用相同旋转；没有加入合成数据或支撑。

PointNet 从头训练，所有层可训练；batch {args.batch_size}，每件每轮 16 个朝向，每轮 {config['steps_per_epoch']} 次更新；AdamW、无 weight decay、无 dropout、梯度裁剪 1，余弦损失。初始学习率 {args.learning_rate}；调度及续训记录见 config.json，续训从保存的优化器学习率连续衰减到 1e-5。与八件延长实验的主要区别是数据规模、从头初始化和适应全数据量的训练预算；并非将八件实验的 epoch 数直接复制。

完成 {epoch} 轮／{result['optimizer_steps']} 次更新，用时 {result['seconds']/60:.1f} 分钟。停止原因：{stop_reason}。按 100 个验证件的方向误差选择第 {best_epoch} 轮权重。

| 评估范围 | 平均误差 | 中位误差 | ≤10° | 余弦损失 |
|---|---:|---:|---:|---:|
| 训练的 932 件，各 32 个独立新朝向 | {train_metrics['mean_angle_deg']:.3f}° | {train_metrics['median_angle_deg']:.3f}° | {train_metrics['within_10deg_pct']:.2f}% | {train_metrics['cosine_loss']:.6f} |
| 未训练的 100 件，各 64 个独立新朝向 | {val_metrics['mean_angle_deg']:.3f}° | {val_metrics['median_angle_deg']:.3f}° | {val_metrics['within_10deg_pct']:.2f}% | {val_metrics['cosine_loss']:.6f} |

每轮在线训练 loss 在 `train_history.csv`，定期同件新朝向／新件误差在 `evaluation_history.json`，曲线在 `training_curve.png`，完整配置、划分、每件指标分别保存在 `config.json`、`split.json`、`metrics.json`。源面法向审计在 `normal_attachment_audit.json`；非水密或法向绕序不一致的模型保留源面的符号并明确标记，未擅自删件或修复几何。

方向仍区分正负，尚未对对称等价 GT 修正。验证零件参与选权重，因此最终新旋转评估是同一批留出零件上的补充评估，不是全新独立零件测试。完整随机旋转与预测保存在对应 runs 目录，便于后续渲染三联图。
'''
    (report / 'README.md').write_text(summary, encoding='utf8')
    print(json.dumps({key: result[key] for key in ['status', 'best_epoch', 'epochs_completed', 'seconds',
          'independent_same_shape_rotations', 'independent_unseen_shape_rotations']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
