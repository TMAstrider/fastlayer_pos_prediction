"""Refit only the final direction layer on training features, then evaluate once."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from train_magics_pointnet import PointNetAxis, angle_metrics, load_clouds, make_validation, random_rotations


@torch.no_grad()
def extract_features(model, points):
    pooled = model.point_mlp(points.transpose(1, 2).contiguous()).amax(dim=2)
    return model.regressor[:-1](pooled)


@torch.no_grad()
def validation_features(model, inputs, batch_size):
    return torch.cat([extract_features(model, inputs[start:start+batch_size]).cpu()
                      for start in range(0, len(inputs), batch_size)])


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=root / 'runs/pointnet_shape_dedup_2026-11-09_val100')
    parser.add_argument('--output', type=Path, default=root / 'runs/pointnet_output_weight_edit_2026-11-09')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_dir = root / 'reports/pointnet_output_weight_edit_2026-11-09'
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = json.loads((args.run / 'config.json').read_text(encoding='utf8'))
    recorded_split = json.loads((args.run / 'split.json').read_text(encoding='utf8'))
    device = torch.device('cuda')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    clouds, split, _ = load_clouds(Path(config['clouds_directory']), device, config['seed'], .2, config['validation_parts'])
    assert split['train_indices'] == recorded_split['train_indices']
    assert split['validation_indices'] == recorded_split['validation_indices']
    assert not set(split['train_shape_group_ids']) & set(split['validation_shape_group_ids'])
    checkpoint_path = args.run / 'best_model.pt'
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = PointNetAxis().to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    old_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    experiment = {
        'method': 'Closed-form ridge least-squares refit of final 256-to-3 affine layer only; fixed coefficient, no validation tuning.',
        'source_checkpoint_sha256': checkpoint_hash,
        'source_checkpoint_epoch': checkpoint['epoch'],
        'training_parts': len(split['train_indices']),
        'validation_parts': len(split['validation_indices']),
        'training_rotations_per_part': 16,
        'ridge_coefficient_per_sample': .001,
        'training_seed': config['seed'] + 61523,
        'loss_for_edit': 'Mean squared error on raw Cartesian direction components plus ridge penalty; output normalized for evaluation.',
        'edited_parameters': ['regressor.8.weight', 'regressor.8.bias'],
        'validation_use': 'Evaluation after the single predetermined fit; source checkpoint was already selected using this validation set.',
    }
    (args.output / 'config.json').write_text(json.dumps(experiment, indent=2), encoding='utf8')
    generator = torch.Generator(device=device).manual_seed(experiment['training_seed'])
    training_ids = torch.tensor(split['train_indices'], device=device).repeat_interleave(16)
    features, targets = [], []
    with torch.no_grad():
        for start in range(0, len(training_ids), 64):
            full = clouds[training_ids[start:start+64]]
            sample_ids = torch.rand(full.shape[:2], device=device, generator=generator).topk(config['point_count'], dim=1, largest=False).indices
            sampled = torch.gather(full, 1, sample_ids.unsqueeze(-1).expand(-1, -1, 3))
            rotations = random_rotations(len(full), generator, device)
            inputs = torch.bmm(sampled, rotations.transpose(1, 2))
            features.append(extract_features(model, inputs).cpu())
            targets.append(rotations[:, :, 2].cpu())
            if start % 2048 == 0:
                print(f'Training-only features {start}/{len(training_ids)}', flush=True)
    train_features, train_targets = torch.cat(features), torch.cat(targets)
    design = torch.cat([train_features.double(), torch.ones((len(train_features), 1), dtype=torch.float64)], dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * (len(design) * experiment['ridge_coefficient_per_sample'])
    penalty[-1, -1] = 0.
    coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ train_targets.double())
    with torch.no_grad():
        model.regressor[-1].weight.copy_(coefficients[:-1].T.to(device=device, dtype=torch.float32))
        model.regressor[-1].bias.copy_(coefficients[-1].to(device=device, dtype=torch.float32))
    changed = [key for key, value in model.state_dict().items() if not torch.equal(value, old_state[key])]
    assert sorted(changed) == sorted(experiment['edited_parameters']), changed
    torch.save({'model_state_dict': model.state_dict(), 'source_epoch': checkpoint['epoch'], 'weight_edit': experiment}, args.output / 'edited_model.pt')
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    val_inputs, val_targets, part_ids, rotation_ids = make_validation(clouds, split['validation_indices'], config['validation_rotations_per_part'], config['point_count'], config['seed'])
    val_features = validation_features(model, val_inputs, 64)
    with torch.no_grad():
        base_pred = nn.functional.linear(val_features, old_state['regressor.8.weight'].cpu(), old_state['regressor.8.bias'].cpu()).numpy()
        edit_pred = model.regressor[-1](val_features.to(device)).cpu().numpy()
        base_train = nn.functional.linear(train_features, old_state['regressor.8.weight'].cpu(), old_state['regressor.8.bias'].cpu()).numpy()
        edit_train = model.regressor[-1](train_features.to(device)).cpu().numpy()
    target = val_targets.cpu().numpy()
    with np.load(args.run / 'best_validation_predictions.npz') as stored:
        assert np.array_equal(target, stored['target_axis'])
        saved_axis = stored['predicted_axis']
    base_unit = base_pred / np.linalg.norm(base_pred, axis=1, keepdims=True).clip(1e-8)
    max_reproduction_error = float(np.abs(base_unit-saved_axis).max())
    assert max_reproduction_error < 1e-4, max_reproduction_error
    rng = np.random.default_rng(config['seed'] + 301)
    variants = {'original': base_pred, 'edited': edit_pred,
                'random_unit_direction': rng.normal(size=target.shape),
                'constant_input_plus_z': np.tile([0., 0., 1.], (len(target), 1))}
    results = {name: {'all_views': angle_metrics(prediction, target),
                      'random_rotations_only': angle_metrics(prediction[rotation_ids != 0], target[rotation_ids != 0])}
               for name, prediction in variants.items()}
    edit_unit = edit_pred / np.linalg.norm(edit_pred, axis=1, keepdims=True).clip(1e-8)
    original_angles = np.degrees(np.arccos(np.sum(base_unit*target, axis=1).clip(-1, 1)))
    edited_angles = np.degrees(np.arccos(np.sum(edit_unit*target, axis=1).clip(-1, 1)))
    part_delta = np.array([(edited_angles-original_angles)[part_ids == part_id].mean() for part_id in np.unique(part_ids)])
    bootstrap = np.random.default_rng(config['seed']+477).choice(part_delta, size=(10000, len(part_delta)), replace=True).mean(1)
    result = {'experiment': experiment, 'metrics': results,
              'training_fit_views': {'original': angle_metrics(base_train, train_targets.numpy()),
                                     'edited': angle_metrics(edit_train, train_targets.numpy())},
              'edited_minus_original_mean_deg': float(part_delta.mean()),
              'part_bootstrap_95pct_interval_deg': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
              'source_checkpoint_unchanged': True,
              'changed_state_keys': changed,
              'original_prediction_reproduction_max_abs_error': max_reproduction_error,
              'elapsed_seconds': time.monotonic()-started,
              'limitations': ['Same validation set previously selected the source checkpoint; no independent test.',
                             'Editing the output layer does not add missing geometry or process information.',
                             'Training fit metrics are in-sample; random baseline is one fixed-seed realization.']}
    np.savez_compressed(args.output / 'validation_predictions.npz', original_axis=base_unit, edited_axis=edit_unit,
                        target_axis=target, part_ids=part_ids, rotation_ids=rotation_ids)
    for folder in (args.output, report_dir):
        (folder / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    summary = ['# 输出层权重编辑实验（2026-11-09）', '',
               '仅使用 932 个训练零件、每件 16 个随机旋转，冻结特征与 BatchNorm，以固定正则系数 0.001 的岭回归重算最后 256→3 层的 771 个参数。原始最佳权重未覆盖。未用验证标签拟合或挑选正则系数。', '',
               '| 模型 | 平均误差 | ≤10° | ≤30° |', '| --- | ---: | ---: | ---: |']
    for name, metrics in results.items():
        value = metrics['all_views']
        summary.append(f"| {name} | {value['mean_angle_deg']:.3f}° | {value['within_10deg_pct']:.3f}% | {value['within_30deg_pct']:.3f}% |")
    summary += ['', f"编辑减原始的平均误差：{part_delta.mean():.4f}°，按零件 bootstrap 95% 区间 {result['part_bootstrap_95pct_interval_deg']}°。负值表示误差减少。", '',
                '验证为原有 100 件、3200 个固定视图；包含 100 个原始摆法，另单独报告排除这些视图后的 3100 个随机旋转。原始 checkpoint 本身由这一验证集选择，所以这不是独立测试。完整训练拟合、随机朝向指标与复现校验见 metrics.json。', '',
                '均匀随机单位方向的理论平均误差是 90°；≤10°的理论概率约 0.760%，≤30°约 6.699%。固定输出 +Z 在 identity 视图上天然正确，不能把它的≤3°比例当成模型泛化能力。', '',
                '本机编辑权重：runs/pointnet_output_weight_edit_2026-11-09/edited_model.pt；脚本：tools/edit_direction_output_weights.py。']
    (report_dir / 'README.md').write_text('\n'.join(summary)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
