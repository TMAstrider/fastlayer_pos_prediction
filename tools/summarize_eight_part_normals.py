"""Compare paired XYZ and XYZ+normal eight-shape direction experiments."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    baseline = root / 'runs/pointnet_random_so3_8parts_extended_2026-11-09'
    normal = root / 'runs/pointnet_random_so3_8parts_normals_2026-11-09'
    report = root / 'reports' / normal.name
    configs = [json.loads((folder / 'config.json').read_text(encoding='utf-8-sig')) for folder in (baseline, normal)]
    for key in ('selected_indices', 'seed', 'max_epochs', 'initial_learning_rate', 'minimum_learning_rate',
                'training_generator_seed', 'final_test_seed', 'initial_checkpoint'):
        assert configs[0][key] == configs[1][key], key
    metrics = [json.loads((folder / 'metrics.json').read_text(encoding='utf8')) for folder in (baseline, normal)]
    saved = []
    for folder in (baseline, normal):
        with np.load(folder / 'independent_predictions.npz') as data:
            saved.append({key: data[key].copy() for key in data.files})
    np.testing.assert_array_equal(saved[0]['part_indices'], saved[1]['part_indices'])
    np.testing.assert_array_equal(saved[0]['target_axis'], saved[1]['target_axis'])
    epochs = []
    for item in metrics:
        epochs.append([row['epoch'] for row in item['history']])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for number, (label, item) in enumerate(zip(('XYZ', 'XYZ + normals'), metrics)):
        axes[0].plot(epochs[number], [row['fixed_probe']['cosine_loss'] for row in item['history']], label=label)
        axes[1].plot(epochs[number], [row['fixed_probe']['mean_angle_deg'] for row in item['history']], label=label)
    for axis in axes:
        axis.set_xlabel('Additional epochs (4 updates each)')
        axis.legend()
    axes[0].set_ylabel('Same-shape fixed-probe cosine loss')
    axes[1].set_ylabel('Same-shape fixed-probe mean angle (deg)')
    figure.savefig(report / 'normals_comparison.png', dpi=160)
    plt.close(figure)
    rows = []
    for index in configs[0]['selected_indices']:
        values = []
        for data in saved:
            mask = data['part_indices'] == index
            predicted = data['predicted_axis'][mask]
            target = data['target_axis'][mask]
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
            target /= np.linalg.norm(target, axis=1, keepdims=True)
            dots = np.clip((predicted*target).sum(-1), -1, 1)
            values.append({'mean_angle_deg': float(np.degrees(np.arccos(dots)).mean()),
                           'cosine_loss': float((1-dots).mean()),
                           'unsigned_axis_mean_angle_deg': float(np.degrees(np.arccos(np.abs(dots))).mean())})
        rows.append({'part_index': index, 'xyz': values[0], 'xyz_normals': values[1],
                     'angle_delta_deg_normals_minus_xyz': values[1]['mean_angle_deg']-values[0]['mean_angle_deg']})
    result = {'paired_targets_identical': True,
              'baseline': metrics[0]['independent_same_shape_512_rotations'],
              'xyz_normals': metrics[1]['independent_same_shape_512_rotations'],
              'per_part': rows,
              'minimum_online_train_loss': min(row['online_train_loss'] for row in metrics[1]['train_history']),
              'best_additional_epoch': metrics[1]['best_epoch'],
              'scope': 'Same eight training shapes, same 512 independent rotation targets; not unseen-shape accuracy.'}
    (report / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    normal_manifest = json.loads((root / 'outputs/eight_part_normals/manifest.json').read_text(encoding='utf8'))
    (report / 'normal_attachment_audit.json').write_text(json.dumps(normal_manifest, indent=2), encoding='utf8')
    table = '\n'.join(f'| {row["part_index"]} | {row["xyz"]["mean_angle_deg"]:.3f}° | '
                      f'{row["xyz_normals"]["mean_angle_deg"]:.3f}° | '
                      f'{row["angle_delta_deg_normals_minus_xyz"]:+.3f}° |' for row in rows)
    baseline_metrics = result['baseline']
    normal_metrics = result['xyz_normals']
    other_baseline = np.mean([row['xyz']['mean_angle_deg'] for row in rows if row['part_index'] != 69])
    other_normals = np.mean([row['xyz_normals']['mean_angle_deg'] for row in rows if row['part_index'] != 69])
    text = f'''\n\n## 与 XYZ 的配对法向对照

开始前的阶段提交为 `047544b`。本轮仍为相同 8 个零件，不加入新的训练形状，不改变 69 号的有向目标。XYZ 2048 点、训练/探针/最终旋转、B 起始 checkpoint、4000 轮预算、学习率和选模规则均与延长 B 一致。最终评测逐样本 part ID 和 target 数组已逐元素核对完全一致。

仅首层卷积由 3 通道扩展为 6 通道，原 XYZ 权重拷贝，新增法向权重置零，其他参数保持一致；程序已检查启用法向前后初始预测一致。训练期间所有参数可学习。此对照检验增加法向后微调的效果，不是 6 通道从头训练。

法向来自原 STL 最近三角面，不改变原 XYZ。8 个源 mesh 均闭合、绕序一致、有正整体体积；最大点到 mesh 表面残差见 normal_attachment_audit.json。边上的法向可能在邻接三角面间有歧义。位置和法向同乘 R.T，法向不参与平移和半径缩放。与几何对称等价监督是两回事，本轮没有使用 abs(dot)。

| 零件索引 | XYZ 平均误差 | XYZ＋法向平均误差 | 差值（负数改善） |
| --- | ---: | ---: | ---: |
{table}

总体独立朝向平均误差 {baseline_metrics['mean_angle_deg']:.3f}° → {normal_metrics['mean_angle_deg']:.3f}°，中位 {baseline_metrics['median_angle_deg']:.3f}° → {normal_metrics['median_angle_deg']:.3f}°，P90 {baseline_metrics['p90_angle_deg']:.3f}° → {normal_metrics['p90_angle_deg']:.3f}°；cosine loss {baseline_metrics['cosine_loss']:.6f} → {normal_metrics['cosine_loss']:.6f}。≤10° {baseline_metrics['within_10deg_pct']:.3f}% → {normal_metrics['within_10deg_pct']:.3f}%。

改善主要集中在 69 号；其余 7 件的平均误差仅 {other_baseline:.3f}° → {other_normals:.3f}°，不能称为全体零件普遍改善。69 号无符号轴误差仍约 46.985°，并未可靠学会环面法线。当前固定点采样、三角面法向与单种子结果也不能证明模型学到了可迁移工艺规则。

最低逐轮在线训练 loss：{result['minimum_online_train_loss']:.6f}。最佳模型为新增 epoch {result['best_additional_epoch']}。总体指标见 comparison.json；所有 8 件都计入总体。单随机种子、小形状集，不能推断新零件泛化或断言法向普遍有效。

![XYZ 与法向对照曲线](normals_comparison.png)

三联图：`outputs/eight_part_normals_pose_comparison/index.html`，每件取相同最终独立旋转集合的第一项，与 XYZ 三联图一一对应。标题误差是单次角误差，表格为 64 朝向平均。无 W&B，本轮记录保存在本地，且属于阶段提交之后的工作。
'''
    with (report / 'README.md').open('a', encoding='utf8') as stream:
        stream.write(text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
