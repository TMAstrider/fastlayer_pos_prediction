"""Archive the deduplicated PointNet experiment once its final metrics exist."""

import argparse
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--run', type=Path, default=root / 'runs/pointnet_shape_dedup_2026-11-09_val100')
    parser.add_argument('--output', type=Path, default=root / 'reports/pointnet_shape_dedup_2026-11-09_val100')
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 3*3600
    while not (run / 'metrics.json').exists():
        if not args.wait:
            raise FileNotFoundError(run / 'metrics.json')
        if time.monotonic() > deadline:
            raise TimeoutError('No completed training metrics within three hours')
        time.sleep(30)
    m = json.loads((run / 'metrics.json').read_text(encoding='utf8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf8'))
    split = json.loads((run / 'split.json').read_text(encoding='utf8'))
    assert config['deduplicated_by_shape']
    assert not set(split['train_shape_group_ids']) & set(split['validation_shape_group_ids'])
    for name in ('config.json', 'split.json', 'metrics.json'):
        shutil.copy2(run / name, output / name)
    history = m['history']
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(epochs, [r['train_cosine_loss'] for r in history])
    axes[0].set(xlabel='Epoch', ylabel='Train cosine loss')
    axes[1].plot(epochs, [r['validation']['mean_angle_deg'] for r in history], label='Validation mean')
    axes[1].axvline(m['best_epoch'], color='gray', linestyle=':', label='Selected epoch')
    axes[1].set(xlabel='Epoch', ylabel='Angle (degrees)')
    axes[1].legend()
    fig.savefig(output / 'training_curve.png', dpi=160)
    plt.close(fig)
    best = m['best_validation']
    report = f'''# 形状去重后 PointNet 基线（2026-11-09）

训练完成，共 {m['epochs_completed']} epochs，最佳 epoch {m['best_epoch']}，用时 {m['training_seconds']/60:.1f} 分钟。

| 最佳验证指标 | 数值 |
| --- | ---: |
| 平均方向误差 | {best['mean_angle_deg']:.4f}° |
| 中位方向误差 | {best['median_angle_deg']:.4f}° |
| P90 | {best['p90_angle_deg']:.4f}° |
| ≤3° | {best['within_3deg_pct']:.4f}% |
| ≤10° | {best['within_10deg_pct']:.4f}% |
| ≤30° | {best['within_30deg_pct']:.4f}% |

训练 {m['training_parts']} 个代表零件，验证 {m['validation_parts']} 个代表零件、{m['validation_views']} 个固定朝向。训练、验证的 shape_group_id 无交集。

网络为从头训练的 PointNet，XYZ 2048 点，batch32，每件每轮 16 个新 SO(3) 朝向，单位三维方向输出；目标随点云旋转，采用余弦方向损失。每个去重组只用一个历史参考位姿。

数据构建及保守筛选规则见 [去重记录](../shape_deduplication_2026-11-09/README.md)。这不是 CAD 身份认证或独立零件族测试，未处理对称等价方向。新旧划分不同，不能把均值差直接归因于去重；角误差也不是打印失败率。

完整历史、配置、划分快照位于本目录。曲线见 [training_curve.png](training_curve.png)；权重位于 `{run.relative_to(root).as_posix()}/best_model.pt`。数据 GT 和其他来源摆法的映射均保留，原始网格未删除。
'''
    (output / 'README.md').write_text(report, encoding='utf8')
    (output / 'STATUS.md').write_text('训练已完成，完整指标及曲线见 [README.md](README.md)。\n', encoding='utf8')
    print(json.dumps({'status': 'archived', 'report': str(output / 'README.md'), 'best_epoch': m['best_epoch'], 'best_validation': best}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
