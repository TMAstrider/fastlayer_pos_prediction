"""Archive the completed Magics PointNet run; optionally wait for its metrics."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    run = root / 'runs/magics_pointnet_scratch_2026-11-09'
    deadline = time.monotonic() + 3*3600
    while not (run / 'metrics.json').exists():
        if not args.wait:
            raise FileNotFoundError(run / 'metrics.json')
        if time.monotonic() > deadline:
            raise TimeoutError('No completed training metrics after three hours')
        time.sleep(30)
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf8'))
    split = json.loads((run / 'split.json').read_text(encoding='utf8'))
    assert not set(split['train_indices']) & set(split['validation_indices'])
    output = root / 'reports/magics_pointnet_scratch'
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('metrics.json', 'config.json', 'split.json'):
        shutil.copy2(run / filename, output / filename)
    history = metrics['history']
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(epochs, [r['train_cosine_loss'] for r in history])
    axes[0].set(xlabel='Epoch', ylabel='Train cosine loss')
    axes[1].plot(epochs, [r['validation']['mean_angle_deg'] for r in history], label='Validation mean')
    axes[1].axvline(metrics['best_epoch'], color='gray', linestyle=':', label='Selected epoch')
    axes[1].set(xlabel='Epoch', ylabel='Angle (degrees)')
    axes[1].legend()
    fig.savefig(output / 'training_curve.png', dpi=160)
    plt.close(fig)
    records = json.loads((root / 'outputs/magics_pointclouds/manifest.json').read_text(encoding='utf8'))['records']
    with np.load(run / 'best_validation_predictions.npz') as predictions:
        pred, target = predictions['predicted_axis'], predictions['target_axis']
        pred = pred / np.linalg.norm(pred, axis=1, keepdims=True).clip(1e-8)
        target = target / np.linalg.norm(target, axis=1, keepdims=True).clip(1e-8)
        angles = np.degrees(np.arccos(np.sum(pred*target, axis=1).clip(-1, 1)))
        part_ids = predictions['part_indices']
    with (output / 'per_part_validation.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['part_index', 'geometry_sha256', 'source_project', 'source_stl_file', 'views', 'mean_angle_deg', 'median_angle_deg', 'max_angle_deg'])
        for part_id in np.unique(part_ids):
            record = records[int(part_id)]
            values = angles[part_ids == part_id]
            writer.writerow([int(part_id), record['geometry_sha256'], record['source_project'], record['source_stl_file'], len(values), float(values.mean()), float(np.median(values)), float(values.max())])
    best = metrics['best_validation']
    report = f'''# Magics PointNet 从头训练（2026-11-09）

## 配置与数据

沿用此前真实件 PointNet 最佳配置：XYZ 2048 点、batch32、每件每轮 16 个新 SO(3) 朝向、AdamW lr=0.001 / weight_decay=0.0001、dropout=0.3、余弦方向损失。PointNet 共享点 MLP + 全局最大池化 + 单位三维方向头，从头训练，无预训练、无法向输入。

本轮仅使用新增 Magics：5,534 条候选记录、1,644 个坐标和几何字节唯一网格；训练 {metrics['training_parts']} / 验证 {metrics['validation_parts']}，共 {metrics['validation_views']} 个固定验证朝向。**没有混入旧 193 件或另外新增的 PJF 件。** 未独立核实每个对象的活动状态和零件语义，包含初步位置检查标为待核对的记录。

网格本身就是记录摆法。点坐标按行向量变换 `P @ R.T`，目标同步变为 `R @ [0,0,1]`；预测 up-axis，忽略 yaw。损失为 `mean(1-dot(pred_unit,target_unit))`。

## 完成结果

- 最佳 epoch：{metrics['best_epoch']}；共完成 {metrics['epochs_completed']} epochs。
- 最佳验证平均方向误差：**{best['mean_angle_deg']:.4f}°**；中位数 {best['median_angle_deg']:.4f}°；P90 {best['p90_angle_deg']:.4f}°。
- ≤3°：{best['within_3deg_pct']:.4f}%；≤5°：{best['within_5deg_pct']:.4f}%；≤10°：{best['within_10deg_pct']:.4f}%；≤30°：{best['within_30deg_pct']:.4f}%。
- 固定输出 +Z 的基线均值：{metrics['baseline_constant_up_axis']['mean_angle_deg']:.4f}°。
- 第一轮训练 loss：{history[0]['train_cosine_loss']:.6f}；最佳验证轮训练 loss：{history[metrics['best_epoch']-1]['train_cosine_loss']:.6f}；末轮：{history[-1]['train_cosine_loss']:.6f}。
- 训练用时约 {metrics['training_seconds']/60:.1f} 分钟；参数量 {metrics['parameter_count']:,}。

旧真实件实验 PointNet 最佳为 71.29°，冻结 PTv3 消融最佳为 73.66°；**新旧验证集不同，不能把本轮数值差直接解释成扩大数据带来的提升。** 公开简单金字塔的 PointNet 0.71° / PTv3 scratch 2.00° 不代表工业件性能。

## 指标限制与同形多位姿

本轮按字节唯一网格随机划分，没有按工程、同形零件或零件族隔离。不同平移、旋转、重新索引的同一形状可能跨集合。因此这不是独立零件族泛化测试，也不是可打印率。

训练期间扩大同形检查后发现同一形状可能有多个记录方向，包括明确约 60.02° 的平放/倾斜案例。当前仍按每条记录的单 GT 计算误差，未更改正在运行的实验。角误差可能处罚其他可接受方向；相关记录的生产用途仍需结合工程对象状态确认。详见 [多位姿审计](../magics_pose_conflicts_2026-11-09.md)。

## 保存位置

- 完整逐轮指标、配置、划分快照：本目录 `metrics.json`、`config.json`、`split.json`。
- 训练曲线：[training_curve.png](training_curve.png)。
- 逐验证零件汇总：`per_part_validation.csv`。
- 本机最佳权重：`runs/magics_pointnet_scratch_2026-11-09/best_model.pt`。
- 本机逐视图预测：同目录 `best_validation_predictions.npz`。
- 训练脚本：`tools/train_magics_pointnet.py`；数据预处理：`tools/export_magics_pointclouds.py`。

快照由 `tools/summarize_magics_training.py` 在完整训练指标生成后归档。报告生成不涉及训练配置变更或自动 Git 提交。
'''
    (output / 'README.md').write_text(report, encoding='utf8')
    print(json.dumps({'report': str(output / 'README.md'), 'best_epoch': metrics['best_epoch'], 'best_validation': best}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
