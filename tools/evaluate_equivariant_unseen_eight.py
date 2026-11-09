"""Forward-only evaluation of the eight-part equivariant model on held-out shapes."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from equivariant_axis import EquivariantAxis
from train_eight_equivariant import equivariance_audit, measured
from train_magics_pointnet import angle_metrics


ROOT = Path(__file__).resolve().parents[1]


def main():
    name = 'equivariant_eight_unseen_parts_2026-11-09'
    run, report = ROOT/'runs'/name, ROOT/'reports'/name
    if run.exists() or report.exists():
        raise FileExistsError('Preserve existing evaluation; choose another name')
    training = ROOT/'runs/equivariant_eight_resampled_2026-11-09'
    prior = ROOT/'runs/pointnet_eight_unseen_parts_2026-11-09'
    config = json.loads((training/'config.json').read_text(encoding='utf8'))
    previous = json.loads((prior/'config.json').read_text(encoding='utf8'))
    split = json.loads((ROOT/'runs/pointnet_shape_dedup_2026-11-09_val100/split.json').read_text(encoding='utf8'))
    records = json.loads((ROOT/'outputs/shape_deduplicated_pointclouds/manifest.json').read_text(encoding='utf8'))['records']
    normal_audit = json.loads((ROOT/'outputs/shape_deduplicated_normals/manifest.json').read_text(encoding='utf8'))
    selected, seed = previous['selected_indices'], previous['seed']
    rng = np.random.default_rng(seed)
    assert selected == sorted(rng.choice(split['validation_indices'],8,replace=False).tolist())
    assert not set(selected) & set(config['selected_indices'])
    assert set(selected) <= set(split['validation_indices'])
    assert not {records[i]['shape_group_id'] for i in selected} & {r['shape_group_id'] for r in config['selected_records']}
    variants = {label:[] for label in ('matched_prior_subset','new_subset_1','new_subset_2','new_subset_3')}
    for index in selected:
        record = records[index]
        with np.load(ROOT/'outputs/shape_deduplicated_pointclouds'/record['pointcloud_file']) as data:
            original = data['points'].copy()
        with np.load(ROOT/'outputs/shape_deduplicated_normals'/record['pointcloud_file']) as data:
            np.testing.assert_array_equal(data['points'],original)
            normals = data['normals'].astype(np.float32)
        np.testing.assert_allclose(np.linalg.norm(normals,axis=-1),1,atol=1e-5)
        points = original.astype(np.float32)
        points -= points.mean(0,keepdims=True)
        points /= np.linalg.norm(points,axis=-1).max()
        for label in variants:
            ids = rng.choice(len(points),2048,replace=False) if label == 'matched_prior_subset' else np.random.default_rng(seed+index*17+int(label[-1])).choice(len(points),2048,replace=False)
            variants[label].append(np.concatenate([points[ids],normals[ids]],axis=-1))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    with np.load(prior/'independent_preview_rotations.npz') as data:
        np.testing.assert_array_equal(data['part_indices'],np.repeat(selected,64))
        saved_rotations = data['augmentation_rotations'].copy()
    rotations = torch.from_numpy(saved_rotations.reshape(8,64,3,3)).to(device)
    checkpoint_path = training/'best_model.pt'
    checkpoint_blob = checkpoint_path.read_bytes()
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True)
    model = EquivariantAxis().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    results = []
    started = time.monotonic()
    for label,clouds in variants.items():
        bases = torch.from_numpy(np.stack(clouds)).to(device)
        metrics,predicted,target,restored = measured(model,bases,rotations)
        parts = []
        for local,index in enumerate(selected):
            bounds = slice(local*64,(local+1)*64)
            part_metrics = angle_metrics(predicted[bounds],target[bounds])
            part_metrics['cosine_loss'] = float((1-(predicted[bounds]*target[bounds]).sum(-1)).mean())
            parts.append({'part_index':index,'shape_group_id':records[index]['shape_group_id'],'metrics':part_metrics})
        results.append({'point_subset':label,'metrics':metrics,'per_part':parts})
        if label == 'matched_prior_subset':
            main_prediction,main_target,main_restored = predicted,target,restored
            audit = equivariance_audit(model,bases,seed+901)
    with np.load(prior/'independent_predictions.npz') as data:
        np.testing.assert_array_equal(main_target,data['target_axis'])
    elapsed = time.monotonic()-started
    selected_audit = [row for row in normal_audit['records'] if row['part_index'] in selected]
    evaluation_config = {
        'task':'Forward-only eight unseen shapes with eight-training-part equivariant model',
        'evaluation_scope':'unseen_parts','selected_indices':selected,
        'selected_records':[records[i] for i in selected], 'training_indices':config['selected_indices'],
        'selection':'Reuse prior seeded eight validation shapes; no selection based on current predictions',
        'seed':seed,'points':2048,'fresh_views_per_part':64,'input_channels':6,'use_normals':True,
        'checkpoint':str(checkpoint_path.relative_to(ROOT)),
        'checkpoint_epoch':checkpoint['epoch'],'checkpoint_sha256':hashlib.sha256(checkpoint_blob).hexdigest(),
        'training_performed':False,'checkpoint_selection_on_this_evaluation':False,
        'input':'XYZ+source triangle normals, 4096-point centering and max-radius normalization, fixed 2048 subset per variant',
        'paired_comparison':'Main subset and all 512 rotations match prior PointNet unseen-eight evaluation; models and training differ',
        'target':'P@R.T, N@R.T, target R[:,2]; signed, no symmetry correction, yaw ignored',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wandb':'disabled',
    }
    result = {
        'status':'complete','unseen_parts_512_rotations':results[0]['metrics'],
        'per_part':results[0]['per_part'],'subset_variants':results,'equivariance_audit':audit,
        'inference_and_audit_seconds':elapsed,'prior_targets_exactly_equal':True,
        'limitations':['Only eight held-out shapes; same original validation pool previously inspected, not a pristine final test.',
                      'Fixed original 4096-point caches; no independent fresh mesh resampling.',
                      'No project or family isolation, no symmetry-equivalent direction correction.',
                      'Source face signs on non-watertight meshes are not certified outward.',
                      'Equivariance makes same-subset rotations correlated; 512 views are not 512 independent shapes.'],
    }
    run.mkdir(parents=True)
    report.mkdir(parents=True)
    for folder in (run,report):
        (folder/'config.json').write_text(json.dumps(evaluation_config,ensure_ascii=False,indent=2),encoding='utf8')
        (folder/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (report/'normal_attachment_audit.json').write_text(json.dumps(selected_audit,ensure_ascii=False,indent=2),encoding='utf8')
    np.savez_compressed(run/'independent_predictions.npz',predicted_axis=main_prediction,target_axis=main_target,part_indices=np.repeat(selected,64))
    np.savez_compressed(run/'independent_preview_rotations.npz',augmentation_rotations=saved_rotations,part_indices=np.repeat(selected,64))
    np.savez_compressed(run/'restored_axes.npz',axes=main_restored)
    rows = '\n'.join(f'| {r["part_index"]} | {r["metrics"]["mean_angle_deg"]:.3f}° | {r["metrics"]["within_10deg_pct"]:.1f}% |' for r in result['per_part'])
    variants_table = '\n'.join(f'| {r["point_subset"]} | {r["metrics"]["mean_angle_deg"]:.3f}° | {r["metrics"]["cosine_loss"]:.6f} |' for r in results)
    metrics = results[0]['metrics']
    text = f'''# 八个未见零件：旋转等变网络前向评测（2026-11-09）

使用八件在线抽点训练的等变网络最佳第 {checkpoint['epoch']} 轮权重。没有重训，没有按本次结果选权重或零件；复用此前 PointNet 的八个留出验证件 {selected}，与训练八件及其形状组不相交。保留 932 件 PointNet 停止状态，未启动全量训练。

输入 2048 点 XYZ＋法向，归一化和此前训练一致。主评测的点子集和 512 个 SO(3) 与此前 PointNet 未见八件评测相同，标签已逐元素核对。模型、法向、初始化与训练预算不同，不是单变量消融。

平均误差 **{metrics['mean_angle_deg']:.3f}°**，中位 **{metrics['median_angle_deg']:.3f}°**，≤10° **{metrics['within_10deg_pct']:.2f}%**，≤30° **{metrics['within_30deg_pct']:.2f}%**，余弦 loss **{metrics['cosine_loss']:.6f}**。旋转集中度 {metrics['mean_signed_rotation_concentration']:.6f}；旋转一致不代表方向正确。

| 零件索引 | 64 朝向平均误差 | ≤10° |
|---|---:|---:|
{rows}

## 换点敏感性（不训练）

| 点子集 | 平均误差 | loss |
|---|---:|---:|
{variants_table}

另三组使用事先固定的抽点种子，未用于选模。全部来自原 4096 点缓存，不是从 mesh 独立重采样。源法向审计见 normal_attachment_audit.json，其中非水密模型的外向符号未认证；没有因其结果较差而删件。

前向结果、实际旋转和还原向量保存在同名 runs 目录。标准三联图在 `outputs/equivariant_eight_unseen_pose_comparison/index.html`，每件只取第一保存朝向，输入无平台，GT/预测落在 40% 透明平台上，保留初始方向箭头。GT 不用于网络输入，yaw 对齐只用于显示。

本次是八个未见形状的有限评测，原验证池此前用于其他实验和分析，不能当作从未接触过的最终测试集。同点集的旋转视图受等变性约束，不能将 512 视图当成 512 个独立零件。没有对称等价修正，不直接解释为打印可用率。
'''
    (report/'README.md').write_text(text,encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
