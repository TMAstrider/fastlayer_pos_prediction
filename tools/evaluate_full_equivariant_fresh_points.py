"""Compare cached-pool and fresh-mesh points using the last stopped model, without training."""

import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from equivariant_axis import EquivariantAxis
from train_eight_equivariant import equivariance_audit
from train_full_online_so3 import assess, load_bases, write_json


ROOT = Path(__file__).resolve().parents[1]
NAME = 'equivariant_full_fresh_points_2026-11-09'


def main():
    training = ROOT/'runs/equivariant_full_resampled_2026-11-09'
    run, report = ROOT/'runs'/NAME, ROOT/'reports'/NAME
    if run.exists() or report.exists():
        raise FileExistsError('Preserve earlier evaluation')
    progress = json.loads((training/'progress.json').read_text(encoding='utf8'))
    assert progress['status']=='stopped_by_user'
    config = json.loads((training/'config.json').read_text(encoding='utf8'))
    split = json.loads((training/'split.json').read_text(encoding='utf8'))
    fresh_dir = ROOT/'outputs/equivariant_full_fresh_mesh_samples_2026-11-09'
    fresh_audit = json.loads((fresh_dir/'manifest.json').read_text(encoding='utf8'))
    assert fresh_audit['status']=='complete' and not fresh_audit['failures']
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    bases, manifest, hashes = load_bases(True,config['seed'],device)
    assert hashes==config['fixed_probe_subset_hashes']
    assert len(fresh_audit['records'])==len(bases)==1032
    for scope,ids in [('train',split['train_indices']),('validation',split['validation_indices'])]:
        assert [manifest['records'][index]['geometry_sha256'] for index in ids]==split[f'{scope}_geometry_sha256']
    model = EquivariantAxis().to(device)
    checkpoint_path = training/'last_model.pt'
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True)
    assert checkpoint['epoch']==progress['epochs_completed']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    run.mkdir(parents=True)
    report.mkdir(parents=True)
    settings = {'task':'Frozen current-weight independent mesh resampling diagnosis','training_performed':False,
                'checkpoint':str(checkpoint_path),'checkpoint_epoch':checkpoint['epoch'],
                'checkpoint_sha256':checkpoint_hash,'checkpoint_selection':'Last complete epoch, not best validation',
                'training_parts':932,'validation_parts':100,'point_count':2048,'input':'XYZ+normals',
                'variants':['original_fixed_subset','old_pool_resample_1','old_pool_resample_2','old_pool_resample_3',
                            'fresh_mesh_1','fresh_mesh_2','fresh_mesh_3'],
                'normalization':'Original 4096-point centroid/max radius unchanged for every variant',
                'rotations':'Exactly the fixed training probe 4 SO(3)/part and validation 32 SO(3)/part from the stopped run',
                'normals':'Old cache for old pool; exact triangle normals with same orientation policy for fresh mesh',
                'sampling_seeds':'See copied fresh_mesh_sampling_audit.json; old pool seed=train_seed+610001+1009*part_index',
                'target':'R[:,2], signed cosine loss, yaw ignored, no symmetry adjustment',
                'source_sha256':{name:hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest() for name in
                                 ['evaluate_full_equivariant_fresh_points.py','prepare_fresh_mesh_samples.py','extract_pointclouds.py',
                                  'equivariant_axis.py','train_full_online_so3.py']},'wandb':'disabled'}
    for folder in (run,report):
        write_json(folder/'config.json',settings)
    write_json(report/'split.json',split)
    write_json(report/'fresh_mesh_sampling_audit.json',fresh_audit)
    source = report/'source'
    source.mkdir()
    for name in settings['source_sha256']:
        (source/name).write_bytes((ROOT/'tools'/name).read_bytes())
    scopes = [('train',split['train_indices'],4,config['seed']+73001),
              ('validation',split['validation_indices'],32,config['seed']+83001)]
    results, per_part_rows = [], []
    original_arrays = {}
    started = time.monotonic()
    for label in settings['variants']:
        if label=='original_fixed_subset':
            clouds = bases
        else:
            features = []
            for index,record in enumerate(manifest['records']):
                if label.startswith('old_pool_'):
                    variant = int(label[-1])-1
                    with np.load(ROOT/'outputs/shape_deduplicated_normals'/record['pointcloud_file']) as data:
                        points, normals = data['points'].astype(np.float32), data['normals'].astype(np.float32)
                    points -= points.mean(0,keepdims=True)
                    points /= np.linalg.norm(points,axis=-1).max()
                    rng = np.random.default_rng(config['seed']+610001+1009*index)
                    subsets = [rng.choice(len(points),2048,replace=False) for _ in range(3)]
                    selected = subsets[variant]
                    features.append(np.concatenate([points[selected],normals[selected]],axis=-1))
                else:
                    with np.load(fresh_dir/record['pointcloud_file']) as data:
                        cloud = data['features']
                    assert hashlib.sha256(cloud.tobytes()).hexdigest()==fresh_audit['records'][index]['features_sha256']
                    features.append(cloud[int(label[-1])-1])
            clouds = torch.from_numpy(np.stack(features)).to(device)
        current = {'variant':label}
        for scope,ids,views,seed in scopes:
            metrics,parts,arrays = assess(model,clouds,ids,views,seed,32,True)
            if label=='original_fixed_subset':
                original_arrays[scope] = arrays
                saved = progress['history'][-1]['same_shape_probe' if scope=='train' else 'validation']
                np.testing.assert_allclose(metrics['mean_angle_deg'],saved['mean_angle_deg'],atol=.002)
            original = original_arrays[scope]
            np.testing.assert_array_equal(arrays['target_axis'],original['target_axis'])
            np.testing.assert_array_equal(arrays['rotations'],original['rotations'])
            cosine = np.clip((arrays['predicted_axis']*original['predicted_axis']).sum(-1),-1,1)
            metrics['axis_change_from_original_mean_deg'] = float(np.degrees(np.arccos(cosine)).mean()) if label!='original_fixed_subset' else 0.
            current[scope] = {'metrics':metrics,'per_part':parts}
            np.savez_compressed(run/f'{label}_{scope}_predictions.npz',**arrays)
            for row in parts:
                per_part_rows.append({'variant':label,'scope':scope,'part_index':row['part_index'],**row['metrics']})
            print(f'{label}: {scope}, error {metrics["mean_angle_deg"]:.3f} deg, loss {metrics["cosine_loss"]:.6f}, prediction change {metrics["axis_change_from_original_mean_deg"]:.3f} deg',flush=True)
        if label=='fresh_mesh_1':
            fresh_equivariance = equivariance_audit(model,clouds[split['train_indices'][:8]],config['seed']+710001)
        results.append(current)
        if label!='original_fixed_subset':
            del clouds
        write_json(report/'partial_results.json',results)
    paired = {}
    for scope,ids,_,_ in scopes:
        old = np.array([[row['metrics']['mean_angle_deg'] for row in result[scope]['per_part']] for result in results[1:4]])
        fresh = np.array([[row['metrics']['mean_angle_deg'] for row in result[scope]['per_part']] for result in results[4:7]])
        delta = fresh.mean(0)-old.mean(0)
        paired[scope] = {'old_pool_three_sample_mean_deg':float(old.mean()),'fresh_mesh_three_sample_mean_deg':float(fresh.mean()),
                         'fresh_minus_old_pool_mean_deg':float(delta.mean()),'fresh_minus_old_pool_median_deg':float(np.median(delta)),
                         'parts_worsened_by_over_10deg':int((delta>10).sum()),'parts_improved_by_over_10deg':int((delta< -10).sum()),
                         'per_part':[{'part_index':index,'old_pool_mean_deg':float(old[:,local].mean()),
                                      'fresh_mesh_mean_deg':float(fresh[:,local].mean()),'delta_deg':float(delta[local])}
                                     for local,index in enumerate(ids)]}
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==checkpoint_hash
    result = {'status':'complete','checkpoint_epoch':checkpoint['epoch'],'training_performed':False,
              'checkpoint_unchanged':True,'seconds':time.monotonic()-started,'variants':results,'paired_comparison':paired,
              'fresh_equivariance_audit':fresh_equivariance,
              'limitations':['Only frozen current weights; no retraining on fresh samples.',
                             'Same historical labels and split; validation has been inspected and is not an independent final test.',
                             'Sampling comparisons change point locations and corresponding triangle normals together.',
                             'Near symmetries can magnify directional changes; signed error is not process acceptability.']}
    for folder in (run,report):
        write_json(folder/'metrics.json',result)
    with (report/'per_part.csv').open('w',newline='',encoding='utf8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(per_part_rows[0]))
        writer.writeheader()
        writer.writerows(per_part_rows)
    figure,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for axis,(scope,_,_,_) in zip(axes,scopes):
        rows = paired[scope]['per_part']
        axis.scatter([row['old_pool_mean_deg'] for row in rows],[row['fresh_mesh_mean_deg'] for row in rows],s=9,alpha=.5)
        axis.plot([0,180],[0,180],color='gray',linestyle='--')
        axis.set(xlabel='Old pool: three subsets mean error (deg)',ylabel='Fresh mesh: three samples mean error (deg)',title=scope,xlim=(0,180),ylim=(0,180))
        axis.grid(alpha=.2)
    figure.savefig(report/'point_sampling_comparison.png',dpi=150)
    plt.close(figure)
    table = '\n'.join(f'| {row["variant"]} | {row["train"]["metrics"]["mean_angle_deg"]:.3f}° | {row["train"]["metrics"]["cosine_loss"]:.6f} | {row["validation"]["metrics"]["mean_angle_deg"]:.3f}° |' for row in results)
    train, validation = paired['train'],paired['validation']
    text = f'''# 全量等变模型：从 mesh 独立重新采点（2026-11-09）

训练已按用户要求停止在完整第 {checkpoint['epoch']} 轮。固定 last_model.pt 当前权重，未使用验证最佳或训练最佳权重，没有重训或改权重。checkpoint SHA256 为 {checkpoint_hash}，评测前后完全一致。

全部932训练件、100验证件；每次2048点XYZ＋法向。三组旧4096点池子集与三组独立mesh采样均事先固定种子，没有按结果挑选。mesh按三角形面积选面、面内均匀采点；法向直接取所选三角形，保持此前封闭负体积网格翻转规则。源STL与旧法向审计的文件哈希逐件核对。所有组保持原4096点池的中心与半径，避免把归一化变化误算成采样变化。

沿用停止训练时的同一训练probe每件4个SO(3)、验证每件32个SO(3)，标签和旋转逐元素核对一致。原评估点误差与停训记录核对通过。等变网络的同点集旋转视图高度相关，并非新增独立形状。

| 点来源 | 932训练件平均误差 | 训练loss | 100验证件平均误差 |
|---|---:|---:|---:|
{table}

三次采样平均：训练件旧池 {train['old_pool_three_sample_mean_deg']:.3f}°，新mesh {train['fresh_mesh_three_sample_mean_deg']:.3f}°，差 {train['fresh_minus_old_pool_mean_deg']:+.3f}°；验证旧池 {validation['old_pool_three_sample_mean_deg']:.3f}°，新mesh {validation['fresh_mesh_three_sample_mean_deg']:.3f}°，差 {validation['fresh_minus_old_pool_mean_deg']:+.3f}°。

训练件按三次采样逐件平均，有 {train['parts_worsened_by_over_10deg']} 件变差超过10°，{train['parts_improved_by_over_10deg']} 件改善超过10°。逐件指标和预测方向变化见 metrics.json、per_part.csv，散点图 point_sampling_comparison.png；完整预测及实际旋转见同名 runs 目录。采样审计、源文件哈希、配置和源码快照均已记录。

本实验只诊断当前权重对采样的敏感性；没有验证“每次重新从mesh采样训练”能否改善新件泛化。非水密网格源法向的外向性仍未认证。位置和对应法向共同变化；对称形状也可能放大方向变化。仍使用有向历史GT误差，不等同于打印可用性。
'''
    (report/'README.md').write_text(text,encoding='utf8')
    print(json.dumps({'status':'complete','checkpoint_epoch':checkpoint['epoch'],'seconds':result['seconds'],
                      'paired_comparison':{scope:{key:value for key,value in row.items() if key!='per_part'} for scope,row in paired.items()}},indent=2),flush=True)


if __name__=='__main__':
    main()
