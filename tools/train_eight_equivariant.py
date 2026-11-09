"""Train an equivariant vector direction network on the original eight parts."""

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

from diagnose_pointnet_random_so3 import rotate_features
from equivariant_axis import EquivariantAxis
from train_magics_pointnet import angle_metrics, random_rotations


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


@torch.no_grad()
def measured(model, clouds, rotations, batch_size=32):
    model.eval()
    count = rotations.shape[1]
    inputs = clouds[:, None].expand(-1, count, -1, -1).reshape(-1, *clouds.shape[1:])
    matrices = rotations.reshape(-1, 3, 3)
    predictions = []
    for start in range(0, len(inputs), batch_size):
        predictions.append(model(rotate_features(inputs[start:start+batch_size], matrices[start:start+batch_size])))
    prediction = torch.cat(predictions)
    targets = matrices[:, :, 2]
    metrics = angle_metrics(prediction.cpu().numpy(), targets.cpu().numpy())
    metrics['cosine_loss'] = float((1-(prediction*targets).sum(-1)).mean())
    restored = torch.bmm(prediction[:, None], matrices).squeeze(1).reshape(len(clouds), count, 3)
    metrics['mean_signed_rotation_concentration'] = float(restored.mean(1).norm(dim=-1).mean())
    return metrics, prediction.cpu().numpy(), targets.cpu().numpy(), restored.cpu().numpy()


@torch.no_grad()
def equivariance_audit(model, clouds, seed):
    model.eval()
    device = clouds.device
    generator = torch.Generator(device=device).manual_seed(seed)
    rotations = random_rotations(len(clouds), generator, device)
    reference = model(clouds)
    prediction = model(rotate_features(clouds, rotations))
    expected = torch.bmm(reference[:, None], rotations.transpose(1, 2)).squeeze(1)
    residuals = (prediction-expected).norm(dim=-1)
    batch_reference = torch.cat([model(cloud[None]) for cloud in clouds])
    batch_residual = float((reference-batch_reference).abs().max())
    permutation = torch.randperm(clouds.shape[1], generator=generator, device=device)
    permutation_residual = float((model(clouds[:, permutation])-reference).abs().max())
    maximum = float(residuals.max())
    assert maximum < .003, f'Equivariance residual too large: {maximum}'
    assert batch_residual < .003 and permutation_residual < .003
    return {'rotation_vector_l2_max': maximum, 'rotation_vector_l2_mean': float(residuals.mean()),
            'batch_independence_max_component_error': batch_residual,
            'point_permutation_max_component_error': permutation_residual,
            'passed': True, 'seed': seed, 'parts_checked': len(clouds),
            'note': 'FP32 tolerance; structural equivariance is mathematical, observed numerical residual is finite precision.'}


def plot_curve(report, train_history, history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot([row['epoch'] for row in train_history], [row['online_train_loss'] for row in train_history], alpha=.5, label='Online train')
    axes[0].plot([row['epoch'] for row in history], [row['probe']['cosine_loss'] for row in history], label='Same shapes, new rotations')
    axes[1].plot([row['epoch'] for row in history], [row['probe']['mean_angle_deg'] for row in history], label='All eight')
    axes[1].plot([row['epoch'] for row in history], [row['probe_without_69_mean_deg'] for row in history], label='Other seven (diagnostic only)')
    axes[0].set(xlabel='Epoch (4 updates)', ylabel='Signed cosine loss')
    axes[1].set(xlabel='Epoch (4 updates)', ylabel='Mean angle (degrees)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(report / 'training_curve.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='equivariant_eight_parts_2026-11-09')
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--resample-points', action='store_true')
    args = parser.parse_args()
    if Path(args.name).name != args.name or min(args.epochs, args.eval_every) < 1:
        parser.error('Require positive counts and single-component name')
    run, report = ROOT / 'runs' / args.name, ROOT / 'reports' / args.name
    if run.exists():
        parser.error('Existing run, choose a new name')
    prior = json.loads((ROOT / 'reports/pointnet_small_fit_2026-11-09/config.json').read_text(encoding='utf8'))
    split = json.loads((ROOT / 'runs/pointnet_shape_dedup_2026-11-09_val100/split.json').read_text(encoding='utf8'))
    manifest = json.loads((ROOT / 'outputs/shape_deduplicated_pointclouds/manifest.json').read_text(encoding='utf8'))
    selected, seed = prior['selected_indices'], prior['seed']
    assert not set(selected) & set(split['validation_indices'])
    device = torch.device('cuda')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    assert selected == sorted(rng.choice(split['train_indices'], 8, replace=False).tolist())
    clouds, full_clouds, subset_hashes = [], [], []
    reference_generator = torch.Generator(device=device).manual_seed(seed+7)
    probe_rotations = []
    for index in selected:
        record = manifest['records'][index]
        with np.load(ROOT / 'outputs/shape_deduplicated_pointclouds' / record['pointcloud_file']) as data:
            original = data['points'].copy()
        points = original.astype(np.float32)
        points -= points.mean(0, keepdims=True)
        points /= np.linalg.norm(points, axis=1).max()
        ids = rng.choice(len(points), 2048, replace=False)
        with np.load(ROOT / 'outputs/eight_part_normals' / record['pointcloud_file']) as data:
            np.testing.assert_array_equal(data['points'], original)
            full_normals = data['normals'].astype(np.float32)
            normals = full_normals[ids]
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-5)
        clouds.append(np.concatenate([points[ids], normals], axis=-1))
        full_clouds.append(np.concatenate([points, full_normals], axis=-1))
        subset_hashes.append(hashlib.sha256(ids.tobytes()).hexdigest())
        random_rotations(16, reference_generator, device)
        probe_rotations.append(random_rotations(64, reference_generator, device))
    clouds = torch.from_numpy(np.stack(clouds)).to(device)
    full_clouds = torch.from_numpy(np.stack(full_clouds)).to(device)
    probe_rotations = torch.stack(probe_rotations)
    model = EquivariantAxis().to(device)
    run.mkdir(parents=True)
    report.mkdir(parents=True, exist_ok=True)
    initial_audit = equivariance_audit(model, clouds, seed+91)
    config = {
        **vars(args), 'model': 'Independent vector-neuron-style SO(3)-equivariant network; not paper reproduction',
        'architecture': '3 vector inputs (XYZ, normal, XYZ cross normal); vector blocks 3-16-24-32; invariant norm-attention pooling at each scale; 72-32 vector block; invariant scalar MLP weights -> unit vector',
        'initialization': 'From scratch, all layers trainable, no pretrained checkpoint',
        'selected_indices': selected, 'selected_records': [manifest['records'][i] for i in selected],
        'seed': seed, 'points': 2048, 'input_channels': 6, 'use_normals': True,
        'point_processing': 'Normalize original 4096 points; fresh 2048 subset per appearance during training; fixed original subset for probe/test' if args.resample_points else 'Exactly same fixed subsets and normalization as original eight-part PointNet diagnostics',
        'subset_index_hashes': subset_hashes, 'training_rotations_per_part_per_epoch': 16,
        'batch_size': 32, 'updates_per_epoch': 4, 'weight_decay': 0, 'dropout': 0,
        'optimizer': 'AdamW', 'learning_rate_schedule': f'Cosine {args.learning_rate} to 1e-5 over {args.epochs} epochs',
        'gradient_clip': 1, 'loss': 'mean(1-dot(unit_pred,R[:,2])); signed, no symmetry correction',
        'training_generator_seed': seed+19703, 'final_test_seed': seed+183079,
        'probe': 'Same original eight-part 512 rotation views, checkpoint selection only',
        'final_test': 'Same independent 512 rotations as prior extended XYZ and XYZ+normals',
        'stop': 'Max epochs or signed probe error below 3 degrees for three checks',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'comparison_limits': ['Different initialization and architecture from previous normals continuation; not a single-variable causal ablation.',
                              'Same eight shapes only; no new-shape generalization claim.',
                              'Fixed finite point samples can break geometric symmetry numerically; no manufacturing equivalence implied.'],
        'device': torch.cuda.get_device_name(), 'torch_version': str(torch.__version__),
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256': {name: hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()
                          for name in ('equivariant_axis.py','train_eight_equivariant.py')},
        'reference': 'https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html',
        'wandb': 'disabled',
    }
    for folder in (run, report):
        write_json(folder/'config.json', config)
    write_json(report/'initial_equivariance_audit.json', initial_audit)
    print('Initial equivariance audit:', json.dumps(initial_audit), flush=True)
    print('Parameters:',config['parameter_count'],flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-5)
    generator = torch.Generator(device=device).manual_seed(seed+19703)
    sampling_generator = torch.Generator(device=device).manual_seed(seed+31901)
    training_ids = torch.arange(8,device=device).repeat_interleave(16)
    history, train_history = [], []
    best, best_epoch, streak = float('inf'), 0, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (report/'train_history.csv').open('w',newline='',encoding='utf8') as handle:
        writer = csv.DictWriter(handle,fieldnames=['epoch','online_train_loss','learning_rate'])
        writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train()
            ids = training_ids[torch.randperm(128,generator=generator,device=device)]
            losses = []
            for start in range(0,128,32):
                base = clouds[ids[start:start+32]]
                if args.resample_points:
                    full = full_clouds[ids[start:start+32]]
                    sample_ids = torch.rand(full.shape[:2],generator=sampling_generator,device=device).topk(2048,dim=1,largest=False).indices
                    base = torch.gather(full,1,sample_ids[...,None].expand(-1,-1,6))
                rotations = random_rotations(len(base),generator,device)
                predicted = model(rotate_features(base,rotations))
                loss = (1-(predicted*rotations[:,:,2]).sum(-1)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                losses.append(loss.detach())
            row = {'epoch':epoch,'online_train_loss':float(torch.stack(losses).mean()),
                   'learning_rate':optimizer.param_groups[0]['lr']}
            train_history.append(row)
            writer.writerow(row)
            handle.flush()
            scheduler.step()
            if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
                metrics,predicted,target,_ = measured(model,clouds,probe_rotations)
                per_part = [angle_metrics(predicted[local*64:(local+1)*64],target[local*64:(local+1)*64])
                            for local in range(8)]
                history.append({**row,'probe':metrics,'per_part':per_part,
                                'probe_without_69_mean_deg':float(np.mean([r['mean_angle_deg'] for i,r in zip(selected,per_part) if i!=69]))})
                if metrics['mean_angle_deg']<best:
                    best,best_epoch = metrics['mean_angle_deg'],epoch
                    torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'model':'EquivariantAxis'},run/'best_model.pt')
                streak = streak+1 if metrics['mean_angle_deg']<3 else 0
                progress = {'status':'running','epochs_completed':epoch,'best_epoch':best_epoch,
                            'best_probe_deg':best,'seconds':time.monotonic()-started,
                            'history':history,'train_history':train_history}
                write_json(run/'progress.json',progress)
                write_json(report/'evaluation_history.json',history)
                torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,
                            'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                            'generator_state':generator.get_state(),'rng_state':torch.get_rng_state(),
                            'sampling_generator_state':sampling_generator.get_state(),
                            'cuda_rng_state':torch.cuda.get_rng_state()},run/'last_model.pt')
                plot_curve(report,train_history,history)
                print(f'Epoch {epoch}: loss {row["online_train_loss"]:.6f}, probe {metrics["mean_angle_deg"]:.3f} deg, best {best:.3f} @{best_epoch}, without69 {history[-1]["probe_without_69_mean_deg"]:.3f}, elapsed {time.monotonic()-started:.1f}s',flush=True)
                if streak>=3:
                    break
    checkpoint = torch.load(run/'best_model.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_audit = equivariance_audit(model,clouds,seed+183)
    test_generator = torch.Generator(device=device).manual_seed(seed+183079)
    test_rotations = torch.stack([random_rotations(64,test_generator,device) for _ in selected])
    metrics,predicted,target,restored = measured(model,clouds,test_rotations)
    prior_run = ROOT/'runs/pointnet_random_so3_8parts_normals_2026-11-09'
    with np.load(prior_run/'independent_predictions.npz') as saved:
        np.testing.assert_array_equal(saved['target_axis'],target)
        np.testing.assert_array_equal(saved['part_indices'],np.repeat(selected,64))
    parts = []
    for local,index in enumerate(selected):
        bounds = slice(local*64,(local+1)*64)
        part_metrics = angle_metrics(predicted[bounds],target[bounds])
        part_metrics['cosine_loss'] = float((1-(predicted[bounds]*target[bounds]).sum(-1)).mean())
        parts.append({'part_index':index,'metrics':part_metrics})
    result = {'status':'complete','best_epoch':best_epoch,'epochs_completed':epoch,
              'seconds':time.monotonic()-started,'optimizer_steps':epoch*4,
              'independent_same_shape_512_rotations':metrics,'per_part_independent_rotations':parts,
              'initial_equivariance_audit':initial_audit,'final_equivariance_audit':final_audit,
              'paired_prior_targets_exactly_equal':True,'history':history,'train_history':train_history,
              'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
              'limitations':config['comparison_limits']}
    for folder in (run,report):
        write_json(folder/'metrics.json',result)
    write_json(run/'progress.json',result)
    write_json(report/'final_equivariance_audit.json',final_audit)
    np.savez_compressed(run/'independent_predictions.npz',predicted_axis=predicted,target_axis=target,part_indices=np.repeat(selected,64))
    np.savez_compressed(run/'independent_preview_rotations.npz',augmentation_rotations=test_rotations.cpu().numpy().reshape(-1,3,3),part_indices=np.repeat(selected,64))
    np.savez_compressed(run/'restored_axes.npz',restored_axes=restored)
    rows = '\n'.join(f'| {r["part_index"]} | {r["metrics"]["mean_angle_deg"]:.3f}° | {r["metrics"]["cosine_loss"]:.6f} |' for r in parts)
    text = f'''# 八件旋转等变网络实验（2026-11-09）

Vector Neuron 思想的独立轻量实现，不宣称原论文完整复现。所有可学习向量通道混合无坐标偏置，非线性使用向量投影，注意力与 head 权重只依赖旋转不变量；XYZ×法向叉积适用于 SO(3)。输入 XYZ+法向，固定原八件相同 2048 点；不使用 GT 方向输入、PCA 帧、零件 ID 或支撑。

原八件索引 {selected}，在线 SO(3)，batch32，每件每轮 16 朝向、每轮 4 更新，AdamW lr={args.learning_rate} 余弦到 1e-5，无 dropout/weight decay，有向余弦损失，梯度裁剪 1。在线重新抽点：{args.resample_points}；若启用，从原始 4096 点每次无放回取 2048 点，法向按相同索引取；独立采样 RNG，不改变训练 SO(3) 生成序列。probe/test 保持原固定点。模型从头训练，参数 {config['parameter_count']}。原法向 PointNet 是已有 XYZ 权重续训，初始化和网络容量不同，不能当严格单变量消融。

完成 {epoch} 轮、{epoch*4} 次更新，用时 {result['seconds']/60:.2f} 分钟，按同件 probe 选第 {best_epoch} 轮。没有在新零件上选模。最终独立 512 旋转与旧法向实验逐元素标签一致，平均误差 **{metrics['mean_angle_deg']:.3f}°**，中位 **{metrics['median_angle_deg']:.3f}°**，≤3° {metrics['within_3deg_pct']:.2f}%，≤10° {metrics['within_10deg_pct']:.2f}%，loss **{metrics['cosine_loss']:.6f}**。

| 索引 | 平均误差 | 余弦 loss |
|---|---:|---:|
{rows}

初始和最终旋转等变、点序不变、batch 独立检查均已通过，最大旋转向量残差见审计 JSON。架构等变不保证方向正确，也不能从完美对称的几何唯一推出有向标签；当前每件固定有限点与三角面法向可能带有采样微小不对称，低误差不意味着识别了工艺语义。所有件计入总体，没有对 69 号改标签或取 abs(dot)。本轮只测同件的新旋转，不声称新形状泛化。

曲线 training_curve.png，逐轮 loss train_history.csv，probe evaluation_history.json，完整参数与源码哈希 config.json，预测和实际旋转在同名 runs 目录，可供标准三联图使用。
'''
    (report/'README.md').write_text(text,encoding='utf8')
    print(json.dumps({key:result[key] for key in ('status','best_epoch','epochs_completed','seconds','independent_same_shape_512_rotations','per_part_independent_rotations','final_equivariance_audit')},indent=2),flush=True)


if __name__=='__main__':
    main()
