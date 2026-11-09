"""Train the eight-part equivariant architecture on the unchanged 932/100 split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from diagnose_pointnet_random_so3 import rotate_features
from equivariant_axis import EquivariantAxis
from train_eight_equivariant import equivariance_audit
from train_full_online_so3 import assess, load_bases, plot_curve, write_json
from train_magics_pointnet import random_rotations


ROOT = Path(__file__).resolve().parents[1]


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def sampled_batch(pools, record_ids, generator):
    full = pools[record_ids]
    ids = torch.rand(full.shape[:2], generator=generator, device=full.device).topk(2048, dim=1, largest=False).indices
    return torch.gather(full, 1, ids[..., None].expand(-1, -1, 6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='equivariant_full_resampled_2026-11-09')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_every) < 1 or Path(args.name).name != args.name:
        parser.error('Require positive counts and a single-component name')
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    run, report = ROOT/'runs'/args.name, ROOT/'reports'/args.name
    if run.exists() and not args.resume:
        parser.error('Existing run; choose a new name')
    previous = None
    if args.resume:
        previous = json.loads((run/'config.json').read_text(encoding='utf8'))
        for key in ('seed', 'batch_size', 'eval_every', 'learning_rate'):
            if previous[key] != getattr(args, key):
                parser.error(f'Resume must preserve {key}')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    pools, manifest, pool_hashes = load_bases(True, args.seed, device, point_count=4096)
    bases, same_manifest, subset_hashes = load_bases(True, args.seed, device)
    assert same_manifest == manifest
    split_file = ROOT/'runs/pointnet_shape_dedup_2026-11-09_val100/split.json'
    split = json.loads(split_file.read_text(encoding='utf8'))
    train_ids, val_ids = split['train_indices'], split['validation_indices']
    assert len(train_ids) == 932 and len(val_ids) == 100 and len(pools) == 1032
    assert set(train_ids).isdisjoint(val_ids)
    for scope, indices in [('train', train_ids), ('validation', val_ids)]:
        assert [manifest['records'][index]['geometry_sha256'] for index in indices] == split[f'{scope}_geometry_sha256']
    assert {manifest['records'][index]['shape_group_id'] for index in train_ids}.isdisjoint(
        {manifest['records'][index]['shape_group_id'] for index in val_ids})
    assert set(train_ids) | set(val_ids) == set(range(len(pools)))
    audit = json.loads((ROOT/'outputs/shape_deduplicated_normals/manifest.json').read_text(encoding='utf8'))
    assert not audit['failures'] and len(audit['records']) == len(pools)
    model = EquivariantAxis().to(device)
    assert sum(parameter.numel() for parameter in model.parameters()) == 10576
    architecture_audit = equivariance_audit(model, bases[train_ids[:8]], args.seed+91)
    rotation_generator = torch.Generator(device=device).manual_seed(args.seed+19703)
    sampling_generator = torch.Generator(device=device).manual_seed(args.seed+31901)
    check_generator = torch.Generator(device=device).manual_seed(args.seed+44191)
    check_inputs = sampled_batch(pools, torch.tensor(train_ids[:8], device=device), check_generator)
    np.testing.assert_allclose(check_inputs[:, :, 3:].norm(dim=-1).cpu().numpy(), 1, atol=1e-5)
    check_rotations = torch.eye(3, device=device).expand(8, -1, -1)
    torch.testing.assert_close(rotate_features(check_inputs, check_rotations), check_inputs)
    run.mkdir(parents=True, exist_ok=args.resume)
    report.mkdir(parents=True, exist_ok=True)
    source_files = ['train_full_equivariant.py', 'equivariant_axis.py', 'train_full_online_so3.py',
                    'train_eight_equivariant.py', 'diagnose_pointnet_random_so3.py', 'train_magics_pointnet.py']
    config = {
        **vars(args), 'task':'932-part equivariant orientation learning with online point resampling and SO(3)',
        'model':'Exactly EquivariantAxis from eight-part experiment, independently implemented vector-neuron-style network',
        'parameter_count':10576, 'initialization':'From scratch, all layers trainable; no eight-part warm start',
        'training_parts':932, 'validation_parts':100, 'point_count':2048, 'pool_points_per_part':4096,
        'input_channels':6, 'use_normals':True, 'batch_norm':False, 'dropout':0, 'weight_decay':0,
        'preprocessing':'Center and normalize original 4096 XYZ by max radius; normals unchanged by centering/scale',
        'point_sampling':'Every appearance: independent uniform 2048-point subset without replacement from original 4096 pool; same IDs for normals',
        'point_pool_permutation_hashes':pool_hashes, 'fixed_probe_subset_hashes':subset_hashes,
        'augmentation':'Uniform online SO(3); P@R.T and N@R.T; target R[:,2]',
        'loss':'Mean signed unit-axis cosine loss, 1-dot(pred,target); no symmetry correction',
        'optimizer':'AdamW', 'gradient_clip':1., 'learning_rate_schedule':f'Cosine {args.learning_rate} to 1e-5 over {args.epochs} epochs',
        'training_rotations_per_part_per_epoch':16,
        'steps_per_epoch':(len(train_ids)*16+args.batch_size-1)//args.batch_size,
        'probe':'932 training shapes x 4 fixed SO(3), fixed subset; not backpropagated',
        'validation':'100 held-out shapes x 32 fixed SO(3), fixed subset; selects best model',
        'validation_point_probe':'100 held-out shapes x 4 SO(3), different fixed point subset; secondary diagnostic, not selection',
        'stop':'Max epochs, user signal after a complete epoch, or three validation checks below 3 degrees; no plateau early stopping',
        'final':'Best validation checkpoint: all training shapes x32 and validation shapes x64 new SO(3); three alternate subsets on same validation shapes',
        'split_source':str(split_file), 'split_limitation':split['limitation'],
        'normal_orientation_uncertified_count':sum(not row['outward_orientation_checked'] for row in audit['records']),
        'reference_pointnet_best_validation_deg':64.21351623535156,
        'comparison_limitation':'Network, point resampling, capacity and schedule differ from prior PointNet; not a single-variable causal comparison',
        'device':torch.cuda.get_device_name(), 'torch_version':str(torch.__version__), 'wandb':'disabled',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{name:hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest() for name in source_files},
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    history, train_history = [], []
    start_epoch, prior_seconds, best_epoch, best_train_epoch = 0, 0., 0, 0
    best, best_train, streak = float('inf'), float('inf'), 0
    if args.resume:
        assert previous['point_pool_permutation_hashes'] == pool_hashes
        assert previous['fixed_probe_subset_hashes'] == subset_hashes
        checkpoint = torch.load(run/'last_model.pt', map_location=device, weights_only=True)
        start_epoch = checkpoint['epoch']
        if args.epochs <= start_epoch:
            parser.error('New epoch limit must exceed saved epoch')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.epochs == previous['epochs']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-start_epoch, eta_min=1e-5)
        rotation_generator.set_state(checkpoint['rotation_generator_state'].cpu())
        sampling_generator.set_state(checkpoint['sampling_generator_state'].cpu())
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
        history, train_history = checkpoint['history'], checkpoint['train_history']
        best, best_epoch = checkpoint['best'], checkpoint['best_epoch']
        best_train, best_train_epoch = checkpoint['best_train'], checkpoint['best_train_epoch']
        streak, prior_seconds = checkpoint['threshold_streak'], checkpoint['elapsed_seconds']
        config['resume_history'] = previous.get('resume_history', [])+[
            {'checkpoint_epoch':start_epoch,'previous_limit':previous['epochs'],'new_limit':args.epochs,
             'restored':'Model, optimizer, LR scheduler or continuous extended schedule, both augmentation generators, torch RNG, histories'}]
        write_json(report/f'config_before_resume_epoch{start_epoch}.json',previous)
    for folder in (run,report):
        write_json(folder/'config.json',config)
        write_json(folder/'split.json',split)
    snapshot = report/('source' if not args.resume else f'source_resume_epoch{start_epoch}')
    snapshot.mkdir(exist_ok=True)
    for name in source_files:
        (snapshot/name).write_bytes((ROOT/'tools'/name).read_bytes())
    write_json(report/'normal_attachment_audit.json',audit)
    write_json(report/'initial_equivariance_audit.json',architecture_audit)
    point_probe = bases.clone()
    point_probe_generator = torch.Generator(device=device).manual_seed(args.seed+99101)
    point_probe[val_ids] = sampled_batch(pools,torch.tensor(val_ids,device=device),point_probe_generator)
    requested_stop = {'signal':None}

    def request_stop(signum, frame):
        requested_stop['signal'] = signum

    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if not args.resume:
        train_metrics,_,_ = assess(model,bases,train_ids,4,args.seed+73001,32)
        val_metrics,_,_ = assess(model,bases,val_ids,32,args.seed+83001,32)
        history.append({'epoch':0,'same_shape_probe':train_metrics,'validation':val_metrics})
    print(f'Ready: 932 train / 100 validation; {config["steps_per_epoch"]} updates per epoch, {args.epochs} max; parameters 10576; starting after {start_epoch}',flush=True)
    print('Equivariance check:',json.dumps(architecture_audit),flush=True)
    training_ids = torch.tensor(train_ids,device=device).repeat_interleave(16)
    stop_reason = 'maximum_epochs'
    with (report/'train_history.csv').open('w',newline='',encoding='utf8') as handle:
        writer = csv.DictWriter(handle,fieldnames=['epoch','online_train_loss','online_train_mean_angle_deg','learning_rate','epoch_seconds','optimizer_steps'])
        writer.writeheader()
        writer.writerows(train_history)
        handle.flush()
        for epoch in range(start_epoch+1,args.epochs+1):
            epoch_started = time.monotonic()
            model.train()
            ids = training_ids[torch.randperm(len(training_ids),generator=rotation_generator,device=device)]
            loss_total = torch.zeros((),device=device)
            angle_total = torch.zeros((),device=device)
            for start in range(0,len(ids),args.batch_size):
                base = sampled_batch(pools,ids[start:start+args.batch_size],sampling_generator)
                rotations = random_rotations(len(base),rotation_generator,device)
                predicted = model(rotate_features(base,rotations))
                cosine = (predicted*rotations[:,:,2]).sum(-1)
                loss = (1-cosine).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                loss_total += loss.detach()*len(base)
                angle_total += torch.rad2deg(torch.acos(cosine.detach().clamp(-1,1))).sum()
            row = {'epoch':epoch,'online_train_loss':float(loss_total/len(ids)),
                   'online_train_mean_angle_deg':float(angle_total/len(ids)),
                   'learning_rate':optimizer.param_groups[0]['lr'],
                   'epoch_seconds':time.monotonic()-epoch_started,'optimizer_steps':epoch*config['steps_per_epoch']}
            scheduler.step()
            train_history.append(row)
            writer.writerow(row)
            handle.flush()
            evaluated = epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs or requested_stop['signal'] is not None
            if evaluated:
                train_metrics,_,_ = assess(model,bases,train_ids,4,args.seed+73001,32)
                val_metrics,_,arrays = assess(model,bases,val_ids,32,args.seed+83001,32,True)
                alternate_metrics,_,_ = assess(model,point_probe,val_ids,4,args.seed+93101,32)
                history.append({**row,'same_shape_probe':train_metrics,'validation':val_metrics,'validation_alternate_points':alternate_metrics})
                if val_metrics['mean_angle_deg'] < best:
                    best,best_epoch = val_metrics['mean_angle_deg'],epoch
                    save_checkpoint(run/'best_model.pt',{'model_state_dict':model.state_dict(),'epoch':epoch,'validation':val_metrics,'model':'EquivariantAxis'})
                    np.savez_compressed(run/'best_validation_predictions.npz',**arrays)
                if train_metrics['mean_angle_deg'] < best_train:
                    best_train,best_train_epoch = train_metrics['mean_angle_deg'],epoch
                    save_checkpoint(run/'best_same_shape_model.pt',{'model_state_dict':model.state_dict(),'epoch':epoch,'same_shape_probe':train_metrics,'model':'EquivariantAxis'})
                streak = streak+1 if val_metrics['mean_angle_deg']<3 else 0
                write_json(report/'evaluation_history.json',history)
                plot_curve(report,train_history,history)
                print(f'Epoch {epoch}: loss {row["online_train_loss"]:.5f}, train-online {row["online_train_mean_angle_deg"]:.2f} deg; train-probe {train_metrics["mean_angle_deg"]:.2f}, val {val_metrics["mean_angle_deg"]:.2f}, alternate-points val {alternate_metrics["mean_angle_deg"]:.2f}; best {best:.2f} @{best_epoch}; {row["epoch_seconds"]:.1f}s/epoch; total {prior_seconds+time.monotonic()-started:.1f}s',flush=True)
            elif epoch%2==0:
                print(f'Epoch {epoch}: loss {row["online_train_loss"]:.5f}, online angle {row["online_train_mean_angle_deg"]:.2f} deg; {row["epoch_seconds"]:.1f}s/epoch',flush=True)
            checkpoint = {'model_state_dict':model.state_dict(),'epoch':epoch,'optimizer_state_dict':optimizer.state_dict(),
                          'scheduler_state_dict':scheduler.state_dict(),'rotation_generator_state':rotation_generator.get_state(),
                          'sampling_generator_state':sampling_generator.get_state(),'rng_state':torch.get_rng_state(),
                          'cuda_rng_state':torch.cuda.get_rng_state(),'history':history,'train_history':train_history,
                          'best':best,'best_epoch':best_epoch,'best_train':best_train,'best_train_epoch':best_train_epoch,
                          'threshold_streak':streak,'elapsed_seconds':prior_seconds+time.monotonic()-started}
            save_checkpoint(run/'last_model.pt',checkpoint)
            progress = {'status':'running','epochs_completed':epoch,'best_epoch':best_epoch,
                        'best_validation_mean_angle_deg':best,'best_same_shape_probe_mean_angle_deg':best_train,
                        'seconds':prior_seconds+time.monotonic()-started,'history':history,'train_history':train_history}
            write_json(run/'progress.json',progress)
            if requested_stop['signal'] is not None:
                progress.update(status='stopped_by_user',stop_reason='User signal; complete epoch checkpoint saved')
                for folder in (run,report):
                    write_json(folder/'stopped_progress.json',progress)
                write_json(run/'progress.json',progress)
                print('User stop: saved complete state, no final evaluation',flush=True)
                return
            if streak>=3:
                stop_reason = 'validation_under_3deg_three_checks'
                break
    checkpoint = torch.load(run/'best_model.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_audit = equivariance_audit(model,bases[val_ids[:8]],args.seed+192)
    train_metrics,train_parts,train_arrays = assess(model,bases,train_ids,32,args.seed+183079,32,True)
    val_metrics,val_parts,val_arrays = assess(model,bases,val_ids,64,args.seed+283079,32,True)
    alternate_results = []
    for seed_offset in (391001,392001,393001):
        generator = torch.Generator(device=device).manual_seed(args.seed+seed_offset)
        alternate_bases = bases.clone()
        alternate_bases[val_ids] = sampled_batch(pools,torch.tensor(val_ids,device=device),generator)
        metrics,parts,_ = assess(model,alternate_bases,val_ids,64,args.seed+283079,32)
        alternate_results.append({'point_sampling_seed':args.seed+seed_offset,'metrics':metrics,'per_part':parts})
        del alternate_bases
    np.savez_compressed(run/'independent_same_shape_predictions.npz',**train_arrays)
    np.savez_compressed(run/'independent_unseen_predictions.npz',**val_arrays)
    result = {'status':'complete','epochs_completed':epoch,'best_epoch':best_epoch,'best_same_shape_epoch':best_train_epoch,
              'stop_reason':stop_reason,'training_parts':932,'validation_parts':100,'optimizer_steps':epoch*config['steps_per_epoch'],
              'seconds':prior_seconds+time.monotonic()-started,'independent_same_shape_rotations':train_metrics,
              'independent_unseen_shape_rotations':val_metrics,'per_part_same_shape':train_parts,'per_part_unseen_shape':val_parts,
              'validation_alternate_point_subsets':alternate_results,'final_equivariance_audit':final_audit,
              'history':history,'train_history':train_history,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
              'limitations':[split['limitation'],'Validation selects checkpoint; final new rotations of these same shapes are not an independent final shape test.',
                            'All subsets come from fixed 4096-point pools, not independent fresh mesh sampling.',
                            'Equivariant rotations are correlated views, not independent new shapes; signed GT only.']}
    for folder in (run,report):
        write_json(folder/'metrics.json',result)
    write_json(run/'progress.json',result)
    write_json(report/'final_equivariance_audit.json',final_audit)
    rows = '\n'.join(f'| {label} | {metrics["mean_angle_deg"]:.3f}° | {metrics["median_angle_deg"]:.3f}° | {metrics["within_10deg_pct"]:.2f}% | {metrics["cosine_loss"]:.6f} |'
                     for label,metrics in [('932 训练件，新旋转',train_metrics),('100 验证件，新旋转',val_metrics)])
    summary = f'''# 全量等变网络在线抽点＋SO(3)（2026-11-09）

状态：完成。原划分 932 train / 100 validation，八件相同 EquivariantAxis 网络，10576 参数，从头训练。每次从原 4096 点池重新取 2048 点 XYZ＋法向，再随机 SO(3)，GT=R[:,2]。batch {args.batch_size}，每件每轮 16 视图，{config['steps_per_epoch']} 更新/轮。AdamW lr={args.learning_rate} 余弦到 1e-5，weight decay/dropout=0，梯度裁剪1，有向余弦损失，无对称修正或支撑输入。

共 {epoch} 轮，耗时 {result['seconds']/60:.1f} 分钟，停止原因 {stop_reason}。最佳验证 checkpoint 第 {best_epoch} 轮，固定 probe 平均误差 {best:.3f}°。最终均用此验证最优权重评估。

| 范围 | 平均误差 | 中位误差 | ≤10° | loss |
|---|---:|---:|---:|---:|
{rows}

另三组验证点子集均只作评估，均值依次为 {', '.join(f"{item['metrics']['mean_angle_deg']:.3f}°" for item in alternate_results)}。所有点均来自原4096点池，不能称为独立mesh重采样。源法向 94 件外向性未认证，逐件审计另存。未按件族/项目隔离；验证用于选模，因此不是独立最终测试集。与此前 PointNet 的网络、抽点策略、容量和训练预算均不同，不是单变量消融。

逐轮曲线 training_curve.png、train_history.csv，评估 evaluation_history.json，配置/来源哈希 config.json，完整源码快照 source/，逐件结果 metrics.json，最佳/最近权重与预测在同名 runs 目录。最近 checkpoint 每轮保存模型、优化器、调度、两套随机生成器和历史，可用 --resume 恢复。没有开启 W&B。
'''
    (report/'README.md').write_text(summary,encoding='utf8')
    preview = run/'preview'
    preview.mkdir(exist_ok=True)
    selected = json.loads((ROOT/'reports/pointnet_eight_unseen_parts_2026-11-09/config.json').read_text(encoding='utf8'))['selected_indices']
    mask = np.isin(val_arrays['part_indices'],selected)
    write_json(preview/'config.json',{'selected_indices':selected,'selected_records':[manifest['records'][index] for index in selected],'evaluation_scope':'unseen_parts'})
    np.savez_compressed(preview/'independent_predictions.npz',**{key:value[mask] for key,value in val_arrays.items() if key!='rotations'})
    np.savez_compressed(preview/'independent_preview_rotations.npz',augmentation_rotations=val_arrays['rotations'][mask],part_indices=val_arrays['part_indices'][mask])
    preview_result = subprocess.run([sys.executable,str(ROOT/'tools/render_validation_poses.py'),'--diagnostic',
                                    '--run',str(preview),'--output',str(ROOT/'outputs'/f'{args.name}_preview'),'--workers','3'],cwd=ROOT)
    write_json(report/'preview_status.json',{'returncode':preview_result.returncode,'selected_indices':selected,
                                           'output':str(ROOT/'outputs'/f'{args.name}_preview')})
    print(json.dumps({key:result[key] for key in ('status','best_epoch','epochs_completed','seconds','independent_same_shape_rotations','independent_unseen_shape_rotations')},indent=2),flush=True)


if __name__=='__main__':
    main()
