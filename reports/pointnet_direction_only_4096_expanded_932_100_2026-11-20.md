# 扩展数据上的 4096 点方向-only PointNet

## 数据

- 去重后的零件：1032 件
- 形状隔离训练池：932 件
- 留出 test：100 件
- 从训练池按来源项目再留出 development：10 件
- 最终参数更新：922 件
- 每件候选方向：256 个
- 每件点输入：4096 点，XYZ＋法向
- 候选输入：3 维方向 XYZ
- 训练标签：45° 角度过滤后的支撑投影并集面积

数据沿用 `runs/pointnet_shape_dedup_2026-11-09_val100/split.json`
中的形状隔离划分，训练集、development 和 test 之间没有几何哈希重叠。

## 网络

```text
4096 点（XYZ＋法向）
    ↓
PointNet
    ↓
1024 维形状 embedding
    ↓
与候选方向 XYZ 的 128 维 embedding 拼接
    ↓
标量分数
```

网络不使用原 18 维候选几何特征，不使用 anchor、几何残差或几何 fusion。
几何 baseline 只在训练完成后作独立对照。

## Test 结果

| 指标 | 扩展数据 PointNet | 几何 baseline |
|---|---:|---:|
| Spearman 中位数 | 0.7817 | 0.9536 |
| Kendall 中位数 | 0.5777 | 0.8323 |
| NDCG@5 中位数 | 0.8759 | 0.9997 |
| NDCG@10 中位数 | 0.8879 | 0.9992 |
| Top-1 面积/oracle 中位数 | 2.5315 | 1.0000 |
| Top-1 精确命中 | 0/100 | 85/100 |
| Top-5 精确命中 | 2/100 | 96/100 |
| 面积误差不超过 5% | 0/100 | 92/100 |

几何 baseline 直接使用标签数据里的 `retained_surface_mm2`，没有进入网络，
也不能作为可部署的纯点云方法。它的结果只说明角度过滤目标本身很容易由
方向对齐后的几何量恢复。

训练最佳 epoch 为 111，120 轮全部完成，在 RTX 4090 上用时约
228 秒，参数量 1,113,217。独立加载 checkpoint 后重新计算 100 件测试集，
指标与训练报告一致。

## 结论

扩展数据后，全局 PointNet embedding 加方向 XYZ 已经能学到明显的排序
信号：Spearman 中位数 0.7817，NDCG@5 中位数 0.8759。但 Top-1 面积
仍明显高于 oracle，说明模型主要学到整体趋势，尚不能可靠找到最尖锐的
最低支撑面积方向。

与 193 件小数据集上的方向-only 结果相比，Spearman 从 0.7046 提升到
0.7817；由于测试零件集合不同，这不是严格配对比较。提高训练样本量有帮助，
但当前输入结构还不足以替代方向对齐后的显式几何计算。

## 数据构建说明

常规零件使用之前定义的 45° 角度过滤及小孔豁免规则生成标签。唯一一个约
142 万三角面的训练零件单进程耗时超过一小时，因此按本轮要求改用纯角度
分片计算，省略孔识别和 18 维辅助特征。该零件位于训练池，不在 100 件
test 中；这不会泄漏 test，但意味着后续若要求严格同分布标签，应对全部
零件统一改为纯角度规则。

## 复现

```bash
python tools/build_expanded_projection_score_dataset.py --jobs 88

python tools/train_pointnet_direction_scorer.py \
  --dataset labels/projection_support_area_expanded_932_100_v0_256 \
  --name pointnet_direction_only_4096_expanded_932_100_2026-11-20 \
  --point-count 4096 \
  --batch-shapes 16 \
  --epochs 120 \
  --device cuda:0
```

详细的 CSV、JSON 和 checkpoint 保留在本地
`reports/pointnet_direction_only_4096_expanded_932_100_2026-11-20/`
与 `runs/pointnet_direction_only_4096_expanded_932_100_2026-11-20/`，
不提交到 Git。
