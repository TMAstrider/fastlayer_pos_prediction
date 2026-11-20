# 4096 点纯点云 PointNet 方向评分实验

## 目的

验证 PointNet 能否只利用零件点云，加每个候选方向的 3 维 XYZ，学会按
支撑投影面积对 256 个方向排序。此版本不使用原来 18 维候选几何特征，
不使用几何 baseline、残差或几何融合。

## 数据与网络

- 数据：`labels/projection_support_area_v0_256`
- 划分：171 件参数更新、12 件 development、10 件 held-out test
- 每件点数：4096
- 点输入：XYZ＋法向，共 6 个通道
- 候选输入：`direction_x`、`direction_y`、`direction_z`
- 网络：PointNet 形状 embedding（1024 维）＋方向 embedding（128 维）→ MLP 标量分数
- 训练：MSE＋pairwise ranking loss，并对点云、法向和方向施加同一个随机 SO(3)
- GT：确定性几何计算的 `projected_area_mm2` 所对应的归一化分数

## Test 结果

| 指标 | 4096 点方向-only PointNet | PointNet＋18 特征 | CandidateMLP |
|---|---:|---:|---:|
| Spearman 中位数 | 0.7046 | 0.9530 | 0.9441 |
| Kendall 中位数 | 0.5063 | 0.8056 | 0.7797 |
| NDCG@5 中位数 | 0.8828 | 0.9695 | 0.9546 |
| NDCG@10 中位数 | 0.8937 | 0.9686 | 0.9690 |
| Top-1 面积/oracle 中位数 | 1.6500 | 1.1844 | 1.3754 |
| Top-5 精确命中 | 0/10 | 2/10 | — |

独立几何 baseline 的 Spearman 和 NDCG@5 中位数分别为 0.9610 和
0.9983。它没有进入模型，只用于确认任务本身容易从几何量排序。

训练最佳 epoch 为 18，第 48 轮早停，参数量 1,113,217。development
Spearman 在该轮为 0.4213。独立加载 checkpoint 后，test 指标复算一致。

## 结论

只给全局 PointNet embedding 和方向 XYZ 时，模型能学到一定的排序信号，
NDCG@5 达到 0.8828；但它选出的 Top-1 平均面积仍明显偏大，和带候选
几何特征的模型有明显差距。当前方案不能替代现有 18 维特征的直接评分器。

这个结果也说明收益很大一部分来自方向对齐后的悬垂几何量，而不是仅靠
全局形状 embedding 就能轻易恢复。下一轮若要继续“纯点云”路线，需要让
PointNet 在方向相关的位置和法向上做 pooled feature，或按候选方向旋转点云后
逐一编码；这两者仍是独立的实验，不应把投影面积 GT 或几何 baseline 塞进网络。

## 复现

```bash
python tools/train_pointnet_direction_scorer.py --device cuda:0
```

详细生成结果位于本地
`reports/pointnet_direction_only_4096_2026-11-20/`，其中 checkpoint、
CSV、JSON 和图表不作为源码提交。
