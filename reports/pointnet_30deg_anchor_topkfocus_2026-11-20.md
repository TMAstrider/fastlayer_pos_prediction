# PointNet 30° Anchor Top-K Focus 实验

## 数据与评估范围

- 训练零件：922
- development 零件：10
- test 零件：100
- 每件候选方向：256
- 目标：AlSi10Mg/standard 45° 过滤后的代理支撑投影面积
- 网络输入：4096 个 XYZ＋法向点、候选方向、14 维 30° 候选几何特征和
  7 维全局形状特征
- 30° 几何先验不读取 45° 目标面积

本拆分检查了 development project 隔离，但 100 个 test 零件中仍有 95 个来自训练集中
出现过的项目，因此不能把结果解释为完全跨客户泛化。

## 模型与训练

- 模型：`PointNetGeometryPrior30Scorer`
- 参数量：1,136,193
- 预测：`1.0 × 30° anchor score + 0.5 × PointNet residual`
- PointNet 点 embedding：1024
- shape hidden dim：256
- direction hidden dim：128
- best epoch：8
- completed epochs：38
- 训练耗时：83.62 秒，CUDA
- seed：20261120

这轮加强 Top-K 目标：

```text
mse_weight = 0.05
pairwise_weight = 0.25
listwise_weight = 0.05
topk_weight = 2.0
topk_positive_count = 10
topk_pairwise_samples = 128
```

## Test 结果

| 指标 | 结果 |
|---|---:|
| Spearman 均值 | 0.7460 |
| Spearman 中位数 | 0.7994 |
| Kendall 均值 | 0.5773 |
| NDCG@5 均值 | 0.8184 |
| NDCG@10 均值 | 0.8276 |
| Top-1 regret 中位数 | 0.0765 |
| Top-1 面积/合成 GT 中位数 | 2.1679 |
| Top-1 面积/合成 GT 均值 | 4.8708 |
| Top-1 精确命中 | 6/100 |
| Top-5 精确命中 | 15/100 |
| Top-10 精确命中 | 23/100 |
| Top-10 内 5% 误差内覆盖 | 33/100 |
| Top-10 中位面积/合成 GT | 1.2343 |
| Top-20 精确命中 | 46/100 |
| Top-20 内 5% 误差内覆盖 | 53/100 |

注意：旧汇总曾把 2 个 oracle 面积为 0、且 Top-K 未命中的零件误计为
“≤5% 成功”。修正后的数值见
`reports/topk_recommendation_metrics_audit_2026-11-20.md`。

Top-1 与合成 GT 方向夹角的中位数为 `79.09°`。因此 Top-1 的支撑面积可能接近最小
GT，但方向本身仍可能相差很大。当前结果更适合作为候选生成器并保留 Top-K，而不只输出
单一位姿。

## 与上一版 listwise/top-k 对比

| 指标 | topkfocus | listwise/top-k | 变化 |
|---|---:|---:|---:|
| Spearman 均值 | 0.7460 | 0.7697 | -0.0237 |
| NDCG@5 均值 | 0.8184 | 0.8236 | -0.0052 |
| Top-1 面积/GT 中位数 | 2.1679 | 2.2659 | 改善约 4.3% |
| Top-1 面积/GT 均值 | 4.8708 | 6.3151 | 明显改善 |
| Top-1 精确命中 | 6 | 3 | +3 |
| Top-5 精确命中 | 15 | 10 | +5 |
| Top-10 精确命中 | 23 | 24 | -1 |
| Top-20 精确命中 | 46 | 44 | +2 |

提高 Top-K 损失后，Top-1 和 Top-5 的精确命中与面积误差改善，但整体 Spearman 和
NDCG 小幅下降。这是排序质量与头部候选质量之间的取舍。

## 基线对比

| 方法 | Spearman 均值 | NDCG@5 均值 | Top-1 面积/GT 中位数 |
|---|---:|---:|---:|
| topkfocus PointNet＋30° anchor | 0.7460 | 0.8184 | 2.1679 |
| 方向-only PointNet | 0.7386 | 0.8087 | 2.5315 |
| 30° 规则直接排序 | 0.5739 | 0.7380 | 3.3213 |
| 45° 保留面规则 | 0.8506 | 0.9918 | 1.0000 |

45° 保留面规则直接使用与目标高度一致的几何量，属于规则上界参照，不应用作网络的
独立学习输入。

## 产物

生成结果位于：

```text
reports/pointnet_30deg_anchor_topkfocus_4096_project_isolated_922_10_100_2026-11-20/
runs/pointnet_30deg_anchor_topkfocus_4096_project_isolated_922_10_100_2026-11-20/
```

训练数据、checkpoint、逐零件 CSV 和指标 JSON 按项目惯例保留为本地生成产物。
