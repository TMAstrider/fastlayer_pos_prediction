# 30° anchor residual 与 Top-K 排序实验

日期：2026-11-20

## 目的

把30°代理支撑评分从普通候选特征提升为显式锚点，验证网络能否只学习
相对30°排序的残差，并在45°、小于6 mm水平孔豁免后的投影面积目标上
改善Top-5/Top-10推荐。

训练和评价目标仍是45°目标，30°锚点及候选特征不读取45°面积、排名或
保留悬垂面数组。

## 严格项目隔离划分

旧扩展数据集的100件test中有95件与训练集共享源项目。本轮重新按完整
`source_project` 做确定性哈希划分：

- 数据：1032件，每件256个方向。
- test：100件，来自19个完整项目。
- 开发池：932件，来自182个项目。
- development：`0701-0710/0708/0708-铝1`，10件。
- 参数更新：922件。
- train/test项目交集：0。
- 几何哈希交集：0。

划分生成命令：

```bash
python tools/build_project_isolated_split.py
```

## 模型

显式anchor残差：

```text
final_score =
    anchor_weight × score_from_lower_is_better(projected_area_30)
    + residual_scale × PointNet(shape, direction, 30° features)
```

启用anchor时，打分头最后一层初始化为零，因此训练前预测严格等于30°
规则分数。本轮 `anchor_weight=1`、`residual_scale=0.5`。

新增三种排序损失：

- MSE：保留45°归一化分数的回归信号。
- Listwise softmax：匹配同一零件256个候选上的目标分数分布。
- Top-K pairwise：把45°目标Top-10候选排到随机负例之前。

## Test 结果

全局排序：

| 方法 | Spearman 中位数 | NDCG@5 中位数 | NDCG@10 中位数 |
|---|---:|---:|---:|
| 30°直接排序 | 0.6188 | 0.7905 | 0.7960 |
| 30°特征、无anchor | 0.7979 | **0.8957** | **0.9060** |
| 30° anchor残差 | **0.8145** | 0.8960 | 0.8910 |
| anchor＋平衡listwise/Top-K | 0.8142 | 0.8734 | 0.8954 |
| anchor＋强Top-K权重 | 0.7994 | 0.8726 | 0.8647 |

Top-5：

| 方法 | Regret 中位数 | 面积/oracle 中位数 | 精确命中 | ≤5% | ≤10% | ≤20% |
|---|---:|---:|---:|---:|---:|---:|
| 30°直接排序 | 0.0558 | 1.8934 | **18/100** | **24/100** | **28/100** | **31/100** |
| 30°特征、无anchor | 0.0571 | 1.9048 | 5/100 | 12/100 | 15/100 | 21/100 |
| 30° anchor残差 | 0.0561 | 2.0415 | 7/100 | 13/100 | 15/100 | 22/100 |
| anchor＋平衡listwise/Top-K | **0.0271** | 1.5921 | 10/100 | 17/100 | 23/100 | 28/100 |
| anchor＋强Top-K权重 | 0.0325 | **1.5413** | 15/100 | 21/100 | 25/100 | 31/100 |

Top-10：

| 方法 | Regret 中位数 | 面积/oracle 中位数 | 精确命中 | ≤5% | ≤10% | ≤20% |
|---|---:|---:|---:|---:|---:|---:|
| 30°直接排序 | 0.0119 | 1.2212 | **33/100** | **42/100** | **44/100** | **49/100** |
| 30°特征、无anchor | 0.0238 | 1.7655 | 11/100 | 20/100 | 23/100 | 30/100 |
| 30° anchor残差 | 0.0254 | 1.7370 | 15/100 | 23/100 | 25/100 | 34/100 |
| anchor＋平衡listwise/Top-K | **0.0096** | 1.2567 | 24/100 | 33/100 | 36/100 | 44/100 |
| anchor＋强Top-K权重 | 0.0119 | **1.2343** | 23/100 | 35/100 | 41/100 | **49/100** |

面积比只统计98件正oracle面积；精确命中和容差命中按全部100件统计。
2件最优投影面积为0，网络在Top-K内都没有找到零面积方向，因此均记为失败。

## 分析

1. 显式anchor比同划分、同输入的无anchor模型提高了Spearman
   `0.7979→0.8145`、Top-5精确命中 `5→7`、Top-10精确命中 `11→15`。
2. 普通anchor残差仍不能阻止网络把部分30°强候选挤出Top-5。
3. 平衡listwise/Top-K模型把Top-5 regret降低约一半，达到 `0.0271`；
   Top-10 regret为 `0.0096`。同时保留anchor残差模型的全局Spearman。
4. 强Top-K权重继续提升Top-5`≤5%`命中到`21/100`，代价是Spearman降到
   `0.7994`。它适合更重视少量推荐候选的场景。
5. 30°直接排序仍是Top-10精确命中的强对照，`33/100`，高于所有网络模型。
   这说明30°锚点本身提供了有效的候选召回，网络主要改善整体排序和面积
   接近程度。
6. 逐点后验λ扫描中，平衡模型在`λ=0.05`时可达到Top-5精确命中`21/100`、
   `≤5%`命中`29/100`，同时Spearman为`0.6931`。由于λ是在看过test后扫描，
   该结果只作为下一轮预注册实验依据，不作为本轮选定配置。

当前默认推荐为“anchor＋平衡listwise/Top-K”：

- 它保留anchor的全局修正能力；
- Top-5 regret和Top-10 regret都明显低于显式anchor基线；
- 没有为了少量Top-K命中完全牺牲Spearman和NDCG。

## 复现

```bash
python tools/train_pointnet_direction_scorer.py \
  --dataset labels/projection_support_area_expanded_932_100_v0_256 \
  --split-file labels/projection_support_area_expanded_932_100_v0_256/split_project_isolated_100_seed30.json \
  --geometry-features labels/geometry_prior_30deg_expanded_932_100_v1 \
  --name pointnet_30deg_anchor_listwise_topk_4096_project_isolated_922_10_100_2026-11-20 \
  --epochs 120 \
  --batch-shapes 16 \
  --point-count 4096 \
  --anchor-weight 1.0 \
  --residual-scale 0.5 \
  --mse-weight 0.25 \
  --pairwise-weight 0.25 \
  --listwise-weight 0.10 \
  --listwise-temperature 0.1 \
  --topk-weight 0.75 \
  --topk-positive-count 10 \
  --topk-pairwise-samples 64

python tools/evaluate_anchor_residual_scales.py \
  --run runs/pointnet_30deg_anchor_listwise_topk_4096_project_isolated_922_10_100_2026-11-20 \
  --dataset labels/projection_support_area_expanded_932_100_v0_256 \
  --split-file labels/projection_support_area_expanded_932_100_v0_256/split_project_isolated_100_seed30.json \
  --geometry-features labels/geometry_prior_30deg_expanded_932_100_v1 \
  --name pointnet_30deg_anchor_listwise_topk_scale_sweep_project_isolated_2026-11-20
```

checkpoint、逐零件CSV和生成标签保留在本地，不提交到Git。

