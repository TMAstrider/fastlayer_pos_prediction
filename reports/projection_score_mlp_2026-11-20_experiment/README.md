# 投影面积候选评分器首轮实验

## 数据与隔离

193 个零件各生成 256 个 Fibonacci 球面方向，投影栅格为 `0.5 mm`。
训练标签只使用：

- `45°` 向下悬垂阈值；
- 严格 `< 6 mm` 水平孔豁免；
- 每件内部的投影面积 min-max 归一化；面积越小分数越高。

训练不使用用户 query，也不回归历史人工姿态。为避免在测试集上选模：

| 子集 | 零件 | 项目 |
|---|---:|---|
| 参数更新 | 171 | 除 development 和 test 外的训练项目 |
| development | 12 | `AI-0705-铝1第二版` |
| test | 10 | `AI-0716-铝7` |

## 对照方法

1. **无锚点 MLP**：直接根据 18 个候选方向特征预测归一化面积分数。
2. **锚点 MLP**：把保留悬垂面积分数加入输出，网络只学习投影重叠修正。
3. **几何 fallback**：直接按 `-retained_surface_mm2` 排序。
4. **排名融合**：在同一零件内融合 MLP 排名与几何 fallback 排名；
   权重仅由 development 集选择。

## Test 结果

下表均为 10 个 test 零件的中位数。

| 方法 | Spearman | NDCG@5 | Top-1 归一化面积损失 | Top-1 面积 / oracle |
|---|---:|---:|---:|---:|
| 锚点 MLP + 排名融合 | 0.9656 | 0.9913 | 0.000026 | 1.0003 |
| 锚点 MLP 原始输出 | 0.9640 | 0.9873 | 0.0060 | 1.0505 |
| 几何 fallback | 0.9610 | 0.9983 | 0.000485 | 1.0062 |
| 无锚点 MLP + 排名融合 | 0.9654 | 0.9817 | 0.00943 | 1.0932 |

锚点 MLP 的 development 融合权重为 `0.3`，该权重随后原样用于 test。
锚点版本比无锚点版本的全局排序和 Top-1 都更稳定。

## 结论

- 从几何特征预测投影面积排序是可学习的；500 个左右方向的选择不需要强化学习。
- 锚点 MLP 融合后的 Top-1 已接近 oracle，Top-5 在 10 件 test 上均命中 oracle 面积。
- 简单几何 fallback 的 NDCG@5 仍略高。当前不能宣称 MLP 全面超过几何规则。
- 第一版推理接口应同时返回 `mlp_hybrid` 和 `geometry_fallback`，
  用更大的项目级测试集继续验证后再决定默认排序器。

## 产物

- `reports/projection_score_mlp_anchored_2026-11-20/`：选定模型、训练曲线和逐件结果；
- `reports/projection_score_mlp_unanchored_2026-11-20/`：直接回归对照；
- `tools/train_projection_score_mlp.py`：可复现训练、项目隔离选模和排名融合；
- `tests/test_projection_score_training.py`：标签隔离、排序指标和融合逻辑测试。
