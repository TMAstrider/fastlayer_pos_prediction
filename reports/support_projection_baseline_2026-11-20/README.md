# 单零件支撑投影面积基线（2026-11-20）

## 目的

先不训练网络，验证固定几何规则下的评分定义是否可用：

1. 一个零件输入；
2. 近似均匀采样 512 个球面构建方向；
3. 每个方向只用 45° 悬垂阈值和 `<6 mm` 水平孔豁免；
4. 计算保留悬垂面在构建平面上的投影并集面积；
5. 把面积转换成“越小越好”的评分。

用户 query 不参与本实验。

## 评分量化

投影面积记为 `A_proj`，评分使用单个零件全部候选方向的 min-max 归一化：

```text
score = 1 - (A_proj - min(A_proj)) / (max(A_proj) - min(A_proj))
```

- 最小投影面积对应 `score = 1`；
- 最大投影面积对应 `score = 0`；
- 相同面积对应相同分数和排名；
- 投影采用 `0.25 mm` 栅格，重复投影位置只计算一次。

因此当前得到的 score 是确定性代理指标。未来 MLP 要预测的是这个 score 或 `A_proj`，再检查网络预测排序与这里 oracle 排名的一致性。

## 实验配置

| 项目 | 值 |
|---|---|
| 零件 | `outputs/meshes_stl/parts/project_0558e01b77/part_10001__1.stl` |
| 材料 / 模式 | AlSi10Mg / standard |
| 角度阈值 | 45° |
| 水平孔豁免 | 严格 `<6 mm` |
| 球面方向 | 512 |
| 投影栅格 | 0.25 mm |

复现命令：

```bash
python tools/score_support_directions.py \
  outputs/meshes_stl/parts/project_0558e01b77/part_10001__1.stl \
  --material AlSi10Mg --mode standard \
  --directions 512 --projection-grid-mm 0.25 \
  --output-dir reports/support_projection_baseline_2026-11-20/part_10001_512
```

## 结果

| 指标 | 数值 |
|---|---:|
| 最小投影面积 | 163.063 mm² |
| 最大投影面积 | 9203.000 mm² |
| 中位投影面积 | 4564.344 mm² |
| 均值投影面积 | 4328.686 mm² |
| 最小 / 最大面积比 | 约 1 / 56.4 |
| 不同面积值 | 510 / 512 |
| 最低面积方向 | `S0421` |
| 第二低方向 | `S0441`，165.375 mm² |
| 第三低方向 | `S0102`，299.625 mm² |
| 最大面积方向 | `S0297` |
| 识别内圆柱 | 8 个 |
| 水平小孔豁免生效方向 | 89 / 512 |

前两个方向的面积只差约 `2.31 mm²`，第三名开始出现明显断层。这说明代理评分并不是平缓、无信息的标量场，而是由具体朝向下的面片集合形成较清晰的低成本区域。

辅助相关性：

- 投影并集面积与原悬垂表面面积的 Spearman 为 `0.941`；
- 投影并集面积与逐三角面投影面积之和的 Spearman 为 `0.970`；
- score 与投影面积的 Spearman 固定为 `-1`，这是评分公式本身决定的，不作为模型性能。

最佳方向中投影并集面积为 `163.0625 mm²`，解析三角面投影面积之和为 `162.56 mm²`。两者相差约 `0.31%`，来自 0.25 mm 栅格离散误差；在可接受范围内。

## 结论

当前量化方式已经可以生成稳定、可排序的训练标签：

- 支撑规则固定且可复现；
- query 与训练目标解耦；
- 面积越小评分越高的关系明确；
- 同一零件的 512 个方向存在明显的低成本区和较大范围的高成本区；
- 当前只验证了 oracle 几何评分，尚未验证 MLP 预测。

下一步是在全部零件上生成相同的“零件-方向-投影面积”数据，再训练候选评分网络，并重点比较 Top-K、Spearman/Kendall、NDCG 和相对 oracle 的面积差。

## 产物

- [`part_10001_512/README.md`](part_10001_512/README.md)
- `part_10001_512/direction_scores.csv`
- `part_10001_512/direction_scores.json`
- `part_10001_512/projected_area_sphere.png`
- `part_10001_512/projected_area_ranking.png`
- `part_10001_512/best_projected_mask.png`

代码：

- `tools/support_area_scoring.py`
- `tools/score_support_directions.py`
- `tests/test_support_area_scoring.py`
