# 支撑投影面积最小排序 MLP 蓝图

更新日期：2026-11-20

## 目标

在球面上采样候选构建方向，用可替换的代理支撑生成器计算每个方向的支撑投影面积，再训练 MLP/PointNet 学习候选方向的评分和排序。推理时，从网络高分方向开始，用用户 query 检查禁支撑区域，返回第一个无违规的高分位姿。

```text
训练：
零件 → 球面方向 → 代理支撑 → 投影面积 → 面积/排序标签 → MLP 评分器

推理：
零件 + query → 球面方向 → MLP 评分排序 → query 过滤 → 推荐位姿
```

## 固定方案

1. 支撑生成器是可替换模块。
2. 训练阶段只使用角度阈值和 `<6 mm` 水平孔豁免。
3. 训练阶段不考虑用户 query，也不回归人工历史姿态。
4. 每个候选方向计算一个支撑投影面积，面积越小评分越高。
5. 网络输入为零件几何和候选方向，输出该方向的面积、代价或分数。
6. query 只在推理阶段作为几何硬约束使用。
7. 网络按分数从高到低推荐；方向违反 query 时舍弃并检查下一名。
8. 低阈值30°评分作为显式锚点，网络优先学习相对锚点的残差。

## 代理支撑与评分

对候选构建方向 `u`：

1. 将零件旋转到 `u = +Z`。
2. 按外法向筛选角度不超过阈值的向下悬垂面。
3. 水平内孔直径严格小于 `6 mm` 时豁免。
4. 生成简化代理支撑。
5. 将保留的支撑区域投影到构建平面，去除重叠后得到投影面积：

```text
A_proj(u) = projected_support_area(u)
cost(u) = A_proj(u)
score(u) = -cost(u)
```

第一版只优化投影面积。支撑柱数量、体积、打印高度等可以记录，但不进入主训练目标。

## 训练数据

每个样本是“一个零件 + 一个候选方向”：

```text
shape_id
split
build_up
angle_threshold_deg
hole_diameter_threshold_mm
support_projection_area_mm2
normalized_cost
```

标签由代理支撑算法自动生成：

- 所有零件使用相同的球面候选集合；
- 训练标签中不包含 query；
- 同一零件按投影面积生成 oracle 排名；
- 相同面积视为并列；
- 划分按独立形状或零件族进行，避免旋转增强造成数据泄漏。

## 网络与训练

- 输入：采样点、法向、候选构建方向。
- 模型：复用现有 PointNet/MLP 候选评分器结构。
- 输出：候选方向的预测面积、代价或分数。
- 损失：弱回归损失 + listwise ranking loss + 同一零件候选之间的pairwise
  ranking loss + 45°目标Top-K候选对随机负例的损失。
- 排序约束：`A_proj(i) < A_proj(j)` 时，`score(i) > score(j)`。

30°锚点后的评分为：

```text
final_score =
    anchor_weight × score_from_lower_is_better(projected_area_30)
    + residual_scale × network_residual(shape, build_direction)
```

启用anchor时，残差打分头最后一层零初始化，使训练前输出严格等于30°
规则分数。45°目标只进入损失和评价，不进入输入。

现有 MLP 以人工历史方向为正样本，属于旧任务。新训练器需要替换标签、负样本构造和评价指标，复用编码器与球面候选框架。

## Query 过滤

网络先给出完整候选排名，然后从高到低检查：

- 支撑源面是否进入禁支撑区域；
- 支撑落点或支撑柱是否进入禁支撑区域；
- 是否侵入需要保护的孔及 clearance volume。

违反 query 的方向直接舍弃，继续检查下一名。若所有方向均违规，明确返回“无可行方向”，并保留最小违规候选用于诊断。

## 评价

主要验证网络能否复现代理几何评分：

- 投影面积回归 MAE / RMSE；
- Spearman、Kendall 排序相关；
- NDCG@5、NDCG@10；
- Top-1 与 oracle 最小面积的差距；
- Top-3/Top-5/Top-10/Top-20 regret；
- Top-K最小面积/oracle面积比；
- Top-K内面积误差`≤5%`、`≤10%`、`≤20%`的零件比例；
- 不同输入旋转下的评分和排名稳定性；
- query 过滤前后的排名、面积损失和推理耗时。

人工历史姿态可以作为外部对照，但不是本任务的训练真值。

## 实施顺序

1. 在简化支撑结果中增加统一的 `support_projection_area_mm2`。
2. 对全部零件和球面候选生成无 query 的面积标签与 oracle 排名。
3. 运行确定性穷举基线，确认最佳方向和面积分布。
4. 训练候选评分 MLP，并比较回归、排序和 Top-K 结果。
5. 接入 query 后过滤，输出首个可行位姿。
6. 公司接口可用后替换支撑后端，并重新生成标签和训练模型。

## 当前进度

- 2026-11-20：已实现角度筛选、小孔豁免和栅格化投影并集面积评估。
- 2026-11-20：已在零件 10001 上完成 512 个球面方向的确定性评分，结果见
  `reports/support_projection_baseline_2026-11-20/`。
- 2026-11-20：已为 193 个零件生成每件 256 个球面方向的投影面积标签，
  结果见 `labels/projection_support_area_v0_256/`。
- 2026-11-20：已完成候选评分 MLP 首轮实验。按项目隔离为
  171 件参数更新、12 件 development、10 件 test；test 不参与训练或早停。
- 选定的锚点 MLP 加几何排名融合，在 test 上的 Spearman 为 0.9656，
  Top-1 面积 / oracle 面积中位数为 1.0003；简单保留悬垂面积基线分别为
  0.9610 和 1.0062。基线仍在 NDCG@5 上更强，第一版保留两种分数作为
  `MLP` 与 `fallback`，实验见
  `reports/projection_score_mlp_2026-11-20_experiment/`。
- 2026-11-20：已为 1032 件、每件 256 个方向生成独立的 30°候选几何特征。
  特征不使用小孔豁免，也不读取 45°目标数组；包含悬垂面积、角度统计、
  投影支撑面积、凸包投影面积及 7 个旋转不变全局形状量。
- 2026-11-20：完成 4096 点 PointNet＋30°先验实验。30°先验使 Spearman
  均值从 0.7386 提高到 0.7762，NDCG@5 从 0.8087 提高到 0.8430，但
  Top-1 面积中位数基本不变；30°规则直接排序仍有更高的 Top-5 命中。
  实验见
  `reports/pointnet_geometry_prior_30deg_4096_expanded_932_100_2026-11-20.md`。
- 下一步：以 30°排序为锚点训练残差排序器，并改用项目隔离 test 验证新零件族泛化。
- 2026-11-20：已实现显式30° anchor residual、listwise loss、Top-K pairwise loss，
  以及Top-3/5/10/20 regret和面积容差指标。重新按完整源项目生成
  922件参数更新、10件development、100件test的严格隔离划分，项目交集为0。
- 2026-11-20：在严格划分上，无anchor模型的Spearman中位数为0.7979，
  anchor残差为0.8145；叠加平衡listwise/Top-K损失后，Top-5 regret从
  0.0561降到0.0271，Top-10 regret降到0.0096。实验见
  `reports/pointnet_30deg_anchor_residual_2026-11-20.md`。
- 下一步：在未参与本轮λ扫描的新development项目上预注册并验证残差尺度，
  同时尝试保留30°直接排序候选的混合候选集。
- 下一步：接入 query 后过滤，在真实用户 query 上验证禁支撑区命中率。

## 复用代码

- `tools/generate_simplified_supports.py`：角度筛面、小孔豁免、支撑代理和 query 检查；
- `tools/recognize_part_features.py`：内圆柱识别与 query 解析；
- `tools/search_support_orientations.py`：球面方向搜索；
- `tools/train_build_direction_scorer.py`：候选评分网络框架；
- `configs/support_rules_v0.json`：工艺规则；
- `configs/support_query.example.json`：query 输入格式。

## 当前不做

- query 条件训练；
- 回归人工历史位姿；
- 复制 Magics/FastLayer 完整支撑算法；
- 热应力、残余变形或真实可拆性仿真；
- 将代理投影面积等同于真实支撑体积。
