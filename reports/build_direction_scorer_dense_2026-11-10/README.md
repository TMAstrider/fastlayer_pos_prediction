# 打分器实验 5：点密度加倍（8192 池，抽取 4096）（2026-11-10）

状态：完成。**结论：不是点太少。分辨率加倍对指标几乎没有影响（≤20° 25% vs 28%），
瓶颈仍是标签多解 + 分数场太平。**

## 目的

当前每个形状用 4096 点池、每步抽 2048 点。问题是：角度误差降不下来，会不会是采样太稀、
小特征（窄缝、倒角、支撑接触区）没被分辨？

## 做法

1. `tools/export_dense_pointclouds.py`：从每个形状的源 STL 重新采 **8192 点**（面积加权，
   三角面法向，按已有 signed volume 修正朝外方向），写成与原池同构的目录
   `outputs/shape_deduplicated_*_8192/`，1032 件，0 失败。
2. 训练仍每步抽 **4096 点**（相对原实验 2048 点，分辨率 ×2；相对池子 ×0.5）。
3. 其余与实验 3/4 一致（去中心、随机 yaw、物理特征、listwise softmax）。

训练性能（单卡 4090，24 GB）：

| 项 | 值 |
|---|---|
| 单步（batch 128 形状 × 9 候选 × 4096 点，bf16） | **92 ms** |
| 每 epoch（932 形状 / 128 = 8 步） | GPU 约 1.1 s，数据约 2.8 s，合计 **3.9 s** |
| 每次验证（100 形状 × 512 候选 × yaw2） | 约 **31 s**（占整轮时间的大部分） |
| 显存占用 | 约 13 GB |
| GPU 利用率 | 20–30%（点 MLP 受带宽限制，且验证阶段占比高） |
| 数据加载 | 16 workers 时 111 ms/批，**32 workers 时 16 ms/批**（差 7 倍） |

## 结果（100 留出件，第 5 轮权重）

| 指标 | 4096 池 / 抽 2048（实验 3） | **8192 池 / 抽 4096（本实验）** |
|---|---:|---:|
| 细化 top-1 中位 | 48.2° | 46.4° |
| 多起点 ≤10° / ≤20° | 23% / 24% | 22% / 25% |
| top-5 含 ≤20° 候选 | 52% | 52% |
| 人工方向分位（中位） | 0.0137 | **0.0107** |
| 人工进入前 10% / 前 5% | 78% / 68% | 77% / 70% |
| 距人工 3° 分数下降 | 0.0058 | 0.0069 |
| 训练 loss（末轮） | 1.02 | 1.35（仍未收敛） |

## 解读

1. **分辨率不是限制因素**：点数翻倍后所有指标都在噪声范围内（≤20° 25% vs 28%、
   top-5 52% vs 52%、人工排名 1.07% vs 1.37%）。方向决策是**全局几何属性**，
   2048 点已经足够；对照实验里金字塔用 2048 点就能到 0.46°。
2. **仍然欠拟合**：训练 loss 1.35（随机为 ln 9 ≈ 2.20），模型还没把训练集拟合到接近 0，
   说明容量/优化也还有余量，但这不是主要矛盾——加负样本（实验 4）和加点数（本实验）
   都没有改变“场太平、argmax 找不到峰”的结构。
3. **点密度真正会有用的地方**：窄缝/孔的可达性（隔离支撑）、法向质量、
   悬垂面积估计噪声（∝1/√N）。这些属于“可清除性”指标，需要的是 mesh 级光线投射，
   而不是更多表面采样点。

## 结论

- 不必再把点数往 8192 加；继续加只会更慢，指标不动。
- 4090 足够：训练 3.9 s/epoch、显存 13 GB；想再提吞吐就减少验证频率或把规范化搬到 GPU。
- 下一步应改**目标形式**（迭代式位姿修正 / 集合式标签），而不是数据密度或采样器。

## 产物与复现

```bash
python tools/export_dense_pointclouds.py --points 8192
python tools/train_build_direction_scorer.py --name build_direction_scorer_dense_2026-11-10 \
  --epochs 40 --batch-shapes 128 --workers 32 --points 4096 \
  --points-dir outputs/shape_deduplicated_pointclouds_8192 \
  --normals-dir outputs/shape_deduplicated_normals_8192 --yaw-average 2
python tools/evaluate_build_direction_scorer.py \
  --checkpoint runs/build_direction_scorer_dense_2026-11-10/best_model.pt --restarts 8 \
  --points 4096 --points-dir outputs/shape_deduplicated_pointclouds_8192 \
  --normals-dir outputs/shape_deduplicated_normals_8192
```

产物：`outputs/shape_deduplicated_pointclouds_8192/`、`runs/build_direction_scorer_dense_2026-11-10/`。
