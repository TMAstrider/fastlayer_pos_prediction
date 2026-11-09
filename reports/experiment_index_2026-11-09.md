# 阶段性实验索引（2026-11-09）

## 到目前为止完成的工作

1. **数据解析与整理**：只读审计 PJF 工程，识别 193 个主零件、234 个支撑候选和 11 个待复核 Box；导出 STL、4096 点表面采样和法向，并保留工程/零件索引。原始数据、导出网格及点云在本机 `datasets/`、`outputs/`，不进入 Git。
2. **任务定义与公开资料**：整理公开论文和可迁移方法；将第一阶段定义为从单个任意朝向零件预测打印构建方向（up-axis），yaw 暂不预测。公开金字塔数据验证了旋转标签和方向损失的实现。
3. **PTv3 真实件实验**：使用预训练 Utonia Tiny PTv3 冻结编码器训练方向 Head；实现法向/朝向覆盖消融、批量特征提取加速，以及单 STL 姿态预测导出。
4. **PointNet 真实件实验**：在同一初步 173/20 零件划分上用 XYZ 从头训练 PointNet；以同一批验证朝向和标签与之前冻结 PTv3 作配对比较。
5. **可视化**：生成 20 个验证零件的随机输入、历史 GT 和预测姿态三栏预览；红箭头标记输入坐标系的 +Z，并固定在零件上随姿态旋转。预览网页和图片在本机 `outputs/validation_pose_comparison/`，不进入 Git。

## 实验结果

| 实验 | 数据与模型 | 主要结果 | 详细记录 |
| --- | --- | --- | --- |
| PJF / 网格 / 点云 | 18 个工程；193 主零件、234 支撑候选、11 Box 候选 | 网格提取、STL 回读校验、4096 点/零件与法向导出已完成；168/193 网格通过闭合检查 | [README 数据核查与脚本](../README.md#数据初步核查)、本机 `reports/dataset_audit.json`、`tools/audit_training_meshes.py` |
| PTv3 首轮真实件 | 冻结 PTv3，XYZ，173/20 | 最佳验证平均方向误差 **80.205°**（epoch 23） | [首轮记录](ptv3_first_training_2026-11-08.md) |
| PTv3 法向/旋转消融 | 冻结 PTv3，16/64 训练朝向，有/无采样法向，173/20 | 最佳均值 **77.09°、75.03°、77.66°、73.66°**；当前最佳为 64 朝向+法向，但单种子、小验证集，朝向数也改变每轮更新数 | [消融结果](ptv3_normals_rotation_ablation/README.md) |
| PointNet 真实件从头训练 | XYZ、2048 点、173/20，16 个新旋转/训练零件/epoch | 640 个固定验证视图最佳平均 **71.29°**（epoch 13）；同一 160 朝向上 PointNet **73.32°**，PTv3+法向 **73.66°** | [PointNet 真实件记录](pointnet_real_parts_scratch_2026-11-08.md) |
| 公开金字塔方向验证 | 10,000 个合成金字塔；PointNet 式网络 | 单独复现运行测试均值 **1.331°**，98.5% 在 3° 内 | [公开数据复现](public_pyramids_reproduction_2026-11-08.md) |
| 公开金字塔模型比较 | 同一数据及官方划分；PointNet scratch、PTv3 scratch、PTv3 全量微调 | 测试均值分别 **0.71°、2.00°、2.19°** | [模型比较](public_pyramids_model_comparison_2026-11-08.md) |

公开数据中的两个 PointNet 数值来自不同实现/训练运行，不能混作同一 checkpoint 的结果。合成几何结果不代表真实工业件表现。真实件尚未达到 3°；当前验证划分也不是独立最终测试集。

## 关键限制与下一阶段检查

- 历史模型原始坐标中的 +Z 是否确实对应 FastLayer 的构建向上方向，仍要通过 FastLayer 导入/导出的小零件往返验证。
- 193 个零件尚未完成几何去重、零件族和对称等价方向审计；当前验证集只有 20 个零件。
- 所有现有真实件方向指标都依赖上述标签假设，且没有按对称等价方向修正；不能解释为可打印率。
- 当前实验只预测 up-axis，没有预测 yaw、支撑结构或真实失败风险。推荐姿态仍需 FastLayer 与工艺员核验。

建议下一步先做 FastLayer 坐标闭环和重复/对称件审计，再决定是重做标签、调整按零件族划分，还是追加模型实验。当前结果表明只换 PointNet/PTv3 编码器没有让真实件角误差降到可用范围。

## 产物和版本管理

本阶段提交包含可复现脚本、依赖清单、README、本目录下实验摘要，以及 [22 项小型实验记录快照](experiment_records/2026-11-09/manifest.json)（配置、完整逐 epoch 指标、固定真实件划分、逐件配对结果、小型曲线、预览元数据）。快照保留相对路径并记录 SHA256，后续运行覆盖本机 `runs/` 时不会丢失这次提交对应的结果。原始工程、数据集下载、导出网格/点云、训练 checkpoint、缓存和验证渲染网页在本机存在，但由 `.gitignore` 排除；清单见 [`.gitignore`](../.gitignore)。具体本机位置：

- 点云与 manifest：`outputs/pointclouds/`
- 网格 STL 与 manifest：`outputs/meshes_stl/`
- PTv3 训练权重、缓存和完整逐 epoch 指标：`runs/utonia_tiny_direction*/`
- PointNet checkpoint、完整逐 epoch 指标和配对验证结果：`runs/pointnet_real_parts_scratch/`
- 20 件姿态图与 HTML gallery：`outputs/validation_pose_comparison/`
- 公开数据训练与模型比较结果：`runs/public_pyramids_axis_v2/`、`runs/public_pyramids_model_comparison/`
