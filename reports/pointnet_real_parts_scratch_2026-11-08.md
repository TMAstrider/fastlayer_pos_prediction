# PointNet 从头训练真实工业零件

## 实验回答

前面对比的 PointNet、从头训练 PTv3 和预训练 PTv3 三项，都是在公开的合成金字塔数据上做的。现在另跑了一次 PointNet 从头训练，只用项目自己的真实打印零件数据，没有混入合成数据。

输入是单个零件的 2048 个 XYZ 表面点，点云先居中并缩放到单位球；不输入法向。每轮从每个训练零件随机抽 16 个视图，对每个视图重新采点并施加独立的均匀 SO(3) 旋转。监督目标是同一旋转下的历史 +Z 打印方向，网络以 PointNet 的逐点 MLP、全局最大池化和三维单位向量 head 预测方向，损失为 `1 - cosine`。193 个零件沿用项目级 173 训练 / 20 验证划分。

## 结果

| 评估 | PointNet 从头训练 | 冻结 PTv3 + XYZ/法向 |
|---|---:|---:|
| 同一组 160 个验证朝向，平均角误差 | **73.32°** | 73.66° |
| 同一组 160 个验证朝向，中位角误差 | 74.15° | **66.46°** |
| PointNet 自己固定的 640 个验证视图，最佳平均误差 | **71.29°**（第 13 轮） | — |

同一组 160 个朝向的目标轴、随机旋转和采样点索引已逐项核对一致。配对评估中，PointNet 和 PTv3 的均值只差约 0.34°，这不足以支持某个架构更好的结论：两边的输入预处理不同，PTv3 另有法向输入；验证样本数也小。PointNet 自身 640 视图的验证均值在第 13 轮达到最佳，训练到第 43 轮时触发了连续 30 轮无改善的提前停止，最终训练损失仍在降，但验证没有跟着改善。

常量 +Z 方向在随机旋转验证目标上的平均误差为 88.13°，因此网络学到了部分方向规律；但 71° 左右的误差距离可用还很远，3° 阈值没有达到。真实零件上的结论与合成金字塔不同：PointNet 在合成数据上的高精度没有直接迁移到这批真实件，换 PointNet 或 PTv3 本身也没有解决真实件预测问题。

## 当前可以得出的判断

这轮实验只说明：按当前标签解释和划分，PointNet 小模型可以从随机旋转点云中学到一点信号，但尚不能准确复现单个历史打印方向。它不能据此证明 PointNet 不适合，也不能说明真实数据不存在可学习规律。当前验证集只有 20 个零件；重复件、零件族和几何对称性尚未审计，而且误差按有向轴直接计算，没有把等价对称方向折算。历史网格坐标的 +Z 也仍需与 FastLayer 项目往返核验。因此在解释误差前，这些都是需要补齐的数据与标签检查。

运行时长约 120 秒，最大 CUDA 显存占用约 1.68 GiB。检查曲线见 [validation_trend.png](../runs/pointnet_real_parts_scratch/paired_ptv3_validation/validation_trend.png)；每个验证零件的配对误差见 [per_part.csv](../runs/pointnet_real_parts_scratch/paired_ptv3_validation/per_part.csv)。模型和完整指标分别在 `runs/pointnet_real_parts_scratch/best_model.pt`、`runs/pointnet_real_parts_scratch/metrics.json`，匹配验证对比在 [metrics.json](../runs/pointnet_real_parts_scratch/paired_ptv3_validation/metrics.json)。
