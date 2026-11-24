# 历史摆放偏好 Head

## 设置

- 初始化：当前逐点方向交互面积模型
- 冻结：共享点云/方向编码器和面积 head
- 训练目标：历史朝上方向在额外候选中的 cross entropy
- history temperature：0.1
- 候选近邻监督权重：1.0
- 目标余弦温度：0.1
- seed：20261124
- 训练零件：922
- development 零件：10
- test 零件未参与训练和选模

## 结果

| 指标 | 最佳 development checkpoint |
|---|---:|
| epoch | 10 |
| exact history loss | 2.1735 |
| exact history rank 中位数 | 1.5 |
| 候选 Top-1 与历史轴夹角中位数 | 60.04° |
| 候选 Top-1 在历史轴 20° 内 | 3/10 |

训练耗时 `195.97` 秒，完成 `18` 个 epoch。

模型把真实历史方向作为第 257 个候选输入共享评分函数。融合评估需另外运行历史偏好
与面积分数的权重扫描；本文件不预先挑选 test 上的融合权重。
