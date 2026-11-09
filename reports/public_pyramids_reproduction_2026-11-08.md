# 公开金字塔数据集：z-invariant 方向回归验证

## 数据和实验范围

- 数据源：https://huggingface.co/datasets/sebius/pbflbm-part-orientation
- 论文 DOI：https://doi.org/10.1007/s40964-025-00960-6
- 下载日期：2026-11-08；固定数据版本：`24d4f3f5f05cad5488fcf9a4cbc95e28ff528a8a`。
- 本次只下载 `pyramids_10000.zip`，14,829,698 字节，不包含其余三类或混合数据包。
- SHA256：`31fd0676a4f608bc87caa0540d38da52fcaf269b7a7f99838ef9d4727c12c4f5`，与 Hugging Face LFS OID 一致。
- 数据卡声明许可为 Apache-2.0；本地数据和来源记录在 `datasets/public_pbflbm/`，由 Git 忽略。
- 使用原始 STL 划分：训练 8,100 个、验证 900 个、测试 1,000 个；不把同一几何的增强旋转重新分到不同集合。

这是论文数据集 A 的独立实现验证。论文没有充分给出网络层宽、学习率、优化器等细节，本脚本使用无 T-Net 的 PointNet 式共享 MLP 和最大池化；不能称为作者代码的精确复现，也不能把此实验当成混合数据集 1.26° 的复现。

## z-invariant 的含义

采用列向量时，用 `C` 表示从任意输入恢复到参考打印朝向的旋转。输入坐标系中的构建向上方向为 `u = C.T @ [0,0,1]`，也就是 `C` 的第三行转为列向量。此方向在单位球面上有两个自由度。

若恢复朝向后，再在平台内绕打印 Z 轴旋转 `Z(psi)`，完整旋转变为 `C' = Z(psi) @ C`。因为 `Z(psi).T @ [0,0,1] = [0,0,1]`，所以 `C'.T @ [0,0,1] = u`。这就是忽略打印坐标系 yaw 的含义。它不忽略翻面，不把 `u` 与 `-u` 视为相同方向，也不让点云对所有三维旋转都不变。

平台内转动保持构建高度、水平投影面积和按竖直方向计算的悬垂关系。平台宽长约束、排版、定向铺粉、扫描/气流等因素仍可能依赖 yaw，因此 yaw 应由后续模块或工艺员确定；不能说这个自由度在生产中永远无用。

本项目历史网格已在参考姿态，输入增强为 `p_input = A @ p_history`，恢复旋转为 `C=A.T`。现有目标 `A @ [0,0,1]` 正好等于 `C.T @ [0,0,1]`，与上述 z-invariant 方向是同一表示。现有模型使用单位化输出和余弦损失，而本次基线采用 tanh 输出和分量 MSE；模型架构及训练方式也不同。

## 公开数据的矩阵约定

数据包中的 `target_rotational_matrix = M` 使用行顶点右乘约定：`V_reference = V_input @ M`。换成本项目列向量约定，恢复矩阵是 `C=M.T`，因此正确输入方向是 **`M[:, 2]`，等价于 `C[2, :]`**。

已通过金字塔元数据的 `height` 核查：将每个网格顶点投影到该方向，高度差应恢复为元数据高度。预处理对全部 10,000 个样本执行此检查；另抽查恢复后的网格，底部为单一尖端、顶部为平面。100 个随机旋转的数值检查中，额外输出坐标系 yaw 对该方向的影响小于 `2.3e-16`。

首次试跑误把 `M[2, :]` 当成目标，训练 33 轮后验证和测试误差均约 65°。该次运行仅是坐标适配失败的调试记录，保存在 `runs/public_pyramids_z_invariant/`，不能用于论文方法的性能结论。修正后的缓存和结果单独保存在 `runs/public_pyramids_axis_v2/`。修正依据是网格与旋转的几何关系，而非在两种标签中按测试误差择优。

## 实现与运行

`tools/train_public_orientation.py` 直接从 ZIP 读取 STL 和 JSON，不需要展开 20,000 个文件。每个样本按三角面面积加权采样 2,048 个表面点，采样固定随机种子、中心化并按最大半径归一化，缓存所有样本。输入仅含 XYZ。

PointNet 式逐点通道为 `3→64→64→64→128→1024`，全局最大池化后为 `1024→512→256→3`，中间使用 BatchNorm、ReLU，回归头含 dropout，最终为 tanh。端到端训练，batch size 48，Adam 初始学习率 0.001，使用验证 MSE 调整学习率；按验证方向角均值选最佳 checkpoint。训练上限 160 epoch，25 epoch 未改善则早停，低于论文设置的 400 epoch 上限。

单位化后的预测和目标以 `acos(dot(u_pred,u_target))` 计算方向角误差。若两个向量已是单位向量，则 `||u_pred-u_target||² = 2(1-dot(u_pred,u_target))`；现有模型的余弦损失与单位向量 MSE 成比例。本次 tanh 输出不强制单位长度，因此训练目标还包含长度偏差，评估角度时才单位化。

在 WSL 中运行：

```bash
cd /mnt/d/proj/pos_set
OMP_NUM_THREADS=4 ~/envs/fastlayer-ptv3/bin/python -u \
  tools/train_public_orientation.py --epochs 160 --patience 25 --workers 2
```

下载数据可固定版本：

```powershell
Invoke-WebRequest 'https://huggingface.co/datasets/sebius/pbflbm-part-orientation/resolve/24d4f3f5f05cad5488fcf9a4cbc95e28ff528a8a/pyramids_10000.zip' -OutFile 'datasets/public_pbflbm/pyramids_10000.zip'
Get-FileHash 'datasets/public_pbflbm/pyramids_10000.zip' -Algorithm SHA256
```

## 结果

本轮完成 160 epoch，训练耗时约 919 秒（不含点云预处理）。最佳模型为 epoch 144，按验证集平均方向角选定；正式测试在选定后执行一次。结果保存于 `runs/public_pyramids_axis_v2/metrics.json`，权重为同目录 `best_model.pt`，测试预测为 `test_predictions.npz`。

| 指标 | 验证集（900 个） | 测试集（1,000 个） |
| --- | --- | --- |
| 平均方向角误差 | 1.320° | 1.331° |
| 中位方向角误差 | 1.272° | 1.296° |
| 90 分位方向角误差 | 2.128° | 2.156° |
| 1° 内比例 | 32.33% | 32.20% |
| 3° 内比例 | 98.78% | 98.50% |
| 10° 内比例 | 100% | 100% |

独立从已保存测试预测重算，平均角误差约 1.331°，最大误差约 4.388°。该结果验证了简单 PointNet 式网络可以从规则合成几何学习构建方向，但没有复现到论文表 1 的金字塔验证误差 0.7°。本次 1.331° 是金字塔测试集结果，不能与论文混合数据集 1.26° 当成同任务横向比较，也不代表真实工业件可打印率。

已完成脚本编译、102 个方向对齐（含正反向）、PointNet 点排列不变性、正反方向的角度指标检查；数据预处理对全部 10,000 个网格核验方向恢复后的高度。单一种子、单一几何类别和独立网络实现限制了结论。下一步可在其余类别上复验，或将该方向网络迁移到真实零件，在相同工程/零件族划分上与冻结 PTv3 方案比较。
