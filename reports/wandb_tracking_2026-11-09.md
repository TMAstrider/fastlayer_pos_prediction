# W&B 指标记录（2026-11-09）

用户已要求暂停 W&B，默认 `WANDB_MODE=disabled`，只使用本地 JSON、曲线与权重。此前曾创建一个补传 run，但表格上传因 403 失败，不能称为完整补传。凭据保存在训练环境的 WSL 用户登录凭据中，不写入代码、配置或 Git。

以后如需启用，项目默认名为 `fastlayer-pos-prediction`。设置 `WANDB_MODE=online` 开启在线记录，或 `WANDB_MODE=offline` 保留本地 W&B 记录供之后同步。只上传指标和实验配置，不上传零件、点云或权重，不保存代码快照。可以通过 `WANDB_PROJECT`、`WANDB_ENTITY` 切换项目或空间。

接入脚本：`tools/train_magics_pointnet.py`、`tools/diagnose_pointnet_random_so3.py`。后者每轮记录在线训练 loss，每 10 轮记录固定同件新朝向探针指标。历史实验只补传原有记录行，不补造中间轮次。

指标范围：

- `train/cosine_loss`：训练时当前批次的平均方向余弦损失。
- `same_shape_probe/*`：同一批训练形状、未参与反向传播的固定旋转；用于选模。
- `recent_train_eval/*`：近期训练输入，在 eval 模式下计算的指标。
- `independent_same_shape_512_rotations/*`：选模后，独立生成的同件随机朝向，只在最终评估一次。
- `unseen_parts_validation/*`：主训练脚本的未见零件验证，和同件新朝向分开。
- 各范围包含平均/中位/P90 角误差、3/5/10/30 度以内比例；新实验同时记录学习率及可用的余弦损失。逐件独立旋转结果保存为 W&B 表格。

历史 B 实验中，已记录最低训练 loss 为 **0.22873897（epoch 900）**，探针最低 loss 为 **0.24130765（epoch 910）**。按探针平均角误差选择的是 epoch 920，其探针 loss 为 **0.2441126**，最终独立同件朝向 loss 为 **0.2540708**、平均误差 **32.801°**。原来只每 10 轮记录一次，无法追溯没有保存的轮次或宣称这是每一轮的绝对最低值。

单样本 `loss = 1 - cos(theta)`。0.244 对应单角度约 40.9°；30° 对应约 0.134。平均余弦损失不可直接转换成平均角误差，应看实际角误差指标。

当前阶段先检查同一批零件对新 SO(3) 朝向的泛化是否足够准确，再研究新零件泛化。已有 B 实验仍为 8 个形状、固定基础点，尚未达到低误差目标；接入监控本身不改变模型或已有训练结果。

在 WSL 训练环境中补传已有实验：

```bash
python tools/wandb_tracking.py reports/pointnet_small_fit_2026-11-09 reports/pointnet_random_so3_8parts_2026-11-09 runs/pointnet_shape_dedup_2026-11-09_val100
```

每次补传的链接保存在 `runs/wandb_backfill/<实验名>/wandb_run.json`。同一目录再次运行会跳过已有导入记录。
