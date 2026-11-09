# 服务器迁移与阶段交接（2026-11-10）

本次提交保存代码、配置、训练划分、法向审计、逐轮指标、曲线、实验总结与源码快照。数据和权重仍由 `.gitignore` 排除；clone仓库不会自动获得它们。没有执行服务器部署或新的训练。

## 当前实验状态

| 实验 | 结果与范围 | 状态 |
|---|---|---|
| 八件PointNet，在线旋转＋法向 | 同件新朝向9.720°；不能当成未见形状成绩 | 完成 |
| 全量PointNet，932／100，XYZ＋法向 | 最佳验证64.214°；记录到377轮，最后完整checkpoint375 | 用户停止 |
| 八件等变网络，固定点 | 同点新旋转0.121°；换剩余点46.288° | 完成 |
| 八件等变网络，在线旧池抽点 | 原评估点1.606°；另外点子集约2.3–3.4°，仍来自4096点池 | 完成 |
| 八件等变模型，对8个未见形状 | 平均79.143°；只做前向 | 完成 |
| 全量等变网络，932／100 | 第55轮训练probe24.400°、验证72.604°；最佳验证63.955°在第1轮 | 用户停止，完整状态已保存 |
| 当前第55轮权重，从mesh独立重新采点 | 尚无完整指标，初次Windows多进程采样因内存压力中断，随后WSL服务无响应 | 未完成，待迁移后继续 |

详细结果及其比较限制见 [阶段索引](data_expansion_experiment_index_2026-11-09.md)。验证100件参与选模，并非独立最终测试集。有关标签、多解及输入充分性的分析保存在 [可学习性分析](upright_learnability_analysis_2026-11-09/README.md)；解释性判断不应当作已证实所有形状都不可预测。

## 必须单独复制的本机目录

从 `D:/proj/pos_set/` 复制到服务器仓库根目录下相同的相对位置。不要只复制Git文件，也不要把全部checkpoint提交进Git。

| 相对路径 | 用途 |
|---|---|
| `outputs/shape_deduplicated_pointclouds/` | 1032件、每件4096个原始采样点及manifest |
| `outputs/shape_deduplicated_normals/` | 对应XYZ和法向、逐件与总审计manifest |
| `runs/pointnet_shape_dedup_2026-11-09_val100/split.json` | 当前脚本读取的原932／100划分；Git中的报告也有备份 |
| `runs/equivariant_full_resampled_2026-11-09/` | 第55轮当前权重、最佳权重、config、progress、恢复训练所需状态 |
| `outputs/magics_parts_stl/` | Magics来源网格，用于从mesh独立重新采样 |
| `outputs/meshes_stl/` | 初始PJF来源网格，用于重新采样 |

网格文件的准确集合由 `outputs/shape_deduplicated_pointclouds/manifest.json` 内每条 `source_stl_file` 决定。本次核查1032个引用全部存在：1019件来自 `outputs/magics_parts_stl/`，13件来自 `outputs/meshes_stl/`。若存储允许，直接复制整个 `outputs/`、`runs/`、`datasets/` 最完整，保留原始工程可追溯性；这些目录可能很大，当前未打包。复现较早的八件实验还需 `outputs/eight_part_normals/` 和相应 `runs/` 子目录，完整复制可覆盖它们。

**当前权重核对**：`runs/equivariant_full_resampled_2026-11-09/last_model.pt` 应为第55轮，SHA256：

```text
858c237e09bbe570bc8c37a4de8d791cdbb46f64d989d5ca80d6fd2a987075ae
```

训练最佳、验证最佳、当前last权重含义不同；本次重新采点诊断使用last，不能替换成第1轮验证最佳。

## Linux环境

当前等变模型是轻量PyTorch实现，不依赖PTv3、spconv、Utonia或预训练权重。建议独立Python3.12环境；本机训练版本为PyTorch2.8.0+cu128。下例CUDA包需与服务器驱动匹配，已有可用PyTorch时直接复用：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt matplotlib pillow trimesh rtree
python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
export WANDB_MODE=disabled
```

不要复制Windows或WSL的虚拟环境到服务器。更早PTv3实验的依赖另见 `requirements-training.txt` 和 `tools/setup_training.sh`，当前诊断无需安装那套完整环境。

## 先完成当前权重的重新采点评测

准备的 `tools/run_fresh_points_audit.sh` 已去掉固定 `/home/panxin` 路径，可使用激活的环境，也可显式指定：

```bash
FASTLAYER_TRAIN_ENV="$PWD/.venv" bash tools/run_fresh_points_audit.sh
```

它使用systemd用户scope限制**本次任务CPU内存4GiB、swap256MiB**，检查有效cgroup限制后才继续；要求cgroup v2、systemd user会话及内存controller权限。不满足时会退出；不要去掉检查后无上限运行。该限制不覆盖CUDA显存或其他任务。在不支持systemd用户scope的服务器上，应通过容器或作业调度器提供同等内存上限，再适配启动器；当前未验证此类环境。

采样器单进程、每次最多读取32768个三角面，文件哈希分块计算，逐件写审计JSON支持恢复。不加载整个大网格到trimesh。Windows中断残留的983个NPZ没有完整manifest、新采样器审计，不算完成；恢复时将重生成。若未复制这批残留，也可直接从空目录生成。

评测固定第55轮权重，依次比较原评估点、旧4096点池三次抽点、mesh三次独立采样；中心、尺度、旋转和有向GT保持一致。覆盖932训练件和100验证件，不做反向传播。结果将写入 `reports/equivariant_full_fresh_points_2026-11-09/`，预测数组在同名 `runs/` 目录。

启动器仅做过bash语法检查；流式采样器及整套诊断尚未完成运行验证，不能声称迁移后必然无故障。此阶段建议先完成诊断，再决定是否重新长训。

## 可选：恢复训练或从头训练

用户目前要求停止训练，因此以下仅为迁移后的操作说明，并未执行：

```bash
# 使用复制的第55轮完整checkpoint，恢复原600轮上限
python -u tools/train_full_equivariant.py --resume

# 新实验需不同目录名，避免覆盖已记录实验
python -u tools/train_full_equivariant.py --name equivariant_server_run --epochs 600
```

训练会在GPU加载点池，batch32、每次2048点XYZ＋法向、在线SO(3)。若资源不足先调整batch并用新实验名，恢复已有run时脚本要求batch等设置一致。重新训练需要另行采用上述系统或作业资源限制。

manifest中的Windows反斜杠路径，流式采样器已转换为Linux分隔符。更早的法向重建/渲染脚本仍可能直接拼接旧路径；本次迁移优先复制已有法向缓存，不宣称所有历史入口都已完成Linux移植。报告中的历史绝对路径只用于溯源，不需要按原D盘或WSL用户名创建目录。
