# 当前权重从mesh重新采样：中断与恢复准备（2026-11-09）

训练按用户要求停止在完整第55轮，last_model.pt保留当前权重及优化器等状态。训练probe24.3998°、验证72.6043°。当前权重SHA256为858c237e09bbe570bc8c37a4de8d791cdbb46f64d989d5ca80d6fd2a987075ae。本轮诊断尚未完成，不能报告重新采点后的准确率。

初次使用Windows原生Python三进程加载完整trimesh，用户报告机器内存压力严重、几乎卡死。该批任务已停止；目录中残留983个NPZ，没有完成manifest，也没有产生任何前向评测结果。这些残留文件不能作为已完成实验使用。

用户要求后续在WSL中执行并限制内存。脚本已改为单进程，按32768个三角面块顺序读取binary STL，文件哈希按1MiB块计算，不再完整展开网格；每完成一件保存独立审计JSON，可恢复。旧NPZ缺少新采样器审计，不会被直接认作完成，将重生成。源STL哈希与既有法向审计核对，沿用同一法向翻转策略。三个独立点集各2048点，中心和尺度保持训练时原4096点池的值。

准备的WSL启动脚本 tools/run_fresh_points_audit.sh 使用systemd用户scope，MemoryMax=4GiB、MemorySwapMax=256MiB，检查实际cgroup memory.max后才运行；若不能创建scope或限制无效，会退出，不降级到无上限运行。此上限约束本次任务的CPU内存，不是整台WSL或CUDA显存。线程数均为1。启动脚本bash -n语法检查通过，但尚未在WSL验证，因为当前WSL启动、状态、终止及shutdown命令都无响应；Restart-Service WslService因当前会话权限不足而失败。已结束本次挂起的wsl.exe调用，没有继续运行本次Python训练或采样。

恢复WSL后，在项目目录运行：

```bash
bash tools/run_fresh_points_audit.sh
```

诊断将固定第55轮last_model.pt，并核对前后checkpoint哈希，比较原评估点、旧点池三次重抽、mesh三次独立采样。全部932训练件和100验证件使用停止训练时同一旋转。实验尚未重训，也未改权重。需要先恢复WSL服务，再实际验证启动脚本、完成采样和纯前向。
