# 工业零件打印位姿推荐：公开工作与可迁移方法调研

调研日期：2026-11-08。范围：单零件、任意初始朝向、金属激光粉末床熔融（LPBF/SLM），目标是从已打印验证的历史朝向中推荐可用姿态。优先核对论文原文、作者代码仓库及公开数据；没有将搜索摘要中的速度或精度数字直接视为项目可达到的效果。

## 结论

1. **最相近的公开论文就是 PointNet + 随机旋转监督。** Zelder 等（2025）用 PointNet 从点云预测旋转，以回到参考打印朝向，并使用随机初始朝向和已知逆旋转标签。我们的“已打印验证网格 + 任意旋转输入 + 回到历史朝向”与其问题定义高度一致。这篇论文应作为神经基线的首要参考，但其验证主要是规则化合成几何，并非复杂真实工业件；其 0.7°/1.26° 等结果不能外推到当前数据。
2. **没有在本次检索中找到一个可以直接替代 FastLayer、已广泛验证的金属 LPBF 位姿推荐开源包。** 公开成果更多是可组合的算法：面片聚类生成候选、支撑/悬垂几何评分、PSO 或强化学习优化、PointNet 姿态回归。通用 FDM 切片器可参考实现，不能直接作为金属 LPBF 的工艺决策器。
3. **第一版建议采用“几何候选 + 学习排序”，PointNet 作为对比基线。** 几何模块能给出每个候选为什么被选中，历史零件用于校准和排序。约 193 个主零件候选经过族去重后，形状数仍有限，不适合把效果押在端到端大模型上。论文的 PointNet 路线仍值得做，因为复现成本低、输入管线已存在。
4. **姿态标签应先定义为构建方向（up-axis），而不是默认完整 SO(3) 欧拉角。** 当绕竖直轴旋转不改变支撑与打印风险时，可将 yaw 留给平台约束或历史姿态。若长方平台、气流方向或历史摆放经验要求特定 yaw，再单独预测/优化这一自由度。对称件要按等价旋转或工艺评分评估，不能只看角度误差。

## 一、最直接相关的论文

### 1. Zelder et al., 2025：从零件数据学习金属 PBF 朝向

- **论文：** [Learning automatic part orientation for powder bed fusion of metal from part data](https://doi.org/10.1007/s40964-025-00960-6)，[公开 PDF](https://d-nb.info/1365324745/34)，Progress in Additive Manufacturing 10, 1713–1719。
- **任务与输入：** 三角网格表面采样点云，每份 2,048 点并归一化至单位球；PointNet 回归旋转表示，将任意初始朝向恢复为参考朝向。论文明确采用随机朝向几何和逆变换标签，这与本项目的监督样本构造方式一致。
- **主要发现：** 比较旋转矩阵、四元数、6D 连续表示与 z-invariant 表示。z-invariant 忽略绕竖直轴的旋转，在几何存在旋转对称时避免把等价解当成错误。论文报告的混合数据集 MARE 为 1.26°，但数据是参数化合成的规则几何；作者明确指出真实生产件、多个可行姿态和不同操作员风格仍待研究。
- **公开资源：** [Hugging Face 数据集](https://huggingface.co/datasets/sebius/pbflbm-part-orientation)，Apache-2.0，约 4.29 GB，包含 5 个压缩包，STL 与随机旋转/逆变换 JSON 标签。可用于先打通训练管线和回归测试。当前 Hugging Face 数据预览存在列结构不一致错误，应直接下载对应 ZIP 读取，不依赖网页预览。
- **开源状态：** 本次通过论文正文、标题检索和 GitHub 搜索，**未确认作者公开训练代码或模型权重**。只能按论文复现，不能说已有可运行实现。
- **对本项目的价值：** 最高。先复现一个小 PointNet，输出构建轴或 z-invariant 旋转，再与本项目真实复杂件的历史姿态做族外测试。不要把合成数据训练结果当作工业可用证明。

### 2. Qin et al., 2020–2021：面片聚类候选 + 多指标选择

- **论文：** [Automatic generation of alternative build orientations for LPBF based on facet clustering](https://doi.org/10.1080/17452759.2020.1756086)（2020）与 [Automatic determination of part build orientation for LPBF](https://doi.org/10.1080/17452759.2020.1832793)（2021，[公开 PDF](https://pure.hud.ac.uk/files/27776800/Automatic_determination_of_part_build_orientation_for_laser_powder_bed_fusion.pdf)）。
- **方法：** 按面片法向层次聚类，组合和细化少量候选朝向；再对支撑体积、体积误差、表面粗糙度、打印时间和成本估算、归一化、加权排序。输入可包含材料、层厚、重涂时间、扫描间距和扫描速度。
- **对本项目的价值：** 几何基线的主框架。已有 PJF 网格和工艺配置可以提供输入，候选可以保留 Pareto 前沿而非强行只有一个“最优”。评价公式来自特定论文和工艺条件，尤其是表面粗糙度模型不能直接换材料/层厚使用。
- **开源状态：** 未确认官方代码；支撑体积示例使用 Autodesk Meshmixer，这不是可直接集成的开源 LPBF 支撑引擎。

### 3. Sheng et al., 2023：多特征机械件 SLM 多目标决策

- **论文：** [Build orientation determination of multi-feature mechanical parts in selective laser melting via multi-objective decision making](https://doi.org/10.1007/s11465-022-0737-8)。
- **价值：** 问题对象与我们的工业机械件更接近，强调按特征组及加工精度需求进行多目标决策，而非只最小化总支撑。提示后续应允许工艺员标注关键面、禁止支撑面或局部优先级；当前 PJF 若缺少这些标注，不能假装仅凭网格恢复全部工艺意图。

### 4. Eranpurwala et al., 2020：机械加工特征 + 深度学习

- **论文：** [Predicting Build Orientation of Additively Manufactured Parts With Mechanical Machining Features Using Deep Learning](https://researchconnect.buffalo.edu/en/publications/predicting-build-orientation-of-additively-manufactured-parts-wit/)。
- **方法线索：** 分类与回归组合，把机械加工特征与由支撑体积优化得到的朝向角关联；公开介绍提及约 1,000 个体素化 STL 样本。
- **价值：** 说明“先有几何规则构造标签，再学习加速/泛化”的路线已有先例。但它学习的是特定规则最优解，不等于真实打印验证；本项目更有价值的监督信号是工艺员实际采用且打印成功的姿态。

### 5. Shi et al., 2023：可泛化强化学习 Learn to Rotate

- **论文：** [Learn to rotate: Part orientation for reducing support volume via generalizable reinforcement learning](https://doi.org/10.1109/TII.2023.3249751)，[作者机构页面](https://research.manchester.ac.uk/en/publications/learn-to-rotate-part-orientation-for-reducing-support-volume-via-/)。
- **方法：** 将连续试旋转视为序贯决策，通过 GPU 加速环境离线学习搜索策略，以支撑体积为奖励，目标是减少在线搜索成本并改善局部最优。论文报告相对随机搜索加速，但依赖其支撑估计环境与基准。
- **对本项目的价值：** 若以后 FastLayer 提供可批量调用的支撑评估 API，可将其作为离线环境；目前不建议作为第一版，因为奖励函数、训练环境和真实 LPBF 工艺还有差距。
- **开源状态：** 本次未确认作者官方公开仓库；机构论文页面不代表代码已发布。

### 6. Schmalzel et al., 2026：真实 SLM 牙科生产数据的 up-axis 学习

- **预印本：** [Predicting build orientation for SLM dental parts](https://arxiv.org/abs/2609.15710)，2026-09-14，当前应作为预印本而非已完成同行评审的结论引用。
- **方法：** 约 2,400 个由技师标注的生产件，比较 ResNet-50 多视图与 PointNeXt-S 点云骨干，重点预测 S² 上的构建轴；对已知旋转做测试时增强并聚合结果。
- **启示：** 真实生产数据可以采用“形状 → up-axis”的监督定义；多视图可能优于点云，测试时旋转集成也是便宜有效的稳健性手段。牙科形状和工艺约束与工业件不同，数值结果不可直接比较。

## 二、可复用的开源项目与资源

| 项目 | 能借鉴什么 | 不能直接解决什么 | 状态/许可核查 |
| --- | --- | --- | --- |
| [OrcaSlicer](https://github.com/OrcaSlicer/OrcaSlicer) | `src/libslic3r/Orient.cpp` 中有法向/面积候选、凸包、底部接触、悬垂、轮廓、投影面积与 unprintability 评分，是候选筛选的实用参考 | FDM 权重、支撑规则和冷却假设不等于 LPBF；不宜原样给工业金属件下结论 | 活跃 C++ 项目，AGPL-3.0；复制/集成代码前评估许可，独立实现算法思想更灵活 |
| [Hugging Face pbflbm-part-orientation](https://huggingface.co/datasets/sebius/pbflbm-part-orientation) | 现成 STL + 旋转标签，可先验证点云采样、旋转逆变换和训练代码 | 合成规则几何，不代表复杂工业零件 | Apache-2.0，5 个 ZIP，约 4.29 GB；与 2025 PointNet 论文配套 |
| [Hitchhiking Rotations](https://github.com/martius-lab/hitchhiking-rotations) | PyTorch 旋转表示、损失、点云到姿态训练示例；论文建议完整 SO(3) 回归使用 R9+SVD 或 R6+GSO | 通用旋转学习，不包含 LPBF 工艺评分 | [ICML 2024 论文](https://proceedings.mlr.press/v235/geist24a.html)与作者代码均有公开页面；本次未逐项核查依赖及子模块许可 |
| [Vector Neurons](https://github.com/FlyingGiraffe/vnn) | SO(3) 等变点云网络，VN-PointNet/VN-DGCNN；用于让预测随输入旋转一致变化 | 不能直接输出“可打印姿态”，仍需历史标签和旋转目标设计 | [ICCV 2021 论文](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html)及作者实现；后续实验再用 |
| [Orient Anything](https://github.com/SpatialVision/Orient-Anything) | 多视图/视觉姿态预测、预训练模型、测试时集成；V2 还处理旋转对称与相对旋转 | 它学习日常物体的“正立/前方”语义，不是金属打印方向，预训练结果不能直接采用 | 作者代码与模型公开；仓库 GitHub 元数据标为 CC-BY-4.0，模型/数据使用还应分别核对条款 |
| [AMVER 论文](https://doi.org/10.1007/s00170-024-14039-z) | FreeCAD + 改进 PSO、射线法近似支撑体积；论文给出悬垂面积、底面接触、层截面积等指标 | 验证以 FDM 案例为主，PSO 平均运行时间仍较长；不是已找到的公共开源仓库 | 论文说明有 FreeCAD Python 实现，但本次未找到官方代码地址，故标为方法参考而非可安装软件 |
| [PartOR_fused_ShapeOptimization](https://github.com/JosephJay95/PartOR_fused_ShapeOptimization) | 可微分/梯度式朝向与形状联合优化的 MATLAB 代码 | 面向挤出打印和形状优化，不需要修改零件形状的当前任务不宜引入 | 作者仓库公开，CC0；依赖 MATLAB 与 gptoolbox |

补充：GPU 凸包方法 [Wade et al., SCF 2023](https://doi.org/10.1145/3623264.3624447)（[作者页面与 PDF](https://matterassembly.org/publications/wade2023determining/)）可用于加速“支撑较少且首层接触稳定”的候选筛选。但其论文明确面向 FFF 的附着与支撑目标，凸包也会忽略凹腔和内部通道，不能独立用于复杂 LPBF 件。本次未确认官方代码仓库。

## 三、其他领域最值得迁移的方法

### 1. 姿态回归：预测构建轴，再决定是否预测 yaw

完整旋转属于 SO(3)，普通欧拉角 MSE 会受周期性和表示不连续影响。可先让网络输出零件坐标系中的构建方向 `u∈S²`，使用球面角度损失；再通过候选 yaw 的几何评分或平台约束确定完整姿态。必须完整 SO(3) 时，采用 6D 连续表示 + Gram–Schmidt，或 9D + SVD，并评估 geodesic 角误差，而不是直接回归三个独立欧拉角。

注意：当前点云已在历史已验证朝向中，对历史网格随机旋转 `A` 后，恢复历史朝向的监督为 `A⁻¹`。若训练时把点云统一归一化/旋转到 PCA 坐标系，必须将同一条变换链记录下来并正确反算标签。中心化可去掉无意义平移，但不能把朝向信息归一化掉。

### 2. SO(3) 等变网络与测试时旋转集成

普通 PointNet 不保证输入旋转后输出按相同规律变化。Vector Neurons 等 SO(3) 等变结构适合这个归纳偏置；它们是值得比较的第二阶段模型，不是首要工程依赖。更便宜的基线是：训练时做均匀 SO(3) 增强；测试时对多个已知旋转的输入分别预测，将输出转回同一坐标系后做球面均值/投票。2026 牙科预印本也显示测试时增强有稳定收益。

旋转不变描述子（PCA 比例、点间距离、形状分布）适合做相似零件检索或零件族划分，但**单靠旋转不变特征不能恢复完整朝向**。检索姿态时要保留相似件在其坐标系中的构建方向，再做坐标对齐。

### 3. 机器人抓取/姿态推荐：学习“候选评分”而非只回归唯一角度

打印朝向与抓取位姿相似：同一物体常有多个可行解。可以生成一组候选，由点云网络编码形状，再编码候选构建轴，输出该候选的接受分数或工艺代价。这样更贴近 Top-k 推荐，也允许工艺员选择。历史已验证朝向是正样本；未打印候选不能自动标负，需要几何弱标签、工艺员成对比较或后续实际反馈。

### 4. 学习排序与多目标优化

将历史位姿视为“工艺员优先选择”，比较历史位姿与几何候选的特征差，训练轻量 pairwise ranking 模型。先按平台尺寸、关键面约束等硬条件过滤，再保留支撑需求、构建高度、孤岛/桥接等指标互不支配的 Pareto 候选。相比全局端到端角度回归，这条路线更适合少量、异构、含重复族的真实数据，也更容易向工艺员解释。

### 5. 热变形/风险：后续作为独立复核模块

[2025 LPBF 朝向与热变形联合框架](https://pmc.ncbi.nlm.nih.gov/articles/PMC12654590/)再次说明支撑、时间、表面质量与变形存在取舍。此类仿真可用于少量高分候选的复核，不应让粗略几何代理直接输出“打印失败概率”。历史全部为已验证成功姿态，不能单独提供失败类别或概率校准所需的数据。

## 四、对当前项目的推荐实验

### 优先级 A：可解释的几何基线

1. 用已导出的主零件三角网格计算面片面积/法向，按面片聚类、主轴和球面采样生成候选构建轴；去除近似重复候选。
2. 对每个候选计算构建高度、平台占用、底部接触、悬垂面积与投影面积；第二阶段用切片/射线补充孤岛、桥接和支撑包络代理量。
3. 在历史朝向和各随机旋转输入上运行同一算法，确认推荐在旋转协变意义下稳定；比较历史姿态与几何高分姿态，找“历史工艺偏好无法由当前指标解释”的样例。
4. 输出 Top-3 候选和理由，交给 FastLayer/工艺员复核。这是后续学习排序的训练数据入口。

### 优先级 B：PointNet 直接回归基线

- 复现 Zelder 等的训练形式，先用公开小数据子集验证随机旋转与逆变换标签正确，再接本项目 4096 点点云。
- 比较 `S² up-axis` 与完整 `R6+GSO`；前者由几何搜索补 yaw，后者仅在工艺确认 yaw 有必要时使用。
- 按零件族和工程项目分组交叉验证，排除同族、镜像件、重复件和随机旋转副本泄漏；单独报告 AlSi10Mg 与 316L 的适用边界。
- 报告角度误差只是辅助指标，主指标是 Top-1/Top-3 工艺接受率、硬约束违规数和不同初始朝向下的推荐一致性。

### 优先级 C：候选排序模型

- 用 A 的候选作为输入，点云/网格编码器提供形状特征，候选方向及工艺指标作为条件。
- 以历史已验证姿态为正例，结合工艺员对候选的排序/接受反馈训练；不要把未尝试的候选都视为失败。
- 如果 PointNet 在族外测试明显弱于几何基线，应优先发展这个方向，而不是继续加深端到端网络。

## 五、明确的研究空白，也是本项目可形成的贡献

- 2025 PointNet 论文的作者也指出真实复杂生产件、多可行姿态、操作员差异仍待验证。当前数据正好提供“已打印验证的真实工业件姿态”，可检验合成几何结论能否迁移。
- 可研究 **SO(3) 等变/旋转集成 + 真实工业件 up-axis 学习**，以及 **历史工艺偏好约束下的候选排序**，而不是再做一个仅最小化悬垂面积的演示器。
- 当前 18 个工程、约 193 个主零件候选的规模仍小，首要证据是分组留出测试与工艺员盲评。应继续积累不同零件族的标注，尤其记录同一零件多个可接受姿态及关键面/禁支撑面，才能超越单一历史姿态模仿。

本报告只做公开资料与代码可得性调研，没有下载运行大型仓库或公开数据，也没有用当前 PJF 数据复现论文数值。开源许可栏用于初筛，正式集成前仍需核查具体文件、依赖和模型/数据条款。
