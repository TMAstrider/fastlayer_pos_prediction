# Anchor 模型 Top-10 位姿与支撑面高亮

对上一轮的 4 个测试零件，每个零件取平衡版 anchor + listwise/Top-K 模型评分最高的
10 个候选方向。每个位姿用两个视角渲染，并按参与训练标签完全相同的 AlSi10Mg/standard
45° 规则高亮保留下来的向下悬垂候选面。橙色面就是简化代理计入支撑投影的位置。

| 零件 | Top-1 面积/合成 GT | Top-10 最小 | Top-10 中位数 | Top-10 最大 | 合成 GT 的模型名次 |
|---|---:|---:|---:|---:|---:|
| 285 | 1.09× | 1.00× | 1.56× | 2.37× | 3 |
| 822 | 1.00× | 1.00× | 1.19× | 1.87× | 1 |
| 130 | 1.02× | 1.02× | 1.89× | 3.26× | Top-10 外 |
| 348 | 3.74× | 1.00× | 4.19× | 7.02× | 5 |

渲染时使用 `evaluate_support_projection` 重新计算每个方向的候选面，并同时校验投影面积和
候选面数量与数据集标签一致。每个候选方向只做绕平台 Z 的显示 yaw 对齐，不改变方向、支撑面
判定或面积。

生成目录：

```text
reports/anchor_top10_support_highlight_2026-11-20/
├── index.html
├── manifest.json
├── ranking.csv
├── overview.jpg
├── part_285_top_10.jpg
├── part_822_top_10.jpg
├── part_130_top_10.jpg
├── part_348_top_10.jpg
├── images/
└── anchor_top10_support_faces.zip
```

复现命令：

```bash
python tools/render_anchor_top10_support_faces.py --workers 4
```
