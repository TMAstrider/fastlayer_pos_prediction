# Magics 零件与构建方向导出

- magics_projects_scanned: 326
- stored_meshes: 7072
- part_candidate_records: 5534
- unique_part_geometries: 1644
- duplicate_part_records: 3890
- platforms_excluded: 1200
- support_boxes_excluded: 338
- projects_with_part_records: 326
- projects_without_part_records: 0
- records_inside_420mm_plate_bbox: 3746
- records_needing_review: 1788
- errors: 0
- stl_paths_verified: 5534
- hardlinks_verified: 3890
- source_content_spot_checks: 15
- geometry_screened_unique_candidates: 1044
- review_reasons: {'xy_outside_420mm_plate': 1618, 'xy_partially_outside_420mm_plate': 170, 'below_build_plate': 2}
- all_source_projects_represented: True

## 文件

- outputs/magics_parts_stl/manifest.json：完整源对象、分类和导出记录。
- gt_all.jsonl：每个工程的每条零件候选及其方向标签，保留重复记录。
- gt_index.csv：可在表格软件中查看的索引。
- gt_unique_screened_candidates.jsonl：去掉精确重复、平台范围异常和负 Z 的候选清单。
- gt_review.jsonl：需要复核的候选。

## 标签坐标

STL 保留存储坐标、毫米单位和三角形，不重新旋转、不简化。该坐标中的构建向上方向为 [0,0,1]；yaw 不监督。
这是已摆放网格自身的方向标签，不是从原始 CAD 坐标恢复的旋转矩阵；当前没有原始 CAD 到工程的变换。
训练施加主动旋转 A 时，点和法向一起旋转，方向标签设为 A@[0,0,1]；使用行向量时点写为 P@A.T。
grounding_translation_mm 仅供单件预览贴地：将最低点移至 Z=0，不改变方向标签，也不修改导出的 STL。

## 数据状态

几何筛选不是对象激活状态确认。工程头元数据当前不可读，无法逐件确认隐藏对象、原始名称及是否属于实际打印任务；
因此所有导出标签均明确记为 stored_build_direction_candidate，不能把候选数量宣称为已确认新增独立训练零件数量。
平台和 Box 排除依据几何签名；其他类型的辅助对象仍可能存在。
精确哈希包含坐标和平移，不等同于零件族识别。合并训练前仍需识别平移/旋转副本，并避免跨训练验证集泄漏。
此次仅导出和核验，原始 193 件的训练划分及模型没有变动。

## 复用

在 D:/proj/pos_set 目录运行：

```powershell
python tools/extract_magics_parts.py datasets
python tools/finalize_magics_gt.py
```

本机 Python 已有 NumPy。建议在 Windows 本机执行，避免 WSL 对 /mnt/d 中大量小文件的路径检查开销。
gt_all.jsonl 中 stl_file 相对于该 GT 文件所在目录；source_project 相对于 datasets 目录。
