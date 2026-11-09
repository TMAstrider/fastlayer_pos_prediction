"""Validate exported Magics STL records and write portable build-direction labels.

Candidate labels describe the stored mesh frame, not an unobserved original CAD
frame. Object activity/role is not recovered from the unreadable project header.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from audit_expanded_data import MTArchive
from export_meshes import verify_stl


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize(manifest_path, dataset_root, report_path):
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document["status"] != "complete" or document["errors"]:
        raise ValueError("Extraction must finish successfully before GT export")
    root = manifest_path.parent.resolve()
    records = document["records"]
    seen_paths = set()
    rows = []
    hardlinks_verified = 0
    for record in records:
        relative = record["stl_file"]
        if relative in seen_paths:
            raise ValueError(f"Repeated output path: {relative}")
        seen_paths.add(relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"STL outside export directory: {relative}")
        expected_size = 84 + 50 * record["triangles"]
        if path.stat().st_size != expected_size:
            raise ValueError(f"Incorrect STL byte count: {relative}")
        with path.open("rb") as handle:
            handle.seek(80)
            if struct.unpack("<I", handle.read(4))[0] != record["triangles"]:
                raise ValueError(f"Incorrect STL triangle count: {relative}")
        if record["duplicate_of_stl_file"]:
            if not path.samefile(root / record["duplicate_of_stl_file"]):
                raise ValueError(f"Duplicate is not hardlinked: {relative}")
            hardlinks_verified += 1
        if record["up_axis_in_stored_mesh_frame"] != [0.0, 0.0, 1.0]:
            raise ValueError(f"Unexpected up axis: {relative}")
        rows.append({
            "sample_id": hashlib.sha256(
                f'{record["project_relative"]}/{record["member"]}'.encode("utf-8")
            ).hexdigest()[:24],
            "stl_file": relative,
            "source_project": record["project_relative"] + ".magics",
            "source_member": record["member"],
            "geometry_sha256": record["geometry_sha256"],
            "gt_up_axis": [0.0, 0.0, 1.0],
            "yaw_unconstrained": True,
            "units": "mm",
            "coordinate_frame": "stored Magics coordinates; no mesh rotation applied",
            "grounding_translation_mm": [0.0, 0.0, -record["z_min_mm"]],
            "label_status": "stored_build_direction_candidate",
            "object_activity_verified": False,
            "needs_review": record["needs_review"],
            "review_reasons": record["review_reasons"],
            "duplicate_of_stl_file": record["duplicate_of_stl_file"],
            "geometry_screen_pass": not record["needs_review"],
            "vertices": record["vertices"],
            "triangles": record["triangles"],
            "bbox_min_mm": record["bbox_min_mm"],
            "bbox_max_mm": record["bbox_max_mm"],
        })

    projects = {record["project_relative"] for record in records}
    expected_projects = {
        path.relative_to(dataset_root).with_suffix("").as_posix()
        for path in dataset_root.rglob("*.magics")
    }
    if projects != expected_projects:
        raise ValueError(f"Project coverage mismatch: {sorted(expected_projects - projects)}")
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*.stl")}
    if actual_paths != seen_paths:
        raise ValueError("Output STL files and manifest entries differ")

    # Content verification re-reads original MT archives for representatives
    # across the full collection, plus review candidates and duplicate records.
    indices = set(np.linspace(0, len(records) - 1, 12, dtype=int).tolist())
    for predicate in (
        lambda item: item["below_build_plate"],
        lambda item: item["needs_review"],
        lambda item: item["duplicate_of_stl_file"] is not None,
    ):
        match = next((i for i, item in enumerate(records) if predicate(item)), None)
        if match is not None:
            indices.add(match)
    source_checks = []
    for index in sorted(indices):
        record = records[index]
        source = dataset_root / (record["project_relative"] + ".magics")
        with MTArchive(source) as archive:
            vertices = np.frombuffer(archive.read(record["member"]), dtype="<i4").reshape(-1, 3)
            vertices = vertices.astype(np.float32) / np.float32(10000)
            faces = np.frombuffer(archive.read(record["surface_member"]), dtype="<u4").reshape(-1, 3)
            verify_stl(root / record["stl_file"], vertices, faces)
        source_checks.append({"stl_file": record["stl_file"], "source_triangles_equal": True})

    unique = {}
    for row in rows:
        if row["geometry_screen_pass"]:
            unique.setdefault(row["geometry_sha256"], row)
    candidates = list(unique.values())
    review = [row for row in rows if row["needs_review"]]
    write_jsonl(root / "gt_all.jsonl", rows)
    write_jsonl(root / "gt_unique_screened_candidates.jsonl", candidates)
    write_jsonl(root / "gt_review.jsonl", review)
    with (root / "gt_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["sample_id", "stl_file", "source_project", "gt_up_x", "gt_up_y", "gt_up_z",
                  "yaw_unconstrained", "needs_review", "review_reasons", "duplicate_of_stl_file"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row[key] for key in fields if key in row},
                "gt_up_x": 0, "gt_up_y": 0, "gt_up_z": 1,
                "review_reasons": ";".join(row["review_reasons"]),
            })

    summary = {
        **document["summary"],
        "stl_paths_verified": len(rows),
        "hardlinks_verified": hardlinks_verified,
        "source_content_spot_checks": len(source_checks),
        "geometry_screened_unique_candidates": len(candidates),
        "review_reasons": dict(Counter(reason for row in review for reason in row["review_reasons"])),
        "all_source_projects_represented": True,
    }
    audit = {"summary": summary, "source_checks": source_checks}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_suffix(".json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text("\n".join([
        "# Magics 零件与构建方向导出", "",
        *[f"- {key}: {value}" for key, value in summary.items()], "",
        "## 文件", "",
        "- outputs/magics_parts_stl/manifest.json：完整源对象、分类和导出记录。",
        "- gt_all.jsonl：每个工程的每条零件候选及其方向标签，保留重复记录。",
        "- gt_index.csv：可在表格软件中查看的索引。",
        "- gt_unique_screened_candidates.jsonl：去掉精确重复、平台范围异常和负 Z 的候选清单。",
        "- gt_review.jsonl：需要复核的候选。", "",
        "## 标签坐标", "",
        "STL 保留存储坐标、毫米单位和三角形，不重新旋转、不简化。该坐标中的构建向上方向为 [0,0,1]；yaw 不监督。",
        "这是已摆放网格自身的方向标签，不是从原始 CAD 坐标恢复的旋转矩阵；当前没有原始 CAD 到工程的变换。",
        "训练施加主动旋转 A 时，点和法向一起旋转，方向标签设为 A@[0,0,1]；使用行向量时点写为 P@A.T。",
        "grounding_translation_mm 仅供单件预览贴地：将最低点移至 Z=0，不改变方向标签，也不修改导出的 STL。", "",
        "## 数据状态", "",
        "几何筛选不是对象激活状态确认。工程头元数据当前不可读，无法逐件确认隐藏对象、原始名称及是否属于实际打印任务；",
        "因此所有导出标签均明确记为 stored_build_direction_candidate，不能把候选数量宣称为已确认新增独立训练零件数量。",
        "平台和 Box 排除依据几何签名；其他类型的辅助对象仍可能存在。",
        "精确哈希包含坐标和平移，不等同于零件族识别。合并训练前仍需识别平移/旋转副本，并避免跨训练验证集泄漏。",
        "此次仅导出和核验，原始 193 件的训练划分及模型没有变动。", "",
        "## 复用", "",
        "在 D:/proj/pos_set 目录运行：", "",
        "```powershell",
        "python tools/extract_magics_parts.py datasets",
        "python tools/finalize_magics_gt.py",
        "```", "",
        "本机 Python 已有 NumPy。建议在 Windows 本机执行，避免 WSL 对 /mnt/d 中大量小文件的路径检查开销。",
        "gt_all.jsonl 中 stl_file 相对于该 GT 文件所在目录；source_project 相对于 datasets 目录。", "",
    ]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/magics_parts_stl/manifest.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--report", type=Path, default=Path("reports/magics_gt_export_2026-11-09.md"))
    args = parser.parse_args()
    finalize(args.manifest.resolve(), args.dataset.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
