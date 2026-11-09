"""Append newly extracted PJF parts to the original train split, preserving validation."""

import argparse
import json
from pathlib import Path
import re


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identifiers(record):
    return set(re.findall(r"(?<!\d)\d{9}(?!\d)", record["source_part_name"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("outputs/pointclouds"))
    parser.add_argument("--added", type=Path, default=Path("outputs/expanded_pjf_pointclouds"))
    parser.add_argument("--split", type=Path, default=Path("runs/utonia_tiny_direction_ablation/with_normals_r64/split.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expanded_real_pointclouds"))
    args = parser.parse_args()
    original = read_json(args.original / "manifest.json")
    added = read_json(args.added / "manifest.json")
    split = read_json(args.split)
    if original["errors"] or added["errors"]:
        raise ValueError("Resolve extraction errors before merging")
    records = original["records"]
    if [r["pointcloud_file"] for r in records] != [r["pointcloud_file"] for r in split["records"]]:
        raise ValueError("Original split and manifest do not match")
    held_out = [records[index] for index in split["validation_indices"]]
    validation_projects = {record["source_project"] for record in held_out}
    validation_ids = set().union(*(identifiers(record) for record in held_out))
    excluded_projects = {
        record["source_project"] for record in added["records"]
        if record["source_project"] in validation_projects or identifiers(record) & validation_ids
    }
    seen_hashes = {record["mesh_hash_sha256"] for record in records}
    merged, excluded = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    import os
    for source_dir, source_records in [(args.original, records), (args.added, added["records"])]:
        for record in source_records:
            if source_dir == args.added:
                reason = (
                    "project shares a validation project or nine-digit part identifier"
                    if record["source_project"] in excluded_projects
                    else "exact raw-mesh hash duplicate" if record["mesh_hash_sha256"] in seen_hashes else None
                )
                if reason:
                    excluded.append({"record": record, "reason": reason})
                    continue
                seen_hashes.add(record["mesh_hash_sha256"])
            copied = dict(record)
            copied["pointcloud_file"] = Path(os.path.relpath(
                (source_dir / record["pointcloud_file"]).resolve(), args.output.resolve()
            )).as_posix()
            merged.append(copied)
    split = dict(split)
    split["train_indices"] = split["train_indices"] + list(range(len(records), len(merged)))
    split["records"] = [
        {key: record[key] for key in ("source_project", "source_part_id", "source_part_name", "pointcloud_file")}
        for record in merged
    ]
    split["method"] += " Original validation indices unchanged; append eligible added PJF records to training only."
    split["limitation"] += " Added records filtered by raw hash and held-out project/parsed identifier; rotated or remeshed duplicates are not certified absent."
    manifest = dict(original)
    manifest.update(records=merged, sample_count=len(merged), excluded_added=excluded,
                    dataset="original real parts plus eligible additional PJF root parts")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"original_parts": len(records), "added_eligible": len(merged)-len(records),
                      "excluded": len(excluded), "train": len(split["train_indices"]),
                      "validation": len(split["validation_indices"])}, indent=2))


if __name__ == "__main__":
    main()
