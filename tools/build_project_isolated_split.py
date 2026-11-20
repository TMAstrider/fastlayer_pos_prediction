#!/usr/bin/env python3
"""Build a deterministic source-project-isolated train/test split."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def unique_records(split: dict[str, Any]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in split["train_records"] + split["validation_records"]:
        key = str(record.get("mesh_hash_sha256") or record["job_id"])
        records.setdefault(key, record)
    return list(records.values())


def select_project_prefix(
    project_counts: dict[str, int],
    target_parts: int,
    seed: int,
) -> tuple[list[str], int]:
    if target_parts < 1:
        raise ValueError("target_parts must be positive")
    ordered = sorted(
        project_counts,
        key=lambda project: hashlib.sha256(
            f"{seed}|{project}".encode("utf-8")
        ).hexdigest(),
    )
    best: tuple[list[str], int] | None = None
    selected: list[str] = []
    total = 0
    for project in ordered[:-1]:
        selected.append(project)
        total += project_counts[project]
        if best is None or abs(total - target_parts) < abs(best[1] - target_parts):
            best = (selected.copy(), total)
    if best is None:
        raise ValueError("At least two source projects are required")
    return best


def build_split(
    records: list[dict[str, Any]],
    target_parts: int,
    seed: int,
) -> dict[str, Any]:
    project_counts = Counter(
        str(record["source_project"]) for record in records
    )
    validation_projects, validation_count = select_project_prefix(
        dict(project_counts),
        target_parts,
        seed,
    )
    selected = set(validation_projects)
    train_records = [
        record for record in records
        if record["source_project"] not in selected
    ]
    validation_records = [
        record for record in records
        if record["source_project"] in selected
    ]
    train_project_set = {
        record["source_project"] for record in train_records
    }
    validation_project_set = {
        record["source_project"] for record in validation_records
    }
    train_hashes = {
        record["mesh_hash_sha256"] for record in train_records
    }
    validation_hashes = {
        record["mesh_hash_sha256"] for record in validation_records
    }
    if train_project_set & validation_project_set:
        raise ValueError("Generated split contains source-project leakage")
    if train_hashes & validation_hashes:
        raise ValueError("Generated split contains duplicate geometry")
    if len(train_records) + len(validation_records) != len(records):
        raise ValueError("Generated split dropped records")
    return {
        "schema_version": 1,
        "split_name": "project_isolated",
        "selection": (
            "source projects sorted by sha256('{seed}|{project}'); "
            "prefix with part count closest to target"
        ),
        "seed": seed,
        "target_validation_parts": target_parts,
        "train_count": len(train_records),
        "validation_count": validation_count,
        "train_project_count": len(train_project_set),
        "validation_project_count": len(validation_project_set),
        "train_projects": sorted(train_project_set),
        "validation_projects": sorted(validation_project_set),
        "train_records": train_records,
        "validation_records": validation_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "labels/projection_support_area_expanded_932_100_v0_256/"
            "split_project_isolated_100_seed30.json"
        ),
    )
    parser.add_argument("--validation-parts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_dir = (root / args.dataset).resolve()
    output = (root / args.output).resolve()
    source = json.loads(
        (dataset_dir / "split.json").read_text(encoding="utf-8")
    )
    split = build_split(
        unique_records(source),
        target_parts=args.validation_parts,
        seed=args.seed,
    )
    output.write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "train_count": split["train_count"],
        "validation_count": split["validation_count"],
        "train_project_count": split["train_project_count"],
        "validation_project_count": split["validation_project_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
