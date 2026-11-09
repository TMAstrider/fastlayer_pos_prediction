from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from orientation_model import compatible_batch_key, encode_prepared_batch, prepare_cloud


def prepare_task(task, clouds, arguments):
    record_id, rotation_index, phase_seed = task
    original, normals = clouds[record_id]
    sample_seed = arguments.seed + phase_seed + record_id * 1000003 + rotation_index
    rotation = Rotation.random(random_state=np.random.default_rng(sample_seed)).as_matrix()
    sample_rng = np.random.default_rng(sample_seed + 2000000011)
    prepared = prepare_cloud(
        original @ rotation.T, arguments.points, arguments.grid_size, sample_rng,
        normals=None if normals is None else normals @ rotation.T,
    )
    return prepared, rotation[:, 2].astype(np.float32), compatible_batch_key(prepared[1], arguments.patch_size)


def encode_grouped(encoder, prepared, batch_size, device):
    buckets = defaultdict(list)
    for index, (_, _, signature) in enumerate(prepared):
        buckets[signature].append(index)
    features = [None] * len(prepared)
    batch_counts = []

    def encode_indices(indices):
        try:
            result = encode_prepared_batch(encoder, [prepared[index][0] for index in indices], device)
        except torch.cuda.OutOfMemoryError:
            if len(indices) == 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(indices) // 2
            encode_indices(indices[:midpoint])
            encode_indices(indices[midpoint:])
            return
        for index, feature in zip(indices, result):
            features[index] = feature
        batch_counts.append(len(indices))

    for indices in buckets.values():
        for start in range(0, len(indices), batch_size):
            encode_indices(indices[start:start + batch_size])
    return np.stack(features), batch_counts


def extract_features_fast(arguments, records, split, encoder, device, cache_path, cache_metadata):
    started_all = time.monotonic()
    clouds = {}
    for index, record in enumerate(records):
        with np.load(arguments.pointclouds / record["pointcloud_file"]) as cloud:
            original = cloud["points"].astype(np.float64)
            original -= original.mean(axis=0)
            clouds[index] = (original, cloud["normals"].astype(np.float64) if arguments.use_normals else None)
    fingerprint = hashlib.sha256(json.dumps(cache_metadata, sort_keys=True).encode()).hexdigest()[:16]
    chunk_directory = cache_path.parent / f"{cache_path.stem}_chunks" / fingerprint
    chunk_directory.mkdir(parents=True, exist_ok=True)
    result = {}
    timings = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        for phase, rotations in [("train", arguments.train_rotations), ("validation", arguments.validation_rotations)]:
            phase_seed = 1000000007 if phase == "validation" else 0
            tasks = [(index, rotation, phase_seed) for index in split[f"{phase}_indices"] for rotation in range(rotations)]
            phase_features, phase_targets = [], []
            pending = {}
            for start in range(0, len(tasks), arguments.extraction_chunk_size):
                chunk_tasks = tasks[start:start + arguments.extraction_chunk_size]
                chunk_file = chunk_directory / f"{phase}_{start:06d}_{len(chunk_tasks)}.npz"
                started = time.monotonic()
                for prefetch_start in [start, start + arguments.extraction_chunk_size]:
                    prefetch_tasks = tasks[prefetch_start:prefetch_start + arguments.extraction_chunk_size]
                    prefetch_file = chunk_directory / f"{phase}_{prefetch_start:06d}_{len(prefetch_tasks)}.npz"
                    if prefetch_tasks and prefetch_start not in pending and not prefetch_file.exists():
                        pending[prefetch_start] = [
                            executor.submit(prepare_task, task, clouds, arguments) for task in prefetch_tasks
                        ]
                if chunk_file.exists():
                    with np.load(chunk_file) as chunk:
                        features, targets = chunk["features"], chunk["targets"]
                    reused = True
                    batch_counts = []
                    preprocessing_seconds = 0.0
                    encoding_seconds = 0.0
                else:
                    prepared = [future.result() for future in pending.pop(start)]
                    preprocessing_seconds = time.monotonic() - started
                    encoding_started = time.monotonic()
                    features, batch_counts = encode_grouped(encoder, prepared, arguments.encoder_batch_size, device)
                    encoding_seconds = time.monotonic() - encoding_started
                    targets = np.stack([item[1] for item in prepared])
                    temporary = chunk_file.with_suffix(".tmp.npz")
                    np.savez(temporary, features=features, targets=targets)
                    temporary.replace(chunk_file)
                    reused = False
                    del prepared
                if len(features) != len(chunk_tasks) or not np.isfinite(features).all():
                    raise ValueError(f"Invalid feature chunk: {chunk_file}")
                phase_features.append(features)
                phase_targets.append(targets)
                timings.append({
                    "phase": phase, "start": start, "samples": len(chunk_tasks),
                    "reused": reused, "seconds": time.monotonic() - started,
                    "preprocessing_seconds": preprocessing_seconds, "encoding_seconds": encoding_seconds,
                    "mean_gpu_batch_size": float(np.mean(batch_counts)) if batch_counts else None,
                })
                print(f"{phase}: {start + len(chunk_tasks)}/{len(tasks)} samples; chunk={timings[-1]['seconds']:.2f}s; cached={reused}", flush=True)
            result[f"{phase}_features"] = np.concatenate(phase_features)
            result[f"{phase}_targets"] = np.concatenate(phase_targets)
            result[f"{phase}_record_ids"] = np.array([task[0] for task in tasks])
    (arguments.output / "extraction_timing.json").write_text(json.dumps({
        "total_seconds": time.monotonic() - started_all,
        "workers": arguments.workers, "encoder_batch_size": arguments.encoder_batch_size,
        "chunks": timings,
    }, indent=2), encoding="utf-8")
    return result
