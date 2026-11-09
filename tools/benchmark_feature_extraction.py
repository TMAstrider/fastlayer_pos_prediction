import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from feature_extraction import prepare_task, encode_grouped
from orientation_model import ROOT, load_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/feature_extraction_benchmark.json")
    arguments = parser.parse_args()
    arguments.points = 2048
    arguments.grid_size = 0.02
    arguments.seed = 20261108
    arguments.patch_size = 128
    records = json.loads((ROOT / "outputs/pointclouds/manifest.json").read_text())["records"]
    clouds = {}
    for index, record in enumerate(records):
        with np.load(ROOT / "outputs/pointclouds" / record["pointcloud_file"]) as cloud:
            original = cloud["points"].astype(np.float64)
            original -= original.mean(axis=0)
            clouds[index] = original, cloud["normals"].astype(np.float64)
    tasks = [(position % len(clouds), position // len(clouds), 0) for position in range(arguments.samples)]
    preprocessing = []
    for workers in [1, 4, 8]:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = list(executor.map(lambda task: prepare_task(task, clouds, arguments), tasks))
        preprocessing.append({"workers": workers, "seconds": time.monotonic() - started})
    torch.set_num_threads(1)
    encoder, _ = load_encoder(ROOT / "checkpoints/utonia_tiny.pth", "cuda", fast_attention=False)
    encode_grouped(encoder, prepared[:8], 1, "cuda")
    measurements = []
    reference = None
    for batch_size in [1, 8, 16, 32]:
        if batch_size == 8:
            from batched_attention import enable_batched_attention

            enable_batched_attention(encoder)
        torch.cuda.synchronize()
        started = time.monotonic()
        features, sizes = encode_grouped(encoder, prepared, batch_size, "cuda")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if reference is None:
            reference = features
        error = float(np.abs(features - reference).max())
        np.testing.assert_allclose(features, reference, atol=2e-5, rtol=2e-5)
        measurements.append({
            "batch_size": batch_size, "mean_actual_batch": float(np.mean(sizes)),
            "seconds": elapsed, "samples_per_second": len(tasks) / elapsed,
            "max_absolute_feature_error": error,
            "peak_torch_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        })
        print(json.dumps(measurements[-1]), flush=True)
    report = {"samples": len(tasks), "cpu_preprocessing": preprocessing, "gpu_encoding": measurements}
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
