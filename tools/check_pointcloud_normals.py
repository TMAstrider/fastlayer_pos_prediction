import argparse
import json
from pathlib import Path

import numpy as np

from orientation_model import ROOT


def main():
    parser = argparse.ArgumentParser(description="Validate exported surface-normal samples and optionally run a CUDA PTv3 forward pass")
    parser.add_argument("--pointclouds", type=Path, default=ROOT / "outputs/pointclouds")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/utonia_tiny.pth")
    parser.add_argument("--cuda-forward", action="store_true")
    arguments = parser.parse_args()
    manifest = json.loads((arguments.pointclouds / "manifest.json").read_text(encoding="utf-8"))
    sample = None
    checked = 0
    for record in manifest["records"]:
        with np.load(arguments.pointclouds / record["pointcloud_file"]) as cloud:
            points = cloud["points"]
            normals = cloud["normals"]
            if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Point/normal shape mismatch: {record['pointcloud_file']}")
            if not np.isfinite(points).all() or not np.isfinite(normals).all():
                raise ValueError(f"Nonfinite point or normal: {record['pointcloud_file']}")
            lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
            if np.any(lengths < 0.99) or np.any(lengths > 1.01):
                raise ValueError(f"Non-unit sampled normal: {record['pointcloud_file']}")
            if sample is None:
                sample = (points.copy(), normals.copy())
            checked += 1
    print(f"Validated finite XYZ samples and unit-length corresponding normals for {checked} parts")

    if arguments.cuda_forward:
        import torch
        from scipy.spatial.transform import Rotation
        from orientation_model import encode_cloud, load_encoder

        if not torch.cuda.is_available():
            parser.error("--cuda-forward requires CUDA")
        torch.set_num_threads(4)
        rotation = Rotation.from_rotvec([0.45, -0.2, 0.7]).as_matrix()
        encoder, _ = load_encoder(arguments.weights, torch.device("cuda"))
        for use_normals in (False, True):
            features = encode_cloud(
                encoder,
                sample[0] @ rotation.T,
                2048,
                0.02,
                np.random.default_rng(42),
                torch.device("cuda"),
                normals=sample[1] @ rotation.T if use_normals else None,
            )
            if features.shape != (972,) or not np.isfinite(features).all():
                raise ValueError(f"Invalid PTv3 features with use_normals={use_normals}")
            print(f"CUDA PTv3 forward passed; use_normals={use_normals}, features={features.shape}")


if __name__ == "__main__":
    main()
