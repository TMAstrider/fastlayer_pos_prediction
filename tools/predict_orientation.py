import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from extract_pointclouds import sample_surface
from export_meshes import write_stl, verify_stl
from orientation_model import ROOT, DirectionHead, encode_cloud, file_sha256, load_encoder, place_mesh


def main():
    parser = argparse.ArgumentParser(description="Predict a build-up direction and export a transformed single-part STL")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/utonia_tiny_direction/best_head.pt")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/utonia_tiny.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/predicted_pose")
    parser.add_argument("--clearance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261108)
    arguments = parser.parse_args()
    if arguments.clearance < 0:
        parser.error("Clearance must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("The sparse-convolution encoder requires CUDA")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    if file_sha256(arguments.weights) != configuration["weights_sha256"]:
        raise ValueError("Encoder weights differ from the training run")
    encoder, _ = load_encoder(arguments.weights, device, configuration["patch_size"])
    head = DirectionHead(checkpoint["feature_dimension"]).to(device)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()
    mesh = trimesh.load(arguments.input, force="mesh", process=False)
    rng = np.random.default_rng(arguments.seed)
    points, normals, _, _ = sample_surface(
        mesh.vertices, mesh.faces, max(4096, configuration["points"]), rng
    )
    features = encode_cloud(
        encoder,
        points,
        configuration["points"],
        configuration["grid_size"],
        rng,
        device,
        normals=normals if configuration.get("use_normals", False) else None,
    )
    with torch.no_grad():
        direction = head(torch.from_numpy(features).unsqueeze(0).to(device))[0].cpu().numpy()
    vertices, transform = place_mesh(mesh.vertices, direction, arguments.clearance)
    arguments.output.mkdir(parents=True, exist_ok=True)
    stl_path = arguments.output / f"{arguments.input.stem}_oriented.stl"
    write_stl(stl_path, vertices.astype(np.float32), mesh.faces)
    verify_stl(stl_path, vertices.astype(np.float32), mesh.faces)
    result = {
        "input": str(arguments.input.resolve()),
        "output_stl": str(stl_path.resolve()),
        "predicted_build_up_direction_input_frame": direction.tolist(),
        "rotation_matrix": transform[:3, :3].tolist(),
        "translation": transform[:3, 3].tolist(),
        "transform_4x4": transform.tolist(),
        "convention": "Column vector: output = rotation @ input + translation. Row vertex array: output = vertices @ rotation.T + translation.",
        "yaw": "Minimum-angle alignment of predicted up to +Z; yaw is not learned.",
        "clearance_raw_units": arguments.clearance,
        "units": "Unchanged from input STL; STL does not encode units.",
        "model_checkpoint": str(arguments.checkpoint.resolve()),
        "validation_metrics": checkpoint["validation_metrics"],
        "status": "Experimental prediction; requires process engineer review and does not certify printability.",
    }
    (arguments.output / f"{arguments.input.stem}_pose.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
