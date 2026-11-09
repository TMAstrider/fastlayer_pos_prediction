import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


ROOT = Path(__file__).resolve().parents[1]
UTONIA_REVISION = "5fce241d3fd943bdef2db14fdd41fff8b090d98e"
FEATURE_PIPELINE_VERSION = "deterministic-pooling-masked-batched-attention-v3"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(weights, device, patch_size=128, fast_attention=True):
    sys.path.insert(0, str(ROOT / "external" / "Utonia"))
    import utonia

    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    configuration = dict(checkpoint["config"])
    configuration.update(
        enable_flash=False,
        enc_patch_size=[patch_size] * len(configuration["enc_depths"]),
        shuffle_orders=False,
    )
    encoder = utonia.model.PointTransformerV3(**configuration)
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    for module in encoder.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False
    encoder.requires_grad_(False)
    encoder.eval().to(device)
    if fast_attention:
        from batched_attention import enable_batched_attention

        enable_batched_attention(encoder)
    return encoder, configuration


def compatible_batch_key(grid, patch_size=128):
    signature = []
    for stage in range(5):
        coarse = grid // (2 ** stage)
        signature.append(int(coarse.max() + 1).bit_length())
    return tuple(signature)


@torch.inference_mode()
def encode_prepared_batch(encoder, prepared_clouds, device):
    import torch_scatter

    sizes = [len(cloud[0]) for cloud in prepared_clouds]
    coordinates = torch.from_numpy(np.concatenate([cloud[0] for cloud in prepared_clouds])).to(device)
    grid = torch.from_numpy(np.concatenate([cloud[1] for cloud in prepared_clouds])).to(device)
    normals = torch.from_numpy(np.concatenate([
        np.zeros_like(cloud[0]) if cloud[2] is None else cloud[2].astype(np.float32)
        for cloud in prepared_clouds
    ])).to(device)
    point = encoder({
        "coord": coordinates,
        "grid_coord": grid,
        "feat": torch.cat([coordinates, torch.zeros_like(coordinates), normals], dim=1),
        "offset": torch.tensor(np.cumsum(sizes), device=device),
    })
    pooled = []
    while True:
        offsets = torch.cat([point.offset.new_zeros(1), point.offset])
        pooled.extend([
            torch_scatter.segment_csr(point.feat.float(), offsets, reduce="mean"),
            torch_scatter.segment_csr(point.feat.float(), offsets, reduce="max"),
        ])
        if "pooling_parent" not in point:
            break
        point = point.pooling_parent
    return torch.cat(pooled, dim=1).cpu().numpy()


def prepare_cloud(points, point_count, grid_size, rng, normals=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N x 3 points")
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError("Expected finite per-point normals matching the points")
        normal_lengths = np.linalg.norm(normals, axis=1)
        if np.any(normal_lengths <= 1e-12):
            raise ValueError("Per-point normals must be nonzero")
        normals = normals / normal_lengths[:, None]
    if len(points) > point_count:
        selected = rng.choice(len(points), point_count, replace=False)
        points = points[selected]
        if normals is not None:
            normals = normals[selected]
    points = points - points.mean(axis=0)
    radius = np.linalg.norm(points, axis=1).max()
    if radius <= 0:
        raise ValueError("Point cloud has zero extent")
    points = (points / radius).astype(np.float32)
    grid = np.floor((points - points.min(axis=0)) / grid_size).astype(np.int32)
    _, indices = np.unique(grid, axis=0, return_index=True)
    return points[indices], grid[indices], None if normals is None else normals[indices]


@torch.no_grad()
def encode_cloud(encoder, points, point_count, grid_size, rng, device, normals=None):
    coordinates, grid, sampled_normals = prepare_cloud(
        points, point_count, grid_size, rng, normals=normals
    )
    coordinates = torch.from_numpy(coordinates).to(device)
    normal_features = (
        torch.zeros_like(coordinates)
        if sampled_normals is None
        else torch.from_numpy(sampled_normals.astype(np.float32)).to(device)
    )
    point = encoder({
        "coord": coordinates,
        "grid_coord": torch.from_numpy(grid).to(device),
        "feat": torch.cat([coordinates, torch.zeros_like(coordinates), normal_features], dim=1),
        "offset": torch.tensor([len(coordinates)], device=device),
    })
    pooled = []
    while True:
        pooled.extend([point.feat.float().mean(dim=0), point.feat.float().amax(dim=0)])
        if "pooling_parent" not in point:
            break
        point = point.pooling_parent
    return torch.cat(pooled).cpu().numpy()


class DirectionHead(nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(feature_dimension))
        self.register_buffer("feature_std", torch.ones(feature_dimension))
        self.layers = nn.Sequential(
            nn.Linear(feature_dimension, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_std
        return functional.normalize(self.layers(normalized), dim=-1, eps=1e-8)


def direction_metrics(predicted, target):
    cosine = np.clip(np.sum(predicted * target, axis=-1), -1.0, 1.0)
    angles = np.rad2deg(np.arccos(cosine))
    return {
        "mean_angle_deg": float(angles.mean()),
        "median_angle_deg": float(np.median(angles)),
        "p90_angle_deg": float(np.percentile(angles, 90)),
        "within_15_deg_fraction": float((angles <= 15).mean()),
        "within_30_deg_fraction": float((angles <= 30).mean()),
    }


def align_up_rotation(direction):
    direction = np.asarray(direction, dtype=np.float64)
    length = np.linalg.norm(direction)
    if not np.isfinite(direction).all() or length < 1e-8:
        raise ValueError("Invalid predicted direction")
    direction = direction / length
    target = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(direction @ target, -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(direction, target)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def place_mesh(vertices, direction, clearance=0.0):
    rotation = align_up_rotation(direction)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    rotated = (vertices - center) @ rotation.T
    shift = np.array([0.0, 0.0, clearance - rotated[:, 2].min()])
    translation = shift - rotation @ center
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return vertices @ rotation.T + translation, transform
