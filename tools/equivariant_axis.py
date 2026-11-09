"""Vector-neuron-style SO(3)-equivariant point-cloud direction regressor."""

import torch
from torch import nn


class VectorLinear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.mix = nn.Linear(inputs, outputs, bias=False)

    def forward(self, vectors):
        return self.mix(vectors.transpose(-1, -2)).transpose(-1, -2)


class VectorActivation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.direction = VectorLinear(channels, channels)

    def forward(self, vectors):
        direction = self.direction(vectors)
        projection = (vectors*direction).sum(-1, keepdim=True)
        denominator = direction.square().sum(-1, keepdim=True).clamp_min(1e-8)
        return vectors-.8*projection.clamp_max(0)/denominator*direction


class VectorBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = VectorLinear(inputs, outputs)
        self.activation = VectorActivation(outputs)

    def forward(self, vectors):
        return self.activation(self.linear(vectors))


class EquivariantAxis(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_blocks = nn.ModuleList([VectorBlock(3, 16), VectorBlock(16, 24), VectorBlock(24, 32)])
        self.global_block = VectorBlock(72, 32)
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))

    def forward(self, features):
        points, normals = features[..., :3], features[..., 3:6]
        vectors = torch.stack([points, normals, torch.cross(points, normals, dim=-1)], dim=-2)
        pooled = []
        for block in self.point_blocks:
            vectors = block(vectors)
            lengths = vectors.square().sum(-1)
            weights = torch.softmax(lengths/(lengths.mean(1, keepdim=True)+1e-6), dim=1)
            pooled.append((vectors*weights[..., None]).sum(1))
        global_vectors = self.global_block(torch.cat(pooled, dim=1))
        invariants = torch.linalg.vector_norm(global_vectors, dim=-1)
        weights = self.head(invariants)
        output = (global_vectors*weights[..., None]).sum(1)
        return nn.functional.normalize(output, dim=-1, eps=1e-8)
