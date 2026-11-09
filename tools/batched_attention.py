from types import MethodType

import numpy as np
import torch
from torch.nn import functional


def padded_attention_forward(self, point):
    if self.training or self.enable_rpe:
        raise RuntimeError("The batched inference backend requires eval mode and disabled relative-position bias")
    patch_size = self.patch_size_max
    cache_key = f"inference_padding_{patch_size}"
    if cache_key not in point:
        offsets = point.offset.detach().cpu().numpy()
        counts = np.diff(np.concatenate(([0], offsets)))
        padded_indices = []
        unpadding = []
        valid_keys = []
        source_offset = 0
        destination_offset = 0
        for count in counts:
            count = int(count)
            padded_count = ((count + patch_size - 1) // patch_size) * patch_size
            local = np.zeros(padded_count, dtype=np.int64)
            local[:count] = np.arange(count)
            valid = np.ones(padded_count, dtype=bool)
            if count < patch_size:
                valid[count:] = False
            elif count % patch_size:
                remainder = count % patch_size
                pad_start = padded_count - patch_size + remainder
                source_start = padded_count - 2 * patch_size + remainder
                local[pad_start:] = np.arange(source_start, padded_count - patch_size)
            padded_indices.append(local + source_offset)
            unpadding.append(np.arange(count) + destination_offset)
            valid_keys.append(valid)
            source_offset += count
            destination_offset += padded_count
        device = point.feat.device
        point[cache_key] = (
            torch.from_numpy(np.concatenate(padded_indices)).to(device),
            torch.from_numpy(np.concatenate(unpadding)).to(device),
            torch.from_numpy(np.concatenate(valid_keys).reshape(-1, 1, 1, patch_size)).to(device),
        )
    pad, unpad, mask = point[cache_key]
    order = point.serialized_order[self.order_index][pad]
    inverse = unpad[point.serialized_inverse[self.order_index]]
    head_dimension = self.channels // self.num_heads
    qkv = self.qkv(point.feat)[order].reshape(-1, 3, self.num_heads, head_dimension)
    query, key, value = qkv.unbind(dim=1)
    query, key = self.rope(query, key, point.coord[order])
    query, key, value = [
        tensor.reshape(-1, patch_size, self.num_heads, head_dimension).permute(0, 2, 1, 3)
        for tensor in (query, key, value)
    ]
    if self.upcast_attention:
        query, key = query.float(), key.float()
    value = value.to(query.dtype)
    pad_dimension = -head_dimension % 8
    if pad_dimension:
        query, key, value = [functional.pad(tensor, (0, pad_dimension)) for tensor in (query, key, value)]
    features = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, scale=self.scale,
    )[..., :head_dimension]
    features = features.to(qkv.dtype).transpose(1, 2).reshape(-1, self.channels)[inverse]
    point.feat = self.proj_drop(self.proj(features))
    return point


def enable_batched_attention(encoder):
    from utonia.model import SerializedAttention

    for module in encoder.modules():
        if isinstance(module, SerializedAttention):
            if module.enable_flash or module.enable_rpe:
                raise ValueError("Unsupported attention configuration for frozen batched extraction")
            module.forward = MethodType(padded_attention_forward, module)
