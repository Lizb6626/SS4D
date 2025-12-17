from typing import *
import torch
import math
from . import DEBUG, BACKEND

if BACKEND == 'xformers':
    import xformers.ops as xops
elif BACKEND == 'flash_attn':
    import flash_attn
elif BACKEND == 'sdpa':
    from torch.nn.functional import scaled_dot_product_attention as sdpa
elif BACKEND == 'naive':
    pass
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")



__all__ = [
    'windowed_scaled_dot_product_self_attention',
]


def windowed_scaled_dot_product_self_attention(
        qkv: torch.Tensor,
        window_size: int,
        shift_window: int = 0,
) -> torch.Tensor:
    """
    Only apply to single num_frames
    Args:
        q (torch.Tensor): [b, T, L, 3, num_heads, head_dim]
    """
    assert len(qkv.shape) == 6, f"mismatched shape for windowed attntion: {qkv.shape}"
    B, T, L, _, num_heads, _ = qkv.shape

    shifted_coords = torch.arange(T, dtype=torch.int32) + shift_window
    shifted_coords = shifted_coords // window_size
    seq_lens = torch.bincount(shifted_coords).repeat(B) * L
    cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens, dim=0)], dim=0).to(qkv.device).int()

    assert BACKEND == 'flash_attn', "Only support flash attn for windowed attention"
    qkv = qkv.flatten(0, 2)
    out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_lens.max())
    out = out.reshape(B, T, L, num_heads, -1)
    return out

