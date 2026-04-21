# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side per-head Q/K RMSNorm (Qwen3-class models).

Qwen3 inserts a per-head RMSNorm on Q and K BEFORE RoPE:

    q_per_head = q.reshape(seq, n_heads,    head_dim)
    k_per_head = k.reshape(seq, n_kv_heads, head_dim)
    q_normed   = rms_norm(q_per_head, q_norm_weight, eps)  # along last dim
    k_normed   = rms_norm(k_per_head, k_norm_weight, eps)
    # ... then RoPE on q_normed, k_normed

The existing predecessor split ELFs (`rms_attn_gemms` and `rope_qk`) emit Q/K
WITHOUT RoPE in the first launch and apply RoPE in the second. Inserting Q/K
Norm between these two launches on the host (BF16 numpy) is the simplest path
to integrate Qwen3 without a new on-tile kernel — RMSNorm doesn't commute with
RoPE for asymmetric weights, so we cannot fuse it into existing kernels via the
linearity trick used for QKV bias on Qwen2.

This helper is BF16-throughout; the float32 promotion happens inside numpy and
the result is cast back to BF16 before being handed to the next NPU launch.
"""

import numpy as np
from ml_dtypes import bfloat16


def rms_norm_per_head(
    x_bf16: np.ndarray, weight_bf16: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Per-head RMSNorm. x: (seq, n_heads, head_dim) BF16. weight: (head_dim,) BF16.

    Returns BF16 with same shape as x.
    """
    x = np.asarray(x_bf16, dtype=np.float32)
    w = np.asarray(weight_bf16, dtype=np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    out = (x / rms) * w
    return out.astype(bfloat16)


def apply_qk_norm(
    q_flat_bf16: np.ndarray,
    k_flat_bf16: np.ndarray,
    q_norm_weight: np.ndarray,
    k_norm_weight: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> tuple:
    """Apply Q and K Norm to the seq-first concatenated Q/K outputs.

    Args:
      q_flat_bf16: (seq, n_heads*head_dim)    BF16 — output of Q GEMM
      k_flat_bf16: (seq, n_kv_heads*head_dim) BF16 — output of K GEMM
      q_norm_weight: (head_dim,) BF16
      k_norm_weight: (head_dim,) BF16
      n_heads, n_kv_heads, head_dim, eps: model config

    Returns:
      (q_normed_flat, k_normed_flat) — same shapes as inputs, BF16.
    """
    seq = q_flat_bf16.shape[0]
    q_per_head = q_flat_bf16.reshape(seq, n_heads, head_dim)
    k_per_head = k_flat_bf16.reshape(seq, n_kv_heads, head_dim)
    q_normed = rms_norm_per_head(q_per_head, q_norm_weight, eps=eps)
    k_normed = rms_norm_per_head(k_per_head, k_norm_weight, eps=eps)
    return (
        q_normed.reshape(seq, n_heads * head_dim),
        k_normed.reshape(seq, n_kv_heads * head_dim),
    )
