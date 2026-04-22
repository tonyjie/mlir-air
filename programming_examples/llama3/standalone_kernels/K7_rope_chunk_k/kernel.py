# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K7: RoPE chunk K (M=64, n_kv_heads=8, head_dim=64) BF16 standalone.

Llama-3.2-1B chunked-prefill kernel — K-side RoPE for one chunk.
Mirrors K6 with n_heads=8 (KV heads, GQA: 32 Q heads / 4 = 8 KV heads),
which makes emb_dim = kv_dim = 8 * 64 = 512.

Same half-split convention and LUT layout as K6.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))

from llama3.multi_launch_builder.rms_gemms_rope_multi import _build_rope_2d
from _llm_shared.kernel_builder.external_kernels import compile_rope

from air.backend.xrt_runner import XRTRunner

# Chunked-prefill RoPE K shape (KV-side, GQA)
M = 64  # chunk size
N_KV_HEADS = 8  # KV heads
HEAD_DIM = 64
KV_DIM = N_KV_HEADS * HEAD_DIM  # 512
ROPE_BASE = 500_000.0


def gen_halfsplit_lut(seq_len, head_dim, theta=ROPE_BASE):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
    pos = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(pos, inv_freq)
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=-1).astype(bfloat16)


def cpu_rope_halfsplit(x, lut_per_pos, n_heads, head_dim):
    M_ = x.shape[0]
    half = head_dim // 2
    x_f = x.astype(np.float32).reshape(M_, n_heads, head_dim)
    lut = lut_per_pos.astype(np.float32)
    cos = lut[:, :half][:, None, :]
    sin = lut[:, half:][:, None, :]
    x1, x2 = x_f[..., :half], x_f[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return (
        np.concatenate([y1, y2], axis=-1)
        .reshape(M_, n_heads * head_dim)
        .astype(bfloat16)
    )


def main():
    print(
        f"K7 rope_chunk_k: M={M}, n_kv_heads={N_KV_HEADS}, head_dim={HEAD_DIM}, kv_dim={KV_DIM}"
    )

    print("  Compiling rope_halfsplit.cc -> rope.o ...")
    compile_rope()

    mlir_module = _build_rope_2d(
        outer_rows=M,
        outer_cols=KV_DIM,
        embed_dim=HEAD_DIM,
        np_dtype=bfloat16,
        herd_x=1,
    )

    np.random.seed(1)
    x = (np.random.randn(M, KV_DIM) * 0.1).astype(bfloat16)
    lut_per_pos = gen_halfsplit_lut(M, HEAD_DIM)
    lut_flat = (
        np.repeat(lut_per_pos.astype(np.float32), N_KV_HEADS, axis=0)
        .flatten()
        .astype(bfloat16)
    )
    assert lut_flat.shape[0] == M * KV_DIM

    y_expected = cpu_rope_halfsplit(x, lut_per_pos, N_KV_HEADS, HEAD_DIM)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K7_rope_chunk_k",
    )
    rc = runner.run_test(
        mlir_module,
        inputs=[x, lut_flat],
        expected_outputs=[y_expected],
        rtol=5e-2,
        atol=5e-2,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
