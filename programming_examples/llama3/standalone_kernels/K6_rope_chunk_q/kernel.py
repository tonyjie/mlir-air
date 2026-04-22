# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K6: RoPE chunk Q (M=64, n_heads=32, head_dim=64) BF16 standalone.

Llama-3.2-1B chunked-prefill kernel — Q-side RoPE for one chunk.
Uses production builder _build_rope_2d (defined inline in
multi_launch_builder/rms_gemms_rope_multi.py) and the production
rope_halfsplit.cc external kernel (compiled to rope.o here).

Half-split RoPE convention (HuggingFace Llama):
  out_head[i]      = in_head[i] * cos[i]   - in_head[i + hd/2] * sin[i]
  out_head[i + hd/2] = in_head[i] * sin[i] + in_head[i + hd/2] * cos[i]
LUT layout per position: [cos_0..cos_{hd/2-1}, sin_0..sin_{hd/2-1}]
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))

# Production rope builder + external-kernel compile.
from llama3.multi_launch_builder.rms_gemms_rope_multi import _build_rope_2d
from llama3.kernel_builder.external_kernels import compile_rope

from air.backend.xrt_runner import XRTRunner

# Chunked-prefill RoPE Q shape
M = 64  # chunk size
N_HEADS = 32  # Q heads
HEAD_DIM = 64
EMB_DIM = N_HEADS * HEAD_DIM  # 2048
ROPE_BASE = 500_000.0  # Llama-3 RoPE theta


def gen_halfsplit_lut(seq_len, head_dim, theta=ROPE_BASE):
    """Generate (seq_len, head_dim) bf16 LUT in half-split layout.
    Per position p: [cos(p*freq_0)..cos(p*freq_{hd/2-1}),
                     sin(p*freq_0)..sin(p*freq_{hd/2-1})]"""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
    pos = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(pos, inv_freq)  # (seq_len, half)
    cos = np.cos(angles)
    sin = np.sin(angles)
    return np.concatenate([cos, sin], axis=-1).astype(bfloat16)  # (seq_len, head_dim)


def cpu_rope_halfsplit(x, lut_per_pos, n_heads, head_dim):
    """CPU half-split RoPE reference.
    x:           (M, n_heads * head_dim) bf16
    lut_per_pos: (M, head_dim) bf16 — per-position LUT
    Returns (M, n_heads * head_dim) bf16."""
    M_ = x.shape[0]
    half = head_dim // 2
    x_f = x.astype(np.float32).reshape(M_, n_heads, head_dim)
    lut = lut_per_pos.astype(np.float32)  # (M, head_dim)
    cos = lut[:, :half][:, None, :]  # (M, 1, half)
    sin = lut[:, half:][:, None, :]  # (M, 1, half)
    x1, x2 = x_f[..., :half], x_f[..., half:]  # each (M, n_heads, half)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return (
        np.concatenate([y1, y2], axis=-1)
        .reshape(M_, n_heads * head_dim)
        .astype(bfloat16)
    )


def main():
    print(
        f"K6 rope_chunk_q: M={M}, n_heads={N_HEADS}, head_dim={HEAD_DIM}, emb_dim={EMB_DIM}"
    )

    # 1. Pre-compile rope.o into CWD (build_peano/) so the linker finds it.
    print("  Compiling rope_halfsplit.cc -> rope.o ...")
    compile_rope()

    # 2. Build the module. Same call shape as production rms_gemms_rope_multi:
    #    _build_rope_2d(outer_rows, outer_cols, embed_dim, dtype, herd_x)
    mlir_module = _build_rope_2d(
        outer_rows=M,
        outer_cols=EMB_DIM,
        embed_dim=HEAD_DIM,
        np_dtype=bfloat16,
        herd_x=1,
    )

    # 3. Generate inputs.
    np.random.seed(0)
    x = (np.random.randn(M, EMB_DIM) * 0.1).astype(bfloat16)

    # Per-position LUT (M positions, half-split layout).
    lut_per_pos = gen_halfsplit_lut(M, HEAD_DIM)  # (M, head_dim)

    # The kernel's LUT input is 1D total = M * EMB_DIM = M * (n_heads * head_dim)
    # = rope_rows * head_dim. Each rope_row r at position p (= r // n_heads) reads
    # LUT[r * head_dim : (r+1) * head_dim] — i.e. position p's (cos, sin) pair.
    # So we tile per-position LUT n_heads times.
    lut_flat = (
        np.repeat(lut_per_pos.astype(np.float32), N_HEADS, axis=0)
        .flatten()
        .astype(bfloat16)
    )
    assert lut_flat.shape[0] == M * EMB_DIM

    y_expected = cpu_rope_halfsplit(x, lut_per_pos, N_HEADS, HEAD_DIM)

    # 4. Run on NPU and validate.
    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K6_rope_chunk_q",
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
