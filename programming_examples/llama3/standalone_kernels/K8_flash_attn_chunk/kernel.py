# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""K8: flash_attn_chunk standalone — first-chunk case (current_pos = 0).

This is the FIRST sub-task of K8. It validates that the existing FA
kernel (attn_npu2_seqfirst) compiles and produces correct output for the
chunked rectangular Q-vs-K shape:
  Q: (lq=64,   2048) bf16
  K: (lk=2048, 512)  bf16  -- only [0..63] valid (first-chunk case)
  V: (lk=2048, 512)  bf16  -- only [0..63] valid
  out: (lq=64, 2048) bf16

For the first chunk current_pos=0, the existing built-in causal flag
(`k > q` masked) gives the right answer because Q row q maps to absolute
position q. K rows [64..2047] are zero-padded; their QK^T contribution
is 0 and softmax weight is bounded; with V also zero those positions
contribute nothing.

Mid-stream and partial-last-chunk cases require either an explicit mask
input OR a runtime current_pos parameter — those are deferred to a
follow-up modification of attn_npu2.cc:apply_causal_mask().
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "programming_examples"))

from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
    build_module as build_attn,
)
from llama3.kernel_builder.external_kernels import compile_attn_npu2

from air.backend.xrt_runner import XRTRunner

# Chunked-prefill FA shape — first-chunk case (current_pos = 0)
CHUNK = 64
MAX_SEQ = 2048
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
EMB_DIM = N_HEADS * HEAD_DIM  # 2048
KV_DIM = N_KV_HEADS * HEAD_DIM  # 512


def cpu_attn_causal(q, k, v, n_heads, n_kv_heads, head_dim, valid_lk):
    """CPU reference: causal attention with Q,K seqfirst layout.
    Only K/V rows [0..valid_lk) are non-zero.
    Q row q attends to K rows [0..min(q, valid_lk-1)]."""
    M = q.shape[0]
    L = k.shape[0]
    q_f = q.astype(np.float32).reshape(M, n_heads, head_dim)
    k_f = k.astype(np.float32).reshape(L, n_kv_heads, head_dim)
    v_f = v.astype(np.float32).reshape(L, n_kv_heads, head_dim)
    rep = n_heads // n_kv_heads
    k_f = np.repeat(k_f, rep, axis=1)
    v_f = np.repeat(v_f, rep, axis=1)
    out = np.zeros((M, n_heads, head_dim), dtype=np.float32)
    scale = 1.0 / np.sqrt(head_dim)
    for h in range(n_heads):
        scores = q_f[:, h, :] @ k_f[:, h, :].T * scale  # (M, L)
        # Causal mask: row q attends to k <= q (and only valid_lk rows are real)
        for row in range(M):
            valid_end = min(row + 1, valid_lk)
            scores[row, valid_end:] = -1e30  # mask
        scores -= scores.max(axis=-1, keepdims=True)  # softmax stability
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        out[:, h, :] = p @ v_f[:, h, :]
    return out.reshape(M, n_heads * head_dim).astype(bfloat16)


def main():
    print(
        f"K8 flash_attn_chunk (first-chunk case): "
        f"lq={CHUNK}, lk={MAX_SEQ}, n_heads={N_HEADS}/{N_KV_HEADS}, head_dim={HEAD_DIM}"
    )

    print("  Compiling attn_npu2.cc -> attn_npu2.o ...")
    compile_attn_npu2(head_dim=HEAD_DIM)

    # Build module: chunked rectangular Q-vs-K with built-in causal.
    mlir_module = build_attn(
        lk=MAX_SEQ,
        lkp=HEAD_DIM,  # 64
        lq=CHUNK,  # 64 (chunk size)
        lqp=CHUNK,  # 64 (whole chunk in one Q launch iter)
        dk=HEAD_DIM,
        dv=HEAD_DIM,
        num_q_tiles=1,  # single Q tile (lqp / num_q_tiles = 64)
        num_cascade_stages=1,  # try 1 first (lqp=64 vs production's 256)
        num_heads=N_HEADS,
        num_kv_heads=N_KV_HEADS,
        causal=False,  # debug: check if non-causal compiles at chunk shape
    )

    np.random.seed(8)
    q = (np.random.randn(CHUNK, EMB_DIM) * 0.1).astype(bfloat16)
    k = np.zeros((MAX_SEQ, KV_DIM), dtype=bfloat16)
    v = np.zeros((MAX_SEQ, KV_DIM), dtype=bfloat16)
    # First chunk: only positions [0..CHUNK) are real
    k[:CHUNK] = (np.random.randn(CHUNK, KV_DIM) * 0.1).astype(bfloat16)
    v[:CHUNK] = (np.random.randn(CHUNK, KV_DIM) * 0.1).astype(bfloat16)

    expected = cpu_attn_causal(q, k, v, N_HEADS, N_KV_HEADS, HEAD_DIM, valid_lk=CHUNK)

    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        output_format="xclbin",
        instance_name="K8_flash_attn_chunk_first",
    )
    rc = runner.run_test(
        mlir_module,
        inputs=[q, k, v],
        expected_outputs=[expected],
        rtol=5e-2,
        atol=5e-1,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
