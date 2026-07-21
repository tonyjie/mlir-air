# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Layer-0 bring-up harness for the NPU vision encoder (A3-5 Step 2).

Runs ONLY encoder layer 0 on NPU against oracle layer_hidden[0], plus a
sub-op localization (LN1/qkv/attn/o/LN2/fc1/gelu/fc2) vs a cpu_vit_layer run on
the same input, so a low cosine points at the offending sub-op. Not the gate
(test_full_vit.py is); a scaffold to de-risk before scaling to 12 layers.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from vision_weights import load_vision_weights, SigLIPVisionConfig
from vision_cpu_helpers import cpu_vit_layer, layer_norm, mha_bidirectional
from vision_prefill import (
    compile_all_kernels,
    run_vit_block,
    _run_layer_norm,
    _run_gemm,
    _run_gelu,
    _run_flash_attention,
)
from shared.infra.cache import KernelCache, Profiler


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    cfg = SigLIPVisionConfig()
    weights = load_vision_weights("lerobot/smolvla_base", dtype=bfloat16, config=cfg)
    o = np.load(_HERE / "vision_oracle.npz")
    patch_embed = o["patch_embed"]  # (1024, 768) f32
    oracle_l0 = o["layer_hidden"][0]  # (1024, 768) f32

    cache = KernelCache("vision_kernel_cache", verbose=False, profiler=Profiler())
    compile_all_kernels(cache, cfg, 1024)

    lw = weights.layers[0]
    x_bf16 = patch_embed.astype(bfloat16)
    seq, emb, hidden = 1024, cfg.emb_dim, cfg.hidden_dim

    # --- NPU sub-ops (localization) ---
    h_npu = _run_layer_norm(cache, x_bf16, lw.ln1_w, lw.ln1_b, seq, emb, "ln1_L0")
    h_cpu = layer_norm(patch_embed, lw.ln1_w, lw.ln1_b, cfg.layer_norm_eps)
    print(f"LN1        cos = {cosine(h_npu, h_cpu):.6f}")

    q = _run_gemm(cache, "gemm_qkvo", h_npu, lw.wq, seq, emb, "wq_L0")
    q = (q.astype(np.float32) + lw.bq.astype(np.float32)).astype(bfloat16)
    q_cpu = h_cpu @ lw.wq.astype(np.float32) + lw.bq.astype(np.float32)
    print(f"q proj     cos = {cosine(q, q_cpu):.6f}")

    k = _run_gemm(cache, "gemm_qkvo", h_npu, lw.wk, seq, emb, "wk_L0")
    k = (k.astype(np.float32) + lw.bk.astype(np.float32)).astype(bfloat16)
    v = _run_gemm(cache, "gemm_qkvo", h_npu, lw.wv, seq, emb, "wv_L0")
    v = (v.astype(np.float32) + lw.bv.astype(np.float32)).astype(bfloat16)

    attn = _run_flash_attention(cache, q, k, v, cfg, seq)
    attn_cpu = mha_bidirectional(
        q.astype(np.float32),
        k.astype(np.float32),
        v.astype(np.float32),
        cfg.n_heads,
        cfg.head_dim,
        cfg.attn_scale,
    )
    print(f"FA         cos = {cosine(attn, attn_cpu):.6f}")

    out_npu = run_vit_block(x_bf16, lw, cfg, cache, layer_idx=0)
    out_cpu = cpu_vit_layer(patch_embed, lw, cfg)
    print(f"block vs cpu_layer  cos = {cosine(out_npu, out_cpu):.6f}")
    c = cosine(out_npu, oracle_l0)
    print(f"block vs ORACLE l0  cos = {c:.6f}  {'PASS' if c > 0.98 else 'FAIL'}")


if __name__ == "__main__":
    main()
