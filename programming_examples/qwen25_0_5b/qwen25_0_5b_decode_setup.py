# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-0.5B decode infrastructure: mv_k4864.o + 10-partition LM Head GEMV.

Decode uses ORIG shapes (emb_dim=896, hidden_dim=4864, n_heads=14,
n_kv_heads=2, head_dim=64). At M=1 the BD-pool exhaustion that forced
prefill padding doesn't apply, so we keep decode at the natural shapes.
CPU prefill seeds the KV cache.

Differences from qwen25_1_5b decode setup:
- mv_k4864.o (vs mv_k8960.o) — Down GEMV at K=hidden_dim=4864.
  K=4864 < 8160 so Rule B not engaged → NO down_k_split needed.
  But Rule D at default tile_m=8 exceeds L2 cap (4864×8×8×2 = 608 KB > 512 KB)
  → use tile_m=2 (DIM_M_OUTPUT=2 in mv.cc compile).
- 10 partitions × 16384 LM head shape unchanged (vocab=151936 same as 1.5B).

Public API (mirrors qwen25_decode_setup.py):
- `ensure_mv_k4864_o()`
- `compile_qwen25_0_5b_decode_kernels(cache, config)`
- `qwen25_0_5b_npu_lm_head_gemv(cache, weights, config, x)`
- `preload_qwen25_0_5b_lm_head(cache, weights, config)`
"""

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Shared constants — Qwen2.5 LM Head partition scheme (vocab=151936 → 10 × 16384)
QWEN25_0_5B_LM_N_PART = 16384
QWEN25_0_5B_LM_N_PARTITIONS = 10  # 10 × 16384 = 163840, padded from vocab=151936


def ensure_mv_k4864_o():
    """Compile mv.cc with K=4864-tuned tile flags + renamed symbols.

    Mirrors qwen25_1_5b's `ensure_mv_k8960_o` but for our hidden_dim=4864:
      -DDIM_M_OUTPUT=2  (smaller output tile to fit Rule D L2 cap)
      -Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16
      -Dlinalg_fill_bf16=dg_linalg_fill_bf16

    Idempotent — skips if mv_k4864.o exists in CWD.
    """
    from _llm_shared.kernel_builder.external_kernels import (
        _compile_kernel,
        KERNEL_OUT_DIR,
    )

    if (KERNEL_OUT_DIR / "mv_k4864.o").exists():
        return

    mv_src = (
        Path(__file__).parent.parent / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    )
    if not mv_src.exists():
        raise FileNotFoundError(f"Cannot find mv.cc at {mv_src}")

    print(f"  Compiling mv_k4864.o (Down GEMV K=4864 renamed symbols)...")
    _compile_kernel(
        mv_src,
        "mv_k4864.o",
        extra_flags=[
            "-DDIM_M_OUTPUT=2",
            "-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16",
            "-Dlinalg_fill_bf16=dg_linalg_fill_bf16",
        ],
    )


def _swap_mv_kernel_name(module, src_name="mv_k8192.o", dst_name="mv_k4864.o"):
    """Replace the mv_k8192.o link_with strings in a built module's IR.

    The shared o_gemv_ffn_multi builder hardcodes 'mv_k8192.o'. For
    Qwen2.5-0.5B we need 'mv_k4864.o' (K=4864 with DIM_M_OUTPUT=2).
    """
    from air.ir import Module

    ir = str(module)
    if src_name not in ir:
        raise RuntimeError(f"Expected '{src_name}' in o_gemv_ffn IR but didn't find it")
    ir = ir.replace(src_name, dst_name)
    return Module.parse(ir, context=module.context)


def compile_qwen25_0_5b_decode_kernels(cache, config):
    """Compile the three decode ELFs at orig Qwen2.5-0.5B shapes.

    - rms_gemv_rope: 6 launches (RMSNorm + Q/K/V GEMV + RoPE Q/K)
    - o_gemv_ffn:    8 launches (O + add + RMSNorm + Gate/Up + SiLU + Down + add)
                     uses mv_k4864.o for Down GEMV
    - lm_head_gemv:  10 partitions × 16384 (vocab=151936)
    """
    from llama3.multi_launch_builder.rms_gemv_rope_multi import (
        build_rms_gemv_rope_module,
    )
    from llama3.multi_launch_builder.o_gemv_ffn_multi import build_o_gemv_ffn_module
    from llama3.multi_launch_builder.lm_head_gemv_multi import (
        build_lm_head_gemv_module,
    )

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    ensure_mv_k4864_o()

    if "rms_gemv_rope" not in cache.artifacts:
        print(
            f"  Building rms_gemv_rope at Qwen2.5-0.5B shapes "
            f"(emb={emb_dim}, kv={kv_dim}, heads={n_heads}/{n_kv_heads}, hd={head_dim})..."
        )
        cache.compile_and_cache(
            "rms_gemv_rope",
            build_rms_gemv_rope_module(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim),
            {
                "verbose": cache.verbose,
                "output_format": "elf",
                "instance_name": "rms_gemv_rope",
                "omit_while_true_loop": False,
            },
        )

    if "o_gemv_ffn" not in cache.artifacts:
        print(
            f"  Building o_gemv_ffn at Qwen2.5-0.5B shapes "
            f"(emb={emb_dim}, hidden={hidden_dim}; mv_k4864.o for Down GEMV)..."
        )
        # Down GEMV at K=4864: K < 8160 → no down_k_split needed (Rule B not engaged).
        # But Rule D at default tile_m=8 exceeds: 4864×8×8×2 = 608 KB > 512 KB cap.
        # → use down's own tile_m=2 path via mv_k4864.o (DIM_M_OUTPUT=2).
        # Default tile_m=8, m_input=4 fine for O/Gate/Up at our smaller K=896 / hidden=4864.
        # Rule C combined channel check:
        #   Q/O at M=896: 896/(8*8) × (8/4) = 14 × 2 = 28
        #   Gate/Up at M=4864: 4864/(8*8) × 2 = 152
        #   Combined per channel: well under 255. No tile_m bump needed.
        mod = build_o_gemv_ffn_module(emb_dim, hidden_dim)
        mod = _swap_mv_kernel_name(mod)
        cache.compile_and_cache(
            "o_gemv_ffn",
            mod,
            {
                "verbose": cache.verbose,
                "output_format": "elf",
                "instance_name": "o_gemv_ffn",
                "omit_while_true_loop": False,
                "omit_pingpong": "all",
            },
        )

    if "lm_head_gemv" not in cache.artifacts:
        print(
            f"  Building lm_head_gemv with {QWEN25_0_5B_LM_N_PARTITIONS} partitions "
            f"× {QWEN25_0_5B_LM_N_PART} (vocab={config.vocab_size})..."
        )
        # Same B-DMA-fires constraint as qwen25_1_5b: per partition the input
        # vector is read launch_count × (tile_m/m_input) times. Default
        # (tile_m=8, m_input=4) gives 16384/(8*8) × 2 = 512 → exceeds 255.
        # tile_m=16, m_input=16 → 128 × 1 = 128 ✓.
        cache.compile_and_cache(
            "lm_head_gemv",
            build_lm_head_gemv_module(
                emb_dim,
                n_partitions=QWEN25_0_5B_LM_N_PARTITIONS,
                tile_m=16,
                m_input=16,
                herd_m=8,
            ),
            {
                "verbose": cache.verbose,
                "output_format": "elf",
                "instance_name": "lm_head_gemv",
                "omit_while_true_loop": False,
            },
        )

    cache._save_manifest()


def preload_qwen25_0_5b_lm_head(cache, weights, config):
    """Pre-load the 10 LM-head partition weights into BOs.

    Issues a real warm-up `cache.load_and_run` so per-bo_key BOs get
    allocated AND the partition weights get written via the same path
    `qwen25_0_5b_npu_lm_head_gemv` uses later (with `bo_key=...
    + static_input_indices`).
    """
    emb_dim = config.emb_dim
    vocab = config.vocab_size

    lm_head = np.asarray(weights.lm_head, dtype=bfloat16)  # (vocab, emb_dim)
    lm_partitions = []
    for p in range(QWEN25_0_5B_LM_N_PARTITIONS):
        n_start = p * QWEN25_0_5B_LM_N_PART
        n_end = min(n_start + QWEN25_0_5B_LM_N_PART, vocab)
        w = np.zeros((QWEN25_0_5B_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = lm_head[n_start:n_end, :]
        lm_partitions.append(w)

    full_args = [np.zeros(emb_dim, dtype=bfloat16)]
    for p in range(QWEN25_0_5B_LM_N_PARTITIONS):
        full_args.append(lm_partitions[p])
        full_args.append(np.zeros(QWEN25_0_5B_LM_N_PART, dtype=bfloat16))

    backend_kwargs = {
        "verbose": cache.verbose,
        "output_format": "elf",
        "instance_name": "lm_head_gemv",
        "omit_while_true_loop": False,
    }
    cache.load_and_run(
        "lm_head_gemv",
        backend_kwargs,
        *full_args,
        output_indices=[2 + 2 * p for p in range(QWEN25_0_5B_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(QWEN25_0_5B_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(QWEN25_0_5B_LM_N_PARTITIONS)},
        bo_key="lm_head_gemv_qwen25_0_5b",
    )


def qwen25_0_5b_npu_lm_head_gemv(cache, weights, config, x_normed_bf16):
    """Run the 10-partition NPU LM Head GEMV on a single hidden state vector.

    Returns logits (vocab_size,) as float32.
    """
    emb_dim = config.emb_dim
    vocab = config.vocab_size

    lm_inputs = [x_normed_bf16.flatten().astype(bfloat16)]
    for p in range(QWEN25_0_5B_LM_N_PARTITIONS):
        n_start = p * QWEN25_0_5B_LM_N_PART
        n_end = min(n_start + QWEN25_0_5B_LM_N_PART, vocab)
        w = np.zeros((QWEN25_0_5B_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = np.asarray(
                weights.lm_head[n_start:n_end], dtype=bfloat16
            )
        lm_inputs.append(w)
        lm_inputs.append(np.zeros(QWEN25_0_5B_LM_N_PART, dtype=bfloat16))

    backend_kwargs = {
        "verbose": cache.verbose,
        "output_format": "elf",
        "instance_name": "lm_head_gemv",
        "omit_while_true_loop": False,
    }
    results = cache.load_and_run(
        "lm_head_gemv",
        backend_kwargs,
        *lm_inputs,
        output_indices=[2 + 2 * p for p in range(QWEN25_0_5B_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(QWEN25_0_5B_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(QWEN25_0_5B_LM_N_PARTITIONS)},
        bo_key="lm_head_gemv_qwen25_0_5b",
    )
    parts = [
        np.asarray(results[2 + 2 * p], dtype=np.float32)
        for p in range(QWEN25_0_5B_LM_N_PARTITIONS)
    ]
    logits = np.concatenate(parts, axis=0)[:vocab]
    return logits
