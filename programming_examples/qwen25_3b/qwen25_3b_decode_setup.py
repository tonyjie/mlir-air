# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-3B decode infrastructure: mv_k11008.o + 10-partition LM Head GEMV.

Decode uses ORIG shapes (emb_dim=2048, hidden_dim=11008, n_heads=16,
n_kv_heads=2, head_dim=128). At M=1 the BD-pool exhaustion that forced
prefill padding to 12288 doesn't apply.

Differences from qwen25_1_5b decode setup:
- mv_k11008.o (vs mv_k8960.o) — Down GEMV at K=hidden_dim=11008.
  K=11008 > 8160 → Rule B engaged → down_k_split=86 (86×128=11008).
  Rule D at default tile_m=8 also exceeds (11008×8×8×2 = 1.4 MB > 512 KB)
  → use tile_m=2 (DIM_M_OUTPUT=2).
- 10 partitions × 16384 LM head shape unchanged (vocab=151936 same).
- emb=2048 (vs 1.5B's 1536) — Q/O GEMV M=2048, K=2048.
"""

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# vocab=151936 → 11 × 13824 = 152064 (padded from 151936; 128 padded rows).
# 11 partitions instead of qwen25_1_5b's 10×16384 because at K=emb=2048 the
# tile_m=16 herd_m=8 config used by 1.5B exceeds Rule D L2 cap by exactly 256B
# (C buffer overhead). 13824 = 216×64 lets us drop to tile_m=8 m_input=8 herd_m=8
# (L2: 2048×64×2=256KB ✓) while keeping per-partition launches=216 ≤ 255 (Rule B).
QWEN25_3B_LM_N_PART = 13824
QWEN25_3B_LM_N_PARTITIONS = 11


def ensure_mv_k11008_o():
    """Compile mv.cc with K=11008-tuned tile flags + renamed symbols.

    Mirrors qwen25_1_5b's `ensure_mv_k8960_o` but for hidden_dim=11008:
      -DDIM_M_OUTPUT=2  (tile_m=2 to fit Rule D L2 cap)
      -Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16
      -Dlinalg_fill_bf16=dg_linalg_fill_bf16

    Idempotent — skips if mv_k11008.o exists.
    """
    from _llm_shared.kernel_builder.external_kernels import (
        _compile_kernel,
        KERNEL_OUT_DIR,
    )

    if (KERNEL_OUT_DIR / "mv_k11008.o").exists():
        return

    mv_src = (
        Path(__file__).parent.parent / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    )
    if not mv_src.exists():
        raise FileNotFoundError(f"Cannot find mv.cc at {mv_src}")

    print(f"  Compiling mv_k11008.o (Down GEMV K=11008 renamed symbols)...")
    _compile_kernel(
        mv_src,
        "mv_k11008.o",
        extra_flags=[
            "-DDIM_M_OUTPUT=2",
            "-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16",
            "-Dlinalg_fill_bf16=dg_linalg_fill_bf16",
        ],
    )


def _swap_mv_kernel_name(module, src_name="mv_k8192.o", dst_name="mv_k11008.o"):
    from air.ir import Module

    ir = str(module)
    if src_name not in ir:
        raise RuntimeError(f"Expected '{src_name}' in o_gemv_ffn IR but didn't find it")
    ir = ir.replace(src_name, dst_name)
    return Module.parse(ir, context=module.context)


def compile_qwen25_3b_decode_kernels(cache, config):
    """Compile the three decode ELFs at orig Qwen2.5-3B shapes.

    - rms_gemv_rope: 6 launches (RMSNorm + Q/K/V GEMV + RoPE Q/K)
    - o_gemv_ffn:    8 launches; uses mv_k11008.o for Down GEMV
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

    ensure_mv_k11008_o()

    if "rms_gemv_rope" not in cache.artifacts:
        print(
            f"  Building rms_gemv_rope at Qwen2.5-3B shapes "
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
            f"  Building o_gemv_ffn at Qwen2.5-3B shapes "
            f"(emb={emb_dim}, hidden={hidden_dim}; mv_k11008.o for Down GEMV)..."
        )
        # Down GEMV at K=hidden_dim=11008:
        #   - K > 8160 → Rule B engaged. down_k_split=86 (86×128=11008).
        #   - Rule D at default tile_m=8 exceeds. mv_k11008.o has DIM_M_OUTPUT=2.
        # O/Gate/Up at K=2048: defaults (tile_m=8, m_input=4) work standalone
        # (Phase 1 verified). At K=2048 + tile_m=16 herd_m=8, L2 cap is exceeded
        # by exactly 256B (C buffer overhead), so we CAN'T use tile_m=16.
        # Trying defaults; Rule C may fire on Gate/Up combined channel reads
        # (344 each > 255), but worth testing first since Phase 1 standalone OK.
        # tile_m=8, m_input=8, herd_m=8: Rule B repeat = M/(m_input*herd_m).
        # Gate/Up M=11008 → 11008/64 = 172 ≤ 255 ✓.
        # O at M=2048 → 2048/64 = 32 ✓. L2: 2048×8×8×2 = 256KB ✓.
        # tile_m=m_input → inner_loop=1 (simpler arithmetic).
        mod = build_o_gemv_ffn_module(
            emb_dim,
            hidden_dim,
            tile_m=8,
            m_input=8,
            down_k_split=86,
        )
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
            f"  Building lm_head_gemv with {QWEN25_3B_LM_N_PARTITIONS} partitions "
            f"× {QWEN25_3B_LM_N_PART} (vocab={config.vocab_size})..."
        )
        # tile_m=8 m_input=8 herd_m=8 + 11 partitions × 13824:
        # - L2: 2048×8×8×2 = 256KB (well under 512KB cap)
        # - launches per partition: 13824/(8*8) = 216 ≤ 255 (Rule B ✓)
        # - tile_m=m_input → inner_loop=1 (cleanest Rule C arithmetic)
        cache.compile_and_cache(
            "lm_head_gemv",
            build_lm_head_gemv_module(
                emb_dim,
                n_partitions=QWEN25_3B_LM_N_PARTITIONS,
                tile_m=8,
                m_input=8,
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


def preload_qwen25_3b_lm_head(cache, weights, config):
    """Pre-load the 10 LM-head partition weights into BOs."""
    emb_dim = config.emb_dim
    vocab = config.vocab_size

    lm_head = np.asarray(weights.lm_head, dtype=bfloat16)
    lm_partitions = []
    for p in range(QWEN25_3B_LM_N_PARTITIONS):
        n_start = p * QWEN25_3B_LM_N_PART
        n_end = min(n_start + QWEN25_3B_LM_N_PART, vocab)
        w = np.zeros((QWEN25_3B_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = lm_head[n_start:n_end, :]
        lm_partitions.append(w)

    full_args = [np.zeros(emb_dim, dtype=bfloat16)]
    for p in range(QWEN25_3B_LM_N_PARTITIONS):
        full_args.append(lm_partitions[p])
        full_args.append(np.zeros(QWEN25_3B_LM_N_PART, dtype=bfloat16))

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
        output_indices=[2 + 2 * p for p in range(QWEN25_3B_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(QWEN25_3B_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(QWEN25_3B_LM_N_PARTITIONS)},
        bo_key="lm_head_gemv_qwen25_3b",
    )


def qwen25_3b_npu_lm_head_gemv(cache, weights, config, x_normed_bf16):
    """Run the 10-partition NPU LM Head GEMV on a single hidden state."""
    emb_dim = config.emb_dim
    vocab = config.vocab_size

    lm_inputs = [x_normed_bf16.flatten().astype(bfloat16)]
    for p in range(QWEN25_3B_LM_N_PARTITIONS):
        n_start = p * QWEN25_3B_LM_N_PART
        n_end = min(n_start + QWEN25_3B_LM_N_PART, vocab)
        w = np.zeros((QWEN25_3B_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = np.asarray(
                weights.lm_head[n_start:n_end], dtype=bfloat16
            )
        lm_inputs.append(w)
        lm_inputs.append(np.zeros(QWEN25_3B_LM_N_PART, dtype=bfloat16))

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
        output_indices=[2 + 2 * p for p in range(QWEN25_3B_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(QWEN25_3B_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(QWEN25_3B_LM_N_PARTITIONS)},
        bo_key="lm_head_gemv_qwen25_3b",
    )
    parts = [
        np.asarray(results[2 + 2 * p], dtype=np.float32)
        for p in range(QWEN25_3B_LM_N_PARTITIONS)
    ]
    logits = np.concatenate(parts, axis=0)[:vocab]
    return logits
