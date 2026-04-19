# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-1.5B decode infrastructure: mv_k8960.o, 10-partition LM Head GEMV.

Uses ORIG shapes for decode (emb_dim=1536, hidden_dim=8960, n_heads=12,
n_kv_heads=2, head_dim=128). At M=1 the BD-pool exhaustion that forced the
prefill padding doesn't apply, so we keep decode at the natural shapes for
simplicity. CPU prefill seeds the KV cache.

Public API
----------
- `ensure_mv_k8960_o()`: compile mv.cc with -DDIM_M_OUTPUT=2 and renamed
  symbols, output `mv_k8960.o`.
- `compile_qwen25_decode_kernels(cache, config)`: compiles
  rms_gemv_rope, o_gemv_ffn (with mv_k8960.o linkage), lm_head_gemv
  (10 partitions × 16384 = 163840, padded from vocab=151936).
- `qwen25_npu_lm_head_gemv(cache, weights, config, x)`: invoke the
  10-partition LM head GEMV, return logits trimmed to vocab_size.
- `preload_qwen25_decode_weights(cache, weights, config)`: per-layer +
  LM-head BO preload (mirrors llama3_inference._preload_decode_weights but
  with 10 partitions and orig shapes).
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Constants — Qwen2.5 LM Head partition scheme (vocab=151936 → 10 × 16384)
# ---------------------------------------------------------------------------

QWEN25_LM_N_PART = 16384
QWEN25_LM_N_PARTITIONS = 10  # 10 × 16384 = 163840, padded from vocab=151936


# ---------------------------------------------------------------------------
# mv_k8960.o — Down GEMV at K=hidden_dim=8960
# ---------------------------------------------------------------------------


def ensure_mv_k8960_o():
    """Compile mv.cc with K=8960-tuned tile flags + renamed symbols.

    Mirrors `llama3_decode._ensure_mv_k8192_o` but for Qwen2.5's hidden_dim:
      -Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16
      -DDIM_M_OUTPUT=2  (smaller output tile for K=8960)

    Idempotent — skips if mv_k8960.o exists in CWD.
    """
    if Path("mv_k8960.o").exists():
        return

    mv_src = (
        Path(__file__).parent.parent / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    )
    if not mv_src.exists():
        raise FileNotFoundError(f"Cannot find mv.cc at {mv_src}")

    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    clang = os.path.join(peano_dir, "bin", "clang++") if peano_dir else "clang++"

    aieopt_dir = os.path.dirname(
        os.path.dirname(
            subprocess.check_output(["which", "aie-opt"], text=True).strip()
        )
    )
    flags = [
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-DNDEBUG",
        f"-I{aieopt_dir}/include",
        # DIM_M_OUTPUT=2 matches default down_tile_m=2 (the K-loop fits via
        # the new down_k_split=14 knob in matvec, not via tile_m bump).
        "-DDIM_M_OUTPUT=2",
        "-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16",
        "-Dlinalg_fill_bf16=dg_linalg_fill_bf16",
        "-c",
        str(mv_src),
        "-o",
        "mv_k8960.o",
    ]
    print(f"  Compiling mv_k8960.o (Down GEMV K=8960 renamed symbols)...")
    subprocess.run([clang] + flags, check=True)


# ---------------------------------------------------------------------------
# Module compile helpers
# ---------------------------------------------------------------------------


def _swap_mv_kernel_name(module, src_name="mv_k8192.o", dst_name="mv_k8960.o"):
    """Replace the mv_k8192.o link_with strings in a built module's IR.

    The shared o_gemv_ffn_multi builder hardcodes 'mv_k8192.o' in two places
    (link_with attr in the herd body and in the private func declarations).
    For Qwen2.5 we need 'mv_k8960.o'. We round-trip through string IR.
    """
    from air.ir import Module

    ir = str(module)
    if src_name not in ir:
        raise RuntimeError(f"Expected '{src_name}' in o_gemv_ffn IR but didn't find it")
    ir = ir.replace(src_name, dst_name)
    # Re-parse in the original module's context (already has air dialect registered).
    return Module.parse(ir, context=module.context)


def compile_qwen25_decode_kernels(cache, config):
    """Compile the three decode ELFs at orig Qwen2.5 shapes.

    - rms_gemv_rope: 6 launches (RMSNorm + Q/K/V GEMV + RoPE Q/K)
    - o_gemv_ffn:    8 launches (O GEMV + add + RMSNorm + Gate/Up GEMV +
                     SiLU+mul + Down GEMV + add) — uses mv_k8960.o
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

    ensure_mv_k8960_o()

    if "rms_gemv_rope" not in cache.artifacts:
        print(
            f"  Building rms_gemv_rope at Qwen2.5 shapes "
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
            f"  Building o_gemv_ffn at Qwen2.5 shapes "
            f"(emb={emb_dim}, hidden={hidden_dim}; mv_k8960.o for Down GEMV)..."
        )
        # Down GEMV at K=hidden_dim=8960: auto-split would give outer=280
        # > 255 HW limit. Use the new `down_k_split=14` knob in matvec
        # (additive, back-compat default-None) to pre-split as
        # 14 × 640 — outer 14 ✓, inner 640 ✓. L2 stays at default
        # (tile_m=2, herd_m=8) → 8960*8*2*2 = 287KB ✓ < 512KB.
        # Default tile_m and herd_m work for O/Gate/Up too.
        # Two friction points at Qwen2.5 hidden_dim=8960:
        # (1) The B-input shim DMA fires `launch_count × (tile_m/m_input)`
        #     times per GEMV. For Gate/Up at M=8960 with default
        #     (tile_m=8, m_input=4): 8960/(8*8) × 2 = 280 → exceeds 255.
        #     Fix: tile_m=16, m_input=16 → 70 launches × 1 inner = 70 each.
        #     Combined Gate+Up = 140 ✓; with O = 152 ✓.
        #     L2 fits: K=1536 × herd=8 × tile_m=16 × 2 = 384KB ✓ < 512KB.
        # (2) Down GEMV K=8960 DMA auto-splits to outer 280. Fixed by
        #     down_k_split=70 (70 × 128 = 8960).
        mod = build_o_gemv_ffn_module(
            emb_dim,
            hidden_dim,
            tile_m=16,
            m_input=16,
            down_k_split=70,
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
            f"  Building lm_head_gemv with {QWEN25_LM_N_PARTITIONS} partitions "
            f"× {QWEN25_LM_N_PART} (vocab={config.vocab_size})..."
        )
        # Same B-DMA-fires constraint as o_gemv_ffn: per partition the input
        # vector is read launch_count × (tile_m/m_input) times. Default
        # (tile_m=8, m_input=4) gives 16384/(8*8) × 2 = 512 per partition →
        # exceeds 255. tile_m=16, m_input=16 → 128 × 1 = 128 ✓.
        cache.compile_and_cache(
            "lm_head_gemv",
            build_lm_head_gemv_module(
                emb_dim,
                n_partitions=QWEN25_LM_N_PARTITIONS,
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


# ---------------------------------------------------------------------------
# Pre-load + invoke for the 10-partition LM Head GEMV
# ---------------------------------------------------------------------------


def preload_qwen25_lm_head(cache, weights, config):
    """Pre-load the 10 LM-head partition weights into BOs."""
    emb_dim = config.emb_dim
    vocab = config.vocab_size

    lm_head = np.asarray(weights.lm_head, dtype=bfloat16)  # (vocab, emb_dim)
    lm_partitions = []
    for p in range(QWEN25_LM_N_PARTITIONS):
        n_start = p * QWEN25_LM_N_PART
        n_end = min(n_start + QWEN25_LM_N_PART, vocab)
        w = np.zeros((QWEN25_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = lm_head[n_start:n_end, :]
        lm_partitions.append(w)

    # Build static input dict (arg index -> tensor) for cache.preload_static_inputs.
    static = {1 + 2 * p: lm_partitions[p] for p in range(QWEN25_LM_N_PARTITIONS)}
    full_args = [np.zeros(emb_dim, dtype=bfloat16)]
    for p in range(QWEN25_LM_N_PARTITIONS):
        full_args.append(lm_partitions[p])
        full_args.append(np.zeros(QWEN25_LM_N_PART, dtype=bfloat16))

    backend_kwargs = {
        "verbose": cache.verbose,
        "output_format": "elf",
        "instance_name": "lm_head_gemv",
        "omit_while_true_loop": False,
    }
    try:
        cache.preload_static_inputs(
            "lm_head_gemv",
            backend_kwargs,
            [("lm_head_gemv_qwen25", static, full_args)],
        )
    except Exception as e:
        # Fall back to lazy preload via the first invocation
        print(
            f"  LM head preload via preload_static_inputs failed ({e}); will lazy-load"
        )


def qwen25_npu_lm_head_gemv(cache, weights, config, x_normed_bf16):
    """Run the 10-partition NPU LM Head GEMV on a single hidden state vector.

    Returns logits (vocab_size,) as float32.
    """
    emb_dim = config.emb_dim
    vocab = config.vocab_size

    lm_inputs = [x_normed_bf16.flatten().astype(bfloat16)]
    for p in range(QWEN25_LM_N_PARTITIONS):
        n_start = p * QWEN25_LM_N_PART
        n_end = min(n_start + QWEN25_LM_N_PART, vocab)
        w = np.zeros((QWEN25_LM_N_PART, emb_dim), dtype=bfloat16)
        if n_end > n_start:
            w[: n_end - n_start, :] = np.asarray(
                weights.lm_head[n_start:n_end], dtype=bfloat16
            )
        lm_inputs.append(w)
        lm_inputs.append(np.zeros(QWEN25_LM_N_PART, dtype=bfloat16))

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
        output_indices=[2 + 2 * p for p in range(QWEN25_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(QWEN25_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(QWEN25_LM_N_PARTITIONS)},
        bo_key="lm_head_gemv_qwen25",
    )
    parts = [
        np.asarray(results[2 + 2 * p], dtype=np.float32)
        for p in range(QWEN25_LM_N_PARTITIONS)
    ]
    logits = np.concatenate(parts, axis=0)[:vocab]
    return logits
