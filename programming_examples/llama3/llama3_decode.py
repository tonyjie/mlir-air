#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache.
Runs prefill first to populate KV cache, then decodes token-by-token.

Usage:
    cd build_peano
    python3 ../llama3_decode.py --compile-only
    python3 ../llama3_decode.py --run-only --n-tokens 10 --profile
    python3 ../llama3_decode.py --run-only --n-tokens 1 --verify
"""

import argparse
import os
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama3_weights import LlamaConfig, load_weights, generate_rope_lut
from _llm_shared.kernel_builder.cache import KernelCache, prepare_air_project
from _llm_shared.kernel_builder.gemm_builder import _build_gemm_module

# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------

# Backend kwargs for different kernel types
_GEMV_K2048_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "",  # ping-pong ON for K=2048
    "runtime_loop_tiling_sizes": [16, 16],
    "use_lock_race_condition_fix": False,
}

_GEMV_K8192_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "all",  # ping-pong OFF for K=8192 (L1 too tight)
    "runtime_loop_tiling_sizes": [16, 16],
    "use_lock_race_condition_fix": False,
}

_SIMPLE_BACKEND = {"omit_while_true_loop": False}


def _ensure_mv_k8192_o():
    """Ensure mv_k8192.o exists in CWD for the o_gemv_ffn merged kernel.

    The Down GEMV (K=8192) shares the same C++ source as mv.o but needs a
    renamed entry point (@dg_matvec_vectorized_bf16_bf16) to avoid type
    conflicts with K=2048 GEMVs in the same ELF.  We compile with:
      -Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16
      -DDIM_M_OUTPUT=2  (tile_m=2 for K=8192)
    """
    from pathlib import Path

    if Path("mv_k8192.o").exists():
        return

    mv_src = (
        Path(__file__).parent.parent / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    )
    if not mv_src.exists():
        raise FileNotFoundError(f"Cannot find mv.cc at {mv_src}")

    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    clang = os.path.join(peano_dir, "bin", "clang++") if peano_dir else "clang++"

    import subprocess

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
        "-DDIM_M_OUTPUT=2",
        "-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16",
        "-Dlinalg_fill_bf16=dg_linalg_fill_bf16",
        "-c",
        str(mv_src),
        "-o",
        "mv_k8192.o",
    ]
    print(f"  Compiling mv_k8192.o (Down GEMV K=8192 renamed symbols)...")
    subprocess.check_call([clang] + flags)


def compile_decode_kernels(cache, config):
    """Compile the 3 merged decode kernels."""
    from _llm_shared.kernel_builder.external_kernels import compile_all_external_kernels

    compile_all_external_kernels(head_dim=config.head_dim)

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Compiling decode kernels (2-call merged pipeline)...")
    print(f"{'='*60}\n")

    # Ensure mv_k8192.o exists for the o_gemv_ffn kernel
    _ensure_mv_k8192_o()

    # 1. rms_gemv_rope: RMSNorm + QKV GEMV + RoPE Q+K (6 launches, 13 args)
    from llama3.multi_launch_builder.rms_gemv_rope_multi import (
        build_rms_gemv_rope_module,
    )

    cache.compile_and_cache(
        "rms_gemv_rope",
        build_rms_gemv_rope_module(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim),
        {
            "verbose": cache.verbose,
            "output_format": "elf",
            "instance_name": "rms_gemv_rope",
            **_GEMV_K2048_BACKEND,
        },
    )

    # 2. o_gemv_ffn: O GEMV + Add + RMSNorm + Gate/Up GEMV + SiLU*mul
    #                + Down GEMV + Add (8 launches, 15 args)
    from llama3.multi_launch_builder.o_gemv_ffn_multi import build_o_gemv_ffn_module

    cache.compile_and_cache(
        "o_gemv_ffn",
        build_o_gemv_ffn_module(emb_dim, hidden_dim),
        {
            "verbose": cache.verbose,
            "output_format": "elf",
            "instance_name": "o_gemv_ffn",
            "omit_pingpong": "all",
            **{k: v for k, v in _GEMV_K2048_BACKEND.items() if k != "omit_pingpong"},
        },
    )

    # 3. LM Head GEMV multi-launch: 8-partition GEMV in one ELF
    from llama3.multi_launch_builder.lm_head_gemv_multi import (
        build_lm_head_gemv_module,
    )

    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(emb_dim),
        {
            "verbose": cache.verbose,
            "output_format": "elf",
            "instance_name": "lm_head_gemv",
            **_GEMV_K2048_BACKEND,
        },
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


def decode_attention_cpu(
    q, k_cache, v_cache, current_pos, n_heads, n_kv_heads, head_dim
):
    """Single-query attention with KV cache.

    Args:
        q: (emb_dim,) — query vector for current token
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys [0:current_pos+1]
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values [0:current_pos+1]
        current_pos: current token position (0-indexed)
        n_heads: number of Q heads (32)
        n_kv_heads: number of KV heads (8)
        head_dim: head dimension (64)

    Returns:
        attn_out: (emb_dim,) — attention output
    """
    group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = current_pos + 1

    q_heads = q.astype(np.float32).reshape(n_heads, head_dim)
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)  # (n_kv, seq, hd)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale  # (seq,)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]  # (hd,)

    return out.reshape(-1).astype(bfloat16)


# ---------------------------------------------------------------------------
# Single decode transformer block
# ---------------------------------------------------------------------------


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
):
    """Run one transformer block for a single decode token.

    Args:
        x_bf16: (emb_dim,) input token embedding
        layer_weights: LayerWeights for this layer
        cache: KernelCache
        config: LlamaConfig
        k_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's K cache
        v_cache_layer: (n_kv_heads, max_seq, head_dim) — this layer's V cache
        current_pos: current token position
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT

    Returns:
        output: (emb_dim,) — block output
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = None  # Set by caller via layer_weights._layer_idx
    if hasattr(layer_weights, "_layer_idx"):
        layer_idx = layer_weights._layer_idx

    def _run(name, backend, *inputs, static_indices=None, **kwargs):
        # Per-layer BO key: same XRT context, separate BOs for weight isolation
        bk = (
            f"{name}_L{layer_idx}" if static_indices and layer_idx is not None else None
        )
        return cache.load_and_run(
            name,
            backend,
            *inputs,
            bo_key=bk,
            static_input_indices=static_indices,
            **kwargs,
        )

    # --- Call 1: rms_gemv_rope (6 launches, 13 args) ---
    # RMSNorm + Q/K/V GEMV + RoPE Q + RoPE K
    _RGR_BACKEND = {
        "output_format": "elf",
        "instance_name": "rms_gemv_rope",
        **_GEMV_K2048_BACKEND,
    }

    x_in = x_bf16.flatten().astype(bfloat16)
    w_norm = layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16)
    normed_buf = np.zeros(emb_dim, dtype=bfloat16)
    wq = layer_weights._wq_t
    q_buf = np.zeros(emb_dim, dtype=bfloat16)
    wk = layer_weights._wk_t
    k_buf = np.zeros(kv_dim, dtype=bfloat16)
    wv = layer_weights._wv_t
    v_buf = np.zeros(kv_dim, dtype=bfloat16)

    # RoPE LUT for current position
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, 64)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)
    q_roped_buf = np.zeros(emb_dim, dtype=bfloat16)
    k_roped_buf = np.zeros(kv_dim, dtype=bfloat16)

    results = _run(
        "rms_gemv_rope",
        _RGR_BACKEND,
        x_in,  # arg0
        w_norm,  # arg1
        normed_buf,  # arg2 (intermediate)
        wq,  # arg3 (static)
        q_buf,  # arg4 (intermediate)
        wk,  # arg5 (static)
        k_buf,  # arg6 (intermediate)
        wv,  # arg7 (static)
        v_buf,  # arg8 (intermediate/output)
        lut_q,  # arg9
        lut_k,  # arg10
        q_roped_buf,  # arg11 (intermediate/output)
        k_roped_buf,  # arg12 (intermediate/output)
        output_indices=[8, 11, 12],
        static_indices={3, 5, 7},
        intermediate_indices={2, 4, 6, 8, 11, 12},
    )
    v = results[8].astype(bfloat16)
    q_roped = results[11].reshape(n_heads, head_dim).astype(bfloat16)
    k_roped = results[12].reshape(n_kv_heads, head_dim).astype(bfloat16)

    # Update KV cache
    k_cache_layer[:, current_pos, :] = k_roped
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- CPU Attention ---
    attn_out = decode_attention_cpu(
        q_roped.flatten(),
        k_cache_layer,
        v_cache_layer,
        current_pos,
        n_heads,
        n_kv_heads,
        head_dim,
    )

    # --- Call 2: o_gemv_ffn (8 launches, 15 args) ---
    # O GEMV + Add + RMSNorm + Gate/Up GEMV + SiLU*mul + Down GEMV + Add
    _OGF_BACKEND = {
        "output_format": "elf",
        "instance_name": "o_gemv_ffn",
        "omit_pingpong": "all",
        **{k: v for k, v in _GEMV_K2048_BACKEND.items() if k != "omit_pingpong"},
    }

    wo = layer_weights._wo_t
    proj_buf = np.zeros(emb_dim, dtype=bfloat16)
    x_residual = x_bf16.flatten().astype(bfloat16)
    res1_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_norm2 = layer_weights.ffn_norm.reshape(emb_dim).astype(bfloat16)
    normed2_buf = np.zeros(emb_dim, dtype=bfloat16)
    w_gate = layer_weights._wgate_t
    gate_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_up = layer_weights._wup_t
    up_buf = np.zeros(hidden_dim, dtype=bfloat16)
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)
    w_down = layer_weights._wdown_t
    down_buf = np.zeros(emb_dim, dtype=bfloat16)
    output_buf = np.zeros(emb_dim, dtype=bfloat16)

    results = _run(
        "o_gemv_ffn",
        _OGF_BACKEND,
        wo,  # arg0 (static)
        attn_out,  # arg1
        proj_buf,  # arg2 (intermediate)
        x_residual,  # arg3
        res1_buf,  # arg4 (intermediate)
        w_norm2,  # arg5
        normed2_buf,  # arg6 (intermediate)
        w_gate,  # arg7 (static)
        gate_buf,  # arg8 (intermediate)
        w_up,  # arg9 (static)
        up_buf,  # arg10 (intermediate)
        swiglu_buf,  # arg11 (intermediate)
        w_down,  # arg12 (static)
        down_buf,  # arg13 (intermediate)
        output_buf,  # arg14 (intermediate/output)
        output_indices=[14],
        static_indices={0, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
    )
    output = results[14].astype(bfloat16)

    return output


# ---------------------------------------------------------------------------
# Full decode generation
# ---------------------------------------------------------------------------


def generate(
    prompt_tokens,
    weights,
    config,
    cache,
    rope_lut_bf16,
    n_tokens=10,
    profile=False,
    verify=False,
):
    """Run prefill + multi-token decode generation.

    Args:
        prompt_tokens: list of token IDs
        weights: LlamaWeights
        config: LlamaConfig
        cache: KernelCache
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT
        n_tokens: number of tokens to generate
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = len(prompt_tokens)
    max_seq = seq_len + n_tokens

    print(f"\n{'='*60}")
    print(f"LLAMA Decode: prompt_len={seq_len}, n_tokens={n_tokens}")
    print(f"{'='*60}\n")

    # --- Prefill phase (CPU reference for KV cache) ---
    print("Running CPU prefill for KV cache...")
    from llama3_reference import transformer_block as cpu_block, rms_norm

    t_prefill_start = time.time()

    x_embed = weights.embed_table[prompt_tokens].astype(bfloat16)
    rope_lut_f32 = rope_lut_bf16[:seq_len].astype(np.float32)

    # KV cache: stores post-RoPE K and raw V
    k_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    x = x_embed.astype(np.float32)
    for layer_idx in range(config.n_layers):
        lw = weights.layers[layer_idx]
        x_out, intermediates = cpu_block(x, lw, rope_lut_f32, config)

        # Use post-RoPE K from CPU reference intermediates
        k_roped = intermediates["k_roped"].astype(np.float32)  # (seq, kv_dim)
        v_raw = intermediates["v"].astype(np.float32)  # (seq, kv_dim)

        # Store in cache: (n_kv_heads, seq, head_dim)
        k_cache[layer_idx, :, :seq_len, :] = (
            k_roped.reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
            .astype(bfloat16)
        )
        v_cache[layer_idx, :, :seq_len, :] = (
            v_raw.reshape(seq_len, n_kv_heads, head_dim)
            .transpose(1, 0, 2)
            .astype(bfloat16)
        )
        x = x_out

    # Final norm + LM Head (CPU)
    x_normed = rms_norm(x, weights.final_norm.astype(np.float32))
    logits = x_normed @ weights.lm_head.astype(np.float32).T
    # Find actual prompt length (before EOS padding)
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    prompt_len = len([t for t in prompt_tokens if t != _tok.eos_token_id])
    pred_pos = prompt_len - 1
    prefill_token = int(np.argmax(logits[pred_pos]))

    t_prefill = time.time() - t_prefill_start
    print(f"Prefill done in {t_prefill:.2f}s. First decode token: {prefill_token}")

    # The decode starts from the prefill's predicted token
    # (not from the hidden state — we embed the predicted token)

    # Pre-transpose all weights for GEMV (done once, not per-token)
    kv_dim = n_kv_heads * head_dim
    hidden_dim = config.hidden_dim
    if not hasattr(weights, "_decode_weights_transposed"):
        print("Pre-transposing weights for GEMV...")
        for lw in weights.layers:
            lw._wq_t = np.ascontiguousarray(
                lw.wq.astype(bfloat16).reshape(emb_dim, emb_dim).T
            )
            lw._wk_t = np.ascontiguousarray(
                lw.wk.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )
            lw._wv_t = np.ascontiguousarray(
                lw.wv.astype(bfloat16).reshape(emb_dim, kv_dim).T
            )
            lw._wo_t = np.ascontiguousarray(
                lw.wo.astype(bfloat16).reshape(emb_dim, emb_dim).T
            )
            lw._wgate_t = np.ascontiguousarray(
                lw.w_gate.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )
            lw._wup_t = np.ascontiguousarray(
                lw.w_up.astype(bfloat16).reshape(emb_dim, hidden_dim).T
            )
            lw._wdown_t = np.ascontiguousarray(
                lw.w_down.astype(bfloat16).reshape(hidden_dim, emb_dim).T
            )
        weights._decode_weights_transposed = True
    # Tag layers with index for per-layer kernel name isolation
    for i, lw in enumerate(weights.layers):
        lw._layer_idx = i

    # Pre-load ALL decode weights into per-layer BOs (outside profiling scope)
    vocab_size = weights.lm_head.shape[0]  # 128256
    n_part = 16384
    n_lm_partitions = 8

    if not hasattr(weights, "_decode_weights_preloaded_to_bos"):
        print("Pre-loading decode weights into per-layer BOs...")

        # 1. Transformer block weights (rms_gemv_rope + o_gemv_ffn per layer)
        _RGR_BACKEND = {
            "output_format": "elf",
            "instance_name": "rms_gemv_rope",
            **_GEMV_K2048_BACKEND,
        }
        _OGEMV_FFN_BACKEND = {
            "output_format": "elf",
            "instance_name": "o_gemv_ffn",
            "omit_while_true_loop": False,
            "omit_pingpong": "all",
            "runtime_loop_tiling_sizes": [16, 16],
            "use_lock_race_condition_fix": False,
        }
        rope_lut_pos_dummy = np.zeros(n_heads * head_dim, dtype=bfloat16)
        rope_lut_k_dummy = np.zeros(n_kv_heads * head_dim, dtype=bfloat16)

        for layer_idx in range(config.n_layers):
            lw = weights.layers[layer_idx]
            # rms_gemv_rope: allocate + write weights
            cache.load_and_run(
                "rms_gemv_rope",
                _RGR_BACKEND,
                np.zeros(emb_dim, dtype=bfloat16),  # x_in
                lw.attn_norm.reshape(emb_dim).astype(bfloat16),  # norm_w
                np.zeros(emb_dim, dtype=bfloat16),  # normed
                lw._wq_t,  # wq
                np.zeros(emb_dim, dtype=bfloat16),  # q
                lw._wk_t,  # wk
                np.zeros(kv_dim, dtype=bfloat16),  # k
                lw._wv_t,  # wv
                np.zeros(kv_dim, dtype=bfloat16),  # v
                rope_lut_pos_dummy,  # lut_q
                rope_lut_k_dummy,  # lut_k
                np.zeros(emb_dim, dtype=bfloat16),  # q_roped
                np.zeros(kv_dim, dtype=bfloat16),  # k_roped
                output_indices=[8, 11, 12],
                static_input_indices={1, 3, 5, 7},
                intermediate_indices={2, 4, 6, 8, 11, 12},
                bo_key=f"rms_gemv_rope_L{layer_idx}",
            )
            # o_gemv_ffn: allocate + write weights
            cache.load_and_run(
                "o_gemv_ffn",
                _OGEMV_FFN_BACKEND,
                lw._wo_t,  # wo
                np.zeros(emb_dim, dtype=bfloat16),  # attn_out
                np.zeros(emb_dim, dtype=bfloat16),  # proj
                np.zeros(emb_dim, dtype=bfloat16),  # x_residual
                np.zeros(emb_dim, dtype=bfloat16),  # res1
                lw.ffn_norm.reshape(emb_dim).astype(bfloat16),  # ffn_norm_w
                np.zeros(emb_dim, dtype=bfloat16),  # normed2
                lw._wgate_t,  # wgate
                np.zeros(hidden_dim, dtype=bfloat16),  # gate
                lw._wup_t,  # wup
                np.zeros(hidden_dim, dtype=bfloat16),  # up
                np.zeros(hidden_dim, dtype=bfloat16),  # swiglu
                lw._wdown_t,  # wdown
                np.zeros(emb_dim, dtype=bfloat16),  # down
                np.zeros(emb_dim, dtype=bfloat16),  # output
                output_indices=[14],
                static_input_indices={0, 5, 7, 9, 12},
                intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
                bo_key=f"o_gemv_ffn_L{layer_idx}",
            )

        # 2. LM Head GEMV weights (8 partitions)
        weights._lm_weight_parts_gemv = []
        for p in range(n_lm_partitions):
            n_start = p * n_part
            n_end = min(n_start + n_part, vocab_size)
            w = np.zeros((n_part, emb_dim), dtype=bfloat16)
            w[: n_end - n_start, :] = np.ascontiguousarray(
                weights.lm_head[n_start:n_end, :]
            ).astype(bfloat16)
            weights._lm_weight_parts_gemv.append(w)

        _LM_GEMV_BACKEND = {
            "output_format": "elf",
            "instance_name": "lm_head_gemv",
            **_GEMV_K2048_BACKEND,
        }
        lm_inputs = [np.zeros(emb_dim, dtype=bfloat16)]
        for p in range(n_lm_partitions):
            lm_inputs.append(weights._lm_weight_parts_gemv[p])
            lm_inputs.append(np.zeros(n_part, dtype=bfloat16))
        cache.load_and_run(
            "lm_head_gemv",
            _LM_GEMV_BACKEND,
            *lm_inputs,
            output_indices=[2 + 2 * p for p in range(n_lm_partitions)],
            static_input_indices={1 + 2 * p for p in range(n_lm_partitions)},
            intermediate_indices={2 + 2 * p for p in range(n_lm_partitions)},
        )

        weights._decode_weights_preloaded_to_bos = True
        total_mb = (
            config.n_layers
            * (
                emb_dim * emb_dim * 2
                + kv_dim * emb_dim * 2 * 2  # wq, wk, wv
                + emb_dim * emb_dim * 2  # wo
                + hidden_dim * emb_dim * 2 * 2
                + emb_dim * hidden_dim * 2  # w_gate, w_up, w_down
            )
            // 1024
            // 1024
        )
        print(f"  Pre-loaded {config.n_layers} layers + LM Head ({total_mb + 512}MB)")

    # --- Decode phase ---
    # Token 0 was generated by prefill (prefill_token)
    generated_tokens = [prefill_token]
    current_pos = prompt_len  # Next position to fill (after actual prompt)

    # Use the prefill_token as input to first decode step
    x_decode = weights.embed_table[prefill_token].astype(bfloat16)

    # NPU warmup: run full decode pass (blocks + LM Head) to warm up
    # NPU power state + XRT kernel contexts + instruction caches
    _LM_GEMV_BACKEND = {
        "output_format": "elf",
        "instance_name": "lm_head_gemv",
        **_GEMV_K2048_BACKEND,
    }
    x_warmup = x_decode.copy()
    for layer_idx in range(config.n_layers):
        x_warmup = run_decode_block(
            x_warmup,
            weights.layers[layer_idx],
            cache,
            config,
            k_cache[layer_idx],
            v_cache[layer_idx],
            current_pos,
            rope_lut_bf16,
        )
    # Also warm up LM Head
    x_normed_warmup = rms_norm(
        x_warmup.astype(np.float32).reshape(1, emb_dim),
        weights.final_norm.astype(np.float32),
    )
    lm_warmup = [x_normed_warmup.flatten().astype(bfloat16)]
    for p in range(n_lm_partitions):
        lm_warmup.append(weights._lm_weight_parts_gemv[p])
        lm_warmup.append(np.zeros(n_part, dtype=bfloat16))
    cache.load_and_run(
        "lm_head_gemv",
        _LM_GEMV_BACKEND,
        *lm_warmup,
        output_indices=[2 + 2 * p for p in range(n_lm_partitions)],
        static_input_indices={1 + 2 * p for p in range(n_lm_partitions)},
        intermediate_indices={2 + 2 * p for p in range(n_lm_partitions)},
    )

    print(f"\nDecoding {n_tokens} tokens (token 1 to {n_tokens})...")
    t_decode_start = time.time()

    for token_idx in range(n_tokens):
        t_token_start = time.perf_counter()

        # Run 16 transformer blocks
        x = x_decode.copy()
        for layer_idx in range(config.n_layers):
            x = run_decode_block(
                x,
                weights.layers[layer_idx],
                cache,
                config,
                k_cache[layer_idx],
                v_cache[layer_idx],
                current_pos,
                rope_lut_bf16,
            )

        # Final RMSNorm (CPU)
        x_normed = rms_norm(
            x.astype(np.float32).reshape(1, emb_dim),
            weights.final_norm.astype(np.float32),
        )

        # LM Head (NPU — 8-partition GEMV, single XRT call)
        _LM_GEMV_BACKEND = {
            "output_format": "elf",
            "instance_name": "lm_head_gemv",
            **_GEMV_K2048_BACKEND,
        }
        x_lm = x_normed.flatten().astype(bfloat16)
        lm_inputs = [x_lm]
        lm_output_indices = []
        for p in range(n_lm_partitions):
            lm_inputs.append(weights._lm_weight_parts_gemv[p])
            lm_inputs.append(np.zeros(n_part, dtype=bfloat16))
            lm_output_indices.append(2 + 2 * p)
        lm_results = cache.load_and_run(
            "lm_head_gemv",
            _LM_GEMV_BACKEND,
            *lm_inputs,
            output_indices=lm_output_indices,
            static_input_indices={1 + 2 * p for p in range(n_lm_partitions)},
            intermediate_indices={2 + 2 * p for p in range(n_lm_partitions)},
        )
        # Assemble logits from partitions
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        for p in range(n_lm_partitions):
            n_start = p * n_part
            n_end = min(n_start + n_part, vocab_size)
            logits[0, n_start:n_end] = lm_results[2 + 2 * p][: n_end - n_start].astype(
                np.float32
            )
        next_token = int(np.argmax(logits[0]))

        t_token = time.perf_counter() - t_token_start

        generated_tokens.append(next_token)
        current_pos += 1
        x_decode = weights.embed_table[next_token].astype(bfloat16)

        if profile:
            # Token 0 = prefill output. Decode tokens start at 1.
            print(
                f"  Token {token_idx + 1}: id={next_token}, time={t_token*1000:.0f}ms"
            )

    t_decode = time.time() - t_decode_start

    print(f"\nGenerated {n_tokens} tokens in {t_decode:.2f}s")
    print(f"Tokens/second: {n_tokens / t_decode:.2f}")
    print(f"Time/token: {t_decode / n_tokens * 1000:.0f}ms")

    return generated_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLAMA-3.2-1B Decode")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
    )
    args = parser.parse_args()

    # External .o files are compiled from source by compile_decode_kernels()
    # via compile_all_external_kernels(). No manual copying needed.
    config = LlamaConfig()

    cache_dir = "decode_kernel_cache"
    cache = KernelCache(cache_dir, verbose=args.verbose)

    if not args.run_only:
        compile_decode_kernels(cache, config)

    if args.compile_only:
        sys.exit(0)

    if args.run_only:
        cache.load_manifest()

    # Load weights and tokenizer
    print("\nLoading weights...")
    weights = load_weights("meta-llama/Llama-3.2-1B")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    prompt_tokens = tokenizer.encode(args.prompt)
    # Pad to seq_len=2048
    seq_len = 2048
    if len(prompt_tokens) < seq_len:
        prompt_tokens = prompt_tokens + [tokenizer.eos_token_id] * (
            seq_len - len(prompt_tokens)
        )

    rope_lut_bf16 = generate_rope_lut(
        config=config,
        seq_len=seq_len + args.n_tokens,
    ).astype(bfloat16)

    generated = generate(
        prompt_tokens,
        weights,
        config,
        cache,
        rope_lut_bf16,
        n_tokens=args.n_tokens,
        profile=args.profile,
        verify=args.verify,
    )

    # Decode tokens
    print(f"\n{'='*60}")
    print(f"Generated text:")
    print(f"{'='*60}")
    all_tokens = prompt_tokens[: len(tokenizer.encode(args.prompt))] + generated
    print(tokenizer.decode(all_tokens))
